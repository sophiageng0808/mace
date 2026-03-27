###########################################################################################
# Radial basis and cutoff
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging
from typing import Optional

import ase
import numpy as np
import torch
from e3nn.util.jit import compile_mode

from mace.tools.scatter import scatter_sum

KE = 14.3996454784255  # k_e in eV·Å; ZBL term uses k_e * Z_i * Z_j * φ(r/a) / r


@compile_mode("script")
class BesselBasis(torch.nn.Module):
    """
    Equation (7)
    """

    def __init__(self, r_max: float, num_basis=8, trainable=False):
        super().__init__()

        bessel_weights = (
            np.pi
            / r_max
            * torch.linspace(
                start=1.0,
                end=num_basis,
                steps=num_basis,
                dtype=torch.get_default_dtype(),
            )
        )
        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "prefactor",
            torch.tensor(np.sqrt(2.0 / r_max), dtype=torch.get_default_dtype()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., 1]
        numerator = torch.sin(self.bessel_weights * x)  # [..., num_basis]
        return self.prefactor * (numerator / x)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={len(self.bessel_weights)}, "
            f"trainable={self.bessel_weights.requires_grad})"
        )


@compile_mode("script")
class ChebychevBasis(torch.nn.Module):
    """
    Equation (7)
    """

    def __init__(self, r_max: float, num_basis=8):
        super().__init__()
        self.register_buffer(
            "n",
            torch.arange(1, num_basis + 1, dtype=torch.get_default_dtype()).unsqueeze(
                0
            ),
        )
        self.num_basis = num_basis
        self.r_max = r_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        x = x.repeat(1, self.num_basis)
        n = self.n.repeat(len(x), 1)
        return torch.special.chebyshev_polynomial_t(x, n)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={self.num_basis},"
        )


@compile_mode("script")
class GaussianBasis(torch.nn.Module):
    """
    Gaussian basis functions
    """

    def __init__(self, r_max: float, num_basis=128, trainable=False):
        super().__init__()
        gaussian_weights = torch.linspace(
            start=0.0, end=r_max, steps=num_basis, dtype=torch.get_default_dtype()
        )
        if trainable:
            self.gaussian_weights = torch.nn.Parameter(
                gaussian_weights, requires_grad=True
            )
        else:
            self.register_buffer("gaussian_weights", gaussian_weights)
        self.coeff = -0.5 / (r_max / (num_basis - 1)) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., 1]
        x = x - self.gaussian_weights
        return torch.exp(self.coeff * torch.pow(x, 2))


@compile_mode("script")
class PolynomialCutoff(torch.nn.Module):
    """Polynomial cutoff function that goes from 1 to 0 as x goes from 0 to r_max.
    Equation (8) -- TODO: from where?
    """

    p: torch.Tensor
    r_max: torch.Tensor

    def __init__(self, r_max: float, p=6):
        super().__init__()
        self.register_buffer("p", torch.tensor(p, dtype=torch.int))
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.calculate_envelope(x, self.r_max, self.p.to(torch.int))

    @staticmethod
    def calculate_envelope(
        x: torch.Tensor, r_max: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        r_over_r_max = x / r_max
        envelope = (
            1.0
            - ((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(r_over_r_max, p)
            + p * (p + 2.0) * torch.pow(r_over_r_max, p + 1)
            - (p * (p + 1.0) / 2) * torch.pow(r_over_r_max, p + 2)
        )
        return envelope * (x < r_max)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, r_max={self.r_max})"


@compile_mode("script")
class ZBLBasis(torch.nn.Module):
    """Implementation of the Ziegler-Biersack-Littmark (ZBL) potential
    with a polynomial cutoff envelope.
    """

    p: torch.Tensor

    def __init__(self, p=6, trainable=False, **kwargs):
        super().__init__()
        if "r_max" in kwargs:
            logging.warning(
                "r_max is deprecated. r_max is determined from the covalent radii."
            )

        # Pre-calculate the p coefficients for the ZBL potential
        self.register_buffer(
            "c",
            torch.tensor(
                [0.1818, 0.5099, 0.2802, 0.02817], dtype=torch.get_default_dtype()
            ),
        )
        self.register_buffer("p", torch.tensor(p, dtype=torch.int))
        self.register_buffer(
            "covalent_radii",
            torch.tensor(
                ase.data.covalent_radii,
                dtype=torch.get_default_dtype(),
            ),
        )
        if trainable:
            self.a_exp = torch.nn.Parameter(torch.tensor(0.300, requires_grad=True))
            self.a_prefactor = torch.nn.Parameter(
                torch.tensor(0.4543, requires_grad=True)
            )
        else:
            self.register_buffer("a_exp", torch.tensor(0.300))
            self.register_buffer("a_prefactor", torch.tensor(0.4543))

    def forward(
        self,
        x: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        node_atomic_numbers = atomic_numbers[torch.argmax(node_attrs, dim=1)].unsqueeze(
            -1
        )
        Z_u = node_atomic_numbers[sender].to(torch.int64)
        Z_v = node_atomic_numbers[receiver].to(torch.int64)
        a = (
            self.a_prefactor
            * 0.529
            / (torch.pow(Z_u, self.a_exp) + torch.pow(Z_v, self.a_exp))
        )
        r_over_a = x / a
        phi = (
            self.c[0] * torch.exp(-3.2 * r_over_a)
            + self.c[1] * torch.exp(-0.9423 * r_over_a)
            + self.c[2] * torch.exp(-0.4028 * r_over_a)
            + self.c[3] * torch.exp(-0.2016 * r_over_a)
        )
        v_edges = (KE * Z_u * Z_v) / x * phi
        r_max = self.covalent_radii[Z_u] + self.covalent_radii[Z_v]
        envelope = PolynomialCutoff.calculate_envelope(x, r_max, self.p)
        v_edges = 0.5 * v_edges * envelope
        V_ZBL = scatter_sum(v_edges, receiver, dim=0, dim_size=node_attrs.size(0))
        return V_ZBL.squeeze(-1)

    def __repr__(self):
        return f"{self.__class__.__name__}(c={self.c})"


def _node_atomic_numbers_from_onehot(
    node_attrs: torch.Tensor,
    atomic_numbers: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Atomic number Z per node from one-hot species and the model's element table."""
    return (
        node_attrs.to(dtype=dtype, device=device)
        * atomic_numbers.to(dtype=dtype, device=device)
    ).sum(dim=1)


def _poly_cutoff(
    r: torch.Tensor,
    r_max: torch.Tensor,
    p: int,
    apply_cutoff: bool,
) -> torch.Tensor:
    """Polynomial outer envelope (1 − r/r_max)^p used by ZBL and r^-12 at the neighbor cutoff."""
    if not apply_cutoff:
        return torch.ones_like(r)
    x = torch.clamp(r / r_max, 0.0, 1.0)
    return (1.0 - x).pow(int(p))


def _split_edge_energy_to_nodes(
    V_edge: torch.Tensor,
    edge_index: torch.Tensor,
    n_nodes: int,
    assume_directed_double: bool = True,
) -> torch.Tensor:
    """Scatter half of each edge energy to each endpoint so the bond is counted once per atom.

    If the graph lists both (i→j) and (j→i), ``assume_directed_double`` scales by 1/2 first
    so the physical pair energy is not doubled.
    """
    src = edge_index[0]
    dst = edge_index[1]

    if assume_directed_double:
        V_edge = 0.5 * V_edge

    half = 0.5 * V_edge
    node_E = scatter_sum(half, src, dim=0, dim_size=n_nodes) + scatter_sum(
        half, dst, dim=0, dim_size=n_nodes
    )
    return node_E


@compile_mode("script")
class ZBLRepulsion(torch.nn.Module):
    """ZBL repulsion on edges (k_e Z_i Z_j φ(r/a)/r)"""

    def __init__(
        self,
        p: int,
        scale: float = 1.0,
        r_min: float = 0.2,
        apply_cutoff: bool = True,
        assume_directed_double: bool = True,
    ):
        super().__init__()
        self.p = int(p)
        self.register_buffer(
            "scale", torch.tensor(float(scale), dtype=torch.get_default_dtype())
        )
        self.r_min = float(r_min)
        self.apply_cutoff = bool(apply_cutoff)
        self.assume_directed_double = bool(assume_directed_double)

        self.register_buffer("ke", torch.tensor(KE))
        self.register_buffer("zbl_a", torch.tensor([0.1818, 0.5099, 0.2802, 0.02817]))
        self.register_buffer("zbl_b", torch.tensor([3.2, 0.9423, 0.4029, 0.2016]))
        self.register_buffer("a0", torch.tensor(0.52917721092))

    def _screening_length(self, Zi: torch.Tensor, Zj: torch.Tensor) -> torch.Tensor:
        return 0.88534 * self.a0 / (Zi.pow(0.23) + Zj.pow(0.23))

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        a = self.zbl_a.to(dtype=x.dtype, device=x.device)
        b = self.zbl_b.to(dtype=x.dtype, device=x.device)
        return (
            a[0] * torch.exp(-b[0] * x)
            + a[1] * torch.exp(-b[1] * x)
            + a[2] * torch.exp(-b[2] * x)
            + a[3] * torch.exp(-b[3] * x)
        )

    def forward(
        self,
        lengths: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
        r_max: torch.Tensor,
    ) -> torch.Tensor:
        V = self.edge_energy(
            lengths=lengths,
            node_attrs=node_attrs,
            edge_index=edge_index,
            atomic_numbers=atomic_numbers,
            r_max=r_max,
        )
        n_nodes = node_attrs.shape[0]
        return _split_edge_energy_to_nodes(
            V_edge=V,
            edge_index=edge_index,
            n_nodes=n_nodes,
            assume_directed_double=self.assume_directed_double,
        )

    def edge_energy(
        self,
        lengths: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
        r_max: torch.Tensor,
    ) -> torch.Tensor:
        src = edge_index[0]
        dst = edge_index[1]
        dtype = lengths.dtype
        device = lengths.device
        # Clamp edge length below r_min so 1/r and r^-12 stay bounded when neighbors get very close.
        r = torch.maximum(
            lengths,
            torch.tensor(self.r_min, dtype=dtype, device=device),
        )

        Z_node = _node_atomic_numbers_from_onehot(node_attrs, atomic_numbers, dtype, device)
        Zi = Z_node[src]
        Zj = Z_node[dst]

        # Standard ZBL: screening length a(Zi,Zj), φ(r/a) sum of exponentials, Coulomb prefactor k_e Zi Zj / r.
        a = self._screening_length(Zi, Zj)
        x = r / a
        phi = self._phi(x)

        ke = self.ke.to(dtype=dtype, device=device)
        scale = self.scale.to(dtype=dtype, device=device)
        V = (ke * Zi * Zj) * (phi / r)
        V = V * scale
        cutoff = _poly_cutoff(r, r_max.to(dtype=dtype, device=device), self.p, self.apply_cutoff)
        V = V * cutoff
        return V


@compile_mode("script")
class R12Repulsion(torch.nn.Module):
    """r^-12 pair wall: c12/r^12 with outer (1−r/r_max)^p and optional inner cutoff."""

    def __init__(
        self,
        p: int,
        c12: float,
        r_min: float = 0.2,
        apply_cutoff: bool = True,
        assume_directed_double: bool = True,
        r12_cutoff: Optional[float] = None,
        r12_switch_width: Optional[float] = None,
    ):
        super().__init__()
        self.p = int(p)
        self.c12 = float(c12)
        self.r_min = float(r_min)
        self.apply_cutoff = bool(apply_cutoff)
        self.assume_directed_double = bool(assume_directed_double)
        self.register_buffer(
            "r12_cutoff",
            torch.tensor(-1.0 if r12_cutoff is None else float(r12_cutoff)),
        )
        self.register_buffer(
            "r12_switch_width",
            torch.tensor(0.0 if r12_switch_width is None else float(r12_switch_width)),
        )

    def forward(
        self,
        lengths: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
        r_max: torch.Tensor,
    ) -> torch.Tensor:
        V = self.edge_energy(
            lengths=lengths,
            node_attrs=node_attrs,
            edge_index=edge_index,
            atomic_numbers=atomic_numbers,
            r_max=r_max,
        )
        n_nodes = node_attrs.shape[0]
        return _split_edge_energy_to_nodes(
            V_edge=V,
            edge_index=edge_index,
            n_nodes=n_nodes,
            assume_directed_double=self.assume_directed_double,
        )

    def edge_energy(
        self,
        lengths: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
        r_max: torch.Tensor,
    ) -> torch.Tensor:
        dtype = lengths.dtype
        device = lengths.device
        # Same floor as ZBL so short edges do not explode r^-12.
        r = torch.maximum(
            lengths,
            torch.tensor(self.r_min, dtype=dtype, device=device),
        )

        c12 = torch.tensor(self.c12, dtype=dtype, device=device)
        V = c12 / (r**12)

        cutoff = _poly_cutoff(r, r_max.to(dtype=dtype, device=device), self.p, self.apply_cutoff)
        V = V * cutoff

        # Inner gate: damp repulsion below r12_cutoff (disabled when buffer ≤ 0). Smoothstep if width > 0.
        r12_cutoff = self.r12_cutoff.to(dtype=dtype, device=device)
        width = self.r12_switch_width.to(dtype=dtype, device=device)
        hard = (r <= r12_cutoff).to(dtype)
        t = torch.clamp((r12_cutoff - r) / torch.clamp(width, min=1e-12), 0.0, 1.0)
        smooth = t * t * (3.0 - 2.0 * t)
        cutoff_extra = torch.where(width > 0, smooth, hard)
        V = V * torch.where(r12_cutoff > 0, cutoff_extra, torch.ones_like(r))
        return V


@compile_mode("script")
class PairRepulsionSwitch(torch.nn.Module):
    """
    Single pair-repulsion term on edges: either ZBL or r^-12.
    """

    def __init__(
        self,
        kinds: Optional[list],
        zbl: Optional[ZBLRepulsion],
        r12: Optional[R12Repulsion],
    ):
        super().__init__()
        if kinds is None:
            kinds = ["zbl"]
        self.kinds = [k for k in kinds if k]
        if len(self.kinds) != 1 or self.kinds[0] not in ("zbl", "r12"):
            raise ValueError(
                "PairRepulsionSwitch requires pair_repulsion_kinds to be exactly "
                "['zbl'] or ['r12'], not combined terms."
            )
        # Only the active term is stored as a child module
        if self.kinds[0] == "zbl":
            if zbl is None or r12 is not None:
                raise ValueError("PairRepulsionSwitch(zbl): expected zbl module only.")
            self.zbl = zbl
        else:
            if r12 is None or zbl is not None:
                raise ValueError("PairRepulsionSwitch(r12): expected r12 module only.")
            self.r12 = r12

    def forward(
        self,
        lengths: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
        r_max: torch.Tensor,
    ) -> torch.Tensor:
        V_edge = self.edge_energy(
            lengths=lengths,
            node_attrs=node_attrs,
            edge_index=edge_index,
            atomic_numbers=atomic_numbers,
            r_max=r_max,
        )
        n_nodes = node_attrs.shape[0]
        if self.kinds[0] == "zbl":
            assume_directed_double = self.zbl.assume_directed_double
        else:
            assume_directed_double = self.r12.assume_directed_double
        return _split_edge_energy_to_nodes(
            V_edge=V_edge,
            edge_index=edge_index,
            n_nodes=n_nodes,
            assume_directed_double=assume_directed_double,
        )

    def edge_energy(
        self,
        lengths: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
        r_max: torch.Tensor,
    ) -> torch.Tensor:
        if self.kinds[0] == "zbl":
            return self.zbl.edge_energy(
                lengths, node_attrs, edge_index, atomic_numbers, r_max
            )
        return self.r12.edge_energy(
            lengths, node_attrs, edge_index, atomic_numbers, r_max
        )


@compile_mode("script")
class AgnesiTransform(torch.nn.Module):
    """Agnesi transform - see section on Radial transformations in
    ACEpotentials.jl, JCP 2023 (https://doi.org/10.1063/5.0158783).
    """

    def __init__(
        self,
        q: float = 0.9183,
        p: float = 4.5791,
        a: float = 1.0805,
        trainable=False,
    ):
        super().__init__()
        self.register_buffer("q", torch.tensor(q, dtype=torch.get_default_dtype()))
        self.register_buffer("p", torch.tensor(p, dtype=torch.get_default_dtype()))
        self.register_buffer("a", torch.tensor(a, dtype=torch.get_default_dtype()))
        self.register_buffer(
            "covalent_radii",
            torch.tensor(
                ase.data.covalent_radii,
                dtype=torch.get_default_dtype(),
            ),
        )
        if trainable:
            self.a = torch.nn.Parameter(torch.tensor(1.0805, requires_grad=True))
            self.q = torch.nn.Parameter(torch.tensor(0.9183, requires_grad=True))
            self.p = torch.nn.Parameter(torch.tensor(4.5791, requires_grad=True))

    def forward(
        self,
        x: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        node_atomic_numbers = atomic_numbers[torch.argmax(node_attrs, dim=1)].unsqueeze(
            -1
        )
        Z_u = node_atomic_numbers[sender].to(torch.int64)
        Z_v = node_atomic_numbers[receiver].to(torch.int64)
        r_0: torch.Tensor = 0.5 * (self.covalent_radii[Z_u] + self.covalent_radii[Z_v])
        r_over_r_0 = x / r_0
        return (
            1
            + (
                self.a
                * torch.pow(r_over_r_0, self.q)
                / (1 + torch.pow(r_over_r_0, self.q - self.p))
            )
        ).reciprocal_()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(a={self.a:.4f}, q={self.q:.4f}, p={self.p:.4f})"
        )


@compile_mode("script")
class SoftTransform(torch.nn.Module):
    """
    Tanh-based smooth transformation:
        T(x) = p1 + (x - p1)*0.5*[1 + tanh(alpha*(x - m))],
    which smoothly transitions from ~p1 for x << p1 to ~x for x >> r0.
    """

    def __init__(self, alpha: float = 4.0, trainable=False):
        """
        Args:
            p1 (float): Lower "clamp" point.
            alpha (float): Steepness; if None, defaults to ~6/(r0-p1).
            trainable (bool): Whether to make parameters trainable.
        """
        super().__init__()
        # Initialize parameters
        self.register_buffer(
            "alpha", torch.tensor(alpha, dtype=torch.get_default_dtype())
        )
        if trainable:
            self.alpha = torch.nn.Parameter(self.alpha.clone())
        self.register_buffer(
            "covalent_radii",
            torch.tensor(
                ase.data.covalent_radii,
                dtype=torch.get_default_dtype(),
            ),
        )

    def compute_r_0(
        self,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute r_0 based on atomic information.

        Args:
            node_attrs (torch.Tensor): Node attributes (one-hot encoding of atomic numbers).
            edge_index (torch.Tensor): Edge index indicating connections.
            atomic_numbers (torch.Tensor): Atomic numbers.

        Returns:
            torch.Tensor: r_0 values for each edge.
        """
        sender = edge_index[0]
        receiver = edge_index[1]
        node_atomic_numbers = atomic_numbers[torch.argmax(node_attrs, dim=1)].unsqueeze(
            -1
        )
        Z_u = node_atomic_numbers[sender].to(torch.int64)
        Z_v = node_atomic_numbers[receiver].to(torch.int64)
        r_0: torch.Tensor = self.covalent_radii[Z_u] + self.covalent_radii[Z_v]
        return r_0

    def forward(
        self,
        x: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:

        r_0 = self.compute_r_0(node_attrs, edge_index, atomic_numbers)
        p_0 = (3 / 4) * r_0
        p_1 = (4 / 3) * r_0
        m = 0.5 * (p_0 + p_1)
        alpha = self.alpha / (p_1 - p_0)
        s_x = 0.5 * (1.0 + torch.tanh(alpha * (x - m)))
        return p_0 + (x - p_0) * s_x

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha.item():.4f})"


class RadialMLP(torch.nn.Module):
    """
    Construct a radial MLP (Linear → LayerNorm → SiLU) stack
    given a list of channel sizes, following ESEN / FairChem.
    """

    def __init__(self, channels_list) -> None:
        super().__init__()

        modules = []
        in_channels = channels_list[0]

        for idx, out_channels in enumerate(channels_list[1:], start=1):
            modules.append(torch.nn.Linear(in_channels, out_channels, bias=True))
            in_channels = out_channels
            if idx < len(channels_list) - 1:
                modules.append(torch.nn.LayerNorm(out_channels))
                modules.append(torch.nn.SiLU())

        self.net = torch.nn.Sequential(*modules)
        self.hs = channels_list

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)
