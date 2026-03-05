###########################################################################################
# Radial basis and cutoff
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging
import math

import ase
import numpy as np
import torch
from e3nn.util.jit import compile_mode
from typing import Optional

from mace.tools.scatter import scatter_sum

KE = 14.3996454784255


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., 1]
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
        v_edges = (14.3996 * Z_u * Z_v) / x * phi
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
    """
    ZBL repulsion computed on edges:
      V_ZBL(r) = KE * Zi*Zj / r * phi(r/a)
    """

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

        r = torch.clamp(lengths, min=self.r_min)
        dtype = r.dtype
        device = r.device

        Z_node = _node_atomic_numbers_from_onehot(node_attrs, atomic_numbers, dtype, device)
        Zi = Z_node[src]
        Zj = Z_node[dst]

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
    """
    Pure r^-12 repulsion on edges: V_12(r) = c12 / r^12.
    """

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
        r = torch.clamp(lengths, min=self.r_min)
        dtype = r.dtype
        device = r.device

        c12 = torch.tensor(self.c12, dtype=dtype, device=device)
        V = c12 / (r**12)

        cutoff = _poly_cutoff(r, r_max.to(dtype=dtype, device=device), self.p, self.apply_cutoff)
        V = V * cutoff

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
    TorchScript-safe router:
      - mode 0: sum of selected kinds (from kinds list)
      - mode 1: zbl only
      - mode 2: r12 only
      - mode 3: both (zbl + r12)
    """

    def __init__(
        self,
        kinds: Optional[list],
        zbl: ZBLRepulsion,
        r12: R12Repulsion,
        mode: int = 0,
    ):
        super().__init__()
        if kinds is None:
            kinds = ["zbl"]
        self.kinds = [k for k in kinds if k]
        self.use_zbl = "zbl" in self.kinds
        self.use_r12 = "r12" in self.kinds
        self.mode = int(mode)
        self.zbl = zbl
        self.r12 = r12
        if self.mode < 0 or self.mode > 3:
            raise ValueError(
                "PairRepulsionSwitch mode must be 0(sum),1(zbl),2(r12),3(both)."
            )

    def forward(
        self,
        lengths: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
        r_max: torch.Tensor,
        node_feats: Optional[torch.Tensor] = None,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        V_edge = self.edge_energy(
            lengths=lengths,
            node_attrs=node_attrs,
            edge_index=edge_index,
            atomic_numbers=atomic_numbers,
            r_max=r_max,
        )
        n_nodes = node_attrs.shape[0]
        return _split_edge_energy_to_nodes(
            V_edge=V_edge,
            edge_index=edge_index,
            n_nodes=n_nodes,
            assume_directed_double=True,
        )

    def edge_energy(
        self,
        lengths: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
        r_max: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.zeros_like(lengths)

        if self.mode == 0:
            if self.use_zbl:
                out = out + self.zbl.edge_energy(
                    lengths, node_attrs, edge_index, atomic_numbers, r_max
                )
            if self.use_r12:
                out = out + self.r12.edge_energy(
                    lengths, node_attrs, edge_index, atomic_numbers, r_max
                )
        elif self.mode == 1:
            out = out + self.zbl.edge_energy(
                lengths, node_attrs, edge_index, atomic_numbers, r_max
            )
        elif self.mode == 2:
            out = out + self.r12.edge_energy(
                lengths, node_attrs, edge_index, atomic_numbers, r_max
            )
        elif self.mode == 3:
            out = out + self.zbl.edge_energy(
                lengths, node_attrs, edge_index, atomic_numbers, r_max
            )
            out = out + self.r12.edge_energy(
                lengths, node_attrs, edge_index, atomic_numbers, r_max
            )

        return out


@compile_mode("script")
class EmbeddingConditionedPairRepulsion(torch.nn.Module):
    def __init__(
        self,
        base: PairRepulsionSwitch,
        embedding_dim: int,
        alpha_hidden_dim: int = 32,
        symmetric_pair_feat: bool = True,
        gate: str = "cosine",
        r_on: float = 0.6,
        r_cut: float = 1.2,
        alpha_min: Optional[float] = 0.1,
        alpha_max: Optional[float] = 10.0,
    ):
        super().__init__()
        self.base = base
        self.embedding_dim = int(embedding_dim)
        self.alpha_hidden_dim = int(alpha_hidden_dim)
        self.symmetric_pair_feat = bool(symmetric_pair_feat)
        self.gate = str(gate)
        self.r_on = float(r_on)
        self.r_cut = float(r_cut)
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * self.embedding_dim, self.alpha_hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.alpha_hidden_dim, 1),
        )
        self.softplus = torch.nn.Softplus()
        self.requires_node_feats = True
        self.embedding_conditioned = True
        # Expose base attributes for config extraction
        self.kinds = getattr(base, "kinds", None)
        self.mode = getattr(base, "mode", 0)
        self.zbl = getattr(base, "zbl", None)
        self.r12 = getattr(base, "r12", None)

        if self.gate not in ("cosine", "none"):
            raise ValueError("pair_repulsion_gate must be 'cosine' or 'none'.")
        if self.gate == "cosine" and self.r_cut <= 0:
            raise ValueError("pair_repulsion_r_cut must be > 0 for cosine gate.")
        if self.gate == "cosine" and self.r_on >= self.r_cut:
            raise ValueError("pair_repulsion_r_on must be < pair_repulsion_r_cut.")
        if self.alpha_min is not None and self.alpha_max is not None:
            if float(self.alpha_min) >= float(self.alpha_max):
                raise ValueError("pair_repulsion_alpha_min must be < alpha_max.")

    def _pair_features(
        self, h_i: torch.Tensor, h_j: torch.Tensor
    ) -> torch.Tensor:
        if self.symmetric_pair_feat:
            return torch.cat([h_i + h_j, (h_i - h_j).abs()], dim=-1)
        return torch.cat([h_i, h_j], dim=-1)

    def edge_gate(self, r: torch.Tensor) -> torch.Tensor:
        if self.gate == "none":
            return torch.ones_like(r)
        if self.r_on <= 0.0:
            x = torch.clamp(r / self.r_cut, 0.0, 1.0)
            g = 0.5 * (torch.cos(math.pi * x) + 1.0)
            return torch.where(r < self.r_cut, g, torch.zeros_like(r))
        g = torch.ones_like(r)
        mid = (r > self.r_on) & (r < self.r_cut)
        x = (r[mid] - self.r_on) / (self.r_cut - self.r_on)
        g[mid] = 0.5 * (torch.cos(math.pi * x) + 1.0)
        g = torch.where(r >= self.r_cut, torch.zeros_like(r), g)
        return g

    def edge_alpha(
        self, node_feats: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        src = edge_index[0]
        dst = edge_index[1]
        h_i = node_feats[src]
        h_j = node_feats[dst]
        pair = self._pair_features(h_i, h_j)
        alpha = self.softplus(self.mlp(pair)).squeeze(-1)
        if self.alpha_min is None or self.alpha_max is None:
            return alpha
        return alpha.clamp(min=float(self.alpha_min), max=float(self.alpha_max))

    def forward(
        self,
        lengths: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
        r_max: torch.Tensor,
        node_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if node_feats is None:
            raise ValueError("Embedding-conditioned repulsion requires node_feats.")

        V_edge = self.base.edge_energy(
            lengths=lengths,
            node_attrs=node_attrs,
            edge_index=edge_index,
            atomic_numbers=atomic_numbers,
            r_max=r_max,
        )
        alpha = self.edge_alpha(node_feats=node_feats, edge_index=edge_index)
        gate = self.edge_gate(lengths)
        V_edge = (
            V_edge
            * alpha.to(dtype=V_edge.dtype, device=V_edge.device)
            * gate.to(dtype=V_edge.dtype, device=V_edge.device)
        )

        n_nodes = node_attrs.shape[0]
        return _split_edge_energy_to_nodes(
            V_edge=V_edge,
            edge_index=edge_index,
            n_nodes=n_nodes,
            assume_directed_double=True,
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


class PairRepulsionSoftTransform(SoftTransform):
    """Backward-compatible soft distance transform for legacy checkpoints."""

    def forward(
        self,
        lengths: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return super().forward(lengths, node_attrs, edge_index, atomic_numbers)


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
