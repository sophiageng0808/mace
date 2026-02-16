###########################################################################################
# Radial basis and cutoff
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################                                                                                          

import logging
import os

import ase
import numpy as np
import torch
from e3nn.util.jit import compile_mode

from mace.tools.scatter import scatter_sum


@compile_mode("script")
class BesselBasis(torch.nn.Module):
    """
    Equation (7)

    Modes (set via env var MACE_BESSEL_MODE):
    - "sine"          : sin(k r) / r
    - "cos_over_r"    : cos(k r) / r
    - "cosm1_over_r"  : (cos(k r) - 1) / r
    - "cos"           : cos(k r)

    NOTE (TorchScript-safe):
    - We resolve env/config ONCE in __init__ and store an integer mode_id buffer.
    - forward() uses only tensor ops + integer branching (no os.getenv, no dynamic attrs).
    """

    def __init__(
        self,
        r_max: float,
        num_basis: int = 8,
        trainable: bool = False,
        use_cosine: bool = False,
        radial_mode: str = "sine",
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__()
        self.use_cosine = use_cosine
        self.radial_mode = radial_mode

        mode = os.getenv("MACE_BESSEL_MODE", "").strip().lower()

        if mode == "":
            rm = str(radial_mode).strip().lower() if radial_mode is not None else ""
            if rm not in ("", "sine", "sin"):
                mode = rm
            else:
                mode = "cosm1_over_r" if bool(use_cosine) else "sine"

                              
        if mode in ("sin", "sine"):
            mode = "sine"
        elif mode in ("cos_over_r", "cosoverr", "cos/r"):
            mode = "cos_over_r"
        elif mode in ("cosm1_over_r", "cosminus1_over_r", "cosm1/r", "cos-1_over_r"):
            mode = "cosm1_over_r"
        elif mode in ("cos", "cosine"):
            mode = "cos"
        else:
            raise ValueError(f"Unknown BesselBasis mode: {mode}")

                                                                         
        mode_map = {"sine": 0, "cos_over_r": 1, "cosm1_over_r": 2, "cos": 3}
        self.register_buffer("mode_id", torch.tensor(mode_map[mode], dtype=torch.int64))

        print(f"[BesselBasis __init__] mode={mode} (id={mode_map[mode]})", flush=True)

        cos_family_default = (mode != "sine")

        if cos_family_default:
            bessel_weights = (np.pi / r_max) * (
                torch.arange(num_basis, dtype=torch.get_default_dtype()) + 0.5
            )
        else:
            bessel_weights = (np.pi / r_max) * torch.linspace(
                start=1.0,
                end=float(num_basis),
                steps=num_basis,
                dtype=torch.get_default_dtype(),
            )

        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.get_default_dtype()))
        self.register_buffer(
            "prefactor",
            torch.tensor(np.sqrt(2.0 / r_max), dtype=torch.get_default_dtype()),
        )
        self.register_buffer("eps", torch.tensor(float(eps), dtype=torch.get_default_dtype()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)

                                             
        x_safe = torch.clamp(x, min=self.eps)
        arg = x_safe * self.bessel_weights

        mid = int(self.mode_id.item())
        if mid == 0:        
            out = torch.sin(arg) / x_safe
        elif mid == 1:              
            out = torch.cos(arg) / x_safe
        elif mid == 2:             
            out = (torch.cos(arg) - 1.0) / x_safe
        else:                 
            out = torch.cos(arg)

        return self.prefactor * out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={len(self.bessel_weights)}, "
            f"trainable={self.bessel_weights.requires_grad}, use_cosine={self.use_cosine}, radial_mode={self.radial_mode})"
        )


@compile_mode("script")
class ChebychevBasis(torch.nn.Module):
    """
    Equation (7)
    """

    def __init__(self, r_max: float, num_basis: int = 8):
        super().__init__()
        self.register_buffer(
            "n",
            torch.arange(1, num_basis + 1, dtype=torch.get_default_dtype()).unsqueeze(0),
        )
        self.num_basis = num_basis
        self.r_max = r_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:            
        x = x.repeat(1, self.num_basis)
        n = self.n.repeat(len(x), 1)
        return torch.special.chebyshev_polynomial_t(x, n)

    def __repr__(self):
        return f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={self.num_basis})"


@compile_mode("script")
class GaussianBasis(torch.nn.Module):
    """
    Gaussian basis functions
    """

    def __init__(self, r_max: float, num_basis: int = 128, trainable: bool = False):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:            
        x = x - self.gaussian_weights
        return torch.exp(self.coeff * torch.pow(x, 2))


@compile_mode("script")
class PolynomialCutoff(torch.nn.Module):
    """Polynomial cutoff function that goes from 1 to 0 as x goes from 0 to r_max.
    Equation (8)
    """

    p: torch.Tensor
    r_max: torch.Tensor

    def __init__(self, r_max: float, p: int = 6):
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
class CosineCutoff(torch.nn.Module):
    """Behler–Parrinello cosine cutoff:
    f(r) = 0.5 * (cos(pi*r/r_max) + 1) for r <= r_max else 0
    """
    r_max: torch.Tensor

    def __init__(self, r_max: float):
        super().__init__()
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.get_default_dtype()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r_max = self.r_max
        xr = x / r_max
        f = 0.5 * (torch.cos(torch.pi * xr) + 1.0)
        return torch.where(x < r_max, f, torch.zeros_like(x))

    def __repr__(self):
        return f"{self.__class__.__name__}(r_max={self.r_max})"


@compile_mode("script")
class PolyOverRnEnvelopeCutoff(torch.nn.Module):
    r_max: torch.Tensor
    p: torch.Tensor
    n: torch.Tensor
    eps: torch.Tensor

    """Envelope-style cutoff:
       f_mod(r) = ((r_max - r)^p) / (r^n) for r <= r_max else 0
       eps prevents singularity at r=0.
    """

    def __init__(self, r_max: float, p: int = 6, n: int = 1, eps: float = 1e-3):
        super().__init__()
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.get_default_dtype()))
        self.register_buffer("p", torch.tensor(p, dtype=torch.int))
        self.register_buffer("n", torch.tensor(n, dtype=torch.int))
        self.register_buffer("eps", torch.tensor(float(eps), dtype=torch.get_default_dtype()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r_max = self.r_max
                                
        dr = torch.clamp(r_max - x, min=0.0)
                               
        denom = torch.pow(torch.clamp(x, min=self.eps), self.n.to(torch.int))
        f = torch.pow(dr, self.p.to(torch.int)) / denom
        return torch.where(x < r_max, f, torch.zeros_like(x))

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, n={self.n}, eps={self.eps}, r_max={self.r_max})"


def make_cutoff_fn(
    cutoff_kind: str,
    r_max: float,
    p: int,
    env_n: int = 1,
    env_eps: float = 1e-3,
) -> torch.nn.Module:
    """
    Factory for cutoff functions used by radial embedding.

    cutoff_kind options (case-insensitive):
      - "polynomial" or "poly"     -> PolynomialCutoff(r_max, p)
      - "cosine" or "bp" or "cos"  -> CosineCutoff(r_max)
      - "poly_over_rn" or "envelope" or "rn"
                                 -> PolyOverRnEnvelopeCutoff(r_max, p, n=env_n, eps=env_eps)
    """
    ck = str(cutoff_kind).strip().lower()
    if ck in ("polynomial", "poly", "standard"):
        return PolynomialCutoff(r_max=r_max, p=p)
    if ck in ("cosine", "bp", "cos"):
        return CosineCutoff(r_max=r_max)
    if ck in ("poly_over_rn", "envelope", "rn"):
        return PolyOverRnEnvelopeCutoff(r_max=r_max, p=p, n=env_n, eps=env_eps)
    raise ValueError(f"Unknown cutoff_kind={cutoff_kind!r}")


@compile_mode("script")
class ZBLBasis(torch.nn.Module):
    """Implementation of the Ziegler-Biersack-Littmark (ZBL) potential
    with a polynomial cutoff envelope.
    """

    p: torch.Tensor

    def __init__(self, p: int = 6, trainable: bool = False, **kwargs):
        super().__init__()
        if "r_max" in kwargs:
            logging.warning("r_max is deprecated. r_max is determined from the covalent radii.")

        self.register_buffer(
            "c",
            torch.tensor([0.1818, 0.5099, 0.2802, 0.02817], dtype=torch.get_default_dtype()),
        )
        self.register_buffer("p", torch.tensor(p, dtype=torch.int))
        self.register_buffer(
            "covalent_radii",
            torch.tensor(ase.data.covalent_radii, dtype=torch.get_default_dtype()),
        )
        if trainable:
            self.a_exp = torch.nn.Parameter(torch.tensor(0.300, requires_grad=True))
            self.a_prefactor = torch.nn.Parameter(torch.tensor(0.4543, requires_grad=True))
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
        node_atomic_numbers = atomic_numbers[torch.argmax(node_attrs, dim=1)].unsqueeze(-1)
        Z_u = node_atomic_numbers[sender].to(torch.int64)
        Z_v = node_atomic_numbers[receiver].to(torch.int64)
        a = self.a_prefactor * 0.529 / (torch.pow(Z_u, self.a_exp) + torch.pow(Z_v, self.a_exp))
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
        trainable: bool = False,
    ):
        super().__init__()
        self.register_buffer("q", torch.tensor(q, dtype=torch.get_default_dtype()))
        self.register_buffer("p", torch.tensor(p, dtype=torch.get_default_dtype()))
        self.register_buffer("a", torch.tensor(a, dtype=torch.get_default_dtype()))
        self.register_buffer(
            "covalent_radii",
            torch.tensor(ase.data.covalent_radii, dtype=torch.get_default_dtype()),
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
        node_atomic_numbers = atomic_numbers[torch.argmax(node_attrs, dim=1)].unsqueeze(-1)
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
        return f"{self.__class__.__name__}(a={self.a:.4f}, q={self.q:.4f}, p={self.p:.4f})"


@compile_mode("script")
class SoftTransform(torch.nn.Module):
    """
    Tanh-based smooth transformation:
        T(x) = p1 + (x - p1)*0.5*[1 + tanh(alpha*(x - m))],
    which smoothly transitions from ~p1 for x << p1 to ~x for x >> r0.
    """

    def __init__(self, alpha: float = 4.0, trainable: bool = False):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.get_default_dtype()))
        if trainable:
            self.alpha = torch.nn.Parameter(self.alpha.clone())
        self.register_buffer(
            "covalent_radii",
            torch.tensor(ase.data.covalent_radii, dtype=torch.get_default_dtype()),
        )

    def compute_r_0(
        self,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        node_atomic_numbers = atomic_numbers[torch.argmax(node_attrs, dim=1)].unsqueeze(-1)
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
