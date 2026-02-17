# mace/modules/repulsion.py
###########################################################################################
# Pair repulsion terms computed on MACE graph edges:
#   - ZBL repulsion
#   - C12 / r^12 repulsion
#
# TorchScript-safe switch module supports modes (int):
#   0 -> off
#   1 -> zbl
#   2 -> r12
#   3 -> both
#
###########################################################################################

from __future__ import annotations

import torch
from mace.tools.scatter import scatter_sum

KE = 14.3996454784255


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
    """
    Stable polynomial cutoff: (1 - r/r_max)^p for r<=r_max else 0.
    TorchScript-friendly.
    """
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
    """
    Split each edge energy 50/50 onto src and dst, returning per-node energies [n_nodes].

    If the edge list is directed and contains both i->j and j->i, then each pair is double-counted.
    Set assume_directed_double=True to multiply by 0.5.
    """
    src = edge_index[0]
    dst = edge_index[1]

    if assume_directed_double:
        V_edge = 0.5 * V_edge  # avoid double-counting in directed graphs

    half = 0.5 * V_edge
    node_E = scatter_sum(half, src, dim=0, dim_size=n_nodes) + scatter_sum(
        half, dst, dim=0, dim_size=n_nodes
    )
    return node_E


class ZBLRepulsion(torch.nn.Module):
    """
    ZBL repulsion computed on edges.

    V_ZBL(r) = KE * Zi*Zj / r * phi(r/a)
    a = 0.88534 * a0 / (Zi^0.23 + Zj^0.23)
    phi(x) = sum_k a_k exp(-b_k x)
    """

    def __init__(
        self,
        p: int,
        r_min: float = 0.2,
        apply_cutoff: bool = True,
        assume_directed_double: bool = True,
    ):
        super().__init__()
        self.p = int(p)
        self.r_min = float(r_min)
        self.apply_cutoff = bool(apply_cutoff)
        self.assume_directed_double = bool(assume_directed_double)

        # ZBL constants (buffers for TorchScript)
        self.register_buffer("zbl_a", torch.tensor([0.1818, 0.5099, 0.2802, 0.02817]))
        self.register_buffer("zbl_b", torch.tensor([3.2, 0.9423, 0.4029, 0.2016]))
        self.register_buffer("a0", torch.tensor(0.52917721092))  # Bohr radius in Angstrom

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
        lengths: torch.Tensor,             # [n_edges]
        node_attrs: torch.Tensor,          # [n_nodes, n_elements]
        edge_index: torch.Tensor,          # [2, n_edges]
        atomic_numbers: torch.Tensor,      # [n_elements] channel->Z
        r_max: torch.Tensor,               # scalar
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

        V = (KE * Zi * Zj) * (phi / r)

        cutoff = _poly_cutoff(r, r_max.to(dtype=dtype, device=device), self.p, self.apply_cutoff)
        V = V * cutoff

        n_nodes = node_attrs.shape[0]
        return _split_edge_energy_to_nodes(
            V_edge=V,
            edge_index=edge_index,
            n_nodes=n_nodes,
            assume_directed_double=self.assume_directed_double,
        )


class R12Repulsion(torch.nn.Module):
    """
    Pure r^-12 repulsion on edges:
      V_12(r) = c12 / r^12
    """

    def __init__(
        self,
        p: int,
        c12: float,
        r_min: float = 0.2,
        apply_cutoff: bool = True,
        assume_directed_double: bool = True,
    ):
        super().__init__()
        self.p = int(p)
        self.c12 = float(c12)  # eV * Ang^12
        self.r_min = float(r_min)
        self.apply_cutoff = bool(apply_cutoff)
        self.assume_directed_double = bool(assume_directed_double)

    def forward(
        self,
        lengths: torch.Tensor,             # [n_edges]
        node_attrs: torch.Tensor,          # [n_nodes, n_elements] (unused, kept for signature symmetry)
        edge_index: torch.Tensor,          # [2, n_edges]
        atomic_numbers: torch.Tensor,      # (unused)
        r_max: torch.Tensor,               # scalar
    ) -> torch.Tensor:
        r = torch.clamp(lengths, min=self.r_min)
        dtype = r.dtype
        device = r.device

        c12 = torch.tensor(self.c12, dtype=dtype, device=device)
        V = c12 / (r**12)

        cutoff = _poly_cutoff(r, r_max.to(dtype=dtype, device=device), self.p, self.apply_cutoff)
        V = V * cutoff

        n_nodes = node_attrs.shape[0]
        return _split_edge_energy_to_nodes(
            V_edge=V,
            edge_index=edge_index,
            n_nodes=n_nodes,
            assume_directed_double=self.assume_directed_double,
        )


class PairRepulsionSwitch(torch.nn.Module):
    """
    TorchScript-safe router: branches on integer mode.

    mode:
      0 -> off
      1 -> zbl
      2 -> r12
      3 -> both
    """

    def __init__(self, kinds, zbl=None, r12=None, mode: str = "sum", **kwargs):
        super().__init__()
        self.mode = mode
        self.zbl = zbl
        self.r12 = r12

        if self.mode < 0 or self.mode > 3:
            # In eager mode, fail fast
            raise ValueError("PairRepulsionSwitch mode must be 0(off),1(zbl),2(r12),3(both).")

    def forward(
        self,
        lengths: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
        r_max: torch.Tensor,
    ) -> torch.Tensor:
        n_nodes = node_attrs.shape[0]
        out = torch.zeros((n_nodes,), dtype=lengths.dtype, device=lengths.device)

        # Off
        if self.mode == 0:
            return out

        # ZBL only
        if self.mode == 1:
            return self.zbl(lengths, node_attrs, edge_index, atomic_numbers, r_max)

        # r12 only
        if self.mode == 2:
            return self.r12(lengths, node_attrs, edge_index, atomic_numbers, r_max)

        # both
        # (self.mode == 3)
        return self.zbl(lengths, node_attrs, edge_index, atomic_numbers, r_max) + self.r12(
            lengths, node_attrs, edge_index, atomic_numbers, r_max
        )

