
from __future__ import annotations

import torch

from mace import modules
from mace.tools.scripts_utils import extract_config_mace_model


class PickleSafeMACE(torch.nn.Module):
    """Delegates to ``ScaleShiftMACE``; pickle state is config dict + weights only."""

    def __init__(self, source: torch.nn.Module | None = None):
        super().__init__()
        if source is not None:
            cfg = extract_config_mace_model(source)
            inner = modules.ScaleShiftMACE(**cfg)
            inner.load_state_dict(source.state_dict(), strict=True)
            self.add_module("_mace_inner", inner)

    def forward(self, *args, **kwargs):
        return self._mace_inner(*args, **kwargs)

    def __getattr__(self, name: str):
        if name == "_mace_inner":
            return super().__getattr__(name)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._mace_inner, name)

    def __getstate__(self):
        inner = self._mace_inner
        return {
            "cfg": extract_config_mace_model(inner),
            "sd": inner.state_dict(),
        }

    def __setstate__(self, state: dict):
        torch.nn.Module.__init__(self)
        inner = modules.ScaleShiftMACE(**state["cfg"])
        inner.load_state_dict(state["sd"], strict=True)
        self.add_module("_mace_inner", inner)
