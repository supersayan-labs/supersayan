# convert.py (unchanged API – minor clean‑ups and docstrings)
"""Utility to convert PyTorch modules to their Supersayan counterparts.

Only cosmetic changes were made here; logic is identical to the previous
version. The file is included for completeness because the user explicitly
asked for it.
"""
from __future__ import annotations

import logging
from abc import ABC
from enum import Enum
from typing import Dict, List, Optional, Type, Union, cast

import torch
import torch.nn as nn
from supersayan.core.keygen import generate_secret_key
from supersayan.core.encryption import decrypt, encrypt
from supersayan.nn.layers import Conv2d, Linear

logger = logging.getLogger(__name__)

# Map PyTorch ⇒ Supersayan layers ------------------------------------------------
LAYER_MAPPING = {nn.Linear: Linear, nn.Conv2d: Conv2d}


class ModelType(Enum):
    PURE = "pure"
    HYBRID = "hybrid"


class SupersayanModel(nn.Module, ABC):
    """Base‑class offering *pure* FHE and *hybrid* modes."""

    def __init__(
        self,
        torch_model: nn.Module,
        model_type: ModelType = ModelType.PURE,
        fhe_modules: Optional[Union[List[str], List[Type[nn.Module]]]] = None,
    ) -> None:
        super().__init__()
        self.original_model = torch_model
        self.model_type = model_type

        if model_type == ModelType.HYBRID and not fhe_modules:
            raise ValueError("fhe_modules must be provided for hybrid models")

        # Decide whether *fhe_modules* is a list of names or of types
        self.using_module_types = bool(fhe_modules and all(isinstance(m, type) for m in fhe_modules))

        if self.using_module_types:
            self.fhe_module_types = cast(List[Type[nn.Module]], fhe_modules)
            self.fhe_module_names: List[str] = []
        else:
            self.fhe_module_names = cast(List[str], fhe_modules) if fhe_modules else []
            self.fhe_module_types = []

        # ------------------------------------------------------------------
        # Build *modules_dict* (direct children only)
        # ------------------------------------------------------------------
        self.modules_dict = nn.ModuleDict()

        if model_type == ModelType.PURE:
            for name, module in torch_model.named_children():
                self.modules_dict[name] = self._convert_module(module)
            self.fhe_module_names = list(self.modules_dict.keys())
        else:
            # HYBRID – allow selection by type *or* by dotted‑name
            all_named = dict(torch_model.named_modules())

            if self.using_module_types:
                for full_name, module in torch_model.named_modules():
                    if any(isinstance(module, t) for t in self.fhe_module_types):
                        self.fhe_module_names.append(full_name)
                for name, module in torch_model.named_children():
                    self.modules_dict[name] = (
                        self._convert_module(module) if name in self.fhe_module_names else module
                    )
            else:
                missing = [n for n in self.fhe_module_names if n not in all_named]
                if missing:
                    raise ValueError(f"module(s) not found: {missing}")
                for name, module in torch_model.named_children():
                    self.modules_dict[name] = (
                        self._convert_module(module) if name in self.fhe_module_names else module
                    )

            # Flatten nested FHE modules so they can be uploaded individually
            for full_name in list(self.fhe_module_names):
                if full_name not in self.modules_dict:
                    safe = full_name.replace(".", "_")
                    self.modules_dict[safe] = self._convert_module(all_named[full_name])

            # Normalise names (dotted → underscored)
            self.fhe_module_names = [n.replace(".", "_") for n in self.fhe_module_names]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _convert_module(self, module: nn.Module) -> nn.Module:
        module_type = type(module)
        if module_type in LAYER_MAPPING:
            cls = LAYER_MAPPING[module_type]
            if isinstance(module, nn.Linear):
                out = cls(module.in_features, module.out_features, bias=module.bias is not None)
                out.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    out.bias.data.copy_(module.bias.data)
                return out
            if isinstance(module, nn.Conv2d):
                out = cls(
                    module.in_channels,
                    module.out_channels,
                    module.kernel_size,
                    module.stride,
                    module.padding,
                    module.dilation,
                    module.groups,
                    bias=module.bias is not None,
                )
                out.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    out.bias.data.copy_(module.bias.data)
                return out
        if isinstance(module, nn.Sequential):
            return nn.Sequential(*(self._convert_module(m) for m in module))
        raise ValueError(f"module type {module_type} not supported")

    # ------------------------------------------------------------------
    # Pure‑FHE forward – hybrid handled in SupersayanClient
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.model_type == ModelType.PURE:
            key = generate_secret_key()
            enc = encrypt(x, key)
            for name in self.fhe_module_names:
                enc = self.modules_dict[name](enc)
            return torch.tensor(decrypt(enc, key), dtype=x.dtype, device=x.device)
        raise NotImplementedError("hybrid forward is implemented by SupersayanClient")
