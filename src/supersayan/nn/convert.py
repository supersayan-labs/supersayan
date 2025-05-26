from __future__ import annotations
import logging
from abc import ABC
from enum import Enum
from typing import List, Optional, Type, Union, cast

import torch
import torch.nn as nn
from supersayan.core.keygen import generate_secret_key
from supersayan.nn.layers import LAYER_MAPPING
from supersayan.core.encryption import encrypt_to_lwes, decrypt_from_lwes

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """
    Model type.

    PURE: All layers are executed in FHE.
    HYBRID: Only the specified layers are executed in FHE.
    """

    PURE = "pure"
    HYBRID = "hybrid"


class SupersayanModel(nn.Module, ABC):
    """Base class offering pure FHE and hybrid modes."""

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

        # Initialize module dict
        self.modules_dict = nn.ModuleDict()

        if model_type == ModelType.PURE:
            self.fhe_module_names = []

            for name, module in torch_model.named_children():
                self.modules_dict[name] = self._convert_module(module)
                self.fhe_module_names.append(name)
        else:

            fhe_module_names_set = set()

            # Get all named modules for lookup
            all_named_modules = dict(torch_model.named_modules())

            # Process the fhe_modules list
            for item in fhe_modules or []:
                if isinstance(item, str):
                    if item not in all_named_modules:
                        raise ValueError(f"Module '{item}' not found in model")
                    fhe_module_names_set.add(item)
                elif isinstance(item, type) and issubclass(item, nn.Module):
                    for full_name, module in all_named_modules.items():
                        if isinstance(module, item):
                            fhe_module_names_set.add(full_name)
                else:
                    raise ValueError(f"Invalid item in fhe_modules: {item}")

            self.fhe_module_names = sorted(list(fhe_module_names_set))

            # Now populate modules_dict
            # First add all direct children
            for name, module in torch_model.named_children():
                if name in fhe_module_names_set:
                    self.modules_dict[name] = self._convert_module(module)
                else:
                    self.modules_dict[name] = module

            # Then add any nested FHE modules that aren't direct children
            for full_name in self.fhe_module_names:
                if "." in full_name:  # It's a nested module
                    safe_name = full_name.replace(".", "_")
                    if full_name in all_named_modules:
                        self.modules_dict[safe_name] = self._convert_module(
                            all_named_modules[full_name]
                        )

            # Normalize all names
            self.fhe_module_names = [
                name.replace(".", "_") for name in self.fhe_module_names
            ]

    def _convert_module(self, module: nn.Module) -> nn.Module:
        """
        Convert a PyTorch module to its Supersayan counterpart.

        Args:
            module: The module to convert

        Returns:
            nn.Module: The converted module
        """
        module_type = type(module)

        if module_type in LAYER_MAPPING:
            cls = LAYER_MAPPING[module_type]

            if isinstance(module, nn.Linear):
                out = cls(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                )
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
        elif isinstance(module, nn.Sequential):
            return nn.Sequential(*(self._convert_module(m) for m in module))
        else:
            raise ValueError(f"Module type {module_type} not supported")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: The input tensor

        Returns:
            torch.Tensor: The output tensor
        """
        if self.model_type == ModelType.PURE:
            key = generate_secret_key()
            enc = encrypt_to_lwes(x, key)

            for name in self.fhe_module_names:
                enc = self.modules_dict[name](enc)

            return torch.tensor(
                decrypt_from_lwes(enc, key), dtype=x.dtype, device=x.device
            )

        raise NotImplementedError("Hybrid forward is implemented by SupersayanClient")
