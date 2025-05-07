import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Type, Any, Callable, cast
from enum import Enum

from supersayan.core.encryption import encrypt, decrypt
from supersayan.core.keygen import generate_secret_key
from supersayan.nn.layers import Linear, Conv2d

logger = logging.getLogger(__name__)

# Dictionary mapping PyTorch layer types to their SuperSayan equivalents
LAYER_MAPPING = {
    nn.Linear: Linear,
    nn.Conv2d: Conv2d,
    # Add more mappings as more layers are implemented
}


class ModelType(Enum):
    PURE = "pure"
    HYBRID = "hybrid"


class SupersayanModel(nn.Module):
    def __init__(
        self,
        torch_model: nn.Module,
        model_type: ModelType = ModelType.PURE,
        fhe_modules: Optional[Union[List[str], List[Type[nn.Module]]]] = None,
    ):
        super(SupersayanModel, self).__init__()
        self.original_model = torch_model
        self.model_type = model_type

        if model_type == ModelType.HYBRID and not fhe_modules:
            raise ValueError("For hybrid model, fhe_modules must be provided")

        # Decide if hybrid list is by type or by name
        self.using_module_types = bool(
            fhe_modules and all(isinstance(m, type) for m in fhe_modules)
        )
        if self.using_module_types:
            self.fhe_module_types = cast(List[Type[nn.Module]], fhe_modules)
            self.fhe_module_names: List[str] = []
        else:
            self.fhe_module_names = cast(List[str], fhe_modules) if fhe_modules else []
            self.fhe_module_types: List[Type[nn.Module]] = []

        # Build modules_dict of **direct** children first
        self.modules_dict = nn.ModuleDict()

        if model_type == ModelType.PURE:
            # pure: convert every direct child
            for name, module in torch_model.named_children():
                sup = self._convert_module(module)
                self.modules_dict[name] = sup
            self.fhe_module_names = list(self.modules_dict.keys())

        else:
            # hybrid: pick direct children to convert or leave alone
            all_named = dict(torch_model.named_modules())
            # if by type, collect *all* matching modules (including nested)
            if self.using_module_types:
                # collect nested names too
                for full_name, module in torch_model.named_modules():
                    if any(isinstance(module, t) for t in self.fhe_module_types):
                        self.fhe_module_names.append(full_name)
                # now go through only direct children for modules_dict
                for name, module in torch_model.named_children():
                    if name in self.fhe_module_names:
                        self.modules_dict[name] = self._convert_module(module)
                    else:
                        self.modules_dict[name] = module

            else:
                # by explicit name list
                # validate names
                existing = set(torch_model.named_modules())
                missing = [n for n in self.fhe_module_names if n not in existing]
                if missing:
                    raise ValueError(f"Module(s) not found: {missing}")
                for name, module in torch_model.named_children():
                    if name in self.fhe_module_names:
                        self.modules_dict[name] = self._convert_module(module)
                    else:
                        self.modules_dict[name] = module

            # --- NEW FLATTENING STEP ---
            # bring in _all_ nested FHE modules so they actually get uploaded
            for full_name in list(self.fhe_module_names):
                if full_name not in self.modules_dict:
                    # grab the raw submodule, convert it, and register under a safe key
                    sub = all_named[full_name]
                    sup = self._convert_module(sub)
                    safe = full_name.replace(".", "_")
                    self.modules_dict[safe] = sup

            # sanitize our list of names to the same safe keys
            self.fhe_module_names = [n.replace(".", "_") for n in self.fhe_module_names]

    def _convert_module(self, module: nn.Module) -> nn.Module:
        module_type = type(module)
        if module_type in LAYER_MAPPING:
            cls = LAYER_MAPPING[module_type]
            # handle Linear
            if isinstance(module, nn.Linear):
                m = cls(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                )
                m.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    m.bias.data.copy_(module.bias.data)
                return m
            # handle Conv2d
            elif isinstance(module, nn.Conv2d):
                m = cls(
                    module.in_channels,
                    module.out_channels,
                    module.kernel_size,
                    module.stride,
                    module.padding,
                    module.dilation,
                    module.groups,
                    bias=module.bias is not None,
                )
                m.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    m.bias.data.copy_(module.bias.data)
                return m

        # recursive for Sequential
        if isinstance(module, nn.Sequential):
            seq = nn.Sequential()
            for i, sub in enumerate(module):
                seq.add_module(str(i), self._convert_module(sub))
            return seq

        raise ValueError(
            f"Module type {module_type} not supported; add to LAYER_MAPPING"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that handles both pure and hybrid modes.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        if self.model_type == ModelType.PURE:
            return self._forward_pure(x)
        else:
            return self._forward_hybrid(x)

    def _forward_pure(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for pure model - all layers run in FHE.

        Args:
            x: Input tensor

        Returns:
            Output tensor after all FHE processing
        """
        # For pure models, run all layers in FHE
        output = x

        # Generate key for the entire forward pass
        key = generate_secret_key()

        # Initial encryption
        encrypted_output = encrypt(output, key)

        # Process each module in order
        for name, module in self.original_model.named_children():
            # All modules run in FHE for pure model
            encrypted_output = self.modules_dict[name](encrypted_output)

        # Final decryption
        output = torch.tensor(
            decrypt(encrypted_output, key), dtype=x.dtype, device=x.device
        )

        return output

    def _forward_hybrid(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for hybrid model with automatic encryption/decryption for FHE modules.

        Args:
            x: Unencrypted input data (torch.Tensor)

        Returns:
            Unencrypted output (torch.Tensor)
        """
        output = x

        # Process each module in the order they appear in the original model
        for name, module in self.original_model.named_children():
            if name in self.fhe_module_names:
                # This module should run in FHE
                # 1. Generate key
                key = generate_secret_key()

                # 2. Encrypt input
                encrypted_input = encrypt(output, key)

                # 3. Process with SuperSayan module
                encrypted_output = self.modules_dict[name](encrypted_input)

                # 4. Decrypt output
                output = torch.tensor(
                    decrypt(encrypted_output, key),
                    dtype=output.dtype,
                    device=output.device,
                )
            else:
                # Normal PyTorch module
                output = self.modules_dict[name](output)

        return output


def convert_model(
    torch_model: nn.Module,
    model_type: ModelType = ModelType.PURE,
    fhe_modules: Optional[Union[List[str], List[Type[nn.Module]]]] = None,
) -> SupersayanModel:
    """
    Convert a PyTorch model to a SuperSayan model.

    Args:
        torch_model: The PyTorch model to convert
        model_type: Whether to create a pure (all FHE) or hybrid model
        fhe_modules: Either a list of module names or module types to execute in FHE
                     (required for hybrid mode, ignored for pure mode)

    Returns:
        A SuperSayan model configured according to the specified parameters
    """
    return SupersayanModel(torch_model, model_type, fhe_modules)


# Maintain backwards compatibility
def convert_to_pure_supersayan(torch_model: nn.Module) -> SupersayanModel:
    """
    Convert a PyTorch model to a Pure SuperSayan model.

    Args:
        torch_model: The PyTorch model to convert

    Returns:
        A Pure SuperSayan model where all layers run in FHE
    """
    return convert_model(torch_model, ModelType.PURE)


def convert_to_hybrid_supersayan(
    torch_model: nn.Module, fhe_modules: Union[List[str], List[Type[nn.Module]]]
) -> SupersayanModel:
    """
    Convert a PyTorch model to a Hybrid SuperSayan model.

    Args:
        torch_model: The PyTorch model to convert
        fhe_modules: Either a list of module names or module types to execute in FHE

    Returns:
        A Hybrid SuperSayan model that runs specified modules in FHE and others as native PyTorch
    """
    return convert_model(torch_model, ModelType.HYBRID, fhe_modules)
