import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Type, Any, Callable
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
    """
    Enum defining the types of supersayan models.
    
    PURE: All layers execute in FHE (Fully Homomorphic Encryption).
    HYBRID: Specified layers are executed in FHE, others remain as original PyTorch modules running in plaintext.
    """
    PURE = "pure"
    HYBRID = "hybrid"

class SupersayanModel(nn.Module):
    """
    A unified model that can operate in pure or hybrid mode.
    
    In pure mode, all layers run in FHE.
    In hybrid mode, specified layers are converted to FHE, others remain as original PyTorch modules.
    """
    def __init__(
        self, 
        torch_model: nn.Module, 
        model_type: ModelType = ModelType.PURE,
        fhe_modules: Optional[Union[List[str], List[Type[nn.Module]]]] = None
    ):
        """
        Initialize a SuperSayan model from a PyTorch model.
        
        Args:
            torch_model: The PyTorch model to convert
            model_type: Whether to create a pure (all FHE) or hybrid model
            fhe_modules: Either a list of module names or module types to execute in FHE 
                        (required for hybrid mode, ignored for pure mode)
        
        Raises:
            ValueError: If a layer doesn't have a SuperSayan equivalent or if 
                        fhe_modules are not provided in hybrid mode
        """
        super(SupersayanModel, self).__init__()
        self.original_model = torch_model
        self.model_type = model_type
        
        # For hybrid model, fhe_modules must be provided
        if model_type == ModelType.HYBRID and not fhe_modules:
            raise ValueError("For hybrid model, fhe_modules must be provided")
        
        # Determine if we're using names or types
        self.using_module_types = False
        if fhe_modules and all(isinstance(m, type) for m in fhe_modules):
            self.using_module_types = True
            self.fhe_module_types = fhe_modules
            self.fhe_module_names = []
        else:
            self.fhe_module_names = fhe_modules if fhe_modules else []
            self.fhe_module_types = []
        
        if model_type == ModelType.PURE:
            # For pure mode, we convert all modules to SuperSayan FHE modules
            self.modules_dict = nn.ModuleDict()
            
            # Get all direct children modules
            for name, module in torch_model.named_children():
                try:
                    # Convert all modules to SuperSayan modules
                    supersayan_module = self._convert_module(module)
                    self.modules_dict[name] = supersayan_module
                except ValueError as e:
                    raise ValueError(f"Failed to convert module '{name}' to SuperSayan: {e}")
            
            # Store all module names for FHE execution
            self.fhe_module_names = list(self.modules_dict.keys())
        else:
            # Convert modules selectively for hybrid model
            self.modules_dict = nn.ModuleDict()
            
            if self.using_module_types:
                # When using module types, find all modules of those types
                all_modules = {}
                for name, module in torch_model.named_modules():
                    if any(isinstance(module, module_type) for module_type in self.fhe_module_types):
                        # Skip modules that are nested inside already identified parent modules
                        if not any(other_name != name and name.startswith(other_name) for other_name in all_modules.keys()):
                            all_modules[name] = module
                            self.fhe_module_names.append(name)
                
                # Now process direct children
                for name, module in torch_model.named_children():
                    if name in self.fhe_module_names or any(isinstance(module, module_type) for module_type in self.fhe_module_types):
                        # Convert to SuperSayan module
                        try:
                            supersayan_module = self._convert_module(module)
                            self.modules_dict[name] = supersayan_module
                        except ValueError as e:
                            raise ValueError(f"Failed to convert module '{name}' to SuperSayan: {e}")
                    else:
                        # Keep as PyTorch module
                        self.modules_dict[name] = module
            else:
                # Using module names
                # Validate that the specified module names exist in the model
                all_module_names = dict(torch_model.named_modules())
                for name in self.fhe_module_names:
                    if name not in all_module_names:
                        raise ValueError(f"Module '{name}' not found in the model. Available modules: {list(all_module_names.keys())}")
                
                # Convert modules selectively
                for name, module in torch_model.named_children():
                    if name in self.fhe_module_names:
                        # Convert to SuperSayan module
                        try:
                            supersayan_module = self._convert_module(module)
                            self.modules_dict[name] = supersayan_module
                        except ValueError as e:
                            raise ValueError(f"Failed to convert module '{name}' to SuperSayan: {e}")
                    else:
                        # Keep as PyTorch module
                        self.modules_dict[name] = module
    
    def _convert_module(self, module: nn.Module) -> nn.Module:
        """
        Convert a PyTorch module to its SuperSayan equivalent.
        
        Args:
            module: The PyTorch module to convert
            
        Returns:
            The converted SuperSayan module
            
        Raises:
            ValueError: If the module type doesn't have a SuperSayan equivalent
        """
        module_type = type(module)
        
        if module_type in LAYER_MAPPING:
            # Get the SuperSayan equivalent class
            supersayan_class = LAYER_MAPPING[module_type]
            
            if module_type == nn.Linear:
                # Handle Linear layer specifically
                supersayan_module = supersayan_class(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None
                )
                
                # Copy weights and biases
                supersayan_module.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    supersayan_module.bias.data = module.bias.data.clone()
                
                return supersayan_module
                
            elif module_type == nn.Conv2d:
                # Handle Conv2d layer
                supersayan_module = supersayan_class(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=module.bias is not None
                )
                
                # Copy weights and biases
                supersayan_module.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    supersayan_module.bias.data = module.bias.data.clone()
                
                return supersayan_module
            
            # Add handling for other layer types as they are implemented
            
        elif isinstance(module, nn.Sequential):
            # Handle Sequential containers
            supersayan_sequential = nn.Sequential()
            for i, submodule in enumerate(module.children()):
                supersayan_submodule = self._convert_module(submodule)
                supersayan_sequential.add_module(str(i), supersayan_submodule)
            return supersayan_sequential
        
        else:
            raise ValueError(f"Module of type {module_type.__name__} not supported in SuperSayan. Supported types: {list(LAYER_MAPPING.keys())}")
    
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
        output = torch.tensor(decrypt(encrypted_output, key), dtype=x.dtype, device=x.device)
        
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
                output = torch.tensor(decrypt(encrypted_output, key), dtype=output.dtype, device=output.device)
            else:
                # Normal PyTorch module
                output = self.modules_dict[name](output)
        
        return output


def convert_model(
    torch_model: nn.Module, 
    model_type: ModelType = ModelType.PURE,
    fhe_modules: Optional[Union[List[str], List[Type[nn.Module]]]] = None
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
    torch_model: nn.Module, 
    fhe_modules: Union[List[str], List[Type[nn.Module]]]
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