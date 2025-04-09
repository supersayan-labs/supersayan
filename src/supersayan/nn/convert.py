import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Type, Any

from supersayan.core.encryption import encrypt, decrypt
from supersayan.core.keygen import generate_secret_key
from supersayan.nn.layers import Linear

logger = logging.getLogger(__name__)

# Dictionary mapping PyTorch layer types to their SuperSayan equivalents
LAYER_MAPPING = {
    nn.Linear: Linear,
    # Add more mappings as more layers are implemented
}

class PureSupersayanModel(nn.Module):
    """
    A model that operates entirely on encrypted data.
    
    It converts all layers of a PyTorch model to their Supersayan equivalents
    and performs operations on encrypted data.
    """
    def __init__(self, torch_model: nn.Module):
        """
        Initialize a Pure Supersayan model from a PyTorch model.
        
        Args:
            torch_model: The PyTorch model to convert
        
        Raises:
            ValueError: If a layer in the PyTorch model doesn't have a Supersayan equivalent
        """
        super(PureSupersayanModel, self).__init__()
        self.original_model = torch_model
        self.supersayan_modules = nn.ModuleList()
        
        # Convert each module in the PyTorch model
        for name, module in torch_model.named_children():
            supersayan_module = self._convert_module(module)
            self.supersayan_modules.append(supersayan_module)
    
    def _convert_module(self, module: nn.Module) -> nn.Module:
        """
        Convert a PyTorch module to its Supersayan equivalent.
        
        Args:
            module: The PyTorch module to convert
            
        Returns:
            The converted Supersayan module
            
        Raises:
            ValueError: If the module type doesn't have a Supersayan equivalent
        """
        module_type = type(module)
        
        if module_type in LAYER_MAPPING:
            # Get the Supersayan equivalent class
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
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with encrypted input.
        
        Args:
            x: Encrypted input data (numpy array of LWE objects)
            
        Returns:
            Encrypted output (numpy array of LWE objects)
        """
        # Ensure x is a numpy array
        if not isinstance(x, np.ndarray):
            logger.warning(f"Expected numpy array as input, got {type(x)}. Attempting to convert.")
            x = np.array(x, dtype=object)
            
        output = x
        for module in self.supersayan_modules:
            output = module(output)
            
        return output


class HybridSupersayanModel(nn.Module):
    """
    A model that operates on a mix of encrypted and unencrypted data.
    
    It converts specified layers of a PyTorch model to their Supersayan equivalents
    and performs operations with encryption/decryption as needed.
    """
    def __init__(self, torch_model: nn.Module, fhe_module_names: List[str]):
        """
        Initialize a Hybrid Supersayan model from a PyTorch model.
        
        Args:
            torch_model: The PyTorch model to convert
            fhe_module_names: List of module names to execute in FHE
            
        Raises:
            ValueError: If a module name in fhe_module_names doesn't exist in the model
                        or if a layer to be executed in FHE doesn't have a Supersayan equivalent
        """
        super(HybridSupersayanModel, self).__init__()
        self.original_model = torch_model
        self.fhe_module_names = fhe_module_names
        
        # Validate that the specified module names exist in the model
        all_module_names = dict(torch_model.named_modules())
        for name in fhe_module_names:
            if name not in all_module_names:
                raise ValueError(f"Module '{name}' not found in the model. Available modules: {list(all_module_names.keys())}")
        
        # Convert modules selectively
        self.modules_dict = nn.ModuleDict()
        for name, module in torch_model.named_children():
            if name in fhe_module_names:
                # Convert to Supersayan module
                try:
                    supersayan_module = self._convert_module(module)
                    self.modules_dict[name] = supersayan_module
                except ValueError as e:
                    raise ValueError(f"Failed to convert module '{name}' to Supersayan: {e}")
            else:
                # Keep as PyTorch module
                self.modules_dict[name] = module
    
    def _convert_module(self, module: nn.Module) -> nn.Module:
        """
        Convert a PyTorch module to its Supersayan equivalent.
        
        Args:
            module: The PyTorch module to convert
            
        Returns:
            The converted Supersayan module
            
        Raises:
            ValueError: If the module type doesn't have a Supersayan equivalent
        """
        module_type = type(module)
        
        if module_type in LAYER_MAPPING:
            # Get the Supersayan equivalent class
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
            
            # Add handling for other layer types as they are implemented
            
        elif isinstance(module, nn.Sequential):
            # Handle Sequential containers
            supersayan_sequential = nn.Sequential()
            for i, submodule in enumerate(module.children()):
                supersayan_submodule = self._convert_module(submodule)
                supersayan_sequential.add_module(str(i), supersayan_submodule)
            return supersayan_sequential
        
        else:
            raise ValueError(f"Module of type {module_type.__name__} not supported in Supersayan. Supported types: {list(LAYER_MAPPING.keys())}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with automatic encryption/decryption for FHE modules.
        
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
                key = generate_secret_key()  # Use a standard key size
                
                # 2. Encrypt input
                encrypted_input = encrypt(output, key)
                
                # 3. Process with Supersayan module
                encrypted_output = self.modules_dict[name](encrypted_input)
                
                # 4. Decrypt output
                output = torch.tensor(decrypt(encrypted_output, key), dtype=output.dtype, device=output.device)
            else:
                # Normal PyTorch module
                output = self.modules_dict[name](output)
        
        return output


def convert_to_pure_supersayan(torch_model: nn.Module) -> PureSupersayanModel:
    """
    Convert a PyTorch model to a Pure Supersayan model.
    
    Args:
        torch_model: The PyTorch model to convert
        
    Returns:
        A Pure Supersayan model that takes encrypted inputs and produces encrypted outputs
    """
    return PureSupersayanModel(torch_model)


def convert_to_hybrid_supersayan(torch_model: nn.Module, fhe_module_names: List[str]) -> HybridSupersayanModel:
    """
    Convert a PyTorch model to a Hybrid Supersayan model.
    
    Args:
        torch_model: The PyTorch model to convert
        fhe_module_names: List of module names to execute in FHE
        
    Returns:
        A Hybrid Supersayan model that takes unencrypted inputs, 
        processes specified modules with encryption, and produces unencrypted outputs
    """
    return HybridSupersayanModel(torch_model, fhe_module_names) 