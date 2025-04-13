import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Type, Any, Callable
from enum import Enum

from supersayan.core.encryption import encrypt, decrypt
from supersayan.core.keygen import generate_secret_key
from supersayan.nn.layers import Linear

logger = logging.getLogger(__name__)

# Dictionary mapping PyTorch layer types to their SuperSayan equivalents
LAYER_MAPPING = {
    nn.Linear: Linear,
    # Add more mappings as more layers are implemented
}

class ModelType(Enum):
    """
    Enum defining the types of supersayan models.
    
    PURE: All layers are converted to SuperSayan equivalents.
    HYBRID: Only specified layers are converted to SuperSayan equivalents.
    """
    PURE = "pure"
    HYBRID = "hybrid"

class SupersayanModel(nn.Module):
    """
    A unified model that can operate in pure or hybrid mode.
    
    In pure mode, it converts all layers to their SuperSayan equivalents.
    In hybrid mode, it selectively converts specified layers.
    """
    def __init__(
        self, 
        torch_model: nn.Module, 
        model_type: ModelType = ModelType.PURE,
        fhe_module_names: Optional[List[str]] = None
    ):
        """
        Initialize a SuperSayan model from a PyTorch model.
        
        Args:
            torch_model: The PyTorch model to convert
            model_type: Whether to create a pure or hybrid model
            fhe_module_names: List of module names to execute in FHE (required for hybrid mode)
        
        Raises:
            ValueError: If a layer doesn't have a SuperSayan equivalent or if 
                        fhe_module_names are not provided in hybrid mode
        """
        super(SupersayanModel, self).__init__()
        self.original_model = torch_model
        self.model_type = model_type
        
        # For hybrid model, fhe_module_names must be provided
        if model_type == ModelType.HYBRID and not fhe_module_names:
            raise ValueError("For hybrid model, fhe_module_names must be provided")
        
        self.fhe_module_names = fhe_module_names if fhe_module_names else []
        
        if model_type == ModelType.PURE:
            # Convert all modules for pure model
            self.modules_list = nn.ModuleList()
            for name, module in torch_model.named_children():
                supersayan_module = self._convert_module(module)
                self.modules_list.append(supersayan_module)
        else:
            # Convert only specified modules for hybrid model
            # Validate that the specified module names exist in the model
            all_module_names = dict(torch_model.named_modules())
            for name in self.fhe_module_names:
                if name not in all_module_names:
                    raise ValueError(f"Module '{name}' not found in the model. Available modules: {list(all_module_names.keys())}")
            
            # Convert modules selectively
            self.modules_dict = nn.ModuleDict()
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
    
    def forward(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Forward pass that handles both pure and hybrid modes.
        
        Args:
            x: Input data (torch.Tensor for hybrid mode, numpy array of LWE objects for pure mode)
            
        Returns:
            Output data (torch.Tensor for hybrid mode, numpy array of LWE objects for pure mode)
        """
        if self.model_type == ModelType.PURE:
            return self._forward_pure(x)
        else:
            return self._forward_hybrid(x)
    
    def _forward_pure(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for pure FHE model with encrypted input.
        
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
        for module in self.modules_list:
            output = module(output)
            
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
    fhe_module_names: Optional[List[str]] = None
) -> SupersayanModel:
    """
    Convert a PyTorch model to a SuperSayan model.
    
    Args:
        torch_model: The PyTorch model to convert
        model_type: Whether to create a pure or hybrid model
        fhe_module_names: List of module names to execute in FHE (required for hybrid mode)
        
    Returns:
        A SuperSayan model configured according to the specified parameters
    """
    return SupersayanModel(torch_model, model_type, fhe_module_names)


# Maintain backwards compatibility
def convert_to_pure_supersayan(torch_model: nn.Module) -> SupersayanModel:
    """
    Convert a PyTorch model to a Pure SuperSayan model.
    
    Args:
        torch_model: The PyTorch model to convert
        
    Returns:
        A Pure SuperSayan model that takes encrypted inputs and produces encrypted outputs
    """
    return convert_model(torch_model, ModelType.PURE)


def convert_to_hybrid_supersayan(torch_model: nn.Module, fhe_module_names: List[str]) -> SupersayanModel:
    """
    Convert a PyTorch model to a Hybrid SuperSayan model.
    
    Args:
        torch_model: The PyTorch model to convert
        fhe_module_names: List of module names to execute in FHE
        
    Returns:
        A Hybrid SuperSayan model that takes unencrypted inputs, 
        processes specified modules with encryption, and produces unencrypted outputs
    """
    return convert_model(torch_model, ModelType.HYBRID, fhe_module_names)