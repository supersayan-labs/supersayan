import logging
import torch
import torch.nn as nn
import numpy as np
import requests
import uuid
import pickle
import base64
import os
from typing import Dict, List, Optional, Tuple, Union, Any, Type

from supersayan.core.encryption import encrypt, decrypt
from supersayan.core.keygen import generate_secret_key
from supersayan.nn.convert import SupersayanModel, ModelType
from .serialization import serialize_data, deserialize_data

logger = logging.getLogger(__name__)

class SupersayanClient(SupersayanModel):
    """
    Client for performing remote inference with SuperSayan FHE models.
    
    This class extends SupersayanModel to transparently handle remote execution
    of FHE layers while keeping local execution for non-FHE layers.
    """
    def __init__(
        self, 
        server_url: str, 
        torch_model: nn.Module,
        fhe_modules: Optional[Union[List[str], List[Type[nn.Module]]]] = None,
        model_id: Optional[str] = None
    ):
        """
        Initialize a SuperSayan client that extends SupersayanModel.
        
        Args:
            server_url: The URL of the SuperSayan server
            torch_model: The PyTorch model to convert
            model_type: Whether to create a pure (all FHE) or hybrid (partial FHE) model
            fhe_modules: Either a list of module names or module types to execute in FHE 
                        (required for hybrid mode)
            model_id: Optional ID of a model already on the server
        
        Note:
            Pure models run entirely in FHE (all layers executed remotely).
            For hybrid mode, specified layers run in FHE and are executed remotely,
            while other layers run in plaintext locally.
        """
        # Initialize the parent SupersayanModel
        super(SupersayanClient, self).__init__(
            torch_model=torch_model,
            model_type=ModelType.HYBRID,
            fhe_modules=fhe_modules
        )
        
        # Client-specific initialization
        self.server_url = server_url.rstrip('/')
        self.secret_key = generate_secret_key()
        self.model_id = model_id
        self.uploaded = False
        
        # Both pure and hybrid models need remote execution
        # If model_id is provided, assume the model is already on the server
        if model_id is not None:
            self.uploaded = True
            self._get_remote_layer_names()
    
    def _upload_model_if_needed(self):
        """
        Upload the FHE modules to the server if needed.
        
        Returns:
            model_id: The ID of the model on the server
        """
        if self.uploaded:
            return self.model_id
            
        # Save model to a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
        
        try:
            remote_modules = nn.ModuleDict(
                {
                    name: self.modules_dict[name]
                    for name in self.fhe_module_names
                    if name in self.modules_dict
                }
            )
            
            # Save just the FHE modules
            torch.save(remote_modules, temp_path)
            
            # Upload the model
            with open(temp_path, 'rb') as f:
                model_data = f.read()
            
            encoded_model = base64.b64encode(model_data).decode('utf-8')
            
            response = requests.post(
                f"{self.server_url}/models/upload",
                json={
                    "model_data": encoded_model
                }
            )
            
            if response.status_code != 200:
                raise ValueError(f"Failed to upload model: {response.text}")
            
            self.model_id = response.json()["model_id"]
            logger.info(f"Model uploaded with ID: {self.model_id}")
            
            # Get model structure
            self._get_remote_layer_names()
            
            self.uploaded = True
            return self.model_id
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def _get_remote_layer_names(self):
        """
        Get the layer names from the remote model.
        """
        if self.model_id is None:
            return
        
        response = requests.get(f"{self.server_url}/models/{self.model_id}/structure")
        
        if response.status_code != 200:
            raise ValueError(f"Failed to retrieve model structure: {response.text}")
        
        structure = response.json()["structure"]
        self.remote_layer_names = structure.get("layer_order", [])
    
    def _process_layer(self, layer_name: str, encrypted_input: np.ndarray) -> np.ndarray:
        """
        Process a single encrypted layer on the server.
        
        Args:
            layer_name: Name of the layer to process
            encrypted_input: Encrypted input data
            
        Returns:
            Encrypted output from the layer
        """
        # Ensure model is uploaded
        self._upload_model_if_needed()

        logger.debug(f"Processing layer: {layer_name}")
 
        # Debug info for input
        logger.debug(f"Input type: {type(encrypted_input)}")
        if hasattr(encrypted_input, 'shape'):
            logger.debug(f"Input shape: {encrypted_input.shape}")
        elif hasattr(encrypted_input, '__len__'):
            logger.debug(f"Input length: {len(encrypted_input)}")
        
        # Serialize the encrypted input using the utility
        try:
            encoded_input = serialize_data(encrypted_input)
        except Exception as e:
            logger.exception(f"Serialization error: {e}")
            raise ValueError(f"Failed to serialize input: {e}")
        
        # Send to server
        response = requests.post(
            f"{self.server_url}/inference/{self.model_id}/{layer_name}",
            json={
                "encrypted_input": encoded_input
            }
        )
        
        if response.status_code != 200:
            raise ValueError(f"Layer processing failed: {response.text}")
        
        # Deserialize the result using the utility
        try:
            result = response.json()
            encrypted_output = deserialize_data(result["encrypted_output"])
            
            # Debug info for output
            logger.debug(f"Output received for layer: {layer_name}")
            logger.debug(f"Output type: {type(encrypted_output)}")
            if hasattr(encrypted_output, 'shape'):
                logger.debug(f"Output shape: {encrypted_output.shape}")
            elif hasattr(encrypted_output, '__len__'):
                logger.debug(f"Output length: {len(encrypted_output)}")
                
            return encrypted_output
        except Exception as e:
            logger.exception(f"Deserialization error: {e}")
            raise ValueError(f"Failed to deserialize output: {e}")
    
    def _forward_hybrid(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for hybrid model with remote execution for FHE layers.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Ensure model is uploaded for hybrid mode
        self._upload_model_if_needed()
        
        output = x
        
        # Process each module in the order they appear in the original model
        for name, module in self.original_model.named_children():
            if name in self.fhe_module_names:
                # This module should run in FHE on the remote server
                
                # Encrypt input
                encrypted_input = encrypt(output, self.secret_key)
                
                # Process with remote FHE
                encrypted_output = self._process_layer(name, encrypted_input)
                
                # Decrypt output
                output = torch.tensor(
                    decrypt(encrypted_output, self.secret_key),
                    dtype=output.dtype,
                    device=output.device
                )
            else:
                # Normal PyTorch module - run locally
                output = self.modules_dict[name](output)
        
        return output