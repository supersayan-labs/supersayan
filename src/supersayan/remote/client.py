import logging
import torch
import torch.nn as nn
import numpy as np
import requests
import uuid
import pickle
import base64
import os
from typing import List, Optional

from supersayan.core.encryption import encrypt, decrypt
from supersayan.core.keygen import generate_secret_key
from supersayan.nn.convert import SupersayanModel, ModelType

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
        model_type: ModelType = ModelType.PURE,
        fhe_module_names: Optional[List[str]] = None,
        model_id: Optional[str] = None
    ):
        """
        Initialize a SuperSayan client that extends SupersayanModel.
        
        Args:
            server_url: The URL of the SuperSayan server
            torch_model: The PyTorch model to convert
            model_type: Whether to create a pure (plaintext) or hybrid (partial FHE) model
            fhe_module_names: List of module names to execute in FHE (required for hybrid mode)
            model_id: Optional ID of a model already on the server
        
        Note:
            Pure models run entirely in plaintext (no FHE, no remote execution).
            For hybrid mode, specified layers run in FHE and are executed remotely.
        """
        # Initialize the parent SupersayanModel
        super(SupersayanClient, self).__init__(
            torch_model=torch_model,
            model_type=model_type,
            fhe_module_names=fhe_module_names
        )
        
        # Client-specific initialization
        self.server_url = server_url.rstrip('/')
        self.session_id = str(uuid.uuid4())
        self.secret_key = generate_secret_key()
        self.model_id = model_id
        self.uploaded = False
        self._closed = False  # Flag to track if session is closed
        
        # Only hybrid models need remote execution
        if model_type == ModelType.HYBRID:
            # If model_id is provided, assume the model is already on the server
            if model_id is not None:
                self.uploaded = True
                self._get_remote_layer_names()
    
    def _upload_model_if_needed(self):
        """
        Upload the FHE modules to the server if needed (hybrid mode only).
        
        Returns:
            model_id: The ID of the model on the server
        """
        # Pure models run locally, so no need to upload
        if self.model_type == ModelType.PURE:
            return None
            
        if self.uploaded:
            return self.model_id
        
        # Save model to a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
        
        try:
            # For hybrid models, we only need to upload the FHE modules
            # Create a model dict with just the FHE modules
            remote_modules = nn.ModuleDict()
            for name in self.fhe_module_names:
                if name in self.modules_dict:
                    remote_modules[name] = self.modules_dict[name]
            
            # Save just the FHE modules
            torch.save(remote_modules, temp_path)
            
            # Upload the model
            with open(temp_path, 'rb') as f:
                model_data = f.read()
            
            encoded_model = base64.b64encode(model_data).decode('utf-8')
            
            response = requests.post(
                f"{self.server_url}/models/upload",
                json={
                    "session_id": self.session_id,
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
        
        # Serialize the encrypted input
        serialized_input = pickle.dumps(encrypted_input)
        encoded_input = base64.b64encode(serialized_input).decode('utf-8')
        
        # Send to server
        response = requests.post(
            f"{self.server_url}/inference/{self.model_id}/{layer_name}",
            json={
                "session_id": self.session_id,
                "encrypted_input": encoded_input
            }
        )
        
        if response.status_code != 200:
            raise ValueError(f"Layer processing failed: {response.text}")
        
        # Deserialize the result
        result = response.json()
        decoded_output = base64.b64decode(result["encrypted_output"])
        data = pickle.loads(decoded_output)
        
        # Handle different serialized formats
        from supersayan.core.types import LWE
        
        if isinstance(data, dict) and "type" in data and "data" in data:
            if data["type"] == "LWE":
                # Single LWE object
                encrypted_output = LWE.from_dict(data)
            elif data["type"] == "lwe_array":
                # Array of LWE objects
                flat_lwes = []
                for lwe_dict in data["data"]:
                    if lwe_dict.get("type") == "LWE":
                        flat_lwes.append(LWE.from_dict(lwe_dict))
                    else:
                        logger.warning(f"Unexpected object type in LWE array: {lwe_dict.get('type')}")
                        flat_lwes.append(None)  # Placeholder for invalid data
                
                # Reshape according to original dimensions
                shape = tuple(data.get("shape", (len(flat_lwes),)))
                encrypted_output = np.array(flat_lwes, dtype=object).reshape(shape)
            elif data["type"] == "array":
                # Plain array
                encrypted_output = np.array(data["data"])
            elif data["type"] == "error":
                # Error occurred during serialization on server
                logger.error(f"Server serialization error: {data.get('error')}")
                raise ValueError(f"Failed to deserialize server response: {data.get('error')}")
            else:
                # Unknown format
                logger.warning(f"Unknown data format received: {data['type']}")
                encrypted_output = data
        else:
            # Not a special format, use as-is
            encrypted_output = data
        
        return encrypted_output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that handles both pure and hybrid modes.
        
        This method overrides the parent's forward method to handle remote execution
        for hybrid models when needed.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        if self.model_type == ModelType.PURE:
            # For pure models, use the parent's implementation which runs locally
            return super()._forward_pure(x)
        else:
            # For hybrid models, use our remote execution implementation
            return self._forward_hybrid(x)
    
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
    
    def close(self):
        """
        Clean up resources and end the session.
        Only needed for hybrid models that use remote execution.
        """
        if self.model_type == ModelType.HYBRID and self.model_id is not None:
            # Set flag to avoid duplicate closing in __del__
            self._closed = True
            
            requests.post(
                f"{self.server_url}/sessions/{self.session_id}/close",
                json={"session_id": self.session_id}
            )
            logger.info(f"Closed session {self.session_id}")
    
    def __del__(self):
        """
        Cleanup on object destruction.
        
        Note: This method should only be used as a fallback.
        Applications should explicitly call close() when done.
        """
        # Only close if session wasn't already explicitly closed
        # Add a _closed flag to track this state
        if not hasattr(self, '_closed') or not self._closed:
            try:
                self.close()
            except:
                pass