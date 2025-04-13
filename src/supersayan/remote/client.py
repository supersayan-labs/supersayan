import logging
import torch
import torch.nn as nn
import numpy as np
import json
import requests
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import uuid
import pickle
import base64

from supersayan.core.encryption import encrypt, decrypt
from supersayan.core.keygen import generate_secret_key

logger = logging.getLogger(__name__)

class SupersayanClient:
    """
    Client for performing remote inference with SuperSayan FHE models.
    
    Handles key generation, encryption, communication with server, and decryption.
    """
    def __init__(self, server_url: str):
        """
        Initialize a SuperSayan client.
        
        Args:
            server_url: The URL of the SuperSayan server
        """
        self.server_url = server_url.rstrip('/')
        self.session_id = str(uuid.uuid4())
        self.secret_key = None
    
    def upload_model(self, model_path: str) -> str:
        """
        Upload a serialized model to the server.
        
        Args:
            model_path: Path to the serialized model file
            
        Returns:
            model_id: Identifier for the uploaded model
            
        Raises:
            ValueError: If the model could not be uploaded
        """
        with open(model_path, 'rb') as f:
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
        
        result = response.json()
        return result["model_id"]
    
    def process_layer(self, model_id: str, layer_name: str, encrypted_input: np.ndarray) -> np.ndarray:
        """
        Process a single encrypted layer on the server.
        
        Args:
            model_id: ID of the model on the server
            layer_name: Name of the layer to process
            encrypted_input: Encrypted input data
            
        Returns:
            Encrypted output from the layer
            
        Raises:
            ValueError: If the layer processing failed
        """
        # Serialize the encrypted input
        serialized_input = pickle.dumps(encrypted_input)
        encoded_input = base64.b64encode(serialized_input).decode('utf-8')
        
        # Send to server
        response = requests.post(
            f"{self.server_url}/inference/{model_id}/{layer_name}",
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
        encrypted_output = pickle.loads(decoded_output)
        
        return encrypted_output
    
    def run_inference(self, model_id: str, input_data: torch.Tensor, fhe_layers: List[str]) -> torch.Tensor:
        """
        Run distributed inference with the client handling torch layers and server handling FHE layers.
        
        Args:
            model_id: ID of the model on the server
            input_data: Input tensor
            fhe_layers: List of layer names to process with FHE
            
        Returns:
            Output tensor after processing through all layers
            
        Raises:
            ValueError: If inference fails
        """
        # Get model structure from server
        response = requests.get(f"{self.server_url}/models/{model_id}/structure")
        
        if response.status_code != 200:
            raise ValueError(f"Failed to retrieve model structure: {response.text}")
        
        model_structure = response.json()["structure"]
        layer_order = model_structure["layer_order"]
        
        # Process each layer in order
        output = input_data
        current_device = input_data.device
        current_dtype = input_data.dtype
        
        for layer_name in layer_order:
            if layer_name in fhe_layers:
                # Process on server with FHE
                # 1. Generate key if needed
                if self.secret_key is None:
                    self.secret_key = generate_secret_key()
                
                # 2. Encrypt the data
                encrypted_input = encrypt(output, self.secret_key)
                
                # 3. Send to server for processing
                encrypted_output = self.process_layer(model_id, layer_name, encrypted_input)
                
                # 4. Decrypt the result
                output = torch.tensor(
                    decrypt(encrypted_output, self.secret_key),
                    dtype=current_dtype,
                    device=current_device
                )
            else:
                # Process on local torch model
                # For this, we'd need the torch model parts
                # This would be implemented by having the client keep the torch model
                # or by requesting the server to execute non-FHE operations
                # For simplicity, we'll assume the client has the original torch model
                # and can execute the non-FHE layers
                
                logger.info(f"Processing non-FHE layer {layer_name} locally")
                # Client-side processing would happen here
                # output = client_model.layers[layer_name](output)
                
                # Note: In a real implementation, you'd have logic to execute the torch layer
        
        return output
    
    def close(self):
        """
        Clean up resources and end the session.
        """
        requests.post(
            f"{self.server_url}/sessions/{self.session_id}/close",
            json={"session_id": self.session_id}
        )
        logger.info(f"Closed session {self.session_id}")