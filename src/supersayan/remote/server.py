import logging
import torch
import numpy as np
import uuid
import pickle
import base64
import os
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import datetime, timedelta
import time
from .serialization import serialize_data, deserialize_data

# Add safe_globals for ModuleDict early to avoid reference errors
from torch.nn.modules.container import ModuleDict
import torch.serialization
torch.serialization.add_safe_globals([ModuleDict])

logger = logging.getLogger(__name__)

class ModelStore:
    """
    Simple storage for uploaded models.
    
    Handles model persistence and loading.
    """
    def __init__(self, storage_dir: str = "server_db/models"):
        """
        Initialize a model store.
        
        Args:
            storage_dir: Directory to store model files
        """
        self.storage_dir = storage_dir
        self.models = {}  # model_id -> model object
        os.makedirs(storage_dir, exist_ok=True)
    
    def save_model(self, model_data: bytes) -> str:
        """
        Save a model to storage.
        
        Args:
            model_data: Serialized model data
        
        Returns:
            model_id: Identifier for the stored model
        """
        model_id = str(uuid.uuid4())
        model_path = os.path.join(self.storage_dir, f"{model_id}.pt")
        
        with open(model_path, 'wb') as f:
            f.write(model_data)
        
        return model_id
    
    def get_model(self, model_id: str) -> dict:
        """
        Get a model by ID.
        
        Args:
            model_id: Identifier for the model
        
        Returns:
            Model information
        
        Raises:
            ValueError: If the model is not found
        """
        # Check if model is already loaded in memory
        if model_id in self.models:
            return {"model": self.models[model_id]}
        
        # Try to load the model from disk
        model_path = os.path.join(self.storage_dir, f"{model_id}.pt")
        if not os.path.exists(model_path):
            raise ValueError(f"Model {model_id} not found")
        
        try:
            # First try with weights_only=False (most compatible option)
            model = torch.load(
                model_path,
                weights_only=False,
                map_location=torch.device('cpu')
            )
        except Exception as e:
            try:
                # If that fails, try with weights_only=True and safe_globals
                model = torch.load(
                    model_path,
                    weights_only=True,
                    map_location=torch.device('cpu')
                )
            except Exception as e2:
                logger.error(f"Failed to load model {model_id}: {str(e)} then {str(e2)}")
                raise ValueError(f"Failed to load model {model_id}: {e}")
        
        # Cache the model in memory
        self.models[model_id] = model
        return {"model": model}
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model from storage.
        
        Args:
            model_id: Identifier for the model
        
        Returns:
            True if the model was deleted, False otherwise
        """
        # Remove from memory if present
        if model_id in self.models:
            del self.models[model_id]
        
        # Remove from disk if present
        model_path = os.path.join(self.storage_dir, f"{model_id}.pt")
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
                return True
            except OSError:
                logger.warning(f"Failed to delete model file for {model_id}")
                return False
        
        return False


class SupersayanServer:
    """
    Server for hosting SuperSayan models and performing remote FHE inference.
    
    Handles model deployment and layer execution.
    """
    def __init__(
        self, 
        storage_dir: str = "server_db/models",
    ):
        """
        Initialize a SuperSayan server.
        
        Args:
            storage_dir: Directory to store model files
        """
        self.model_store = ModelStore(storage_dir)
    
    def handle_upload_model(self, model_data_base64: str) -> dict:
        """
        Handle a model upload request.
        
        Args:
            model_data_base64: Base64-encoded serialized model
        
        Returns:
            Response with model ID
        """
        # Decode model data
        try:
            model_data = base64.b64decode(model_data_base64)
        except Exception as e:
            return {"error": f"Invalid model data: {e}"}, 400
        
        # Save the model
        try:
            model_id = self.model_store.save_model(model_data)
            return {"model_id": model_id}
        except Exception as e:
            return {"error": f"Failed to save model: {e}"}, 500
    
    def handle_get_model_structure(self, model_id: str) -> dict:
        """
        Handle a request to get the structure of a model.
        
        Args:
            model_id: Model identifier
        
        Returns:
            Response with model structure
        """
        try:
            model_info = self.model_store.get_model(model_id)
            model = model_info["model"]
            
            # Extract structure information
            layer_order = []
            for name, _ in model.named_children():
                layer_order.append(name)
            
            return {
                "structure": {
                    "layer_order": layer_order
                }
            }
        except ValueError as e:
            return {"error": str(e)}, 404
        except Exception as e:
            return {"error": f"Failed to get model structure: {e}"}, 500
    
    def handle_inference(self, model_id: str, layer_name: str, encrypted_input_base64: str) -> dict:
        """
        Handle an inference request for a specific layer.
        
        Args:
            model_id: Model identifier
            layer_name: Name of the layer to execute
            encrypted_input_base64: Base64-encoded serialized encrypted input
        
        Returns:
            Response with encrypted output
        """
        try:
            # Get model
            model_info = self.model_store.get_model(model_id)
            model = model_info["model"]
            
            # Decode input using the serialization utility
            encrypted_input = deserialize_data(encrypted_input_base64)

            # Debug info for encrypted input (after deserialization)
            logger.debug(f"Encrypted input received for model {model_id}, layer {layer_name}")
            logger.debug(f"Input type: {type(encrypted_input)}")
            if hasattr(encrypted_input, 'shape'):
                logger.debug(f"Input shape: {encrypted_input.shape}")
            elif hasattr(encrypted_input, '__len__'):
                logger.debug(f"Input length: {len(encrypted_input)}")
            
            # Execute layer
            if not hasattr(model, layer_name):
                return {"error": f"Layer {layer_name} not found in model"}, 404
            
            layer = getattr(model, layer_name)
            encrypted_output = layer(encrypted_input)

            # Debug info for encrypted output
            logger.debug(f"Output generated for model {model_id}, layer {layer_name}")
            logger.debug(f"Output type: {type(encrypted_output)}")
            if hasattr(encrypted_output, 'shape'):
                logger.debug(f"Output shape: {encrypted_output.shape}")
            elif hasattr(encrypted_output, '__len__'):
                logger.debug(f"Output length: {len(encrypted_output)}")
            
            try:
                # Serialize the output using the serialization utility
                encrypted_output_base64 = serialize_data(encrypted_output)
                return {"encrypted_output": encrypted_output_base64}
            except Exception as e:
                logger.exception(f"Serialization error: {e}")
                return {"error": f"Failed to serialize output: {e}"}, 500
            
        except ValueError as e:
            return {"error": str(e)}, 404
        except Exception as e:
            logger.exception(f"Inference error: {e}")
            return {"error": f"Inference failed: {e}"}, 500