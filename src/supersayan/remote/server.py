import logging
import torch
import torch.nn as nn
import numpy as np
import json
import uuid
import pickle
import base64
import os
from typing import Dict, List, Optional, Tuple, Union, Any
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import datetime, timedelta
import time

from supersayan.nn.convert import SupersayanModel, ModelType, convert_model

logger = logging.getLogger(__name__)

class ModelStore:
    """
    Storage and management for uploaded models.
    
    Handles model persistence, loading, and cleanup of unused models.
    """
    def __init__(self, storage_dir: str = "/tmp/supersayan/models", cleanup_interval: int = 3600):
        """
        Initialize a model store.
        
        Args:
            storage_dir: Directory to store model files
            cleanup_interval: Interval in seconds for cleaning up unused models
        """
        self.storage_dir = storage_dir
        self.cleanup_interval = cleanup_interval
        self.models = {}  # model_id -> model_info
        self.model_lock = threading.RLock()
        
        os.makedirs(storage_dir, exist_ok=True)
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_task, daemon=True)
        self.cleanup_thread.start()
    
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
        
        with self.model_lock:
            self.models[model_id] = {
                "path": model_path,
                "last_accessed": datetime.now(),
                "model": None  # Lazy loading
            }
        
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
        with self.model_lock:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model_info = self.models[model_id]
            model_info["last_accessed"] = datetime.now()
            
            # Lazy load the model if needed
            if model_info["model"] is None:
                try:
                    model_info["model"] = torch.load(model_info["path"])
                except Exception as e:
                    raise ValueError(f"Failed to load model {model_id}: {e}")
            
            return model_info
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model from storage.
        
        Args:
            model_id: Identifier for the model
        
        Returns:
            True if the model was deleted, False otherwise
        """
        with self.model_lock:
            if model_id not in self.models:
                return False
            
            model_info = self.models[model_id]
            
            try:
                os.remove(model_info["path"])
            except OSError:
                logger.warning(f"Failed to delete model file for {model_id}")
            
            del self.models[model_id]
            return True
    
    def _cleanup_task(self):
        """
        Periodic task to clean up unused models.
        """
        while True:
            time.sleep(self.cleanup_interval)
            
            now = datetime.now()
            to_delete = []
            
            with self.model_lock:
                for model_id, model_info in self.models.items():
                    # Clean up models not accessed in the last day
                    if now - model_info["last_accessed"] > timedelta(days=1):
                        to_delete.append(model_id)
            
            for model_id in to_delete:
                logger.info(f"Cleaning up unused model {model_id}")
                self.delete_model(model_id)


class SessionManager:
    """
    Manages client sessions and associated resources.
    """
    def __init__(self, cleanup_interval: int = 3600):
        """
        Initialize a session manager.
        
        Args:
            cleanup_interval: Interval in seconds for cleaning up expired sessions
        """
        self.sessions = {}  # session_id -> session_info
        self.session_lock = threading.RLock()
        self.cleanup_interval = cleanup_interval
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_task, daemon=True)
        self.cleanup_thread.start()
    
    def create_session(self, session_id: str) -> dict:
        """
        Create a new session or update an existing one.
        
        Args:
            session_id: Unique identifier for the session
        
        Returns:
            Session information
        """
        with self.session_lock:
            if session_id in self.sessions:
                # Update last accessed time
                self.sessions[session_id]["last_accessed"] = datetime.now()
            else:
                # Create new session
                self.sessions[session_id] = {
                    "created": datetime.now(),
                    "last_accessed": datetime.now(),
                    "models": set()  # Track models used by this session
                }
            
            return self.sessions[session_id]
    
    def get_session(self, session_id: str) -> dict:
        """
        Get session information.
        
        Args:
            session_id: Unique identifier for the session
        
        Returns:
            Session information
        
        Raises:
            ValueError: If the session is not found
        """
        with self.session_lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session_info = self.sessions[session_id]
            session_info["last_accessed"] = datetime.now()
            return session_info
    
    def close_session(self, session_id: str) -> bool:
        """
        Close a session and clean up resources.
        
        Args:
            session_id: Unique identifier for the session
        
        Returns:
            True if the session was closed, False if it wasn't found
        """
        with self.session_lock:
            if session_id not in self.sessions:
                return False
            
            del self.sessions[session_id]
            return True
    
    def _cleanup_task(self):
        """
        Periodic task to clean up expired sessions.
        """
        while True:
            time.sleep(self.cleanup_interval)
            
            now = datetime.now()
            to_delete = []
            
            with self.session_lock:
                for session_id, session_info in self.sessions.items():
                    # Clean up sessions not accessed in the last hour
                    if now - session_info["last_accessed"] > timedelta(hours=1):
                        to_delete.append(session_id)
            
            for session_id in to_delete:
                logger.info(f"Cleaning up expired session {session_id}")
                self.close_session(session_id)


class SupersayanServer:
    """
    Server for hosting SuperSayan models and performing remote FHE inference.
    
    Handles model deployment, layer execution, and client session management.
    """
    def __init__(
        self, 
        storage_dir: str = "/tmp/supersayan/models",
        max_workers: int = 10
    ):
        """
        Initialize a SuperSayan server.
        
        Args:
            storage_dir: Directory to store model files
            max_workers: Maximum number of concurrent inference requests
        """
        self.model_store = ModelStore(storage_dir)
        self.session_manager = SessionManager()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def handle_upload_model(self, session_id: str, model_data_base64: str) -> dict:
        """
        Handle a model upload request.
        
        Args:
            session_id: Client session ID
            model_data_base64: Base64-encoded serialized model
        
        Returns:
            Response with model ID
        """
        # Update or create session
        session_info = self.session_manager.create_session(session_id)
        
        # Decode model data
        try:
            model_data = base64.b64decode(model_data_base64)
        except Exception as e:
            return {"error": f"Invalid model data: {e}"}, 400
        
        # Save the model
        try:
            model_id = self.model_store.save_model(model_data)
            session_info["models"].add(model_id)
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
    
    def handle_inference(self, session_id: str, model_id: str, layer_name: str, encrypted_input_base64: str) -> dict:
        """
        Handle an inference request for a specific layer.
        
        Args:
            session_id: Client session ID
            model_id: Model identifier
            layer_name: Name of the layer to execute
            encrypted_input_base64: Base64-encoded serialized encrypted input
        
        Returns:
            Response with encrypted output
        """
        try:
            # Validate session
            self.session_manager.get_session(session_id)
            
            # Get model
            model_info = self.model_store.get_model(model_id)
            model = model_info["model"]
            
            # Decode input
            encrypted_input_bytes = base64.b64decode(encrypted_input_base64)
            encrypted_input = pickle.loads(encrypted_input_bytes)
            
            # Execute layer
            if not hasattr(model, layer_name):
                return {"error": f"Layer {layer_name} not found in model"}, 404
            
            layer = getattr(model, layer_name)
            encrypted_output = layer(encrypted_input)
            
            # Encode output
            encrypted_output_bytes = pickle.dumps(encrypted_output)
            encrypted_output_base64 = base64.b64encode(encrypted_output_bytes).decode('utf-8')
            
            return {"encrypted_output": encrypted_output_base64}
        except ValueError as e:
            return {"error": str(e)}, 404
        except Exception as e:
            logger.exception(f"Inference error: {e}")
            return {"error": f"Inference failed: {e}"}, 500
    
    def handle_close_session(self, session_id: str) -> dict:
        """
        Handle a request to close a session.
        
        Args:
            session_id: Client session ID
        
        Returns:
            Response indicating success
        """
        try:
            success = self.session_manager.close_session(session_id)
            if success:
                return {"message": f"Session {session_id} closed"}
            else:
                return {"error": f"Session {session_id} not found"}, 404
        except Exception as e:
            return {"error": f"Failed to close session: {e}"}, 500