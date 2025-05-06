import logging
import torch
import torch.nn as nn
import numpy as np
import uuid
import pickle
import base64
import os
import socketio
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
        
        # Initialize Socket.IO client with reconnection enabled
        self.sio = socketio.Client(
            reconnection=True,
            reconnection_attempts=5,
            reconnection_delay=1,
            reconnection_delay_max=5,
            randomization_factor=0.5,
            logger=True
        )

        self._setup_socketio()
        self._connect_to_server()
        
        # Both pure and hybrid models need remote execution
        # If model_id is provided, assume the model is already on the server
        if model_id is not None:
            self.uploaded = True
            self._get_remote_layer_names()
    
    def _setup_socketio(self):
        """Setup Socket.IO event handlers"""
        @self.sio.event
        def connect():
            logger.info("Connected to SuperSayan server")
            
        @self.sio.event
        def connect_error(data):
            logger.error(f"Connection error: {data}")
            
        @self.sio.event
        def disconnect():
            logger.info("Disconnected from SuperSayan server")
    
    def _connect_to_server(self):
        """Connect to the Socket.IO server"""
        if not self.sio.connected:
            try:
                self.sio.connect(self.server_url, transports=["websocket"], wait_timeout=30)  # Increased timeout for connection
            except Exception as e:
                logger.error(f"Failed to connect to server: {e}")
                raise ConnectionError(f"Failed to connect to server: {e}")
    
    def _ensure_connected(self):
        """Ensure that the socket is connected, reconnect if needed"""
        if not self.sio.connected:
            logger.warning("Socket disconnected, attempting to reconnect...")
            self._connect_to_server()
    
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
            # upload every flattened FHE module under its safe name
            remote_modules = nn.ModuleDict({
                name: self.modules_dict[name]
                for name in self.fhe_module_names
            })
            
            # Save just the FHE modules
            torch.save(remote_modules, temp_path)
            
            # Upload the model
            with open(temp_path, 'rb') as f:
                model_data = f.read()
            
            encoded_model = base64.b64encode(model_data).decode('utf-8')
            
            # Ensure connection before upload
            self._ensure_connected()
            
            # Use Socket.IO for model upload (synchronous call-and-response)
            response = self.sio.call('upload_model', {'model_data': encoded_model}, timeout=60)

            data, status_code = response
            
            # Check if status code indicates an error
            if status_code != 200:
                if isinstance(data, dict) and 'error' in data:
                    raise ValueError(f"Failed to upload model: {data['error']}")
                else:
                    raise ValueError(f"Failed to upload model with status code: {status_code}")
            
            self.model_id = data["model_id"]
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
        
        # Ensure connection before request
        self._ensure_connected()
        
        # Use Socket.IO for getting model structure
        response = self.sio.call('get_model_structure', {'model_id': self.model_id}, timeout=30)
        
        data, status_code = response
        
        # Check if status code indicates an error
        if status_code != 200:
            if isinstance(data, dict) and 'error' in data:
                raise ValueError(f"Failed to retrieve model structure: {data['error']}")
            else:
                raise ValueError(f"Failed to retrieve model structure with status code: {status_code}")
        
        structure = data["structure"]
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
        
        # Print encoded input size in MB
        input_size_mb = len(encoded_input) / (1024 * 1024)
        logger.info(f"Encoded input size: {input_size_mb:.2f} MB")
        
        # Ensure connection before inference
        self._ensure_connected()
        
        logger.info(f"Sending inference request for layer: {layer_name}")
        
        # Send to server using Socket.IO with extended timeout
        try:
            safe_name = layer_name.replace('.', '_')
            response = self.sio.call('inference', {
                'model_id': self.model_id,
                'layer_name': safe_name,
                'encrypted_input': encoded_input
            }, timeout=50000)

            
            logger.info(f"Received response for layer: {layer_name}")
            
            data, status_code = response
            
            # Check if status code indicates an error
            if status_code != 200:
                if isinstance(data, dict) and 'error' in data:
                    raise ValueError(f"Layer processing failed: {data['error']}")
                else:
                    raise ValueError(f"Layer processing failed with status code: {status_code}")
            
            # Deserialize the result using the utility
            try:
                encrypted_output = deserialize_data(data["encrypted_output"])
                
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
                
        except socketio.exceptions.TimeoutError:
            logger.error(f"Timeout while processing layer {layer_name}")
            # Reconnect after timeout
            if not self.sio.connected:
                logger.info("Reconnecting after timeout...")
                self._connect_to_server()
            raise ValueError(f"Timeout while processing layer {layer_name}")
    
    def _forward_hybrid(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for hybrid model with recursive execution of nested modules,
        dispatching any layer in self.fhe_module_names to the remote FHE server.
        """
        # Make sure the remote model is uploaded once
        self._upload_model_if_needed()

        # Kick off recursion over the top-level children
        out = x
        for name, module in self.original_model.named_children():
            out = self._hybrid_forward_module(name, module, out)
        return out

    def _hybrid_forward_module(
        self,
        prefix: str,
        module: torch.nn.Module,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Recursively forward through `module`.  If `prefix` is in
        self.fhe_module_names, do encrypt→remote→decrypt; otherwise,
        recurse into children or run locally.
        """
        # Full module identifier
        full_name = prefix

        # If this is one of our FHE layers, do it remotely
        if full_name in self.fhe_module_names:
            # encrypt, remote-execute, decrypt
            enc_in = encrypt(x, self.secret_key)
            enc_out = self._process_layer(full_name, enc_in)
            dec = decrypt(enc_out, self.secret_key)
            return torch.tensor(dec, dtype=x.dtype, device=x.device)
        
        # Otherwise: if it has submodules, dive into them in declaration order
        children = list(module.named_children())
        if children:
            out = x
            for child_name, child_mod in children:
                child_full = f"{full_name}.{child_name}"
                out = self._hybrid_forward_module(child_full, child_mod, out)
            return out
        else:
            # Leaf module (Conv, BN, ReLU, etc.) — just run it locally
            return module(x)
        
    def __del__(self):
        """Disconnect from server when client is destroyed"""
        if hasattr(self, 'sio') and self.sio.connected:
            try:
                self.sio.disconnect()
            except:
                pass