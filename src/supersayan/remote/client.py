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
from supersayan.remote.serialization import serialize_data, deserialize_data
from supersayan.remote.chunking import ChunkManager

logger = logging.getLogger(__name__)

# Timeout constants (in seconds)
CONNECTION_TIMEOUT = 30
MODEL_UPLOAD_TIMEOUT = 60
MODEL_STRUCTURE_TIMEOUT = 30
STANDARD_CHUNK_TIMEOUT = 60
INFERENCE_TIMEOUT = 300  # 5 minutes for inference is more reasonable
CLEANUP_TIMEOUT = 10

class SupersayanClient(SupersayanModel):
    """
    Client for performing remote inference with SuperSayan FHE models.
    
    This class extends SupersayanModel to handle remote execution
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
        Initialize a SuperSayan client.
        
        Args:
            server_url: The URL of the SuperSayan server
            torch_model: The PyTorch model to convert
            fhe_modules: List of module names or types to execute in FHE
            model_id: Optional ID of a model already on the server
        """
        super(SupersayanClient, self).__init__(
            torch_model=torch_model,
            model_type=ModelType.HYBRID,
            fhe_modules=fhe_modules
        )
        
        self.server_url = server_url.rstrip('/')
        self.secret_key = generate_secret_key()
        self.model_id = model_id
        self.uploaded = False
        
        self.sio = socketio.Client(
            reconnection=True,
            reconnection_attempts=5,
            reconnection_delay=1,
            reconnection_delay_max=5,
            randomization_factor=0.5,
            logger=True
        )

        self.chunk_manager = ChunkManager()

        self._setup_socketio()
        self._connect_to_server()
        
        if model_id is not None:
            self.uploaded = True
            self._get_remote_layer_names()
    
    def _setup_socketio(self):
        """Setup Socket.IO event handlers."""
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
        """Connect to the Socket.IO server."""
        if not self.sio.connected:
            try:
                self.sio.connect(self.server_url, transports=["websocket"], wait_timeout=CONNECTION_TIMEOUT)
            except Exception as e:
                logger.error(f"Failed to connect to server: {e}")
                raise ConnectionError(f"Failed to connect to server: {e}")
    
    def _ensure_connected(self):
        """Ensure that the socket is connected, reconnect if needed."""
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
            
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
        
        try:
            remote_modules = nn.ModuleDict({
                name: self.modules_dict[name]
                for name in self.fhe_module_names
            })
            
            torch.save(remote_modules, temp_path)
            
            with open(temp_path, 'rb') as f:
                model_data = f.read()
            
            encoded_model = base64.b64encode(model_data).decode('utf-8')
            
            self._ensure_connected()
            
            response = self.sio.call('upload_model', {'model_data': encoded_model}, timeout=MODEL_UPLOAD_TIMEOUT)

            data, status_code = response
            
            if status_code != 200:
                if isinstance(data, dict) and 'error' in data:
                    raise ValueError(f"Failed to upload model: {data['error']}")
                else:
                    raise ValueError(f"Failed to upload model with status code: {status_code}")
            
            self.model_id = data["model_id"]
            logger.info(f"Model uploaded with ID: {self.model_id}")
            
            self._get_remote_layer_names()
            
            self.uploaded = True
            return self.model_id
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def _get_remote_layer_names(self):
        """Get the layer names from the remote model."""
        if self.model_id is None:
            return
        
        self._ensure_connected()
        
        response = self.sio.call('get_model_structure', {'model_id': self.model_id}, timeout=MODEL_STRUCTURE_TIMEOUT)
        
        data, status_code = response
        
        if status_code != 200:
            if isinstance(data, dict) and 'error' in data:
                raise ValueError(f"Failed to retrieve model structure: {data['error']}")
            else:
                raise ValueError(f"Failed to retrieve model structure with status code: {status_code}")
        
        structure = data["structure"]
        self.remote_layer_names = structure.get("layer_order", [])
    
    def _send_chunked_data(self, event_name: str, data: Dict[str, Any], large_data_key: str, timeout: int = STANDARD_CHUNK_TIMEOUT) -> Tuple[Dict[str, Any], int]:
        """
        Send data that may be too large using chunks.
        
        Args:
            event_name: Socket.IO event name
            data: Payload to send
            large_data_key: Key in data dictionary for large data
            timeout: Timeout for each chunk transfer
            
        Returns:
            Tuple containing response data and status code
        """
        large_data = data[large_data_key]
        
        if not self.chunk_manager.needs_chunking(large_data):
            return self.sio.call(event_name, data, timeout=timeout)
        
        metadata = {k: v for k, v in data.items() if k != large_data_key}
        
        start_data = {
            **metadata,
            'chunked': True,
            large_data_key: None
        }
        
        def send_chunk(chunk):
            return self.sio.call("chunk", chunk, timeout=timeout)
        
        logger.info(f"Starting chunked transfer for {event_name}")
        transfer_id, chunks = self.chunk_manager.create_transfer(large_data)
        
        start_data['transfer_id'] = transfer_id
        start_data['total_chunks'] = len(chunks)
        
        start_response = self.sio.call(f"{event_name}_start", start_data, timeout=timeout)
        start_data, status_code = start_response
        
        if status_code != 200:
            logger.error(f"Failed to start chunked transfer: {start_data}")
            return start_response
            
        for chunk in chunks:
            chunk_response = send_chunk(chunk)
            chunk_data, chunk_status = chunk_response
            
            if chunk_status != 200:
                logger.error(f"Failed to send chunk {chunk['chunk_index'] + 1}/{chunk['total_chunks']}")
                return chunk_response
        
        logger.info(f"All chunks sent for transfer {transfer_id}, waiting for completion")
        return self.sio.call(f"{event_name}_complete", {
            'transfer_id': transfer_id
        }, timeout=timeout*2)

    def _process_layer(self, layer_name: str, encrypted_input: np.ndarray) -> np.ndarray:
        """
        Process a single encrypted layer on the server.
        
        Args:
            layer_name: Name of the layer to process
            encrypted_input: Encrypted input data
            
        Returns:
            Encrypted output from the layer
        """
        self._upload_model_if_needed()

        try:
            encoded_input = serialize_data(encrypted_input)
        except Exception as e:
            logger.exception(f"Serialization error: {e}")
            raise ValueError(f"Failed to serialize input: {e}")
        
        self._ensure_connected()
        
        logger.info(f"Sending inference request for layer: {layer_name}")
        
        try:
            safe_name = layer_name.replace('.', '_')
            
            response = self._send_chunked_data('inference', {
                'model_id': self.model_id,
                'layer_name': safe_name,
                'encrypted_input': encoded_input
            }, 'encrypted_input', timeout=INFERENCE_TIMEOUT)
            
            data, status_code = response
            
            if status_code != 200:
                if isinstance(data, dict) and 'error' in data:
                    raise ValueError(f"Layer processing failed: {data['error']}")
                else:
                    raise ValueError(f"Layer processing failed with status code: {status_code}")
            
            encrypted_output_base64 = data.get("encrypted_output")
            if encrypted_output_base64 is None and data.get("chunked", False):
                transfer_id = data.get("transfer_id")
                total_chunks = data.get("total_chunks", 0)
                
                if not transfer_id or not total_chunks:
                    raise ValueError("Invalid chunked response metadata")
                
                logger.info(f"Receiving chunked response with {total_chunks} chunks")
                
                def get_chunk(chunk_index):
                    chunk_response = self.sio.call("get_response_chunk", {
                        'transfer_id': transfer_id, 
                        'chunk_index': chunk_index
                    }, timeout=INFERENCE_TIMEOUT)
                    
                    chunk_data, chunk_status = chunk_response
                    if chunk_status != 200:
                        raise ValueError(f"Failed to retrieve chunk {chunk_index + 1}/{total_chunks}")
                    
                    return chunk_data
                
                encrypted_output_base64 = self.chunk_manager.receive_chunked(
                    transfer_id, 
                    total_chunks, 
                    get_chunk
                )
                
                try:
                    self.sio.call("cleanup_response_chunks", {'transfer_id': transfer_id}, timeout=CLEANUP_TIMEOUT)
                except Exception as e:
                    logger.warning(f"Failed to clean up response chunks: {e}")
            
            encrypted_output = deserialize_data(encrypted_output_base64)
            return encrypted_output
                
        except socketio.exceptions.TimeoutError:
            logger.error(f"Timeout while processing layer {layer_name}")
            if not self.sio.connected:
                self._connect_to_server()
            raise ValueError(f"Timeout while processing layer {layer_name}")
    
    def _forward_hybrid(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for hybrid model with recursive execution of nested modules,
        dispatching any layer in self.fhe_module_names to the remote FHE server.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after processing
        """
        self._upload_model_if_needed()

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
        
        Args:
            prefix: Module prefix
            module: Module to process
            x: Input tensor
            
        Returns:
            Output tensor after processing
        """
        full_name = prefix

        if full_name in self.fhe_module_names:
            enc_in = encrypt(x, self.secret_key)
            enc_out = self._process_layer(full_name, enc_in)
            dec = decrypt(enc_out, self.secret_key)
            return torch.tensor(dec, dtype=x.dtype, device=x.device)
        
        children = list(module.named_children())
        if children:
            out = x
            for child_name, child_mod in children:
                child_full = f"{full_name}.{child_name}"
                out = self._hybrid_forward_module(child_full, child_mod, out)
            return out
        else:
            return module(x)
        
    def __del__(self):
        """Disconnect from server when client is destroyed."""
        if hasattr(self, 'sio') and self.sio.connected:
            try:
                self.sio.disconnect()
            except:
                pass