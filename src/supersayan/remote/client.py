from __future__ import annotations

import pickle
import socket
import time
import sys
from typing import Any, Dict, List, Optional, Type, Union, cast

import numpy as np
import torch
import torch.nn as nn

from supersayan.core.encryption import decrypt_from_lwes, encrypt_to_lwes
from supersayan.core.types import SupersayanTensor
from supersayan.logging_config import get_logger
from supersayan.nn.convert import ModelType, SupersayanModel
from supersayan.remote.socket_utils import REQUEST_TIMEOUT, recv_obj, send_obj

logger = get_logger(__name__)


class LayerTiming:
    """Class to store detailed timing information for a layer."""
    
    def __init__(self, layer_name: str, layer_type: str):
        self.layer_name = layer_name
        self.layer_type = layer_type
        
        # FHE layer timings
        self.encryption_time: float = 0.0
        self.encrypted_input_size: int = 0
        self.send_time: float = 0.0
        self.server_inference_time: float = 0.0
        self.receive_time: float = 0.0
        self.encrypted_output_size: int = 0
        self.decryption_time: float = 0.0
        
        # Non-FHE layer timings
        self.torch_inference_time: float = 0.0
        
        # Sample timing information
        self.sample_timings: List[Dict[str, Any]] = []
    
    def add_fhe_sample(self, encryption_time: float, encrypted_input_size: int,
                      send_time: float, server_inference_time: float,
                      receive_time: float, encrypted_output_size: int,
                      decryption_time: float):
        """Add timing data for one FHE sample."""
        sample_data = {
            "encryption_time": encryption_time,
            "encrypted_input_size": encrypted_input_size,
            "send_time": send_time,
            "server_inference_time": server_inference_time,
            "receive_time": receive_time,
            "encrypted_output_size": encrypted_output_size,
            "decryption_time": decryption_time,
            "total_time": encryption_time + send_time + server_inference_time + receive_time + decryption_time
        }
        self.sample_timings.append(sample_data)
    
    def add_torch_sample(self, torch_inference_time: float):
        """Add timing data for one torch sample."""
        sample_data = {
            "torch_inference_time": torch_inference_time
        }
        self.sample_timings.append(sample_data)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for this layer."""
        if not self.sample_timings:
            return {}
            
        if self.layer_type == "FHE":
            # Average over all samples
            avg_encryption = np.mean([s["encryption_time"] for s in self.sample_timings])
            avg_input_size = np.mean([s["encrypted_input_size"] for s in self.sample_timings])
            avg_send = np.mean([s["send_time"] for s in self.sample_timings])
            avg_server = np.mean([s["server_inference_time"] for s in self.sample_timings])
            avg_receive = np.mean([s["receive_time"] for s in self.sample_timings])
            avg_output_size = np.mean([s["encrypted_output_size"] for s in self.sample_timings])
            avg_decryption = np.mean([s["decryption_time"] for s in self.sample_timings])
            avg_total = np.mean([s["total_time"] for s in self.sample_timings])
            
            return {
                "layer_name": self.layer_name,
                "layer_type": self.layer_type,
                "num_samples": len(self.sample_timings),
                "avg_encryption_time": avg_encryption,
                "avg_encrypted_input_size": avg_input_size,
                "avg_send_time": avg_send,
                "avg_server_inference_time": avg_server,
                "avg_receive_time": avg_receive,
                "avg_encrypted_output_size": avg_output_size,
                "avg_decryption_time": avg_decryption,
                "avg_total_time": avg_total
            }
        else:  # Torch layer
            avg_torch = np.mean([s["torch_inference_time"] for s in self.sample_timings])
            return {
                "layer_name": self.layer_name,
                "layer_type": self.layer_type,
                "num_samples": len(self.sample_timings),
                "avg_torch_inference_time": avg_torch
            }


class SupersayanClient(SupersayanModel):
    """
    Client that communicates with a Supersayan TCP server.

    Only the layers listed in fhe_modules are executed remotely in FHE;
    the rest are executed locally in clear.
    """

    def __init__(
        self,
        server_url: str,
        torch_model: nn.Module,
        fhe_modules: Optional[Union[List[str], List[Type[nn.Module]]]] = None,
        model_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            torch_model=torch_model,
            model_type=ModelType.HYBRID,
            fhe_modules=fhe_modules,
        )
        self.model_id = model_id
        self.uploaded = model_id is not None

        # Note that the server_url should have the following format:
        # "host:port" (don't include http:// or https://)

        if ":" in server_url:
            host, port_str = server_url.split(":", 1)
            port = int(port_str)
        else:
            host, port = server_url, 80  # Default port is HTTP

        self.host = host
        self.port = port

        # Timing data
        self.layer_timings: Dict[str, LayerTiming] = {}
        self.enable_timing = True

        # Populate remote layer names if a pre-existing model_id was given
        if self.uploaded:
            self._get_remote_layer_names()

    def _is_leaf_layer(self, module: nn.Module) -> bool:
        """Check if a module is a leaf layer (has no children)."""
        return len(list(module.children())) == 0

    def _get_leaf_layers(self) -> Dict[str, nn.Module]:
        """Get all leaf layers in the model."""
        leaf_layers = {}
        for name, module in self.original_model.named_modules():
            if self._is_leaf_layer(module):
                leaf_layers[name] = module
        return leaf_layers

    def _get_object_size(self, obj: Any) -> int:
        """Get the size of an object in bytes."""
        return sys.getsizeof(pickle.dumps(obj))

    def _send_request(
        self, payload: Dict[str, Any], timeout: int = REQUEST_TIMEOUT
    ) -> tuple[Dict[str, Any], float, float]:
        """
        Open a short-lived connection, send payload, return response with timing.

        Args:
            payload: The payload to send
            timeout: The timeout for the connection

        Returns:
            tuple: (response, send_time, receive_time)
        """
        sock = None
        send_time = 0.0
        receive_time = 0.0

        try:
            sock = socket.create_connection((self.host, self.port), timeout=timeout)
            
            # Time the send operation
            send_start = time.time()
            conn_id = send_obj(sock, payload)
            send_time = time.time() - send_start
            
            logger.info(
                f"[CONN:{conn_id}] Established connection to {self.host}:{self.port}"
            )
            
            # Time the receive operation
            receive_start = time.time()
            response, _ = recv_obj(sock)
            receive_time = time.time() - receive_start
            
            logger.info(f"[CONN:{conn_id}] Completed request/response cycle")
        finally:
            if sock is not None:
                sock.close()

        if not isinstance(response, dict):
            raise ValueError("Invalid response from server – expected dict")
        if not response.get("status", False):
            raise ValueError(response.get("error", "Unknown server error"))

        return cast(Dict[str, Any], response), send_time, receive_time

    def _upload_model_if_needed(self) -> str:
        """
        Serialise and upload FHE modules to the server (once per session).

        Returns:
            str: The model ID
        """
        if self.uploaded:
            return self.model_id

        remote_modules = nn.ModuleDict(
            {name: self.modules_dict[name] for name in self.fhe_module_names}
        )

        model_bytes = pickle.dumps(remote_modules)

        response, _, _ = self._send_request(
            {"command": "upload_model", "model_data": model_bytes}
        )

        self.model_id = str(response["model_id"])
        self.uploaded = True

        self._get_remote_layer_names()

        return self.model_id

    def _get_remote_layer_names(self) -> None:
        if self.model_id is None:
            return

        resp, _, _ = self._send_request(
            {"command": "get_model_structure", "model_id": self.model_id}
        )
        structure = resp["structure"]

        self.remote_layer_names = structure.get("layer_order", [])

    def _process_layer(self, layer_name: str, encrypted_input: Any) -> tuple[Any, float, float, float]:
        """
        Encrypt → remote FHE layer → decrypt with detailed timing.

        Args:
            layer_name: The name of the layer to process
            encrypted_input: The encrypted input to the layer

        Returns:
            tuple: (output, send_time, server_inference_time, receive_time)
        """
        request = {
            "command": "inference",
            "model_id": self.model_id,
            "layer_name": layer_name,
            "encrypted_input": encrypted_input,
        }

        response, send_time, receive_time = self._send_request(request)
        server_inference_time = response.get("inference_time", 0.0)

        return response["encrypted_output"], send_time, server_inference_time, receive_time

    def _forward_hybrid(self, x: torch.Tensor) -> SupersayanTensor:
        """
        Hybrid forward pass with detailed timing.

        Args:
            x: The input tensor

        Returns:
            SupersayanTensor: The output tensor
        """
        # Clear previous timing data for new forward pass
        if self.enable_timing:
            self.layer_timings.clear()
        
        # Get all leaf layers
        leaf_layers = self._get_leaf_layers()
        
        # Register hooks to intercept layers
        hooks = []

        def make_fhe_hook(layer_name, normalized_name):
            def hook(module, input, output):
                if not self.enable_timing:
                    # Original behavior without timing
                    input_st = SupersayanTensor(input[0])
                    enc_in = encrypt_to_lwes(input_st, self.secret_key)
                    enc_out, _, _, _ = self._process_layer(normalized_name, enc_in)
                    dec = decrypt_from_lwes(enc_out, self.secret_key)
                    return dec
                
                # Initialize timing data for this layer
                if normalized_name not in self.layer_timings:
                    self.layer_timings[normalized_name] = LayerTiming(layer_name, "FHE")
                
                input_st = SupersayanTensor(input[0])

                # Time encryption
                enc_start = time.time()
                enc_in = encrypt_to_lwes(input_st, self.secret_key)
                encryption_time = time.time() - enc_start
                
                # Get encrypted input size
                encrypted_input_size = self._get_object_size(enc_in)

                # Process layer (includes send, inference, receive timing)
                enc_out, send_time, server_inference_time, receive_time = self._process_layer(normalized_name, enc_in)
                
                # Get encrypted output size
                encrypted_output_size = self._get_object_size(enc_out)
                
                # Time decryption
                dec_start = time.time()
                dec = decrypt_from_lwes(enc_out, self.secret_key)
                decryption_time = time.time() - dec_start

                # Store timing data
                self.layer_timings[normalized_name].add_fhe_sample(
                    encryption_time, encrypted_input_size, send_time,
                    server_inference_time, receive_time, encrypted_output_size,
                    decryption_time
                )

                return dec

            return hook

        def make_torch_hook(layer_name):
            def hook(module, input, output):
                if not self.enable_timing:
                    return output
                
                # Initialize timing data for this layer
                if layer_name not in self.layer_timings:
                    self.layer_timings[layer_name] = LayerTiming(layer_name, "Torch")
                
                # Time torch inference
                torch_start = time.time()
                # The actual computation already happened, so we just measure the overhead
                # For proper timing, we need to re-run the computation
                with torch.no_grad():
                    torch_output = module(*input)
                torch_inference_time = time.time() - torch_start
                
                # Store timing data
                self.layer_timings[layer_name].add_torch_sample(torch_inference_time)
                
                return torch_output

            return hook

        # Register hooks for leaf layers only
        for name, module in leaf_layers.items():
            normalized_name = name.replace(".", "_")
            if normalized_name in self.fhe_module_names:
                # This is an FHE layer
                hook = module.register_forward_hook(make_fhe_hook(name, normalized_name))
                hooks.append(hook)
            else:
                # This is a regular PyTorch layer
                if self.enable_timing:
                    hook = module.register_forward_hook(make_torch_hook(name))
                    hooks.append(hook)

        # Run forward pass
        with torch.no_grad():
            torch_output = self.original_model(x)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return SupersayanTensor(torch_output)

    def forward(self, x: torch.Tensor) -> SupersayanTensor:
        """
        Forward pass.

        Args:
            x: The input tensor

        Returns:
            SupersayanTensor: The output tensor
        """
        self._upload_model_if_needed()

        return self._forward_hybrid(x)

    def get_timing_summary(self) -> Dict[str, Any]:
        """
        Get timing summary for all layers.
        
        Returns:
            Dict containing timing summaries for all layers
        """
        summary = {
            "layers": {},
            "total_fhe_layers": 0,
            "total_torch_layers": 0
        }
        
        for layer_name, timing in self.layer_timings.items():
            layer_summary = timing.get_summary()
            if layer_summary:
                summary["layers"][layer_name] = layer_summary
                if timing.layer_type == "FHE":
                    summary["total_fhe_layers"] += 1
                else:
                    summary["total_torch_layers"] += 1
        
        return summary

    def reset_timing(self) -> None:
        """Reset all timing data."""
        self.layer_timings.clear()

    def set_timing_enabled(self, enabled: bool) -> None:
        """Enable or disable timing collection."""
        self.enable_timing = enabled
