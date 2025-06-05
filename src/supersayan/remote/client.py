from __future__ import annotations

import pickle
import socket
import time
from typing import Any, Dict, List, Optional, Type, Union, cast

import numpy as np
import torch
import torch.nn as nn

from supersayan.core.encryption import decrypt_from_lwes, encrypt_to_lwes
from supersayan.core.timing import (
    FHELayerTiming,
    NonFHELayerTiming,
    get_object_size_bytes,
    get_timing_collector,
    is_leaf_module,
)
from supersayan.core.types import SupersayanTensor
from supersayan.logging_config import get_logger
from supersayan.nn.convert import ModelType, SupersayanModel
from supersayan.remote.socket_utils import REQUEST_TIMEOUT, recv_obj, send_obj

logger = get_logger(__name__)


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
        enable_timing: bool = False,
    ) -> None:
        super().__init__(
            torch_model=torch_model,
            model_type=ModelType.HYBRID,
            fhe_modules=fhe_modules,
        )
        self.model_id = model_id
        self.uploaded = model_id is not None
        self.enable_timing = enable_timing

        # Note that the server_url should have the following format:
        # "host:port" (don't include http:// or https://)

        if ":" in server_url:
            host, port_str = server_url.split(":", 1)
            port = int(port_str)
        else:
            host, port = server_url, 80  # Default port is HTTP

        self.host = host
        self.port = port

        # Populate remote layer names if a pre-existing model_id was given
        if self.uploaded:
            self._get_remote_layer_names()

    def _send_request(
        self, payload: Dict[str, Any], timeout: int = REQUEST_TIMEOUT
    ) -> tuple[Dict[str, Any], float, float]:
        """
        Open a short-lived connection, send payload, return response with timing.

        Args:
            payload: The payload to send
            timeout: The timeout for the connection

        Returns:
            tuple[Dict[str, Any], float, float]: The response, send time, and receive time
        """
        sock = None

        try:
            sock = socket.create_connection((self.host, self.port), timeout=timeout)
            
            # Time the send operation
            send_start = time.time()
            conn_id = send_obj(sock, payload)
            send_end = time.time()
            send_time = send_end - send_start
            
            logger.info(
                f"[CONN:{conn_id}] Established connection to {self.host}:{self.port}"
            )
            
            # Time the receive operation
            receive_start = time.time()
            response, _ = recv_obj(sock)
            receive_end = time.time()
            receive_time = receive_end - receive_start
            
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

    def _process_layer(self, layer_name: str, encrypted_input: Any) -> tuple[Any, FHELayerTiming]:
        """
        Encrypt → remote FHE layer → decrypt with detailed timing.

        Args:
            layer_name: The name of the layer to process
            encrypted_input: The encrypted input to the layer

        Returns:
            tuple[Any, FHELayerTiming]: The output from the layer and timing info
        """
        timing = FHELayerTiming(layer_name=layer_name)
        
        if self.enable_timing:
            timing.encrypted_input_size_bytes = get_object_size_bytes(encrypted_input)

        request = {
            "command": "inference",
            "model_id": self.model_id,
            "layer_name": layer_name,
            "encrypted_input": encrypted_input,
        }

        response, send_time, receive_time = self._send_request(request)
        
        if self.enable_timing:
            timing.send_time = send_time
            timing.receive_time = receive_time
            timing.inference_time = response.get("inference_time", 0.0)
            timing.encrypted_output_size_bytes = get_object_size_bytes(response["encrypted_output"])

        return response["encrypted_output"], timing

    def _forward_hybrid(self, x: torch.Tensor) -> SupersayanTensor:
        """
        Hybrid forward pass with detailed timing.

        Args:
            x: The input tensor

        Returns:
            SupersayanTensor: The output tensor
        """
        collector = get_timing_collector() if self.enable_timing else None
        
        # Register hooks to intercept FHE layers and time non-FHE layers
        hooks = []

        def make_fhe_hook(layer_name):
            def hook(module, input, output):
                input_st = SupersayanTensor(input[0])

                timing = FHELayerTiming(layer_name=layer_name)
                
                # Time encryption
                if self.enable_timing:
                    enc_start = time.time()
                    
                enc_in = encrypt_to_lwes(input_st, self.secret_key)
                
                if self.enable_timing:
                    enc_end = time.time()
                    timing.encryption_time = enc_end - enc_start
                
                # Process layer remotely (includes send/receive timing)
                enc_out, layer_timing = self._process_layer(layer_name, enc_in)
                
                # Merge timing info
                if self.enable_timing:
                    timing.encrypted_input_size_bytes = layer_timing.encrypted_input_size_bytes
                    timing.send_time = layer_timing.send_time
                    timing.receive_time = layer_timing.receive_time
                    timing.inference_time = layer_timing.inference_time
                    timing.encrypted_output_size_bytes = layer_timing.encrypted_output_size_bytes
                
                # Time decryption
                if self.enable_timing:
                    dec_start = time.time()
                    
                dec = decrypt_from_lwes(enc_out, self.secret_key)
                
                if self.enable_timing:
                    dec_end = time.time()
                    timing.decryption_time = dec_end - dec_start
                    
                    # Add timing to collector
                    if collector:
                        collector.add_fhe_layer_timing(timing)

                return dec

            return hook

        def make_non_fhe_hook(layer_name):
            def hook(module, input, output):
                if self.enable_timing and collector:
                    # This hook runs after the layer execution
                    # We need to time the layer during execution, but PyTorch hooks
                    # run after the forward pass is complete
                    # For now, we'll use a pre-hook to get accurate timing
                    pass
                return output

            return hook

        def make_non_fhe_pre_hook(layer_name):
            def pre_hook(module, input):
                if self.enable_timing and collector:
                    # Store start time in the module
                    module._timing_start = time.time()
                return input

            return pre_hook

        def make_non_fhe_post_hook(layer_name):
            def post_hook(module, input, output):
                if self.enable_timing and collector and hasattr(module, '_timing_start'):
                    end_time = time.time()
                    torch_inference_time = end_time - module._timing_start
                    delattr(module, '_timing_start')
                    
                    timing = NonFHELayerTiming(
                        layer_name=layer_name,
                        torch_inference_time=torch_inference_time
                    )
                    collector.add_non_fhe_layer_timing(timing)
                return output

            return post_hook

        # Register hooks for all modules (only leaf modules)
        for name, module in self.original_model.named_modules():
            if not is_leaf_module(module):
                continue
                
            normalized_name = name.replace(".", "_")
            if normalized_name in self.fhe_module_names:
                hook = module.register_forward_hook(make_fhe_hook(normalized_name))
                hooks.append(hook)
            else:
                # For non-FHE layers, register both pre and post hooks for accurate timing
                if self.enable_timing:
                    pre_hook = module.register_forward_pre_hook(make_non_fhe_pre_hook(normalized_name))
                    post_hook = module.register_forward_hook(make_non_fhe_post_hook(normalized_name))
                    hooks.extend([pre_hook, post_hook])

        # Run forward pass
        with torch.no_grad():
            torch_output = self.original_model(x)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return SupersayanTensor(torch_output)

    def forward(self, x: torch.Tensor) -> SupersayanTensor:
        """
        Forward pass with optional timing collection.

        Args:
            x: The input tensor

        Returns:
            SupersayanTensor: The output tensor
        """
        self._upload_model_if_needed()

        # Handle batch inputs by processing each sample individually for timing
        if self.enable_timing and x.size(0) > 1:
            collector = get_timing_collector()
            outputs = []
            
            for i in range(x.size(0)):
                collector.start_sample()
                sample_input = x[i:i+1]  # Keep batch dimension
                sample_output = self._forward_hybrid(sample_input)
                outputs.append(sample_output)
                collector.end_sample()
            
            return SupersayanTensor(torch.cat(outputs, dim=0))
        else:
            if self.enable_timing:
                collector = get_timing_collector()
                collector.start_sample()
                result = self._forward_hybrid(x)
                collector.end_sample()
                return result
            else:
                return self._forward_hybrid(x)

    def get_timing_stats(self) -> Dict[str, Any]:
        """
        Get timing statistics from the collector.
        
        Returns:
            Dict[str, Any]: Timing statistics
        """
        if not self.enable_timing:
            return {"error": "Timing not enabled. Create client with enable_timing=True"}
        
        collector = get_timing_collector()
        return collector.get_summary_stats()
    
    def clear_timing_stats(self) -> None:
        """Clear timing statistics."""
        if self.enable_timing:
            collector = get_timing_collector()
            collector.clear()
