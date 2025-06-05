from __future__ import annotations

import pickle
import socket
import time
import sys
from typing import Any, Dict, List, Optional, Type, Union, cast
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from supersayan.core.encryption import decrypt_from_lwes, encrypt_to_lwes
from supersayan.core.types import SupersayanTensor
from supersayan.logging_config import get_logger
from supersayan.nn.convert import ModelType, SupersayanModel
from supersayan.remote.socket_utils import REQUEST_TIMEOUT, recv_obj, send_obj

logger = get_logger(__name__)


class TimingCollector:
    """Collects detailed timing and size metrics for benchmark analysis."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all collected metrics."""
        # Per-layer metrics (lists to store per-sample data)
        self.fhe_layers = {}  # layer_name -> dict of metrics lists
        self.non_fhe_layers = {}  # layer_name -> dict of metrics lists
        
        # Total metrics across all layers (lists to store per-sample data)
        self.total_encryption_time = []
        self.total_send_time = []
        self.total_inference_time = []
        self.total_receive_time = []
        self.total_decryption_time = []
        self.total_torch_inference_time = []
    
    def add_fhe_layer_metrics(self, layer_name: str, encryption_time: float, 
                            encrypted_input_size: int, send_time: float,
                            inference_time: float, receive_time: float,
                            encrypted_output_size: int, decryption_time: float):
        """Add metrics for an FHE layer."""
        if layer_name not in self.fhe_layers:
            self.fhe_layers[layer_name] = {
                'encryption_times': [],
                'encrypted_input_sizes': [],
                'send_times': [],
                'inference_times': [],
                'receive_times': [],
                'encrypted_output_sizes': [],
                'decryption_times': []
            }
        
        self.fhe_layers[layer_name]['encryption_times'].append(encryption_time)
        self.fhe_layers[layer_name]['encrypted_input_sizes'].append(encrypted_input_size)
        self.fhe_layers[layer_name]['send_times'].append(send_time)
        self.fhe_layers[layer_name]['inference_times'].append(inference_time)
        self.fhe_layers[layer_name]['receive_times'].append(receive_time)
        self.fhe_layers[layer_name]['encrypted_output_sizes'].append(encrypted_output_size)
        self.fhe_layers[layer_name]['decryption_times'].append(decryption_time)
    
    def add_non_fhe_layer_metrics(self, layer_name: str, torch_inference_time: float):
        """Add metrics for a non-FHE layer."""
        if layer_name not in self.non_fhe_layers:
            self.non_fhe_layers[layer_name] = {
                'torch_inference_times': []
            }
        
        self.non_fhe_layers[layer_name]['torch_inference_times'].append(torch_inference_time)
    
    def finalize_sample(self):
        """Finalize metrics for the current sample by computing totals."""
        # Sum up FHE layer times for this sample
        sample_encryption_time = 0
        sample_send_time = 0
        sample_inference_time = 0
        sample_receive_time = 0
        sample_decryption_time = 0
        
        for layer_metrics in self.fhe_layers.values():
            if layer_metrics['encryption_times']:
                sample_encryption_time += layer_metrics['encryption_times'][-1]
                sample_send_time += layer_metrics['send_times'][-1]
                sample_inference_time += layer_metrics['inference_times'][-1]
                sample_receive_time += layer_metrics['receive_times'][-1]
                sample_decryption_time += layer_metrics['decryption_times'][-1]
        
        # Sum up non-FHE layer times for this sample
        sample_torch_inference_time = 0
        for layer_metrics in self.non_fhe_layers.values():
            if layer_metrics['torch_inference_times']:
                sample_torch_inference_time += layer_metrics['torch_inference_times'][-1]
        
        self.total_encryption_time.append(sample_encryption_time)
        self.total_send_time.append(sample_send_time)
        self.total_inference_time.append(sample_inference_time)
        self.total_receive_time.append(sample_receive_time)
        self.total_decryption_time.append(sample_decryption_time)
        self.total_torch_inference_time.append(sample_torch_inference_time)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all collected metrics with averages."""
        def avg(lst):
            return sum(lst) / len(lst) if lst else 0
        
        summary = {
            'fhe_layers': {},
            'non_fhe_layers': {},
            'totals': {}
        }
        
        # FHE layer averages
        for layer_name, metrics in self.fhe_layers.items():
            summary['fhe_layers'][layer_name] = {
                'avg_encryption_time': avg(metrics['encryption_times']),
                'avg_encrypted_input_size': avg(metrics['encrypted_input_sizes']),
                'avg_send_time': avg(metrics['send_times']),
                'avg_inference_time': avg(metrics['inference_times']),
                'avg_receive_time': avg(metrics['receive_times']),
                'avg_encrypted_output_size': avg(metrics['encrypted_output_sizes']),
                'avg_decryption_time': avg(metrics['decryption_times']),
                'sample_count': len(metrics['encryption_times'])
            }
        
        # Non-FHE layer averages
        for layer_name, metrics in self.non_fhe_layers.items():
            summary['non_fhe_layers'][layer_name] = {
                'avg_torch_inference_time': avg(metrics['torch_inference_times']),
                'sample_count': len(metrics['torch_inference_times'])
            }
        
        # Total averages
        summary['totals'] = {
            'avg_total_encryption_time': avg(self.total_encryption_time),
            'avg_total_send_time': avg(self.total_send_time),
            'avg_total_inference_time': avg(self.total_inference_time),
            'avg_total_receive_time': avg(self.total_receive_time),
            'avg_total_decryption_time': avg(self.total_decryption_time),
            'avg_total_torch_inference_time': avg(self.total_torch_inference_time),
            'total_samples': len(self.total_encryption_time) if self.total_encryption_time else 0
        }
        
        return summary


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
        self.timing_collector = TimingCollector()

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

    def _get_object_size_bytes(self, obj: Any) -> int:
        """Get the size of an object in bytes."""
        return sys.getsizeof(pickle.dumps(obj))

    def _send_request(
        self, payload: Dict[str, Any], timeout: int = REQUEST_TIMEOUT
    ) -> tuple[Dict[str, Any], float, float, float]:
        """
        Open a short-lived connection, send payload, return response with timing.

        Args:
            payload: The payload to send
            timeout: The timeout for the connection

        Returns:
            Tuple[Dict[str, Any], float, float, float]: The response, send_time, server_inference_time, receive_time
        """
        sock = None

        try:
            sock = socket.create_connection((self.host, self.port), timeout=timeout)
            
            # Time the sending
            send_start = time.time()
            conn_id = send_obj(sock, payload)
            send_end = time.time()
            send_time = send_end - send_start
            
            logger.info(
                f"[CONN:{conn_id}] Established connection to {self.host}:{self.port}"
            )
            
            # Time the total wait (includes server processing + network receive)
            wait_start = time.time()
            response, _ = recv_obj(sock)
            wait_end = time.time()
            total_wait_time = wait_end - wait_start
            
            logger.info(f"[CONN:{conn_id}] Completed request/response cycle")
        finally:
            if sock is not None:
                sock.close()

        if not isinstance(response, dict):
            raise ValueError("Invalid response from server – expected dict")
        if not response.get("status", False):
            raise ValueError(response.get("error", "Unknown server error"))

        # Extract server inference time if available, otherwise assume 0
        server_inference_time = response.get("server_inference_time", 0.0)
        
        # Calculate pure network receive time (total wait minus server processing)
        receive_time = total_wait_time - server_inference_time

        return cast(Dict[str, Any], response), send_time, server_inference_time, receive_time

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

        response, _, _, _ = self._send_request(
            {"command": "upload_model", "model_data": model_bytes}
        )

        self.model_id = str(response["model_id"])
        self.uploaded = True

        self._get_remote_layer_names()

        return self.model_id

    def _get_remote_layer_names(self) -> None:
        if self.model_id is None:
            return

        resp, _, _, _ = self._send_request(
            {"command": "get_model_structure", "model_id": self.model_id}
        )
        structure = resp["structure"]

        self.remote_layer_names = structure.get("layer_order", [])

    def _process_layer(self, layer_name: str, encrypted_input: Any) -> tuple[Any, float, float, float, int]:
        """
        Encrypt → remote FHE layer → decrypt.

        Args:
            layer_name: The name of the layer to process
            encrypted_input: The encrypted input to the layer

        Returns:
            Tuple[Any, float, float, float, int]: output, send_time, inference_time, receive_time, output_size
        """
        request = {
            "command": "inference",
            "model_id": self.model_id,
            "layer_name": layer_name,
            "encrypted_input": encrypted_input,
        }

        # Get properly separated timing from server
        response, send_time, server_inference_time, receive_time = self._send_request(request)

        encrypted_output = response["encrypted_output"]
        output_size = self._get_object_size_bytes(encrypted_output)

        return encrypted_output, send_time, server_inference_time, receive_time, output_size

    def _forward_hybrid(self, x: torch.Tensor) -> SupersayanTensor:
        """
        Hybrid forward pass with detailed timing.

        Args:
            x: The input tensor

        Returns:
            SupersayanTensor: The output tensor
        """
        # Register hooks to intercept FHE layers
        hooks = []

        def make_fhe_hook(layer_name):
            def hook(module, input, output):
                input_st = SupersayanTensor(input[0])

                # Time encryption
                enc_start = time.time()
                enc_in = encrypt_to_lwes(input_st, self.secret_key)
                enc_end = time.time()
                encryption_time = enc_end - enc_start
                encrypted_input_size = self._get_object_size_bytes(enc_in)

                # Process layer (includes send, inference, receive timing)
                enc_out, send_time, inference_time, receive_time, encrypted_output_size = self._process_layer(layer_name, enc_in)
                
                # Time decryption
                dec_start = time.time()
                dec = decrypt_from_lwes(enc_out, self.secret_key)
                dec_end = time.time()
                decryption_time = dec_end - dec_start

                # Record metrics
                self.timing_collector.add_fhe_layer_metrics(
                    layer_name, encryption_time, encrypted_input_size,
                    send_time, inference_time, receive_time,
                    encrypted_output_size, decryption_time
                )

                return dec

            return hook

        def make_non_fhe_hook(layer_name):
            def hook(module, input, output):
                # Time the layer inference
                torch_start = time.time()
                # The actual computation already happened, so we just measure a dummy operation
                # This is a limitation - we can't easily separate individual layer times in PyTorch
                # So we'll use a different approach below
                torch_end = time.time()
                torch_inference_time = torch_end - torch_start
                
                # This won't be accurate, so we'll handle non-FHE timing differently
                return output

            return hook

        # Register hooks for FHE modules
        for name, module in self.original_model.named_modules():
            normalized_name = name.replace(".", "_")
            if normalized_name in self.fhe_module_names:
                hook = module.register_forward_hook(make_fhe_hook(normalized_name))
                hooks.append(hook)

        # For non-FHE layers, we'll time the entire forward pass of non-hooked layers
        # This is simpler and more accurate than trying to time individual layers
        non_fhe_start = time.time()
        
        # Run forward pass
        with torch.no_grad():
            torch_output = self.original_model(x)
            
        non_fhe_end = time.time()
        total_non_fhe_time = non_fhe_end - non_fhe_start

        # Record total non-FHE time as a single "layer"
        self.timing_collector.add_non_fhe_layer_metrics("all_non_fhe_layers", total_non_fhe_time)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Finalize metrics for this sample
        self.timing_collector.finalize_sample()

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
        """Get a summary of timing metrics collected during inference."""
        return self.timing_collector.get_summary()

    def reset_timing(self) -> None:
        """Reset timing metrics collection."""
        self.timing_collector.reset()
