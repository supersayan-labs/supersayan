# client.py
# ------------------------------
# Simple TCP client for hybrid SuperSayan inference.
# ------------------------------
from __future__ import annotations

import hashlib
import logging
import pickle
import socket
import struct
from typing import Any, Dict, Optional, Union, List, Type, cast

import numpy as np
import torch
import torch.nn as nn

from supersayan.core.encryption import decrypt, encrypt
from supersayan.core.keygen import generate_secret_key
from supersayan.core.types import SerializableArray, convert_from_serializable
from supersayan.nn.convert import ModelType, SupersayanModel, LAYER_MAPPING

logger = logging.getLogger(__name__)

# Constants for chunking
_CHUNK_SIZE = 500 * 1024 * 1024
_HEADER_FMT = "!Q"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_COUNT_FMT = "!I"
_COUNT_SIZE = struct.calcsize(_COUNT_FMT)

_connection_counter = 0
_REQUEST_TIMEOUT = 600  # seconds


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("connection closed while receiving data")
        data.extend(chunk)
    return bytes(data)


def _recv_obj(sock: socket.socket) -> Any:
    # Read total payload size
    total_size_bytes = _recv_exact(sock, _HEADER_SIZE)
    total_size = struct.unpack(_HEADER_FMT, total_size_bytes)[0]
    # Read packet count
    count_bytes = _recv_exact(sock, _COUNT_SIZE)
    packet_count = struct.unpack(_COUNT_FMT, count_bytes)[0]
    # Read protocol-level conn_id
    conn_id_bytes = _recv_exact(sock, _HEADER_SIZE)
    conn_id = struct.unpack(_HEADER_FMT, conn_id_bytes)[0]

    logger.info(f"[CONN:{conn_id}] Receiving object: {total_size} bytes in {packet_count} packets")

    # Collect chunks
    payload = bytearray()
    for idx in range(1, packet_count + 1):
        size_bytes = _recv_exact(sock, _HEADER_SIZE)
        chunk_size = struct.unpack(_HEADER_FMT, size_bytes)[0]
        logger.info(f"[CONN:{conn_id}] Receiving packet {idx}/{packet_count}: {chunk_size} bytes")
        payload.extend(_recv_exact(sock, chunk_size))

    # Verify length
    if len(payload) != total_size:
        raise ValueError(f"[CONN:{conn_id}] Length mismatch: expected {total_size} bytes, got {len(payload)}")

    # Read 32-byte ASCII MD5 trailer
    expected_hash = _recv_exact(sock, 32).decode('ascii')
    calc_hash = hashlib.md5(payload).hexdigest()
    logger.info(f"[CONN:{conn_id}] Trailer hash: sent={expected_hash} calc={calc_hash}")
    if expected_hash != calc_hash:
        raise ValueError(f"[CONN:{conn_id}] Hash mismatch: sent={expected_hash} vs calc={calc_hash}")

    obj = pickle.loads(payload)
    return obj


def _send_obj(sock: socket.socket, obj: Any) -> int:
    global _connection_counter
    _connection_counter += 1
    conn_id = _connection_counter

    if isinstance(obj, dict):
        obj["_conn_id"] = conn_id

    payload = pickle.dumps(obj)
    total_size = len(payload)
    chunks = [payload[i : i + _CHUNK_SIZE] for i in range(0, total_size, _CHUNK_SIZE)]
    total_packets = len(chunks)

    # Compute MD5
    content_hash = hashlib.md5(payload).hexdigest()
    logger.info(f"[CONN:{conn_id}] Sending {total_size} bytes in {total_packets} packets, MD5={content_hash}")

    # Send headers
    sock.sendall(struct.pack(_HEADER_FMT, total_size))
    sock.sendall(struct.pack(_COUNT_FMT, total_packets))
    sock.sendall(struct.pack(_HEADER_FMT, conn_id))

    # Send chunks
    for idx, chunk in enumerate(chunks, start=1):
        chunk_size = len(chunk)
        logger.info(f"[CONN:{conn_id}] Sending packet {idx}/{total_packets}: {chunk_size} bytes")
        sock.sendall(struct.pack(_HEADER_FMT, chunk_size))
        sock.sendall(chunk)

    # Send 32-byte ASCII hex MD5 trailer
    sock.sendall(content_hash.encode('ascii'))
    return conn_id


# -----------------------------------------------------------------------------
# SuperSayan TCP client
# -----------------------------------------------------------------------------
_REQUEST_TIMEOUT = 600  # seconds – covers large model uploads / inference


class SupersayanClient(SupersayanModel):
    """Client that communicates with a Supersayan TCP server.

    Only the layers listed in *fhe_modules* are executed remotely in FHE;
    the rest are executed locally in clear.
    """

    def __init__(
        self,
        server_url: str,
        torch_model: nn.Module,
        fhe_modules: Optional[Union[List[str], List[Type[nn.Module]]]] = None,
        model_id: Optional[str] = None,
    ) -> None:
        super().__init__(torch_model=torch_model, model_type=ModelType.HYBRID, fhe_modules=fhe_modules)

        self.secret_key = generate_secret_key()
        self.model_id: Optional[str] = model_id
        self.uploaded = model_id is not None

        # ------------------------------------------------------------------
        # Parse host / port ("127.0.0.1:8000", "localhost", etc.)
        # ------------------------------------------------------------------
        server_url = server_url.replace("http://", "").replace("https://", "")
        if ":" in server_url:
            host, port_str = server_url.split(":", 1)
            port = int(port_str)
        else:
            host, port = server_url, 8000

        self._host: str = host or "127.0.0.1"
        self._port: int = port

        # Populate remote layer names if a pre‑existing model_id was given
        if self.uploaded:
            self._get_remote_layer_names()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _send_request(self, payload: Dict[str, Any], timeout: int = _REQUEST_TIMEOUT) -> Dict[str, Any]:
        """Open a short‑lived connection, send *payload*, return response."""
        sock: socket.socket | None = None
        try:
            sock = socket.create_connection((self._host, self._port), timeout=timeout)
            conn_id = _send_obj(sock, payload)
            logger.info(f"[CONN:{conn_id}] Established connection to {self._host}:{self._port}")
            response = _recv_obj(sock)
            logger.info(f"[CONN:{conn_id}] Completed request/response cycle")
        finally:
            if sock is not None:
                sock.close()

        if not isinstance(response, dict):
            raise ValueError("invalid response from server – expected dict")
        if not response.get("status", False):
            raise ValueError(response.get("error", "unknown server error"))
        return cast(Dict[str, Any], response)

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------
    def _upload_model_if_needed(self) -> str:
        """Serialise and upload FHE modules to the server (once per session)."""
        if self.uploaded:
            return cast(str, self.model_id)

        # Gather *direct* FHE modules and flatten names dotted → underscored
        remote_modules = nn.ModuleDict({name: self.modules_dict[name] for name in self.fhe_module_names})

        model_bytes = pickle.dumps(remote_modules)

        response = self._send_request({"command": "upload_model", "model_data": model_bytes})
        self.model_id = cast(str, response["model_id"])
        self.uploaded = True
        self._get_remote_layer_names()
        return self.model_id

    def _get_remote_layer_names(self) -> None:
        if self.model_id is None:
            return
        resp = self._send_request({"command": "get_model_structure", "model_id": self.model_id})
        structure = resp["structure"]
        self.remote_layer_names = structure.get("layer_order", [])

    # ------------------------------------------------------------------
    # Remote layer execution helpers
    # ------------------------------------------------------------------
    def _process_layer(self, layer_name: str, encrypted_input: Any) -> Any:
        """Encrypt → remote FHE layer → decrypt."""
        self._upload_model_if_needed()
        safe_name = layer_name.replace(".", "_")
        request = {
            "command": "inference",
            "model_id": self.model_id,
            "layer_name": safe_name,
            "encrypted_input": encrypted_input,
        }
        resp = self._send_request(request)
        temp = convert_from_serializable(resp["encrypted_output"])
        if isinstance(temp, SerializableArray):
            return temp.array
        return temp

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _convert_module(self, module: nn.Module) -> nn.Module:
        """Convert a PyTorch module to its Supersayan counterpart."""
        module_type = type(module)
        if module_type in LAYER_MAPPING:
            cls = LAYER_MAPPING[module_type]
            if isinstance(module, nn.Linear):
                out = cls(module.in_features, module.out_features, bias=module.bias is not None)
                out.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    out.bias.data.copy_(module.bias.data)
                return out
            if isinstance(module, nn.Conv2d):
                out = cls(
                    module.in_channels,
                    module.out_channels,
                    module.kernel_size,
                    module.stride,
                    module.padding,
                    module.dilation,
                    module.groups,
                    bias=module.bias is not None,
                )
                out.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    out.bias.data.copy_(module.bias.data)
                return out
        if isinstance(module, nn.Sequential):
            return nn.Sequential(*(self._convert_module(m) for m in module))
        raise ValueError(f"module type {module_type} not supported")

    # ------------------------------------------------------------------
    # Hybrid forward pass
    # ------------------------------------------------------------------
    def _hybrid_forward_module(self, prefix: str, module: nn.Module, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Recurse through *module*; offload FHE layers, run others locally."""
        full_name = prefix

        # If this module is to be executed in FHE based on its name
        if full_name in self.fhe_module_names:
            enc_in = encrypt(x, self.secret_key)
            enc_out = self._process_layer(full_name, enc_in)
            dec = decrypt(enc_out, self.secret_key)
            return torch.tensor(dec, dtype=x.dtype, device=x.device)
        
        # If this module should be executed in FHE based on its type
        if self.using_module_types and any(isinstance(module, t) for t in self.fhe_module_types):
            # Create a safe name for this module if not already in fhe_module_names
            safe_name = full_name.replace(".", "_")
            if safe_name not in self.fhe_module_names:
                # Add this module to the modules_dict and fhe_module_names
                self.modules_dict[safe_name] = self._convert_module(module)
                self.fhe_module_names.append(safe_name)
                # Upload model again since we added a new module
                self.uploaded = False
                self._upload_model_if_needed()
            
            enc_in = encrypt(x, self.secret_key)
            enc_out = self._process_layer(safe_name, enc_in)
            dec = decrypt(enc_out, self.secret_key)
            return torch.tensor(dec, dtype=x.dtype, device=x.device)

        # Otherwise recurse into children (if any) or execute locally
        children = list(module.named_children())
        if children:
            out = x
            for child_name, child_mod in children:
                child_full = f"{full_name}.{child_name}"
                out = self._hybrid_forward_module(child_full, child_mod, out)
            return out
        return module(x)

    def _forward_hybrid(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        self._upload_model_if_needed()
        out = x
        for name, module in self.original_model.named_children():
            out = self._hybrid_forward_module(name, module, out)
        return out

    # ------------------------------------------------------------------
    # Public forward override
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.model_type == ModelType.HYBRID:
            return self._forward_hybrid(x)
        # Pure models handled by the base class
        return super().forward(x)

    # ------------------------------------------------------------------
    # Cleanup – nothing persistent so nothing to do
    # ------------------------------------------------------------------
    def __del__(self) -> None:  # noqa: D401
        pass
