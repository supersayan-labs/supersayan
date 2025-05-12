# client.py
"""Simple TCP client for hybrid SuperSayan inference.

Replaces the previous Socket.IO implementation with a lightweight TCP
protocol that uses pickle for (de)serialisation.
"""
from __future__ import annotations

import hashlib
import io
import logging
import pickle
import socket
import struct
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.container import ModuleDict

from supersayan.core.encryption import decrypt, encrypt
from supersayan.core.keygen import generate_secret_key
from supersayan.nn.convert import ModelType, SupersayanModel

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# TCP helpers
# -----------------------------------------------------------------------------
_HEADER_FMT = "!Q"  # unsigned long long – 8‑byte payload length
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

# Connection counter for tracking requests/responses
_connection_counter = 0


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Receive *exactly* ``n`` bytes from *sock* (blocking)."""
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("connection closed while receiving data")
        data.extend(chunk)
    return bytes(data)


def _send_obj(sock: socket.socket, obj: Any) -> int:
    """pickle.dumps *obj* and stream it preceded by its length header."""
    global _connection_counter
    _connection_counter += 1
    conn_id = _connection_counter

    # Tag with connection ID if dict
    if isinstance(obj, dict):
        obj["_conn_id"] = conn_id

    # 1) Serialize the object *without* a _hash
    payload_content = pickle.dumps(obj)
    payload_size0 = len(payload_content)

    # 2) Compute its MD5
    content_hash = hashlib.md5(payload_content).hexdigest()

    # 3) Inject that hash into the dict
    if isinstance(obj, dict):
        obj["_hash"] = content_hash

    # 4) Serialize final payload (with _hash inside)
    payload = pickle.dumps(obj)
    payload_size = len(payload)

    logger.info(
        f"[CONN:{conn_id}] Sending object: "
        f"size={payload_size} bytes, "
        f"content_hash={content_hash}"
    )

    # 5) Send header + payload
    sock.sendall(struct.pack(_HEADER_FMT, payload_size))
    sock.sendall(payload)

    return conn_id


def _recv_obj(sock: socket.socket) -> Any:
    """Read a length-prefixed pickle payload, verify hash, and return the object."""
    size_bytes = _recv_exact(sock, _HEADER_SIZE)
    size = struct.unpack(_HEADER_FMT, size_bytes)[0]

    payload = _recv_exact(sock, size)
    actual_length = len(payload)
    if actual_length != size:
        logger.error(f"Length mismatch: Expected {size} bytes but received {actual_length} bytes")
        raise ValueError(f"Length mismatch: Expected {size} bytes but received {actual_length} bytes")

    # Compute MD5 of exactly what we received
    received_hash = hashlib.md5(payload).hexdigest()

    # Unpickle
    obj = pickle.loads(payload)

    conn_id = "unknown"
    if isinstance(obj, dict) and "_hash" in obj:
        # first preserve the sent hash & conn_id
        sent_hash = obj["_hash"]
        conn_id   = obj.get("_conn_id", "unknown")

        # recompute MD5 over everything *except* _hash (but still including _conn_id)
        calc_hash = hashlib.md5(
            pickle.dumps({k: v for k, v in obj.items() if k != "_hash"})
        ).hexdigest()

        logger.info(
            f"[CONN:{conn_id}] Received object: "
            f"size={actual_length} bytes, "
            f"sent_hash={sent_hash}, "
            f"calc_hash={calc_hash}"
        )
        if sent_hash != calc_hash:
            logger.warning(f"[CONN:{conn_id}] Hash mismatch!")

        # *now* strip our metadata so downstream code sees only the payload
        obj.pop("_hash", None)
        obj.pop("_conn_id", None)

    elif isinstance(obj, dict):
        # no hash present
        conn_id = obj.pop("_conn_id", "unknown")
        logger.info(
            f"[CONN:{conn_id}] Received object (no hash): "
            f"size={actual_length} bytes, "
            f"recv_hash={received_hash}"
        )

    else:
        logger.info(
            f"[CONN:unknown] Received non-dict object: "
            f"size={actual_length} bytes, "
            f"recv_hash={received_hash}"
        )

    return obj

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
        return resp["encrypted_output"]

    # ------------------------------------------------------------------
    # Hybrid forward pass
    # ------------------------------------------------------------------
    def _hybrid_forward_module(self, prefix: str, module: nn.Module, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Recurse through *module*; offload FHE layers, run others locally."""
        full_name = prefix

        # If this module is to be executed in FHE – encrypt → remote → decrypt
        if full_name in self.fhe_module_names:
            enc_in = encrypt(x, self.secret_key)
            enc_out = self._process_layer(full_name, enc_in)
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
