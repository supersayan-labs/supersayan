from __future__ import annotations
import hashlib
import logging
import pickle
import socket
import struct
from typing import Any, Dict, Optional, Union, List, Type, cast
import torch
import torch.nn as nn

from supersayan.core.encryption import decrypt_from_lwes, encrypt_to_lwes
from supersayan.core.keygen import generate_secret_key
from supersayan.nn.convert import ModelType, SupersayanModel

logger = logging.getLogger(__name__)

# Global connection counter
_connection_counter = 0

# Constants for chunking
_CHUNK_SIZE = 500 * 1024 * 1024  # 500MB per packet
_HEADER_FMT = "!Q"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_COUNT_FMT = "!I"
_COUNT_SIZE = struct.calcsize(_COUNT_FMT)
_REQUEST_TIMEOUT = 600  # seconds


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """
    Receive exactly n bytes from the socket.

    Args:
        sock: The socket to receive from
        n: The number of bytes to receive

    Returns:
        bytes: The bytes received
    """
    data = bytearray()

    while len(data) < n:
        chunk = sock.recv(n - len(data))

        if not chunk:
            raise ConnectionError("connection closed while receiving data")
        data.extend(chunk)

    return bytes(data)


def _recv_obj(sock: socket.socket) -> Any:
    """
    Receive an object from the socket.

    Args:
        sock: The socket to receive from

    Returns:
        Any: The object received
    """
    # Read total payload size
    total_size_bytes = _recv_exact(sock, _HEADER_SIZE)
    total_size = struct.unpack(_HEADER_FMT, total_size_bytes)[0]

    # Read packet count
    count_bytes = _recv_exact(sock, _COUNT_SIZE)
    packet_count = struct.unpack(_COUNT_FMT, count_bytes)[0]

    # Read protocol-level conn_id
    conn_id_bytes = _recv_exact(sock, _HEADER_SIZE)
    conn_id = struct.unpack(_HEADER_FMT, conn_id_bytes)[0]

    logger.info(
        f"[CONN:{conn_id}] Receiving object: {total_size} bytes in {packet_count} packets"
    )

    # Collect chunks
    payload = bytearray()
    for idx in range(1, packet_count + 1):
        size_bytes = _recv_exact(sock, _HEADER_SIZE)
        chunk_size = struct.unpack(_HEADER_FMT, size_bytes)[0]
        logger.info(
            f"[CONN:{conn_id}] Receiving packet {idx}/{packet_count}: {chunk_size} bytes"
        )
        payload.extend(_recv_exact(sock, chunk_size))

    # Verify length
    if len(payload) != total_size:
        raise ValueError(
            f"[CONN:{conn_id}] Length mismatch: expected {total_size} bytes, got {len(payload)}"
        )

    # Read 32-byte ASCII MD5 trailer
    expected_hash = _recv_exact(sock, 32).decode("ascii")
    calc_hash = hashlib.md5(payload).hexdigest()
    logger.info(f"[CONN:{conn_id}] Trailer hash: sent={expected_hash} calc={calc_hash}")
    if expected_hash != calc_hash:
        raise ValueError(
            f"[CONN:{conn_id}] Hash mismatch: sent={expected_hash} vs calc={calc_hash}"
        )

    obj = pickle.loads(payload)

    return obj


def _send_obj(sock: socket.socket, obj: Any) -> int:
    """
    Send an object to the socket.

    Args:
        sock: The socket to send to
        obj: The object to send

    Returns:
        int: The connection ID
    """
    global _connection_counter
    _connection_counter += 1
    conn_id = _connection_counter

    obj["_conn_id"] = conn_id
    obj = pickle.dumps(obj)

    total_size = len(obj)
    chunks = [obj[i : i + _CHUNK_SIZE] for i in range(0, total_size, _CHUNK_SIZE)]
    total_packets = len(chunks)

    # Compute MD5
    content_hash = hashlib.md5(obj).hexdigest()
    logger.info(
        f"[CONN:{conn_id}] Sending {total_size} bytes in {total_packets} packets, MD5={content_hash}"
    )

    # Send headers
    sock.sendall(struct.pack(_HEADER_FMT, total_size))
    sock.sendall(struct.pack(_COUNT_FMT, total_packets))
    sock.sendall(struct.pack(_HEADER_FMT, conn_id))

    # Send chunks
    for idx, chunk in enumerate(chunks, start=1):
        chunk_size = len(chunk)
        logger.info(
            f"[CONN:{conn_id}] Sending packet {idx}/{total_packets}: {chunk_size} bytes"
        )
        sock.sendall(struct.pack(_HEADER_FMT, chunk_size))
        sock.sendall(chunk)

    # Send 32-byte ASCII hex MD5 trailer
    sock.sendall(content_hash.encode("ascii"))
    return conn_id


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

        self.secret_key = generate_secret_key()
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

        # Populate remote layer names if a pre-existing model_id was given
        if self.uploaded:
            self._get_remote_layer_names()

    def _send_request(
        self, payload: Dict[str, Any], timeout: int = _REQUEST_TIMEOUT
    ) -> Dict[str, Any]:
        """
        Open a short-lived connection, send payload, return response.

        Args:
            payload: The payload to send
            timeout: The timeout for the connection

        Returns:
            Dict[str, Any]: The response from the server
        """
        sock = None

        try:
            sock = socket.create_connection((self.host, self.port), timeout=timeout)
            conn_id = _send_obj(sock, payload)
            logger.info(
                f"[CONN:{conn_id}] Established connection to {self.host}:{self.port}"
            )
            response = _recv_obj(sock)
            logger.info(f"[CONN:{conn_id}] Completed request/response cycle")
        finally:
            if sock is not None:
                sock.close()

        if not isinstance(response, dict):
            raise ValueError("Invalid response from server – expected dict")
        if not response.get("status", False):
            raise ValueError(response.get("error", "Unknown server error"))

        return cast(Dict[str, Any], response)

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

        response = self._send_request(
            {"command": "upload_model", "model_data": model_bytes}
        )

        self.model_id = str(response["model_id"])
        self.uploaded = True

        self._get_remote_layer_names()

        return self.model_id

    def _get_remote_layer_names(self) -> None:
        if self.model_id is None:
            return

        resp = self._send_request(
            {"command": "get_model_structure", "model_id": self.model_id}
        )
        structure = resp["structure"]

        self.remote_layer_names = structure.get("layer_order", [])

    def _process_layer(self, layer_name: str, encrypted_input: Any) -> Any:
        """
        Encrypt → remote FHE layer → decrypt.

        Args:
            layer_name: The name of the layer to process
            encrypted_input: The encrypted input to the layer

        Returns:
            Any: The output from the layer
        """
        request = {
            "command": "inference",
            "model_id": self.model_id,
            "layer_name": layer_name,
            "encrypted_input": encrypted_input,
        }

        response = self._send_request(request)

        return response["encrypted_output"]

    def _hybrid_forward_module(
        self, prefix: str, module: nn.Module, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Recurse through module; offload FHE layers, run others locally.

        Args:
            prefix: The prefix of the module
            module: The module to forward
            x: The input tensor

        Returns:
            torch.Tensor: The output tensor
        """
        full_name = prefix

        if full_name in self.fhe_module_names:
            enc_in = encrypt_to_lwes(x, self.secret_key)

            enc_out = self._process_layer(full_name, enc_in)

            dec = decrypt_from_lwes(enc_out, self.secret_key)

            return torch.tensor(dec, dtype=x.dtype, device=x.device)

        children = list(module.named_children())

        if children:
            out = x

            for child_name, child_module in children:
                child_full = f"{full_name}.{child_name}"
                out = self._hybrid_forward_module(child_full, child_module, out)

            return out

        return module(x)

    def _forward_hybrid(self, x: torch.Tensor) -> torch.Tensor:
        """
        Hybrid forward pass.

        Args:
            x: The input tensor

        Returns:
            torch.Tensor: The output tensor
        """
        out = x

        for name, module in self.original_model.named_children():
            out = self._hybrid_forward_module(name, module, out)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: The input tensor

        Returns:
            torch.Tensor: The output tensor
        """
        self._upload_model_if_needed()

        return self._forward_hybrid(x)
