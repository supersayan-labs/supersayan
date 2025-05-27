from __future__ import annotations
import logging
import socket
from typing import Any, Dict, Optional, Union, List, Type, cast
import torch
import pickle
import torch.nn as nn
import numpy as np

from supersayan.core.encryption import decrypt_from_lwes, encrypt_to_lwes
from supersayan.core.keygen import generate_secret_key
from supersayan.nn.convert import ModelType, SupersayanModel
from supersayan.remote.socket_utils import (
    recv_obj,
    send_obj,
    REQUEST_TIMEOUT,
)

logger = logging.getLogger(__name__)


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
        self, payload: Dict[str, Any], timeout: int = REQUEST_TIMEOUT
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
            conn_id = send_obj(sock, payload)
            logger.info(
                f"[CONN:{conn_id}] Established connection to {self.host}:{self.port}"
            )
            response, _ = recv_obj(sock)
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
        self, prefix: str, module: nn.Module, x: np.ndarray[np.float32]
    ) -> np.ndarray[np.float32]:
        """
        Recurse through module; offload FHE layers, run others locally.

        Args:
            prefix: The prefix of the module
            module: The module to forward
            x: The input tensor

        Returns:
            np.ndarray[np.float32]: The output tensor
        """
        full_name = prefix

        # Normalize the name to match fhe_module_names format (dots replaced with underscores)
        normalized_name = full_name.replace(".", "_")

        if normalized_name in self.fhe_module_names:
            enc_in = encrypt_to_lwes(x, self.secret_key)

            enc_out = self._process_layer(normalized_name, enc_in)

            dec = decrypt_from_lwes(enc_out, self.secret_key)

            return dec

        children = list(module.named_children())

        if children:
            out = x

            for child_name, child_module in children:
                child_full = f"{full_name}.{child_name}"
                out = self._hybrid_forward_module(child_full, child_module, out)

            return out

        torch_input = torch.from_numpy(x)
        torch_output = module(torch_input)

        return torch_output.detach().numpy().astype(np.float32)

    def _forward_hybrid(self, x: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        """
        Hybrid forward pass.

        Args:
            x: The input tensor

        Returns:
            np.ndarray[np.float32]: The output tensor
        """
        out = x

        for name, module in self.original_model.named_children():
            out = self._hybrid_forward_module(name, module, out)

        return out

    def forward(
        self, x: np.ndarray[np.float32] | torch.Tensor
    ) -> np.ndarray[np.float32]:
        """
        Forward pass.

        Args:
            x: The input tensor

        Returns:
            np.ndarray[np.float32]: The output tensor
        """
        self._upload_model_if_needed()

        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.float32)

        return self._forward_hybrid(x)
