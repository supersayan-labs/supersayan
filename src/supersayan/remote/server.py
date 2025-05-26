from __future__ import annotations
import hashlib
import logging
import os
import pickle
import uuid
from typing import Any, Dict, List

import torch

logger = logging.getLogger(__name__)


class ModelStore:
    """Helper to load / save PyTorch ``.pt`` blobs on disk."""

    def __init__(self, storage_dir: str = "server_db/models") -> None:
        self.storage_dir = storage_dir
        self.models: Dict[str, torch.nn.Module] = {}

        os.makedirs(storage_dir, exist_ok=True)

    def _path(self, model_id: str) -> str:
        """
        Get the path to the model file.

        Args:
            model_id: The ID of the model

        Returns:
            str: The path to the model file
        """
        return os.path.join(self.storage_dir, f"{model_id}.pt")

    def save_model(self, model_blob: bytes) -> str:
        """
        Save the model blob to disk.

        Args:
            model_blob: The model blob to save

        Returns:
            str: The ID of the saved model
        """
        model_id = str(uuid.uuid4())

        # Calculate hash of model blob
        model_hash = hashlib.md5(model_blob).hexdigest()
        logger.info(f"Saving model with hash: {model_hash}")

        # Store both model and its hash
        with open(self._path(model_id), "wb") as f:
            f.write(model_blob)

        # Store hash separately for verification
        with open(self._path(model_id) + ".hash", "w") as f:
            f.write(model_hash)

        return model_id

    def get_model(self, model_id: str) -> torch.nn.Module:
        """
        Get the model from disk.

        Args:
            model_id: The ID of the model

        Returns:
            torch.nn.Module: The model
        """
        if model_id in self.models:
            return self.models[model_id]

        path = self._path(model_id)
        if not os.path.exists(path):
            raise ValueError(f"model {model_id} not found")

        # Read model bytes
        with open(path, "rb") as f:
            model_blob = f.read()

        # Verify hash if available
        calculated_hash = hashlib.md5(model_blob).hexdigest()
        hash_path = path + ".hash"

        if os.path.exists(hash_path):
            with open(hash_path, "r") as f:
                stored_hash = f.read().strip()

            logger.info(
                f"Model {model_id} - Stored hash: {stored_hash}, Calculated hash: {calculated_hash}"
            )

            if stored_hash != calculated_hash:
                logger.warning(
                    f"Hash mismatch for model {model_id}! Stored: {stored_hash}, Calculated: {calculated_hash}"
                )
        else:
            logger.info(
                f"Model {model_id} - Calculated hash: {calculated_hash} (no stored hash available)"
            )

        # Deserialize model
        model = pickle.loads(model_blob)
        self.models[model_id] = model

        logger.info(
            "Loaded model %s (layers: %s)",
            model_id,
            ", ".join(dict(model.named_children()).keys()),
        )

        return model

    def delete_model(self, model_id: str) -> bool:
        """
        Delete the model from disk.

        Args:
            model_id: The ID of the model

        Returns:
            bool: True if the model was deleted, False otherwise
        """
        self.models.pop(model_id, None)

        try:
            os.remove(self._path(model_id))
            return True
        except FileNotFoundError:
            return False


class SupersayanServer:
    """
    Server that handles model uploads and inference.
    """

    def __init__(self, storage_dir: str = "server_db/models") -> None:
        self.model_store = ModelStore(storage_dir)

    def handle_upload_model_bytes(self, model_bytes: bytes) -> str:
        """
        Save model bytes and return its model_id.

        Args:
            model_bytes: The model bytes to save

        Returns:
            str: The ID of the saved model
        """
        # Calculate hash for logging
        model_hash = hashlib.md5(model_bytes).hexdigest()
        logger.info(f"Handling model upload with hash: {model_hash}")

        return self.model_store.save_model(model_bytes)

    def handle_get_model_structure(self, model_id: str) -> Dict[str, List[str]]:
        """
        Get the model structure.

        Args:
            model_id: The ID of the model

        Returns:
            Dict[str, List[str]]: The model structure
        """
        model = self.model_store.get_model(model_id)
        layer_order = [name for name, _ in model.named_children()]

        return {"layer_order": layer_order}

    def handle_inference(
        self, model_id: str, layer_name: str, encrypted_input: Any
    ) -> Any:
        """
        Handle inference.

        Args:
            model_id: The ID of the model
            layer_name: The name of the layer
            encrypted_input: The encrypted input

        Returns:
            Any: The output from the layer
        """
        model = self.model_store.get_model(model_id)

        if not hasattr(model, layer_name):
            raise ValueError(f"Layer {layer_name!r} not found in model {model_id}")

        layer = getattr(model, layer_name)

        encrypted_output = layer(encrypted_input)

        return encrypted_output
