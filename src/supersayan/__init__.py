"""
supersayan: A Python frontend for Julia Supersayan TFHE Fully Homomorphic Encryption.

This package provides:
  - Core cryptographic functions, key management, and homomorphic operations.
  - Python bindings to a Julia backend for TFHE encryption, decryption, and arithmetic.
  - Neural network modules (in supersayan.nn) that operate on encrypted data using a PyTorch-style interface.
  - Conversion utilities for PyTorch models to Supersayan-compatible encrypted models.
  - Client-server architecture for distributed execution of encrypted models.
"""

from .core import encryption, keygen, bindings, operations
from .nn import layers
from .nn.convert import (
    convert_to_pure_supersayan,
    convert_to_hybrid_supersayan,
    convert_model,
    SupersayanModel,
    ModelType,
)

__all__ = [
    "encryption",
    "keygen",
    "bindings",
    "operations",
    "layers",
    "convert_to_pure_supersayan",
    "convert_to_hybrid_supersayan",
    "convert_model",
    "SupersayanModel",
    "ModelType",
]
