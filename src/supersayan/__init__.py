"""
supersayan: A Python frontend for Julia Supersayan TFHE Fully Homomorphic Encryption.

This package provides:
  - Core cryptographic functions, key management, and homomorphic operations.
  - Python bindings to a Julia backend for TFHE encryption, decryption, and arithmetic.
  - Neural network modules (in supersayan.nn) that operate on encrypted data using a PyTorch-style interface.
"""

from .core import encryption, keygen, bindings, operations
from .nn import layers

__all__ = ["encryption", "keygen", "bindings", "operations", "layers"]
