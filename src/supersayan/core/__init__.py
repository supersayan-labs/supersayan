"""
Core module for supersayan.

This module provides:
  - Bindings to the Julia backend.
  - Encryption and decryption functions.
  - Key generation functions.
  - Homomorphic operations (addition, dot product, etc.).
"""

from .encryption import encrypt, decrypt
from .keygen import generate_secret_key

__all__ = [
    "SupersayanTFHE",
    "encrypt",
    "decrypt",
    "generate_secret_key"
]
