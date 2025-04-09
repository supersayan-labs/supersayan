"""
Core module for supersayan.

This module provides:
  - Bindings to the Julia backend.
  - Encryption and decryption functions.
  - Key generation functions.
  - Homomorphic operations (addition, dot product, etc.).
"""

from .bindings import jl
from .encryption import encrypt, decrypt
from .keygen import generate_secret_key
from .operations import add_lwe, dot_product_lwe

__all__ = [
    "jl",
    "encrypt",
    "decrypt",
    "generate_secret_key",
    "add_lwe",
    "dot_product_lwe",
]
