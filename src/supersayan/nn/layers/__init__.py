"""
Neural network layers implementing encrypted operations.

This subpackage provides:
  - Linear: A fully-connected layer using homomorphic dot products.
"""

from .linear import Linear

__all__ = ["Linear"]
