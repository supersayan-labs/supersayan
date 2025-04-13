"""
Neural network layers implementing encrypted operations.

This subpackage provides:
  - Linear: A fully-connected layer using homomorphic dot products.
  - Conv2d: A 2D convolutional layer using Toeplitz-based encoding and BSGS optimization.
"""

from .linear import Linear
from .conv2d import Conv2d

__all__ = ["Linear", "Conv2d"]
