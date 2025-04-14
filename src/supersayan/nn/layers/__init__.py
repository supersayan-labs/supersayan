"""
Neural network layers implementing encrypted operations.

This subpackage provides:
  - Linear: A fully-connected layer using homomorphic dot products.
  - Conv2d: A convolutional layer using homomorphic operations.
"""

from .linear import Linear
from .conv2d import Conv2d
__all__ = ["Linear", "Conv2d"]
