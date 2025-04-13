"""
Neural network modules for supersayan.

This package provides:
  - PyTorch-style encrypted neural network layers
  - Conversion utilities for PyTorch models to Supersayan models
"""

from .layers import Linear
from .convert import (
    convert_to_pure_supersayan,
    convert_to_hybrid_supersayan,
    convert_model,
    SupersayanModel,
    ModelType
)

__all__ = [
    "convert_to_pure_supersayan",
    "convert_to_hybrid_supersayan",
    "convert_model",
    "SupersayanModel",
    "ModelType",
    "Linear"
]