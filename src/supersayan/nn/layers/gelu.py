import torch
import torch.nn as nn

from supersayan.core.bindings import SupersayanTFHE
from supersayan.core.types import SupersayanTensor
from supersayan.logging_config import get_logger

logger = get_logger(__name__)


class GELU(nn.Module):
    """Supersayan equivalent of torch.nn.GELU."""

    def __init__(self, approximate: str = "none"):
        super(GELU, self).__init__()
        self.approximate = approximate

    def forward(self, input: SupersayanTensor) -> SupersayanTensor:
        """
        Forward pass.

        Args:
            input: The input tensor

        Returns:
            SupersayanTensor: The output tensor
        """
        julia_result = SupersayanTFHE.Layers.GELU.gelu_forward(
            input.to_julia(), self.approximate
        )

        return SupersayanTensor._from_julia(julia_result)

    def __repr__(self):
        return f"GELU(approximate='{self.approximate}')" 