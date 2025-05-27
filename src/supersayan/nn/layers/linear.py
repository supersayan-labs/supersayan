import torch
import torch.nn as nn
import numpy as np
from typing import List
import logging

from supersayan.core.bindings import SupersayanTFHE
from supersayan.core.types import LWE

logger = logging.getLogger(__name__)


class Linear(nn.Module):
    """Supersayan equivalent of torch.nn.Linear."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.randn(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: np.ndarray[LWE]) -> np.ndarray[LWE]:
        """
        Forward pass.

        Args:
            input: The input tensor

        Returns:
            np.ndarray[LWE]: The output tensor
        """

        weight_np = self.weight.detach().cpu().numpy()

        bias_np = self.bias.detach().cpu().numpy() if self.bias is not None else None

        julia_result = SupersayanTFHE.Layers.Linear.linear_forward(
            input, weight_np, bias_np
        )

        return np.asarray(julia_result, dtype=np.float32)

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"
