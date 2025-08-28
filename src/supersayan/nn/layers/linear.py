from typing import List

import numpy as np
import torch
import torch.nn as nn

from supersayan.core.bindings import SupersayanTFHE
from supersayan.core.types import SupersayanTensor
from supersayan.logging_config import get_logger

logger = get_logger(__name__)


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

    def forward(self, input: SupersayanTensor) -> SupersayanTensor:
        """
        Forward pass.

        Args:
            input: The input tensor

        Returns:
            SupersayanTensor: The output tensor
        """

        weight_np = self.weight.detach().cpu().numpy()

        bias_np = self.bias.detach().cpu().numpy() if self.bias is not None else None

        julia_result = SupersayanTFHE.Layers.Linear.linear_forward(
            input.to_julia(), weight_np, bias_np
        )

        return SupersayanTensor._from_julia(julia_result)

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"
