import torch
import torch.nn as nn
import numpy as np
from typing import List
import logging

from supersayan.core.bindings import SupersayanTFHE
from supersayan.core.types import LWE

logger = logging.getLogger(__name__)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights and bias as torch Tensors.
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: np.ndarray[LWE]) -> np.ndarray[LWE]:
        """
        Performs the forward pass using homomorphic operations.

        Args:
            input (np.ndarray[LWE]): A NumPy array of LWE ciphertexts with shape (batch, in_features).

        Returns:
            np.ndarray[LWE]: A NumPy array of LWE ciphertexts with shape (batch, out_features).
        """
        logger.info(f"Linear forward pass with input shape: {input.shape}")
        batch_size = input.shape[0]

        # Detach the weight and bias parameters as plain numbers.
        weight_np = (
            self.weight.detach().cpu().numpy()
        )  # shape: (out_features, in_features)
        bias_np = self.bias.detach().cpu().numpy() if self.bias is not None else None
        
        logger.info(f"About to call Julia implementation directly")
        # Call Julia implementation directly
        print(f"All of the types : input={type(input)}, weight_np={type(weight_np)}, bias_np={type(bias_np)}")
        print(f"Shapes: input={input.shape}, weight_np={weight_np.shape}, bias_np={bias_np.shape if bias_np is not None else None}")
        julia_result = SupersayanTFHE.Layers.Linear.linear_forward(
            input, weight_np, bias_np
        )
        logger.info(f"Julia implementation returned result")

        result = np.asarray(julia_result, dtype=np.float32)

        return result

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"
