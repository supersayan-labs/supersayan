from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from supersayan.core.bindings import SupersayanTFHE
from supersayan.core.types import SupersayanTensor
from supersayan.logging_config import get_logger

logger = get_logger(__name__)


class Conv2d(nn.Module):
    """Supersayan equivalent of torch.nn.Conv2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        tup = lambda v: (v, v) if isinstance(v, int) else v

        self.kernel_size, self.stride = tup(kernel_size), tup(stride)
        self.padding, self.dilation = tup(padding), tup(dilation)
        self.in_channels, self.out_channels, self.groups = (
            in_channels,
            out_channels,
            groups,
        )

        if in_channels % groups or out_channels % groups:
            raise ValueError("In/out channels must be divisible by groups")

        kh, kw = self.kernel_size

        self.weight = nn.Parameter(
            torch.randn(
                out_channels, in_channels // groups, kh, kw, dtype=torch.float32
            )
        )
        self.bias = (
            nn.Parameter(torch.randn(out_channels, dtype=torch.float32))
            if bias
            else None
        )

    def forward(self, x: SupersayanTensor) -> SupersayanTensor:
        """
        Forward pass.

        Args:
            x: The input tensor

        Returns:
            SupersayanTensor: The output tensor
        """
        if x.ndim != 5:
            raise ValueError(
                f"Expected 5-D encrypted tensor (N,C,H,W,lwe_dim); got shape {x.shape}"
            )

        n, c, h, w, _ = x.shape

        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {c}")

        weight_np = self.weight.detach().cpu().numpy()
        bias_np = self.bias.detach().cpu().numpy() if self.bias is not None else None

        julia_result = SupersayanTFHE.Layers.Conv2d.conv2d_forward(
            x.to_julia(),
            weight_np,
            bias_np,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        y = SupersayanTensor._from_julia(julia_result)

        return y

    def __repr__(self):
        return (
            f"Conv2d(in={self.in_channels}, out={self.out_channels}, "
            f"ks={self.kernel_size}, stride={self.stride}, pad={self.padding}, "
            f"dil={self.dilation}, groups={self.groups}, bias={self.bias is not None})"
        )
