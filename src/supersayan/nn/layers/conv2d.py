import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Union
from supersayan.core.bindings import SupersayanTFHE
import logging

logger = logging.getLogger(__name__)


class Conv2d(nn.Module):
    """
    2-D convolution for encrypted feature maps.
    The input **and** output are 5-D NDArrays holding LWE ciphertext vectors:
        (N, C_in, H, W, lwe_dim)  →  (N, C_out, H_out, W_out, lwe_dim)
    Nothing is ever flattened.
    """

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
            raise ValueError("in/out channels must be divisible by groups")

        kh, kw = self.kernel_size
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, kh, kw, dtype=torch.float32)
        )
        self.bias = nn.Parameter(torch.randn(out_channels, dtype=torch.float32)) if bias else None

    # ------------------------------------------------------------
    # forward
    # ------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : np.ndarray
            Encrypted input feature map with **shape**
            (N, C_in, H, W, lwe_dim).

        Returns
        -------
        np.ndarray
            Encrypted output map with shape
            (N, C_out, H_out, W_out, lwe_dim).
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

        # Call Julia implementation directly
        y = SupersayanTFHE.Layers.Conv2d.conv2d_forward(
            x,                                # (N, C_in, H, W, lwe_dim)
            weight_np,                             # (C_out, C_in/groups, kh, kw)
            bias_np,                             # (C_out,) or nothing
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        # Julia returns an Array{Float32,5} so numpy conversion is cheap.
        return np.asarray(y, dtype=np.float32)

    # ------------------------------------------------------------
    def __repr__(self):
        return (
            f"Conv2d(in={self.in_channels}, out={self.out_channels}, "
            f"ks={self.kernel_size}, stride={self.stride}, pad={self.padding}, "
            f"dil={self.dilation}, groups={self.groups}, bias={self.bias is not None})"
        )
