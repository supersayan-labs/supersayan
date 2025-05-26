import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Union
from supersayan.core.bindings import SupersayanTFHE


class Conv2d(nn.Module):
    """
    PyTorch-style 2-D convolution for **encrypted** feature maps.
    Forward delegates to a fast Julia kernel that does a classical
    sliding-window convolution (no Toeplitz tricks).
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

        # --- sanitize hyper-parameters ----
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

        # --- weights ---
        kh, kw = self.kernel_size
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, kh, kw)
        )
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None

    # ------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------
    def _output_spatial_size(self, h_in: int, w_in: int) -> Tuple[int, int]:
        h_out = (
            h_in
            + 2 * self.padding[0]
            - self.dilation[0] * (self.kernel_size[0] - 1)
            - 1
        ) // self.stride[0] + 1
        w_out = (
            w_in
            + 2 * self.padding[1]
            - self.dilation[1] * (self.kernel_size[1] - 1)
            - 1
        ) // self.stride[1] + 1
        return h_out, w_out

    # ------------------------------------------------------------
    # forward
    # ------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args
        ----
        x: numpy array of LWE ciphertexts
           shape = (N, C_in, H, W)

        Returns
        -------
        numpy array of LWE ciphertexts
           shape = (N, C_out, H_out, W_out)
        """
        n, c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {c}")

        h_out, w_out = self._output_spatial_size(h, w)

        # detach → plain numpy for Julia
        w_np = self.weight.detach().cpu().numpy()
        b_np = self.bias.detach().cpu().numpy() if self.bias is not None else None

        y_flat = SupersayanTFHE.Layers.Conv2d.conv2d_forward(
            x.flatten().tolist(),      # ciphertexts
            (n, c, h, w),              # input shape
            w_np,                      # plaintext weights
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            b_np,                      # plaintext bias
        )

        # Julia returns a vector of ciphertexts → reshape
        return np.array(y_flat, dtype=object).reshape(
            n, self.out_channels, h_out, w_out
        )

    def __repr__(self):
        return (
            f"Conv2d(in={self.in_channels}, out={self.out_channels}, "
            f"ks={self.kernel_size}, stride={self.stride}, pad={self.padding}, "
            f"dil={self.dilation}, groups={self.groups}, bias={self.bias is not None})"
        )
