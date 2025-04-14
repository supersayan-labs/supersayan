import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Union
import logging

from supersayan.core.bindings import SupersayanTFHE

logger = logging.getLogger(__name__)

class Conv2d(nn.Module):
    """
    A PyTorch-style Conv2d layer redefined for encrypted data.
    
    This implementation follows the Orion approach using Toeplitz-based encoding,
    single-shot multiplexing, and optimizations based on the BSGS (Baby-Step Giant-Step)
    algorithm for FHE operations.
    
    The forward pass expects a NumPy array of LWE ciphertexts of shape (batch, channels, height, width).
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
        bias: bool = True
    ):
        super(Conv2d, self).__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("in_channels and out_channels must be divisible by groups")
            
        # Initialize weights as torch parameters with shape (out_channels, in_channels//groups, K_h, K_w)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1])
        )
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Pre-compute toeplitz matrices for the BSGS optimization
        self.toeplitz_matrices = None
        self.precomputed = False
            
    def _compute_output_shape(self, input_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Compute the output shape of the convolution operation.
        
        Args:
            input_shape: A tuple (height, width) of the input feature map.
            
        Returns:
            A tuple (height, width) of the output feature map.
        """
        h_in, w_in = input_shape
        
        # Apply formula: H_out = (H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        h_out = int((h_in + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = int((w_in + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        
        return (h_out, w_out)
    
    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass using homomorphic operations.
        
        Args:
            input: A NumPy array of LWE ciphertexts with shape (batch, channels, height, width).
            
        Returns:
            A NumPy array of LWE ciphertexts with shape (batch, out_channels, out_height, out_width).
        """
        batch_size, channels, height, width = input.shape
        
        if channels != self.in_channels:
            raise ValueError(f"Expected input with {self.in_channels} channels, got {channels}")
            
        # Compute output dimensions
        out_height, out_width = self._compute_output_shape((height, width))
        
        # Detach the weight and bias parameters as plain numbers
        weight_np = self.weight.detach().cpu().numpy()
        bias_np = self.bias.detach().cpu().numpy() if self.bias is not None else None
        
        # Build Toeplitz matrices if not already done
        if not self.precomputed or self.toeplitz_matrices is None:
            # Build toeplitz matrices using Julia backend
            self.toeplitz_matrices = SupersayanTFHE.Layers.Conv2d.build_toeplitz_matrices(
                input.shape,
                weight_np,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                self.groups
            )
            self.precomputed = True
        
        # Flatten input for Julia processing using vectorized operations
        flattened_input = input.flatten().tolist()
        
        # Call Julia implementation directly
        julia_result = SupersayanTFHE.Layers.Conv2d.conv2d_forward(
            flattened_input,
            input.shape,
            self.toeplitz_matrices,
            bias_np
        )
        
        # Reshape the result back to (batch_size, out_channels, out_height, out_width) using vectorized operations
        output = np.array(julia_result, dtype=object).reshape(batch_size, self.out_channels, out_height, out_width)
        
        return output
    
    def __repr__(self):
        return (
            f"Conv2d(in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"padding={self.padding}, "
            f"groups={self.groups}, "
            f"bias={self.bias is not None})"
        )