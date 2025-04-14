import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional, Union, Dict
import logging

from supersayan.core.bindings import SupersayanTFHE
from supersayan.core.types import LWE

logger = logging.getLogger(__name__)

class Conv2dOrion(nn.Module):
    """
    A PyTorch-style Conv2d layer redefined for encrypted data.
    
    This implementation follows the Orion approach using:
    1. Toeplitz-based encoding: Converting convolution to matrix-vector product
    2. Single-shot multiplexing: Handling stride > 1 in a single multiplicative depth
    3. BSGS (Baby-Step Giant-Step): Reduces rotations from O(n) to O(sqrt(n))
    4. Double-hoisting: Reuses expensive parts of key-switching across multiple rotations
    
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
        super(Conv2dOrion, self).__init__()
        
        # Convert scalar parameters to tuples
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
        
        # Validate channels and groups relationship
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("in_channels and out_channels must be divisible by groups")
            
        # Initialize weights as torch parameters (out_channels, in_channels//groups, K_h, K_w)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1])
        )
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Precomputation containers
        self.encoding_plan = None
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
        Performs the forward pass using homomorphic operations with
        single-shot multiplexing and BSGS optimization.
        
        Args:
            input: A NumPy array of LWE ciphertexts with shape (batch, channels, height, width).
            
        Returns:
            A NumPy array of LWE ciphertexts with shape (batch, out_channels, out_height, out_width).
        """
        batch_size, channels, height, width = input.shape
        
        if channels != self.in_channels:
            raise ValueError(f"Expected input with {self.in_channels} channels, got {channels}")
            
        # Get output dimensions
        out_height, out_width = self._compute_output_shape((height, width))
        
        # Detach the weight and bias parameters as plain numbers
        weight_np = self.weight.detach().cpu().numpy()
        bias_np = self.bias.detach().cpu().numpy() if self.bias is not None else None
            
        # Build or rebuild the encoding plan if needed
        if not self.precomputed or self.encoding_plan is None:
            # Call Julia implementation to build the encoding plan
            self.encoding_plan = SupersayanTFHE.Layers.Conv2dOrion.build_encoding_plan(
                input.shape,
                weight_np,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                self.groups
            )
            self.precomputed = True
            logger.info(f"Built encoding plan for Conv2dOrion with input shape {input.shape}")
        
        # Flatten input for Julia processing
        flattened_input = []
        for b in range(batch_size):
            for c in range(channels):
                for h in range(height):
                    for w in range(width):
                        flattened_input.append(input[b, c, h, w])
        
        # Call Julia implementation directly
        julia_result = SupersayanTFHE.Layers.Conv2dOrion.conv2d_orion_forward(
            flattened_input,
            input.shape,
            self.encoding_plan,
            bias_np
        )
        
        # Reshape the result back to (batch_size, out_channels, out_height, out_width)
        output = np.empty((batch_size, self.out_channels, out_height, out_width), dtype=object)
        idx = 0
        for b in range(batch_size):
            for c in range(self.out_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        output[b, c, h, w] = julia_result[idx]
                        idx += 1
        
        return output
    
    def __repr__(self):
        return (
            f"Conv2d(in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"padding={self.padding}, "
            f"dilation={self.dilation}, "
            f"groups={self.groups}, "
            f"bias={self.bias is not None})"
        )