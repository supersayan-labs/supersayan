"""
Benchmark for large Conv2d layer configurations.
"""

import torch
import numpy as np
import logging
import pytest

from supersayan.core.keygen import generate_secret_key
from supersayan.core.encryption import encrypt, decrypt
from supersayan.nn.layers.conv2d import Conv2d
from supersayan.nn.layers.conv2d_orion import Conv2dOrion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "layer_type,batch_size,in_channels,out_channels,input_size,kernel_size,stride,padding",
    [
        # Standard Conv2d with larger parameters
        ("standard", 1, 8, 16, 32, 3, 1, 1),
        ("standard", 2, 16, 32, 24, 3, 1, 1),
        ("standard", 1, 32, 64, 16, 3, 1, 1),
        
        # Orion Conv2d with larger parameters
        ("orion", 1, 8, 16, 32, 3, 1, 1),
        ("orion", 2, 16, 32, 24, 3, 1, 1),
        ("orion", 1, 32, 64, 16, 3, 1, 1),
        
        # ResNet-like bottleneck configurations (reduced size)
        ("standard", 1, 64, 64, 16, 1, 1, 0),  # 1x1 conv for channel reduction
        ("orion", 1, 64, 64, 16, 1, 1, 0),     # 1x1 conv for channel reduction
    ]
)
def test_benchmark_conv2d_large(benchmark, layer_type, batch_size, in_channels, out_channels, 
                      input_size, kernel_size, stride, padding):
    """Benchmark Conv2d and Conv2dOrion layers with larger configurations."""
    # Generate secret key
    key = generate_secret_key()
    
    # Define FHE convolutional layer (either standard or Orion)
    if layer_type == "standard":
        conv = Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=True
        )
        layer_name = "Conv2d"
    else:
        conv = Conv2dOrion(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=True
        )
        layer_name = "Conv2dOrion"
    
    # Generate random input
    input_data = torch.randn(batch_size, in_channels, input_size, input_size)
    
    # Log test configuration
    logger.info(f"Running large {layer_name} benchmark with: "
               f"batch_size={batch_size}, "
               f"in_channels={in_channels}, "
               f"out_channels={out_channels}, "
               f"input_size={input_size}x{input_size}")
    
    # Encrypt the input data
    encrypted_input = encrypt(input_data, key)
    
    # Compute expected output shape
    h_out = w_out = int((input_size + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
    expected_shape = (batch_size, out_channels, h_out, w_out)
    
    # Benchmark the forward pass
    result = benchmark(lambda: conv(encrypted_input))
    
    # Verify shape
    decrypted_output = decrypt(result, key)
    assert decrypted_output.shape == expected_shape