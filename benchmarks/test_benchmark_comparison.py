"""
Benchmark comparing Conv2d vs Conv2dOrion implementations.
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
    "in_channels,out_channels,input_size,kernel_size,stride,padding",
    [
        (1, 1, 8, 3, 1, 1),
        (3, 4, 8, 3, 1, 1),
        (3, 4, 16, 3, 1, 1),
        (3, 4, 8, 3, 2, 1),
    ]
)
def test_benchmark_conv2d_comparison(benchmark, in_channels, out_channels, 
                         input_size, kernel_size, stride, padding):
    """Benchmark the standard Conv2d implementation for comparison."""
    
    # Generate secret key
    key = generate_secret_key()
    
    # Generate random input
    batch_size = 1
    input_data = torch.randn(batch_size, in_channels, input_size, input_size)
    encrypted_input = encrypt(input_data, key)
    
    # Standard Conv2d
    conv = Conv2d(
        in_channels=in_channels, 
        out_channels=out_channels, 
        kernel_size=kernel_size, 
        stride=stride, 
        padding=padding, 
        bias=True
    )
    
    # Benchmark
    result = benchmark(lambda: conv(encrypted_input))
    
    # Compute expected output shape
    h_out = w_out = int((input_size + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
    expected_shape = (batch_size, out_channels, h_out, w_out)
    
    # Verify shape
    decrypted_output = decrypt(result, key)
    assert decrypted_output.shape == expected_shape


@pytest.mark.parametrize(
    "in_channels,out_channels,input_size,kernel_size,stride,padding",
    [
        (1, 1, 8, 3, 1, 1),
        (3, 4, 8, 3, 1, 1),
        (3, 4, 16, 3, 1, 1),
        (3, 4, 8, 3, 2, 1),
    ]
)
def test_benchmark_conv2d_orion_comparison(benchmark, in_channels, out_channels, 
                      input_size, kernel_size, stride, padding):
    """Benchmark the optimized Conv2dOrion implementation for comparison."""
    
    # Generate secret key
    key = generate_secret_key()
    
    # Generate random input
    batch_size = 1
    input_data = torch.randn(batch_size, in_channels, input_size, input_size)
    encrypted_input = encrypt(input_data, key)
    
    # Optimized Conv2dOrion
    conv = Conv2dOrion(
        in_channels=in_channels, 
        out_channels=out_channels, 
        kernel_size=kernel_size, 
        stride=stride, 
        padding=padding, 
        bias=True
    )
    
    # Benchmark
    result = benchmark(lambda: conv(encrypted_input))
    
    # Compute expected output shape
    h_out = w_out = int((input_size + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
    expected_shape = (batch_size, out_channels, h_out, w_out)
    
    # Verify shape
    decrypted_output = decrypt(result, key)
    assert decrypted_output.shape == expected_shape