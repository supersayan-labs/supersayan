"""
Verify correctness of FHE implementations compared to PyTorch native operations.

This test focuses on numerical correctness, not performance benchmarking.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import pytest

from supersayan.core.keygen import generate_secret_key
from supersayan.core.encryption import encrypt, decrypt
from supersayan.nn.layers.linear import Linear as FHELinear
from supersayan.nn.layers.conv2d import Conv2d as FHEConv2d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_verify_linear():
    """Verify FHE Linear layer produces results close to PyTorch nn.Linear."""

    with torch.no_grad():
        # Test parameters
        batch_size = 2
        in_features = 10
        out_features = 4

        # Generate random input
        input_data = torch.randn(batch_size, in_features)

        # Create PyTorch Linear layer
        torch_linear = nn.Linear(in_features, out_features)

        # Create equivalent FHE Linear layer with same weights and bias
        fhe_linear = FHELinear(in_features, out_features)
        fhe_linear.weight.data = torch_linear.weight.data.clone()
        fhe_linear.bias.data = torch_linear.bias.data.clone()

        # Get PyTorch result
        torch_output = torch_linear(input_data)

        # Generate secret key and encrypt input for FHE
        key = generate_secret_key()
        encrypted_input = encrypt(input_data, key)

        # Get FHE result
        encrypted_output = fhe_linear(encrypted_input)
        fhe_output = decrypt(encrypted_output, key)

        # Compare results (allowing small numerical differences)
        # Ensure both outputs are torch.Tensor and cast if needed
        torch_output_tensor = torch.as_tensor(torch_output)
        fhe_output_tensor = torch.as_tensor(fhe_output)

        # Calculate difference and get statistics
        diff = (torch_output_tensor - fhe_output_tensor).abs()
        max_diff = diff.max().item()  # Use item() to get Python scalar
        mean_diff = diff.mean().item()  # Use item() to get Python scalar

        print(
            f"Linear layer max difference: {max_diff:.6f}, mean difference: {mean_diff:.6f}"
        )

        # Assert the difference is small (accounting for FHE numerical precision differences)
        assert max_diff < 0.1, f"Max difference too large: {max_diff}"
        assert mean_diff < 0.05, f"Mean difference too large: {mean_diff}"


def test_verify_conv2d():
    """Verify FHE Conv2d layer produces results close to PyTorch nn.Conv2d."""

    with torch.no_grad():
        # Test parameters
        batch_size = 1
        in_channels = 2
        out_channels = 3
        kernel_size = 3
        stride = 1
        padding = 1

        # Generate random input
        input_data = torch.randn(batch_size, in_channels, 8, 8)

        # Create PyTorch Conv2d layer
        torch_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )

        # Create FHE Conv2d layer with same weights and bias
        fhe_conv = FHEConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        fhe_conv.weight.data = torch_conv.weight.data.clone()
        fhe_conv.bias.data = torch_conv.bias.data.clone()

        # Get PyTorch result
        torch_output = torch_conv(input_data)

        # Generate secret key and encrypt input for FHE
        key = generate_secret_key()
        encrypted_input = encrypt(input_data, key)

        # Get FHE result
        encrypted_output = fhe_conv(encrypted_input)
        fhe_output = decrypt(encrypted_output, key)

        # Compare results (allowing small numerical differences)
        # Ensure both outputs are torch.Tensor and cast if needed
        torch_output_tensor = torch.as_tensor(torch_output)
        fhe_output_tensor = torch.as_tensor(fhe_output)

        # Calculate difference and get statistics
        diff = (torch_output_tensor - fhe_output_tensor).abs()
        max_diff = diff.max().item()  # Use item() to get Python scalar
        mean_diff = diff.mean().item()  # Use item() to get Python scalar

        print(
            f"Conv2d layer max difference: {max_diff:.6f}, mean difference: {mean_diff:.6f}"
        )

        # Assert the difference is small (accounting for FHE numerical precision differences)
        assert max_diff < 0.1, f"Max difference too large: {max_diff}"
        assert mean_diff < 0.05, f"Mean difference too large: {mean_diff}"


def test_verify_conv2d_with_stride():
    """Verify FHE Conv2d with stride produces results close to PyTorch nn.Conv2d."""

    with torch.no_grad():
        # Test parameters
        batch_size = 1
        in_channels = 2
        out_channels = 3
        kernel_size = 3
        stride = 2
        padding = 1

        # Generate random input
        input_data = torch.randn(batch_size, in_channels, 8, 8)

        # Create PyTorch Conv2d layer
        torch_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )

        # Create FHE Conv2d layer with same weights and bias
        fhe_conv = FHEConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        fhe_conv.weight.data = torch_conv.weight.data.clone()
        fhe_conv.bias.data = torch_conv.bias.data.clone()

        # Get PyTorch result
        torch_output = torch_conv(input_data)

        # Generate secret key and encrypt input for FHE
        key = generate_secret_key()
        encrypted_input = encrypt(input_data, key)

        # Get FHE result
        encrypted_output = fhe_conv(encrypted_input)
        fhe_output = decrypt(encrypted_output, key)

        # Compare results (allowing small numerical differences)
        # Ensure both outputs are torch.Tensor and cast if needed
        torch_output_tensor = torch.as_tensor(torch_output)
        fhe_output_tensor = torch.as_tensor(fhe_output)

        # Calculate difference and get statistics
        diff = (torch_output_tensor - fhe_output_tensor).abs()
        max_diff = diff.max().item()  # Use item() to get Python scalar
        mean_diff = diff.mean().item()  # Use item() to get Python scalar

        print(
            f"Conv2d (stride=2) max difference: {max_diff:.6f}, mean difference: {mean_diff:.6f}"
        )

        # Assert the difference is small (accounting for FHE numerical precision differences)
        assert max_diff < 0.1, f"Max difference too large: {max_diff}"
        assert mean_diff < 0.05, f"Mean difference too large: {mean_diff}"
