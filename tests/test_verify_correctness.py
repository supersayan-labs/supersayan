import pytest
import torch
import torch.nn as nn

from supersayan.core.encryption import decrypt_from_lwes, encrypt_to_lwes
from supersayan.core.keygen import generate_secret_key
from supersayan.logging_config import configure_logging, get_logger
from supersayan.nn.layers.conv2d import Conv2d
from supersayan.nn.layers.linear import Linear

configure_logging(level="INFO")
logger = get_logger(__name__)


@pytest.fixture(scope="module")
def fhe_secret_key():
    """Generate secret key fixture for FHE operations."""
    logger.info("Generating secret key (fixture)...")
    return generate_secret_key()


def test_verify_linear(fhe_secret_key):
    """Verify FHE Linear layer produces results close to PyTorch nn.Linear."""
    with torch.no_grad():
        batch_size = 2
        in_features = 10
        out_features = 4

        input_data = torch.randn(batch_size, in_features)

        torch_linear = nn.Linear(in_features, out_features)

        fhe_linear = Linear(in_features, out_features)
        fhe_linear.weight.data = torch_linear.weight.data.clone()
        fhe_linear.bias.data = torch_linear.bias.data.clone()

        torch_output = torch_linear(input_data)

        encrypted_input = encrypt_to_lwes(input_data, fhe_secret_key)

        encrypted_output = fhe_linear(encrypted_input)
        fhe_output = decrypt_from_lwes(encrypted_output, key)

        torch_output_tensor = torch.as_tensor(torch_output)
        fhe_output_tensor = torch.as_tensor(fhe_output)

        diff = (torch_output_tensor - fhe_output_tensor).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(
            f"Linear layer max difference: {max_diff:.6f}, mean difference: {mean_diff:.6f}"
        )

        assert max_diff < 0.1, f"Max difference too large: {max_diff}"
        assert mean_diff < 0.05, f"Mean difference too large: {mean_diff}"


def test_verify_conv2d(fhe_secret_key):
    """Verify FHE Conv2d layer produces results close to PyTorch nn.Conv2d."""
    with torch.no_grad():
        batch_size = 1
        in_channels = 2
        out_channels = 3
        kernel_size = 3
        stride = 1
        padding = 1

        input_data = torch.randn(batch_size, in_channels, 8, 8)

        torch_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )

        fhe_conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        fhe_conv.weight.data = torch_conv.weight.data.clone()
        fhe_conv.bias.data = torch_conv.bias.data.clone()

        torch_output = torch_conv(input_data)

        encrypted_input = encrypt_to_lwes(input_data, fhe_secret_key)

        encrypted_output = fhe_conv(encrypted_input)
        fhe_output = decrypt_from_lwes(encrypted_output, fhe_secret_key)

        torch_output_tensor = torch.as_tensor(torch_output)
        fhe_output_tensor = torch.as_tensor(fhe_output)

        diff = (torch_output_tensor - fhe_output_tensor).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(
            f"Conv2d layer max difference: {max_diff:.6f}, mean difference: {mean_diff:.6f}"
        )

        assert max_diff < 0.1, f"Max difference too large: {max_diff}"
        assert mean_diff < 0.05, f"Mean difference too large: {mean_diff}"


def test_verify_conv2d_with_stride(fhe_secret_key):
    """Verify FHE Conv2d with stride produces results close to PyTorch nn.Conv2d."""
    with torch.no_grad():
        batch_size = 1
        in_channels = 2
        out_channels = 3
        kernel_size = 3
        stride = 2
        padding = 1

        input_data = torch.randn(batch_size, in_channels, 8, 8)

        torch_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )

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

        torch_output = torch_conv(input_data)

        encrypted_input = encrypt_to_lwes(input_data, fhe_secret_key)

        encrypted_output = fhe_conv(encrypted_input)
        fhe_output = decrypt_from_lwes(encrypted_output, key)

        torch_output_tensor = torch.as_tensor(torch_output)
        fhe_output_tensor = torch.as_tensor(fhe_output)

        diff = (torch_output_tensor - fhe_output_tensor).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(
            f"Conv2d (stride=2) max difference: {max_diff:.6f}, mean difference: {mean_diff:.6f}"
        )

        assert max_diff < 0.1, f"Max difference too large: {max_diff}"
        assert mean_diff < 0.05, f"Mean difference too large: {mean_diff}"
