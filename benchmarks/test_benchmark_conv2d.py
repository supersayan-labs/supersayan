import torch
import pytest

from supersayan.core.keygen import generate_secret_key
from supersayan.core.encryption import encrypt_to_lwes, decrypt_from_lwes
from supersayan.nn.layers.conv2d import Conv2d
from supersayan.logging_config import get_logger, configure_logging

configure_logging(level="INFO", disable_file_logging=True)
logger = get_logger(__name__)


@pytest.mark.parametrize(
    "batch_size,in_channels,out_channels,input_size,kernel_size,stride,padding",
    [
        (1, 1, 1, 8, 3, 1, 1),
        (1, 3, 4, 8, 3, 1, 1),
        (1, 3, 4, 16, 3, 1, 1),
        (1, 3, 4, 8, 3, 2, 1),
        (1, 3, 4, 16, 3, 2, 1),
    ],
)
def test_benchmark_conv2d(
    benchmark,
    batch_size,
    in_channels,
    out_channels,
    input_size,
    kernel_size,
    stride,
    padding,
):
    """Benchmark Conv2d layer with different configurations."""
    # Generate secret key
    key = generate_secret_key()

    # Define FHE convolutional layer
    conv = Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True,
    )

    # Generate random input
    input_data = torch.randn(batch_size, in_channels, input_size, input_size)

    # Encrypt the input data
    encrypted_input = encrypt_to_lwes(input_data, key)

    # Compute expected output shape
    h_out = w_out = int((input_size + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
    expected_shape = (batch_size, out_channels, h_out, w_out)

    # Benchmark the forward pass
    result = benchmark(lambda: conv(encrypted_input))

    # Verify shape
    decrypted_output = decrypt_from_lwes(result, key)
    assert decrypted_output.shape == expected_shape
