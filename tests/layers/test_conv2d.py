import numpy as np
import pytest
import torch

from supersayan.core.encryption import decrypt_from_lwes, encrypt_to_lwes
from supersayan.core.keygen import generate_secret_key
from supersayan.core.types import SupersayanTensor
from supersayan.logging_config import configure_logging, get_logger
from supersayan.nn.layers.conv2d import Conv2d

configure_logging(level="INFO")
logger = get_logger(__name__)


@pytest.fixture(scope="module")
def fhe_secret_key():
    """Generate secret key fixture for FHE operations."""
    logger.info("Generating secret key (fixture)...")
    return generate_secret_key()


def test_conv2d_layer(fhe_secret_key):
    """Test FHE Conv2d layer forward pass."""
    logger.info("Creating Conv2d layer...")
    conv = Conv2d(
        in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1, bias=True
    )

    input_data = SupersayanTensor(torch.rand(1, 3, 8, 8), device=torch.device("cuda"))
    print("Input shape:", input_data.shape)

    try:
        logger.info("Encrypting input data...")
        encrypted_input = encrypt_to_lwes(input_data, fhe_secret_key)

        logger.info("Processing convolutional layer...")
        output_encrypted = conv(encrypted_input)

        logger.info("Decrypting result...")
        decrypted_output = decrypt_from_lwes(output_encrypted, fhe_secret_key)

        print("\nOutput shape:", decrypted_output.shape)

    except Exception as e:
        logger.error(f"FHE convolution failed: {e}")
        raise

    expected_shape = (1, 4, 8, 8)
    assert (
        decrypted_output.shape == expected_shape
    ), f"Expected shape {expected_shape} but got {decrypted_output.shape}"

    logger.info("Conv2d test completed successfully")


def test_conv2d_with_stride(fhe_secret_key):
    """Test FHE Conv2d layer with stride=2."""
    logger.info("Creating Conv2d layer with stride=2...")
    conv = Conv2d(
        in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=1, bias=True
    )

    input_data = SupersayanTensor(torch.rand(1, 3, 8, 8), device=torch.device("cuda"))
    print("Input shape:", input_data.shape)

    try:
        logger.info("Encrypting input data...")
        encrypted_input = encrypt_to_lwes(input_data, fhe_secret_key)

        logger.info("Processing convolutional layer with stride...")
        output_encrypted = conv(encrypted_input)

        logger.info("Decrypting result...")
        decrypted_output = decrypt_from_lwes(output_encrypted, fhe_secret_key)

        print("\nOutput shape with stride=2:", decrypted_output.shape)
        print("Output should be half the size in each spatial dimension")
        print("Output sample (all values):")
        print(decrypted_output[0, 0])

    except Exception as e:
        logger.error(f"FHE strided convolution failed: {e}")
        raise

    expected_shape = (1, 4, 4, 4)
    assert (
        decrypted_output.shape == expected_shape
    ), f"Expected shape {expected_shape} but got {decrypted_output.shape}"

    logger.info("Strided Conv2d test completed successfully")


# def test_large_conv2d_layer(fhe_secret_key):
#     """
#     Test a larger Conv2d layer with more output channels and a larger kernel size.
#     This test will verify the functionality of a more complex convolutional layer.
#     """
#     logger.info("Creating large Conv2d layer...")
#     conv = Conv2d(
#         in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False
#     )

#     input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
#     print("Input shape for large Conv2d:", input_data.shape)

#     try:
#         logger.info("Encrypting input data for large Conv2d...")
#         encrypted_input = encrypt_to_lwes(input_data, fhe_secret_key)

#         logger.info("Processing large convolutional layer...")
#         output_encrypted = conv(encrypted_input)

#         logger.info("Decrypting result from large Conv2d...")
#         decrypted_output = decrypt_from_lwes(output_encrypted, fhe_secret_key)

#         print("\nOutput shape for large Conv2d:", decrypted_output.shape)
#         print("Output sample (all values):")
#         print(decrypted_output[0, 0])

#     except Exception as e:
#         logger.error(f"FHE large convolution failed: {e}")
#         raise

#     expected_shape = (
#         1,
#         64,
#         112,
#         112,
#     )
#
#     assert (
#         decrypted_output.shape == expected_shape
#     ), f"Expected shape {expected_shape} but got {decrypted_output.shape}"

#     logger.info("Large Conv2d test completed successfully")
