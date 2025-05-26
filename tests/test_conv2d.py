# tests/test_conv2d.py

"""
test_conv2d.py

A test for a 2D convolutional layer using FHE operations
"""

import torch
import numpy as np
import logging
from supersayan.core.keygen import generate_secret_key
from supersayan.core.encryption import encrypt, decrypt
from supersayan.nn.layers.conv2d import Conv2d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_conv2d_layer():
    # Generate secret key
    logger.info("Generating secret key...")
    key = generate_secret_key()

    # Define FHE convolutional layer
    logger.info("Creating Conv2d layer...")
    conv = Conv2d(
        in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1, bias=True
    )

    # import pickle
    # logger.info("Pickling Conv2d layer...")
    # conv = pickle.loads(pickle.dumps(conv))
    # logger.info("Unpickled Conv2d layer")

    # Generate random input with batch size 1 and shape (1, 3, 8, 8)
    input_data = torch.randn(1, 3, 8, 8)
    print("Input shape:", input_data.shape)

    try:
        # Encrypt the input data
        logger.info("Encrypting input data...")
        encrypted_input = encrypt(input_data, key)

        # Forward pass through the convolutional layer
        logger.info("Processing convolutional layer...")
        output_encrypted = conv(encrypted_input)

        # Decrypt the result
        logger.info("Decrypting result...")
        decrypted_output = decrypt(output_encrypted, key)

        print("\nOutput shape:", decrypted_output.shape)
        print("Output sample (first few values):")
        print(decrypted_output[0, 0, :2, :2])

    except Exception as e:
        logger.error(f"FHE convolution failed: {e}")
        raise

    # Verify output has correct shape
    expected_shape = (1, 4, 8, 8)  # Same spatial dimensions due to padding=1
    assert (
        decrypted_output.shape == expected_shape
    ), f"Expected shape {expected_shape} but got {decrypted_output.shape}"

    logger.info("Conv2d test completed successfully")


def test_conv2d_with_stride():
    # Generate secret key
    logger.info("Generating secret key...")
    key = generate_secret_key()

    # Define FHE convolutional layer with stride=2
    logger.info("Creating Conv2d layer with stride=2...")
    conv = Conv2d(
        in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=1, bias=True
    )

    # Generate random input with batch size 1 and shape (1, 3, 8, 8)
    input_data = torch.randn(1, 3, 8, 8)
    print("Input shape:", input_data.shape)

    try:
        # Encrypt the input data
        logger.info("Encrypting input data...")
        encrypted_input = encrypt(input_data, key)

        # Forward pass through the convolutional layer with stride
        logger.info("Processing convolutional layer with stride...")
        output_encrypted = conv(encrypted_input)

        # Decrypt the result
        logger.info("Decrypting result...")
        decrypted_output = decrypt(output_encrypted, key)

        print("\nOutput shape with stride=2:", decrypted_output.shape)
        print("Output should be half the size in each spatial dimension")
        print("Output sample (all values):")
        print(decrypted_output[0, 0])

    except Exception as e:
        logger.error(f"FHE strided convolution failed: {e}")
        raise

    # Verify output has correct shape with stride=2
    expected_shape = (1, 4, 4, 4)  # Half size in each spatial dimension
    assert (
        decrypted_output.shape == expected_shape
    ), f"Expected shape {expected_shape} but got {decrypted_output.shape}"

    logger.info("Strided Conv2d test completed successfully")


if __name__ == "__main__":
    print("=== Testing Regular Conv2d ===")
    test_conv2d_layer()
    print("\n=== Testing Conv2d with Stride ===")
    test_conv2d_with_stride()


def test_large_conv2d_layer():
    """
    Test a larger Conv2d layer with more output channels and a larger kernel size.
    This test will verify the functionality of a more complex convolutional layer.
    """
    # Generate secret key
    logger.info("Generating secret key for large Conv2d test...")
    key = generate_secret_key()

    # Define a larger FHE convolutional layer
    logger.info("Creating large Conv2d layer...")
    conv = Conv2d(
        in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False
    )

    # Generate random input with batch size 1 and shape (1, 3, 16, 16)
    input_data = torch.randn(1, 3, 224, 224)
    print("Input shape for large Conv2d:", input_data.shape)

    try:
        # Encrypt the input data
        logger.info("Encrypting input data for large Conv2d...")
        encrypted_input = encrypt(input_data, key)

        # Forward pass through the larger convolutional layer
        logger.info("Processing large convolutional layer...")
        output_encrypted = conv(encrypted_input)

        # Decrypt the result
        logger.info("Decrypting result from large Conv2d...")
        decrypted_output = decrypt(output_encrypted, key)

        print("\nOutput shape for large Conv2d:", decrypted_output.shape)
        print("Output sample (all values):")
        print(decrypted_output[0, 0])

    except Exception as e:
        logger.error(f"FHE large convolution failed: {e}")
        raise

    # Verify output has correct shape
    expected_shape = (
        1,
        64,
        112,
        112,
    )  # Output shape considering stride=2 and padding=3
    assert (
        decrypted_output.shape == expected_shape
    ), f"Expected shape {expected_shape} but got {decrypted_output.shape}"

    logger.info("Large Conv2d test completed successfully")
