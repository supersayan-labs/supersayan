# tests/layers/test_linear.py

"""
test_linear.py

A test for a linear layer using FHE operations
"""

import torch
import numpy as np
import logging
import pytest
from supersayan.core.keygen import generate_secret_key
from supersayan.core.encryption import encrypt_to_lwes, decrypt_from_lwes
from supersayan.nn.layers.conv2d import Conv2d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from supersayan.nn.layers.linear import Linear

@pytest.fixture(scope="module")
def fhe_secret_key():
    logger.info("Generating secret key (fixture)...")
    return generate_secret_key()

def test_linear_layer(fhe_secret_key):
    # Define FHE linear layer
    logger.info("Creating Linear layer...")
    linear = Linear(in_features=8, out_features=4, bias=True)
    
    weights = np.random.rand(4, 8).astype(np.float32)
    bias = np.random.rand(4).astype(np.float32)
    
    # Generate random input with batch size 1 and shape (1, 8)
    input_data = np.random.rand(1, 8).astype(np.float32)
    print("Input shape:", input_data.shape)

    try:
        # Encrypt the input data
        logger.info("Encrypting input data...")
        encrypted_input = encrypt_to_lwes(input_data, fhe_secret_key)
        
        # Forward pass through the linear layer
        logger.info("Processing linear layer...")
        output_encrypted = linear(encrypted_input)

        # Decrypt the result
        logger.info("Decrypting result...")
        print("Output shape (encrypted):", output_encrypted[0].shape)
        decrypted_output = decrypt_from_lwes(output_encrypted[0], fhe_secret_key)

        print("\nOutput shape:", decrypted_output.shape)
        print("Output sample (all values):")
        print(decrypted_output)

    except Exception as e:
        logger.error(f"FHE linear failed: {e}")
        raise

    # Verify output has correct shape
    expected_shape = (1, 4)
    assert (
        decrypted_output.shape == expected_shape
    ), f"Expected shape {expected_shape} but got {decrypted_output.shape}"

    logger.info("Linear test completed successfully")