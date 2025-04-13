# tests/test_conv2d.py

"""
test_conv2d.py

A test for a 2D convolutional layer using FHE operations with Orion-style optimizations:
1. Toeplitz-based encoding: Converting convolution to matrix-vector product
2. Single-shot multiplexing: Handling stride > 1 in a single multiplicative depth
3. BSGS (Baby-Step Giant-Step): Reduces rotations from O(n) to O(sqrt(n))
4. Double-hoisting: Reuses expensive parts of key-switching across multiple rotations
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
    conv = Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1, bias=True)
    
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
    assert decrypted_output.shape == expected_shape, f"Expected shape {expected_shape} but got {decrypted_output.shape}"
    
    logger.info("Conv2d test completed successfully")

def test_conv2d_with_stride():
    # Generate secret key
    logger.info("Generating secret key...")
    key = generate_secret_key()
    
    # Define FHE convolutional layer with stride=2
    logger.info("Creating Conv2d layer with stride=2...")
    conv = Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=1, bias=True)
    
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
    assert decrypted_output.shape == expected_shape, f"Expected shape {expected_shape} but got {decrypted_output.shape}"
    
    logger.info("Strided Conv2d test completed successfully")

if __name__ == '__main__':
    print("=== Testing Regular Conv2d ===")
    test_conv2d_layer()
    print("\n=== Testing Conv2d with Stride ===")
    test_conv2d_with_stride()