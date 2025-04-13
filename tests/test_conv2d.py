# tests/test_conv2d.py

"""
test_conv2d.py

A simple test for a 2D convolutional layer using FHE operations.
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
        output_data = output_encrypted
        
        decrypted_output = decrypt(output_data, key)
        
        print("\nOutput shape:", decrypted_output.shape)
        print("Output sample (first few values):")
        print(decrypted_output[0, 0, :2, :2])
        
    except Exception as e:
        logger.error(f"FHE convolution failed: {e}")
        raise
    
    logger.info("Conv2d test completed successfully")

if __name__ == '__main__':
    test_conv2d_layer()