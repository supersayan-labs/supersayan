# tests/test_perceptron.py

"""
test_perceptron.py

A simple test for a perceptron network using FHE operations with the Linear layer.
"""

import torch
import numpy as np
import logging
from supersayan.core.keygen import generate_secret_key
from supersayan.core.encryption import encrypt, decrypt
from supersayan.nn.layers.linear import Linear

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_perceptron_network():
    # Generate secret key
    logger.info("Generating secret key...")
    key = generate_secret_key()
    
    # Define FHE layers directly
    logger.info("Creating FHE layers...")
    linear1 = Linear(in_features=3, out_features=4, bias=True)
    linear2 = Linear(in_features=4, out_features=2, bias=True)
    
    # Generate random input with batch size 2
    input_data = torch.randn(2, 3)
    print("Input:")
    print(input_data)
    
    try:
        # Encrypt the input data
        logger.info("Encrypting input data...")
        encrypted_input = encrypt(input_data, key)
                
        # Forward pass through the first linear layer
        logger.info("Processing first layer...")
        hidden = linear1(encrypted_input)
        
        # No activation is performed on encrypted data (placeholder)
        # In a real implementation, you would need FHE-compatible activation
        logger.info("Applying activation (passthrough in FHE)...")
        hidden_activated = hidden  # Placeholder for activation
        
        # Forward pass through the second linear layer
        logger.info("Processing second layer...")
        output_encrypted = linear2(hidden_activated)
        
        # Decrypt the result
        logger.info("Decrypting result...")
        output_data = output_encrypted.data
        
        # Convert memoryview to numpy array if needed
        if isinstance(output_data, memoryview):
            output_data = np.array(output_data)
        
        decrypted_output = decrypt(output_data, key)
        
        print("\nFinal network output (decrypted):")
        print(decrypted_output)
        
    except Exception as e:
        logger.error(f"FHE execution failed: {e}")
        raise
    
    logger.info("Perceptron test completed successfully")

if __name__ == '__main__':
    test_perceptron_network()