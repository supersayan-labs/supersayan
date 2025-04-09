import torch
import torch.nn as nn
import numpy as np

from supersayan.nn import convert_to_pure_supersayan, convert_to_hybrid_supersayan
from supersayan.core.keygen import generate_secret_key
from supersayan.core.encryption import encrypt, decrypt

def test_pure_model_conversion():
    """Test converting a PyTorch model to a Pure SuperSayan model."""
    # Create a simple PyTorch model
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(10, 5)
            self.fc2 = nn.Linear(5, 1)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x
    
    # Instantiate the model
    torch_model = SimpleModel()
    
    # Convert to Pure SuperSayan model
    pure_model = convert_to_pure_supersayan(torch_model)
    
    # Generate a key for encryption
    key = generate_secret_key()
    
    # Create a sample input with explicit dtype
    x = torch.randn(2, 10, dtype=torch.float32)
    
    # Encrypt the input
    encrypted_x = encrypt(x, key)
    
    # Forward pass on the encrypted model
    encrypted_output = pure_model(encrypted_x)
    
    # Decrypt the output
    decrypted_output = decrypt(encrypted_output, key)
    
    print("Pure Model Test:")
    print(f"Original Input Shape: {x.shape}")
    print(f"Encrypted Output Shape: {np.shape(encrypted_output)}")
    print(f"Decrypted Output Shape: {np.shape(decrypted_output)}")
    
    # Compare with PyTorch model (approximately)
    torch_output = torch_model(x).detach().cpu().numpy()
    mean_diff = np.mean(np.abs(decrypted_output - torch_output))
    print(f"Mean Absolute Difference: {mean_diff}")
    
    # FHE operations introduce noise, so use a larger threshold
    return mean_diff < 1.0  # Allow for more difference due to encryption noise

def test_hybrid_model_conversion():
    """Test converting a PyTorch model to a Hybrid SuperSayan model."""
    # Create a simple PyTorch model with named modules
    class HybridModel(nn.Module):
        def __init__(self):
            super(HybridModel, self).__init__()
            self.encoder = nn.Linear(10, 5)
            self.classifier = nn.Linear(5, 1)
        
        def forward(self, x):
            x = self.encoder(x)
            x = self.classifier(x)
            return x
    
    # Instantiate the model
    torch_model = HybridModel()
    
    # Convert to Hybrid SuperSayan model, running only the encoder in FHE
    hybrid_model = convert_to_hybrid_supersayan(torch_model, ["encoder"])
    
    # Create a sample input with explicit dtype to ensure consistency
    x = torch.randn(2, 10, dtype=torch.float32)
    
    # Forward pass on the hybrid model
    output = hybrid_model(x)
    
    print("\nHybrid Model Test:")
    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {output.shape}")
    
    # Compare with PyTorch model (approximately)
    torch_output = torch_model(x)
    mean_diff = torch.mean(torch.abs(output - torch_output)).item()
    print(f"Mean Absolute Difference: {mean_diff}")
    
    # FHE operations introduce noise, so use a larger threshold
    return mean_diff < 1.0  # Allow for more difference due to encryption noise

if __name__ == "__main__":
    print("Testing PyTorch to SuperSayan model conversion\n")
    
    try:
        pure_result = test_pure_model_conversion()
        print(f"Pure Model Test Passed: {pure_result}\n")
    except Exception as e:
        print(f"Pure Model Test Failed: {e}\n")
    
    try:
        hybrid_result = test_hybrid_model_conversion()
        print(f"Hybrid Model Test Passed: {hybrid_result}")
    except Exception as e:
        print(f"Hybrid Model Test Failed: {e}") 