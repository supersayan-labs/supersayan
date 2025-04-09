import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from supersayan.nn import convert_to_hybrid_supersayan
from supersayan.core.keygen import generate_secret_key
from supersayan.core.encryption import encrypt, decrypt

# Simulated dataset: [rooms, area, location_score, age, has_garage]
X = torch.rand(500, 5, dtype=torch.float32)  # 500 houses
y = (
    50000 * X[:, 0] +     # rooms
    300 * X[:, 1] * 100 + # area
    100000 * X[:, 2] +    # location_score
    -2000 * X[:, 3] * 100 + # age
    20000 * X[:, 4]       # has_garage
).unsqueeze(1) + torch.randn(500, 1, dtype=torch.float32) * 10000  # add some noise

# DataLoader
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model: simple DNN with named modules for hybrid conversion
class HousePriceRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        # Use named modules instead of Sequential for better hybrid conversion
        self.linear1 = nn.Linear(5, 16)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x

def test_hybrid_house_price_model():
    print("Testing Hybrid Supersayan House Price Regression Model\n")
    
    # Create and train original PyTorch model
    torch_model = HousePriceRegressor()
    
    # Training setup
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(torch_model.parameters(), lr=1e-3)
    
    # Training loop - brief training just for demonstration
    print("Training original PyTorch model:")
    for epoch in range(10):
        total_loss = 0
        batches = 0
        for batch_x, batch_y in loader:
            pred = torch_model(batch_x)
            loss = loss_fn(pred, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
        
        avg_loss = total_loss / batches
        print(f"Epoch {epoch}: Avg Loss = {avg_loss:.2f}")
    
    # Convert to Hybrid Supersayan model with all linear layers in FHE
    fhe_module_names = ["linear1", "linear2", "linear3"]
    hybrid_model = convert_to_hybrid_supersayan(torch_model, fhe_module_names)
    
    # Test the hybrid model on a small sample
    test_x = torch.rand(5, 5, dtype=torch.float32)  # 5 test samples
    
    print("\nTesting Hybrid Supersayan model (all linear layers in FHE):")
    
    # Original model predictions
    torch_pred = torch_model(test_x)
    
    # Hybrid model predictions
    hybrid_pred = hybrid_model(test_x)
    
    # Compare results
    torch_values = torch_pred.detach().numpy()
    hybrid_values = hybrid_pred.detach().numpy()
    
    print("\nOriginal PyTorch model predictions:")
    print(torch_values)
    print("\nHybrid Supersayan model predictions:")
    print(hybrid_values)
    
    # Calculate mean absolute difference
    mean_diff = np.mean(np.abs(torch_values - hybrid_values))
    print(f"\nMean Absolute Difference: {mean_diff}")
    
    # FHE operations introduce noise, so use a larger threshold
    result = mean_diff < 1000.0  # Larger threshold for house prices
    print(f"Hybrid Model Test Passed: {result}")
    
    return result

if __name__ == "__main__":
    test_hybrid_house_price_model() 