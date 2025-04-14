#!/usr/bin/env python3
"""
Script to test the Supersayan client with a house price regression model.

This script:
1. Trains a simple neural network for house price prediction
2. Tests the hybrid mode where only specified nn.Linear layers run in FHE remotely,
   while other layers (like ReLU) run in plaintext locally

The client-server architecture transparently handles the remote execution
of FHE layers while keeping local execution for non-FHE layers in hybrid mode.
"""

import os
import sys
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from supersayan.nn.convert import ModelType
from supersayan.remote.client import SupersayanClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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


def test_house_price_client(server_url):
    """
    Test the client-server architecture with a house price regression model.
    
    Args:
        server_url: URL of the SuperSayan server
    """
    print("Testing Client for Supersayan House Price Regression Model\n")
    
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
    
    # Test sample data
    test_x = torch.rand(5, 5, dtype=torch.float32)  # 5 test samples
    
    # Get predictions from original model for comparison
    torch_pred = torch_model(test_x)
    torch_values = torch_pred.detach().numpy()
    
    print(f"\nConnecting to server at {server_url}...")
    
    try:
        # Test hybrid model (only Linear layers in FHE run remotely)
        print("\n--- Testing Hybrid Model (Linear layers in FHE run remotely, non-linear layers run in plaintext locally) ---")
        
        # Create a client with hybrid model (only Linear layers in FHE) using module type
        client_hybrid = SupersayanClient(
            server_url=server_url,
            torch_model=torch_model,
            model_type=ModelType.HYBRID,
            fhe_modules=[nn.Linear]  # Convert only Linear layers to FHE
        )
        
        print("Running inference with hybrid SupersayanClient...")
        
        # Perform forward pass - FHE layers run remotely, others run locally
        client_hybrid_pred = client_hybrid(test_x)
        client_hybrid_values = client_hybrid_pred.detach().numpy()
        
        print("\nOriginal PyTorch model predictions:")
        print(torch_values)
        print("\nHybrid SupersayanClient model predictions (Linear layers in FHE, others in plaintext):")
        print(client_hybrid_values)

        print(f"Hybrid Model Test Passed!")
        
        # Close the hybrid client session
        print("Closing hybrid client session...")
        client_hybrid.close()
        
    except Exception as e:
        print(f"Error during client test: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Test SuperSayan client")
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:8000", help="URL of the SuperSayan server")
    args = parser.parse_args()
    
    test_house_price_client(args.server_url)


if __name__ == "__main__":
    main()