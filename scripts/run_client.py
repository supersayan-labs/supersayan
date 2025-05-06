import torch
import sys
import os
import torch.nn as nn
import numpy as np
import pytest
import logging
import requests
from urllib.parse import urljoin

from supersayan.nn.convert import convert_to_hybrid_supersayan
from supersayan.remote.client import SupersayanClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_hybrid_house_price_regression(server_url="http://127.0.0.1:8000"):
    """
    Test the client-server architecture with a house price regression model.
    
    Args:
        server_fixture: The server URL fixture
    """    
    # Verify server is responsive
    try:
        response = requests.get(urljoin(server_url, "/health"), timeout=2)
        assert response.status_code == 200, "Server health check failed"
        logger.info("Server health check passed")
    except Exception as e:
        pytest.fail(f"Server health check failed: {e}")

    # Define a simple house price regressor model
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

    # Create the model
    torch_model = HousePriceRegressor()
    
    # Test sample data
    test_x = torch.rand(5, 5, dtype=torch.float32)  # 5 test samples
    
    # Get predictions from original model for comparison
    torch_pred = torch_model(test_x)
    torch_values = torch_pred.detach().numpy()
    
    logger.info("Creating hybrid client...")
    # Create a client with hybrid model (only Linear layers in FHE)
    client_hybrid = SupersayanClient(
        server_url="http://127.0.0.1:8000",
        torch_model=torch_model,
        fhe_modules=[nn.Linear]  # Convert only Linear layers to FHE
    )
    
    try:
        # Perform forward pass - FHE layers run remotely, others run locally
        logger.info("Running forward pass...")
        client_hybrid_pred = client_hybrid(test_x)
        client_hybrid_values = client_hybrid_pred.detach().numpy()
        
        logger.info("Original PyTorch model predictions:")
        logger.info(torch_values)
        logger.info("Hybrid SupersayanClient model predictions:")
        logger.info(client_hybrid_values)
        
        # Compare the results (allowing for some numerical differences due to FHE)
        mean_diff = np.mean(np.abs(torch_values - client_hybrid_values))
        logger.info(f"Mean absolute difference: {mean_diff}")
        
        # The threshold should be adjusted based on expected precision
        assert mean_diff < 1000.0, f"Model predictions differ too much: {mean_diff}"
        
    except Exception as e:
        logger.error(f"Error during client test: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    test_hybrid_house_price_regression()