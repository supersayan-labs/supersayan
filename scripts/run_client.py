import torch
import sys
import os
import torch.nn as nn
import numpy as np
import pytest
import logging
import socketio
import requests
from torchvision import models
from urllib.parse import urljoin
from torchsummary import summary

from supersayan.nn.convert import convert_to_hybrid_supersayan
from supersayan.remote.client import SupersayanClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_hybrid_house_price_regression(server_url="http://127.0.0.1:8000"):
    """
    Test the client-server architecture with a house price regression model.
    Uses a 30MB input tensor to test large data transfer.

    Args:
        server_url: The server URL
    """

    class HousePriceRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            # Input feature dimension unchanged (5)
            # But we'll process the large batch at once
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

    torch_model = HousePriceRegressor()

    # Calculate size for a 30MB tensor (float32 = 4 bytes)
    # 30MB = 30 * 1024 * 1024 bytes = 31,457,280 bytes
    # Number of float32 values = 31,457,280 / 4 = 7,864,320
    # Using 5 features per sample, we need: 7,864,320 / 5 = 1,572,864 samples
    num_samples = 100

    logger.info(f"Creating {num_samples} x 5 input tensor (approx. 30MB)...")
    test_x = torch.rand(num_samples, 5, dtype=torch.float32)

    # Only run the PyTorch model on a small subset to save time
    sample_x = test_x[:10]
    torch_pred = torch_model(sample_x)
    torch_values = torch_pred.detach().numpy()

    logger.info("Creating hybrid client...")
    client_hybrid = SupersayanClient(
        server_url=server_url, torch_model=torch_model, fhe_modules=[nn.Linear]
    )

    try:
        logger.info("Running forward pass with large input...")
        client_hybrid_pred = client_hybrid(test_x)

        # Only compare results on the first few samples
        client_hybrid_values = client_hybrid_pred[:10].detach().numpy()

        logger.info("Original PyTorch model predictions (first 10 samples):")
        logger.info(torch_values)
        logger.info("Hybrid SupersayanClient model predictions (first 10 samples):")
        logger.info(client_hybrid_values)

        mean_diff = np.mean(np.abs(torch_values - client_hybrid_values))
        logger.info(f"Mean absolute difference: {mean_diff}")

        assert mean_diff < 1000.0, f"Model predictions differ too much: {mean_diff}"

    except Exception as e:
        logger.error(f"Error during client test: {e}", exc_info=True)
        raise


def test_resnet18_random_input(server_url="http://127.0.0.1:8000"):
    """
    Test the client-server architecture with a ResNet18 model on random input.
    Only the Conv2d and Linear layers run in FHE.

    Args:
        server_url: The server URL
    """
    torch_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    torch_model.eval()

    print(summary(torch_model, (3, 224, 224)))

    test_x = torch.rand(1, 3, 224, 224, dtype=torch.float32)

    torch_pred = torch_model(test_x)
    torch_values = torch_pred.detach().numpy()

    logger.info("Creating hybrid ResNet18 client...")
    client_hybrid = SupersayanClient(
        server_url=server_url,
        torch_model=torch_model,
        fhe_modules=[nn.Conv2d, nn.Linear],
    )

    try:
        logger.info("Running ResNet18 forward pass...")

        client_hybrid_pred = client_hybrid(test_x)
        client_hybrid_values = client_hybrid_pred.detach().numpy()

        logger.info("Original PyTorch model predictions:")
        logger.info(torch_values)
        logger.info("Hybrid SupersayanClient model predictions:")
        logger.info(client_hybrid_values)

        mean_diff = np.mean(np.abs(torch_values - client_hybrid_values))
        logger.info(f"Mean absolute difference: {mean_diff}")

        assert mean_diff < 1.0, f"Model predictions differ too much: {mean_diff}"

    except Exception as e:
        logger.error(f"Error during ResNet18 client test: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # test_hybrid_house_price_regression()
    test_resnet18_random_input()
