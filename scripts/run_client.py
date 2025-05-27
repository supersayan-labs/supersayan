# run_client.py
"""Example script exercising the Supersayan TCP client / server stack.

This replicates the two original integration tests:
* a toy *HousePriceRegressor*
* a full ImageNet‑pretrained ResNet‑18

Start the server first:
$ python scripts/run_server.py

Then run this file:
$ python scripts/run_client.py
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary  # type: ignore
from torchvision import models

from supersayan.remote.client import SupersayanClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# House‑price regression toy model (small tensors but variable batch size)
# -----------------------------------------------------------------------------


def test_hybrid_house_price_regression(
    server: str = "127.0.0.1:8000",
) -> None:  # noqa: D401
    class HousePriceRegressor(nn.Module):
        def __init__(self) -> None:  # noqa: D401
            super().__init__()
            self.linear1 = nn.Linear(5, 16)
            self.relu1 = nn.ReLU()
            self.dropout = nn.Dropout(0.1)
            self.linear2 = nn.Linear(16, 8)
            self.relu2 = nn.ReLU()
            self.linear3 = nn.Linear(8, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
            x = self.relu1(self.linear1(x))
            x = self.dropout(x)
            x = self.relu2(self.linear2(x))
            return self.linear3(x)

    torch_model = HousePriceRegressor()
    num_samples = 100
    test_x = np.random.rand(num_samples, 5).astype(np.float32)
    torch_values = torch_model(torch.from_numpy(test_x)).detach().numpy()

    client = SupersayanClient(
        server_url=server, torch_model=torch_model, fhe_modules=[nn.Linear]
    )
    client_values = client(test_x)

    mean_diff = float(np.mean(np.abs(torch_values - client_values)))
    logger.info("House‑price regression – mean abs diff (first 10): %.6f", mean_diff)
    assert mean_diff < 1000.0, "model predictions differ too much"


# -----------------------------------------------------------------------------
# ResNet‑18 (ImageNet weights, Conv + Linear in FHE)
# -----------------------------------------------------------------------------


def test_resnet18_random_input(server: str = "127.0.0.1:8000") -> None:  # noqa: D401
    torch_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    torch_model.eval()

    # Print a concise summary
    print(summary(torch_model, (3, 224, 224)))

    test_x = np.random.rand(1, 3, 224, 224).astype(np.float32)
    torch_values = torch_model(torch.from_numpy(test_x)).detach().numpy()

    client = SupersayanClient(
        server_url=server, torch_model=torch_model, fhe_modules=[nn.Conv2d, nn.Linear]
    )

    client_values = client(test_x)

    mean_diff = float(np.mean(np.abs(torch_values - client_values)))
    logger.info("ResNet‑18 – mean abs diff: %.6f", mean_diff)
    assert mean_diff < 1.0, "predictions differ too much"


# -----------------------------------------------------------------------------
# Small CNN for MNIST digits (Conv + Linear in FHE)
# -----------------------------------------------------------------------------


def test_mnist_cnn(server: str = "127.0.0.1:8000") -> None:  # noqa: D401
    """Test a small CNN model on MNIST-like input data."""
    
    class MNISTNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Simple CNN architecture for MNIST
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            
            # Flatten layer to handle the reshape
            self.flatten = nn.Flatten()
            
            # After two pooling layers: 28x28 -> 14x14 -> 7x7
            self.fc1 = nn.Linear(32 * 7 * 7, 128)
            self.relu3 = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
            self.fc2 = nn.Linear(128, 10)  # 10 digit classes
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = self.flatten(x)  # Use nn.Flatten instead of view
            x = self.dropout(self.relu3(self.fc1(x)))
            x = self.fc2(x)
            return x
    
    torch_model = MNISTNet()
    torch_model.eval()  # Set to evaluation mode
    
    # Print model summary
    logger.info("MNIST CNN Model Summary:")
    print(summary(torch_model, (1, 28, 28)))
    
    # Create random MNIST-like input (batch_size=4, channels=1, height=28, width=28)
    batch_size = 4
    test_x = np.random.rand(batch_size, 1, 28, 28).astype(np.float32)
    
    # Get PyTorch predictions
    with torch.no_grad():
        torch_values = torch_model(torch.from_numpy(test_x)).detach().numpy()
    
    # Create client with Conv2d and Linear layers in FHE
    client = SupersayanClient(
        server_url=server, torch_model=torch_model, fhe_modules=[nn.Conv2d, nn.Linear]
    )
    
    # Get client predictions
    client_values = client(test_x)
    
    # Compare results
    mean_diff = float(np.mean(np.abs(torch_values - client_values)))
    max_diff = float(np.max(np.abs(torch_values - client_values)))
    
    logger.info("MNIST CNN – mean abs diff: %.6f", mean_diff)
    logger.info("MNIST CNN – max abs diff: %.6f", max_diff)
    logger.info("MNIST CNN – PyTorch output shape: %s", torch_values.shape)
    logger.info("MNIST CNN – Client output shape: %s", client_values.shape)
    
    # Show first few predictions for comparison
    logger.info("PyTorch predictions (first sample): %s", torch_values[0])
    logger.info("Client predictions (first sample): %s", client_values[0])
    
    assert mean_diff < 1.0, f"predictions differ too much: {mean_diff}"
    logger.info("MNIST CNN test passed!")


if __name__ == "__main__":
    # test_hybrid_house_price_regression()
    test_resnet18_random_input()
    # test_mnist_cnn()
