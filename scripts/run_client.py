from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
from torchvision import models

from supersayan.logging_config import configure_logging, get_logger
from supersayan.remote.client import SupersayanClient

# Configure logging
configure_logging(
    level="INFO", console_format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = get_logger(__name__)


def test_hybrid_house_price_regression(
    server: str = "127.0.0.1:8000",
) -> None:
    class HousePriceRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear1 = nn.Linear(5, 16)
            self.relu1 = nn.ReLU()
            self.dropout = nn.Dropout(0.1)
            self.linear2 = nn.Linear(16, 8)
            self.relu2 = nn.ReLU()
            self.linear3 = nn.Linear(8, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.relu1(self.linear1(x))
            x = self.dropout(x)
            x = self.relu2(self.linear2(x))
            return self.linear3(x)

    torch_model = HousePriceRegressor()
    num_samples = 100
    test_x = torch.rand(num_samples, 5)
    torch_values = torch_model(test_x)

    client = SupersayanClient(
        server_url=server, torch_model=torch_model, fhe_modules=[nn.Linear]
    )
    client_values = client(test_x)

    mean_diff = float(np.mean(np.abs(torch_values - client_values)))
    logger.info("House‑price regression – mean abs diff (first 10): %.6f", mean_diff)
    assert mean_diff < 1000.0, "model predictions differ too much"


def test_resnet18_random_input(server: str = "127.0.0.1:8000") -> None:
    torch_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    torch_model.eval()

    print(summary(torch_model, (3, 224, 224)))

    test_x = torch.rand(1, 3, 224, 224)
    torch_values = torch_model(test_x)

    client = SupersayanClient(
        server_url=server, torch_model=torch_model, fhe_modules=[nn.Conv2d, nn.Linear]
    )

    start = time.time()
    client_values = client(test_x)
    end = time.time()
    print(f"Time taken: {end - start} seconds")

    mean_diff = float(np.mean(np.abs(torch_values - client_values)))
    logger.info("ResNet‑18 – mean abs diff: %.6f", mean_diff)
    assert mean_diff < 1.0, "predictions differ too much"


def test_mnist_cnn(server: str = "127.0.0.1:8000") -> None:
    """Test a small CNN model on MNIST-like input data."""

    class MNISTNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.flatten = nn.Flatten()

            self.fc1 = nn.Linear(32 * 7 * 7, 128)
            self.relu3 = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = self.flatten(x)
            x = self.dropout(self.relu3(self.fc1(x)))
            x = self.fc2(x)
            return x

    torch_model = MNISTNet()
    torch_model.to("cpu")
    torch_model.eval()

    logger.info("MNIST CNN Model Summary:")
    # print(summary(torch_model, (1, 28, 28)))

    batch_size = 4
    test_x = torch.rand(batch_size, 1, 28, 28, device="cpu")

    with torch.no_grad():
        torch_values = torch_model(test_x)

    client = SupersayanClient(
        server_url=server, torch_model=torch_model, fhe_modules=[nn.Conv2d, nn.Linear]
    )

    start = time.time()
    client_values = client(test_x)
    end = time.time()
    print(f"Time taken: {end - start} seconds")

    mean_diff = float(torch.mean(torch.abs(torch_values - client_values)))
    max_diff = float(torch.max(torch.abs(torch_values - client_values)))

    logger.info("MNIST CNN – mean abs diff: %.6f", mean_diff)
    logger.info("MNIST CNN – max abs diff: %.6f", max_diff)
    logger.info("MNIST CNN – PyTorch output shape: %s", torch_values.shape)
    logger.info("MNIST CNN – Client output shape: %s", client_values.shape)

    logger.info("PyTorch predictions (first sample): %s", torch_values[0])
    logger.info("Client predictions (first sample): %s", client_values[0])

    assert mean_diff < 1.0, f"predictions differ too much: {mean_diff}"
    logger.info("MNIST CNN test passed!")


if __name__ == "__main__":
    # test_hybrid_house_price_regression()
    # test_resnet18_random_input()
    test_mnist_cnn()
