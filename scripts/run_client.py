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
    test_x = torch.rand(num_samples, 5, dtype=torch.float32)

    # Local ground truth (first 10 rows)
    torch_values = torch_model(test_x[:10]).detach().numpy()

    client = SupersayanClient(
        server_url=server, torch_model=torch_model, fhe_modules=[nn.Linear]
    )
    client_values = client(test_x)[:10].detach().numpy()

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

    test_x = torch.rand(1, 3, 224, 224, dtype=torch.float32)
    torch_values = torch_model(test_x).detach().numpy()

    client = SupersayanClient(
        server_url=server, torch_model=torch_model, fhe_modules=[nn.Conv2d, nn.Linear]
    )
    client_values = client(test_x).detach().numpy()

    mean_diff = float(np.mean(np.abs(torch_values - client_values)))
    logger.info("ResNet‑18 – mean abs diff: %.6f", mean_diff)
    assert mean_diff < 1.0, "predictions differ too much"


if __name__ == "__main__":
    # test_hybrid_house_price_regression()
    test_resnet18_random_input()
