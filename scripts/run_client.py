from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from supersayan.core.types import SupersayanTensor
from supersayan.logging_config import configure_logging, get_logger
from supersayan.remote.client import SupersayanClient

# Configure logging
configure_logging(
    level="INFO", console_format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = get_logger(__name__)


def benchmark_hybrid_house_price_regression(
    server: str = "127.0.0.1:8000",
    num_samples: int = 10,
) -> dict:
    """Benchmark house price regression model."""
    logger.info("Starting house price regression benchmark...")

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
    torch_model.to("cuda")
    torch_model.eval()

    test_x = SupersayanTensor(
        torch.rand(num_samples, 5, device=torch.device("cuda")),
        device=torch.device("cuda"),
    )

    # Benchmark PyTorch
    torch_start = time.time()
    with torch.no_grad():
        torch_values = torch_model(test_x)
    torch_end = time.time()
    torch_time = torch_end - torch_start
    torch_time_per_sample = torch_time / num_samples

    # Benchmark client
    client = SupersayanClient(
        server_url=server, torch_model=torch_model, fhe_modules=[nn.Linear]
    )

    client_start = time.time()
    client_values = client(test_x)
    client_end = time.time()
    client_time = client_end - client_start
    client_time_per_sample = client_time / num_samples

    result = {
        "model": "HousePriceRegressor",
        "num_samples": num_samples,
        "input_shape": list(test_x.shape),
        "output_shape": list(client_values.shape),
        "torch_time": torch_time,
        "torch_time_per_sample": torch_time_per_sample,
        "client_time": client_time,
        "client_time_per_sample": client_time_per_sample,
        "speedup": torch_time / client_time if client_time > 0 else 0,
        "timestamp": datetime.now().isoformat(),
    }

    logger.info(
        f"House price regression - PyTorch time: {torch_time:.4f}s ({torch_time_per_sample:.4f}s/sample), Client time: {client_time:.4f}s ({client_time_per_sample:.4f}s/sample)"
    )
    return result


def benchmark_resnet18(server: str = "127.0.0.1:8000", num_samples: int = 1) -> dict:
    """Benchmark ResNet18 model."""
    logger.info("Starting ResNet18 benchmark...")

    torch_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    torch_model.to("cuda")
    torch_model.eval()

    test_x = SupersayanTensor(
        torch.rand(num_samples, 3, 224, 224, device=torch.device("cuda")),
        device=torch.device("cuda"),
    )

    # Benchmark PyTorch
    torch_start = time.time()
    with torch.no_grad():
        torch_values = torch_model(test_x)
    torch_end = time.time()
    torch_time = torch_end - torch_start
    torch_time_per_sample = torch_time / num_samples

    # Benchmark client
    client = SupersayanClient(
        server_url=server, torch_model=torch_model, fhe_modules=[nn.Conv2d, nn.Linear]
    )

    client_start = time.time()
    client_values = client(test_x)
    client_end = time.time()
    client_time = client_end - client_start
    client_time_per_sample = client_time / num_samples

    result = {
        "model": "ResNet18",
        "num_samples": num_samples,
        "input_shape": list(test_x.shape),
        "output_shape": list(client_values.shape),
        "torch_time": torch_time,
        "torch_time_per_sample": torch_time_per_sample,
        "client_time": client_time,
        "client_time_per_sample": client_time_per_sample,
        "speedup": torch_time / client_time if client_time > 0 else 0,
        "timestamp": datetime.now().isoformat(),
    }

    logger.info(
        f"ResNet18 - PyTorch time: {torch_time:.4f}s ({torch_time_per_sample:.4f}s/sample), Client time: {client_time:.4f}s ({client_time_per_sample:.4f}s/sample)"
    )
    return result


def benchmark_mnist_cnn(server: str = "127.0.0.1:8000", num_samples: int = 1) -> dict:
    """Benchmark MNIST CNN model."""
    logger.info("Starting MNIST CNN benchmark...")

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
    torch_model.to("cuda")
    torch_model.eval()

    test_x = SupersayanTensor(
        torch.randn(num_samples, 1, 28, 28, device=torch.device("cuda")),
        device=torch.device("cuda"),
    )

    # Benchmark PyTorch
    torch_start = time.time()
    with torch.no_grad():
        torch_values = torch_model(test_x)
    torch_end = time.time()
    torch_time = torch_end - torch_start
    torch_time_per_sample = torch_time / num_samples

    # Benchmark client
    client = SupersayanClient(
        server_url=server, torch_model=torch_model, fhe_modules=[nn.Conv2d, nn.Linear]
    )

    client_start = time.time()
    client_values = client(test_x)
    client_end = time.time()
    client_time = client_end - client_start
    client_time_per_sample = client_time / num_samples

    result = {
        "model": "MNIST_CNN",
        "num_samples": num_samples,
        "input_shape": list(test_x.shape),
        "output_shape": list(client_values.shape),
        "torch_time": torch_time,
        "torch_time_per_sample": torch_time_per_sample,
        "client_time": client_time,
        "client_time_per_sample": client_time_per_sample,
        "speedup": torch_time / client_time if client_time > 0 else 0,
        "timestamp": datetime.now().isoformat(),
    }

    logger.info(
        f"MNIST CNN - PyTorch time: {torch_time:.4f}s ({torch_time_per_sample:.4f}s/sample), Client time: {client_time:.4f}s ({client_time_per_sample:.4f}s/sample)"
    )
    return result


def run_benchmarks(server: str = "127.0.0.1:8000") -> None:
    """Run all benchmarks and save results to JSON."""
    results = {
        "benchmarks": [],
        "system_info": {
            "server_url": server,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
    }

    # Run benchmarks
    try:
        # House price regression with 10 samples
        results["benchmarks"].append(
            benchmark_hybrid_house_price_regression(server, num_samples=10)
        )
    except Exception as e:
        logger.error(f"House price regression benchmark failed: {e}")
        results["benchmarks"].append({"model": "HousePriceRegressor", "error": str(e)})

    try:
        # ResNet18 with 1 sample
        results["benchmarks"].append(benchmark_resnet18(server, num_samples=1))
    except Exception as e:
        logger.error(f"ResNet18 benchmark failed: {e}")
        results["benchmarks"].append({"model": "ResNet18", "error": str(e)})

    try:
        # MNIST CNN with 1 sample
        results["benchmarks"].append(benchmark_mnist_cnn(server, num_samples=1))
    except Exception as e:
        logger.error(f"MNIST CNN benchmark failed: {e}")
        results["benchmarks"].append({"model": "MNIST_CNN", "error": str(e)})

    # Save results
    output_file = Path("benchmark_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Benchmark results saved to {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    for benchmark in results["benchmarks"]:
        if "error" not in benchmark:
            print(f"\n{benchmark['model']}:")
            print(f"  Samples: {benchmark['num_samples']}")
            print(
                f"  PyTorch time: {benchmark['torch_time']:.4f}s ({benchmark['torch_time_per_sample']:.4f}s/sample)"
            )
            print(
                f"  Client time: {benchmark['client_time']:.4f}s ({benchmark['client_time_per_sample']:.4f}s/sample)"
            )
            print(f"  Speedup: {benchmark['speedup']:.2f}x")
        else:
            print(f"\n{benchmark['model']}: FAILED - {benchmark['error']}")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmarks()
