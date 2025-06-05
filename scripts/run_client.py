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


def print_timing_details(timing_summary: dict, model_name: str) -> None:
    """Print detailed timing breakdown."""
    print(f"\n{'='*60}")
    print(f"DETAILED TIMING BREAKDOWN - {model_name}")
    print(f"{'='*60}")
    
    # FHE Layers
    if timing_summary['fhe_layers']:
        print(f"\nFHE LAYERS:")
        print(f"{'Layer':<20} {'Encrypt':<10} {'Send':<10} {'Inference':<10} {'Receive':<10} {'Decrypt':<10} {'In Size':<12} {'Out Size':<12}")
        print("-" * 110)
        
        for layer_name, metrics in timing_summary['fhe_layers'].items():
            print(f"{layer_name:<20} "
                  f"{metrics['avg_encryption_time']*1000:>8.2f}ms "
                  f"{metrics['avg_send_time']*1000:>8.2f}ms "
                  f"{metrics['avg_inference_time']*1000:>8.2f}ms "
                  f"{metrics['avg_receive_time']*1000:>8.2f}ms "
                  f"{metrics['avg_decryption_time']*1000:>8.2f}ms "
                  f"{metrics['avg_encrypted_input_size']:>10.0f}B "
                  f"{metrics['avg_encrypted_output_size']:>10.0f}B")
    
    # Non-FHE Layers  
    if timing_summary['non_fhe_layers']:
        print(f"\nNON-FHE LAYERS:")
        print(f"{'Layer':<20} {'Torch Time':<12}")
        print("-" * 35)
        
        for layer_name, metrics in timing_summary['non_fhe_layers'].items():
            print(f"{layer_name:<20} {metrics['avg_torch_inference_time']*1000:>10.2f}ms")
    
    # Totals
    totals = timing_summary['totals']
    print(f"\nTOTALS (averaged across {totals['total_samples']} samples):")
    print("-" * 50)
    print(f"Total Encryption Time:     {totals['avg_total_encryption_time']*1000:>8.2f}ms")
    print(f"Total Send Time:           {totals['avg_total_send_time']*1000:>8.2f}ms") 
    print(f"Total Inference Time:      {totals['avg_total_inference_time']*1000:>8.2f}ms")
    print(f"Total Receive Time:        {totals['avg_total_receive_time']*1000:>8.2f}ms")
    print(f"Total Decryption Time:     {totals['avg_total_decryption_time']*1000:>8.2f}ms")
    print(f"Total Torch Inference:     {totals['avg_total_torch_inference_time']*1000:>8.2f}ms")
    
    total_fhe_time = (totals['avg_total_encryption_time'] + 
                      totals['avg_total_send_time'] + 
                      totals['avg_total_inference_time'] + 
                      totals['avg_total_receive_time'] + 
                      totals['avg_total_decryption_time'])
    total_time = total_fhe_time + totals['avg_total_torch_inference_time']
    
    print(f"Total FHE Pipeline Time:   {total_fhe_time*1000:>8.2f}ms")
    print(f"TOTAL TIME:                {total_time*1000:>8.2f}ms")


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

    # Benchmark client with detailed timing
    client = SupersayanClient(
        server_url=server, torch_model=torch_model, fhe_modules=[nn.Linear]
    )

    # Reset timing before benchmarking
    client.reset_timing()
    
    client_start = time.time()
    client_values = client(test_x)
    client_end = time.time()
    client_time = client_end - client_start
    client_time_per_sample = client_time / num_samples

    # Get detailed timing summary
    timing_summary = client.get_timing_summary()
    
    # Print detailed timing breakdown
    print_timing_details(timing_summary, "HousePriceRegressor")

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
        "detailed_timing": timing_summary,  # Add detailed timing data
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

    # Benchmark client with detailed timing
    client = SupersayanClient(
        server_url=server, torch_model=torch_model, fhe_modules=[nn.Conv2d, nn.Linear]
    )

    # Reset timing before benchmarking
    client.reset_timing()

    client_start = time.time()
    client_values = client(test_x)
    client_end = time.time()
    client_time = client_end - client_start
    client_time_per_sample = client_time / num_samples

    # Get detailed timing summary
    timing_summary = client.get_timing_summary()
    
    # Print detailed timing breakdown
    print_timing_details(timing_summary, "ResNet18")

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
        "detailed_timing": timing_summary,  # Add detailed timing data
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

    # Benchmark client with detailed timing
    client = SupersayanClient(
        server_url=server, torch_model=torch_model, fhe_modules=[nn.Conv2d, nn.Linear]
    )

    # Reset timing before benchmarking
    client.reset_timing()

    client_start = time.time()
    client_values = client(test_x)
    client_end = time.time()
    client_time = client_end - client_start
    client_time_per_sample = client_time / num_samples

    # Get detailed timing summary
    timing_summary = client.get_timing_summary()
    
    # Print detailed timing breakdown
    print_timing_details(timing_summary, "MNIST_CNN")

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
        "detailed_timing": timing_summary,  # Add detailed timing data
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
            
            # Add timing breakdown summary
            if "detailed_timing" in benchmark:
                timing = benchmark["detailed_timing"]["totals"]
                total_fhe_time = (timing.get('avg_total_encryption_time', 0) + 
                                timing.get('avg_total_send_time', 0) + 
                                timing.get('avg_total_inference_time', 0) + 
                                timing.get('avg_total_receive_time', 0) + 
                                timing.get('avg_total_decryption_time', 0))
                print(f"  FHE Pipeline time: {total_fhe_time:.4f}s")
                print(f"  Non-FHE time: {timing.get('avg_total_torch_inference_time', 0):.4f}s")
        else:
            print(f"\n{benchmark['model']}: FAILED - {benchmark['error']}")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmarks()
