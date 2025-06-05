from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from supersayan.core.timing import get_timing_collector, reset_timing_collector
from supersayan.core.types import SupersayanTensor
from supersayan.logging_config import configure_logging, get_logger
from supersayan.remote.client import SupersayanClient

# Configure logging
configure_logging(
    level="INFO", console_format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = get_logger(__name__)


def run_timing_benchmark(
    server: str = "127.0.0.1:8000",
    num_samples: int = 5,
) -> dict:
    """Run a timing benchmark with detailed metrics."""
    logger.info("Starting detailed timing benchmark...")

    class SimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.relu1 = nn.ReLU()
            self.linear2 = nn.Linear(20, 15)
            self.relu2 = nn.ReLU()
            self.linear3 = nn.Linear(15, 5)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.relu1(self.linear1(x))
            x = self.relu2(self.linear2(x))
            x = self.linear3(x)
            return x

    torch_model = SimpleNet()
    torch_model.eval()

    # Create test data
    test_x = SupersayanTensor(torch.rand(num_samples, 10))

    # Reset timing collector
    reset_timing_collector()

    # Create client with timing enabled
    client = SupersayanClient(
        server_url=server,
        torch_model=torch_model,
        fhe_modules=[nn.Linear],  # Only Linear layers will be FHE
        enable_timing=True
    )

    logger.info(f"Running inference on {num_samples} samples...")
    
    # Run inference (this will collect timing data)
    start_time = time.time()
    output = client(test_x)
    end_time = time.time()
    total_time = end_time - start_time

    # Get detailed timing statistics
    timing_stats = client.get_timing_stats()

    result = {
        "model": "SimpleNet",
        "num_samples": num_samples,
        "total_time": total_time,
        "average_time_per_sample": total_time / num_samples,
        "timing_stats": timing_stats,
        "timestamp": datetime.now().isoformat(),
    }

    logger.info(f"Completed benchmark in {total_time:.4f}s ({total_time/num_samples:.4f}s/sample)")
    
    return result


def print_timing_results(results: dict) -> None:
    """Print timing results in a readable format."""
    print("\n" + "="*80)
    print("DETAILED TIMING BENCHMARK RESULTS")
    print("="*80)
    
    print(f"Model: {results['model']}")
    print(f"Samples: {results['num_samples']}")
    print(f"Total time: {results['total_time']:.4f}s")
    print(f"Average time per sample: {results['average_time_per_sample']:.4f}s")
    
    timing_stats = results.get("timing_stats", {})
    
    if "fhe_layers" in timing_stats:
        print("\nFHE LAYERS:")
        print("-" * 40)
        for layer_name, stats in timing_stats["fhe_layers"].items():
            print(f"\n  {layer_name}:")
            print(f"    Count: {stats['count']}")
            print(f"    Encryption time: {stats['encryption_time']['mean']:.4f}s (±{stats['encryption_time']['max'] - stats['encryption_time']['min']:.4f}s)")
            print(f"    Send time: {stats['send_time']['mean']:.4f}s (±{stats['send_time']['max'] - stats['send_time']['min']:.4f}s)")
            print(f"    Inference time: {stats['inference_time']['mean']:.4f}s (±{stats['inference_time']['max'] - stats['inference_time']['min']:.4f}s)")
            print(f"    Receive time: {stats['receive_time']['mean']:.4f}s (±{stats['receive_time']['max'] - stats['receive_time']['min']:.4f}s)")
            print(f"    Decryption time: {stats['decryption_time']['mean']:.4f}s (±{stats['decryption_time']['max'] - stats['decryption_time']['min']:.4f}s)")
            print(f"    Encrypted input size: {stats['encrypted_input_size_bytes']['mean']:.0f} bytes")
            print(f"    Encrypted output size: {stats['encrypted_output_size_bytes']['mean']:.0f} bytes")
    
    if "non_fhe_layers" in timing_stats:
        print("\nNON-FHE LAYERS:")
        print("-" * 40)
        for layer_name, stats in timing_stats["non_fhe_layers"].items():
            print(f"\n  {layer_name}:")
            print(f"    Count: {stats['count']}")
            print(f"    Torch inference time: {stats['torch_inference_time']['mean']:.4f}s (±{stats['torch_inference_time']['max'] - stats['torch_inference_time']['min']:.4f}s)")
    
    print("\n" + "="*80)


def save_timing_results(results: dict, filename: str = None) -> None:
    """Save timing results to a JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"timing_benchmark_{timestamp}.json"
    
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Timing results saved to {filepath}")


def main():
    """Run the timing benchmark."""
    try:
        results = run_timing_benchmark(num_samples=3)
        print_timing_results(results)
        save_timing_results(results)
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 