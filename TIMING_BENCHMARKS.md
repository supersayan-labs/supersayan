# Detailed Timing and Benchmarking

This document describes the comprehensive timing and benchmarking system implemented in SuperSayan for analyzing the performance of hybrid FHE/non-FHE neural network inference.

## Overview

The timing system provides detailed metrics for both FHE (Fully Homomorphic Encryption) layers and non-FHE layers, allowing you to understand exactly where time is spent during inference.

## Metrics Collected

### For FHE Layers (per layer, per sample):

- **Encryption Time**: Time to encrypt input data using LWES
- **Encrypted Input Size**: Size in bytes of the encrypted input data
- **Send Time**: Time to send encrypted data to the server
- **Inference Time**: Time spent on the server computing the layer (excluding network overhead)
- **Receive Time**: Time to receive encrypted results from the server
- **Encrypted Output Size**: Size in bytes of the encrypted output data
- **Decryption Time**: Time to decrypt the output data

### For Non-FHE Layers:

- **Torch Inference Time**: Time spent executing PyTorch operations locally

### Aggregated Metrics:

- **Total times**: Sum of each timing category across all layers
- **Averages**: Average time per sample for each metric
- **Per-layer breakdowns**: Individual layer performance analysis

## Usage

### Basic Usage

```python
from supersayan.remote.client import SupersayanClient
import torch.nn as nn

# Create your model
model = YourModel()

# Create client with FHE modules specified
client = SupersayanClient(
    server_url="127.0.0.1:8000",
    torch_model=model,
    fhe_modules=[nn.Linear, nn.Conv2d]  # Specify which layer types to run in FHE
)

# Reset timing before benchmarking
client.reset_timing()

# Run inference
output = client(input_data)

# Get detailed timing summary
timing_summary = client.get_timing_summary()
```

### Timing Summary Structure

The `get_timing_summary()` method returns a dictionary with the following structure:

```python
{
    'fhe_layers': {
        'layer_name': {
            'avg_encryption_time': float,      # Average encryption time (seconds)
            'avg_encrypted_input_size': float, # Average input size (bytes)
            'avg_send_time': float,            # Average send time (seconds)
            'avg_inference_time': float,       # Average inference time (seconds)
            'avg_receive_time': float,         # Average receive time (seconds)
            'avg_encrypted_output_size': float,# Average output size (bytes)
            'avg_decryption_time': float,      # Average decryption time (seconds)
            'sample_count': int                # Number of samples processed
        }
    },
    'non_fhe_layers': {
        'layer_name': {
            'avg_torch_inference_time': float, # Average PyTorch inference time (seconds)
            'sample_count': int                # Number of samples processed
        }
    },
    'totals': {
        'avg_total_encryption_time': float,    # Total encryption time across all layers
        'avg_total_send_time': float,          # Total send time across all layers
        'avg_total_inference_time': float,     # Total inference time across all layers
        'avg_total_receive_time': float,       # Total receive time across all layers
        'avg_total_decryption_time': float,    # Total decryption time across all layers
        'avg_total_torch_inference_time': float, # Total PyTorch time for non-FHE layers
        'total_samples': int                   # Total number of samples processed
    }
}
```

## Running Benchmarks

### Full Benchmark Suite

Run the complete benchmark suite with detailed timing:

```bash
python scripts/run_client.py
```

This will benchmark:

- House Price Regression model (10 samples)
- ResNet18 (1 sample)
- MNIST CNN (1 sample)

### Simple Test

Run a simple timing test:

```bash
python scripts/test_timing.py
```

### Custom Benchmarks

You can create custom benchmarks by following this pattern:

```python
def benchmark_my_model(server: str = "127.0.0.1:8000", num_samples: int = 1):
    # Create your model
    model = MyModel()

    # Create test data
    test_data = torch.randn(num_samples, *input_shape)

    # Create client
    client = SupersayanClient(
        server_url=server,
        torch_model=model,
        fhe_modules=[nn.Linear]  # Specify FHE layers
    )

    # Reset and run timing
    client.reset_timing()
    output = client(test_data)

    # Get results
    timing_summary = client.get_timing_summary()

    # Print detailed breakdown (optional)
    print_timing_details(timing_summary, "MyModel")

    return timing_summary
```

## Example Output

When you run the benchmarks, you'll see detailed output like this:

```
============================================================
DETAILED TIMING BREAKDOWN - HousePriceRegressor
============================================================

FHE LAYERS:
Layer                Encrypt    Send       Inference  Receive    Decrypt    In Size      Out Size
--------------------------------------------------------------------------------------------------------------
linear1              12.34ms    5.67ms     89.12ms    4.23ms     8.90ms      1234B        5678B
linear2              8.76ms     3.45ms     45.67ms    2.89ms     6.78ms      5678B        2345B
linear3              6.54ms     2.34ms     23.45ms    1.67ms     4.32ms      2345B         123B

NON-FHE LAYERS:
Layer                Torch Time
-----------------------------------
all_non_fhe_layers       2.34ms

TOTALS (averaged across 10 samples):
--------------------------------------------------
Total Encryption Time:      27.64ms
Total Send Time:             11.46ms
Total Inference Time:       158.24ms
Total Receive Time:          8.79ms
Total Decryption Time:      20.00ms
Total Torch Inference:       2.34ms
Total FHE Pipeline Time:    226.13ms
TOTAL TIME:                 228.47ms
```

## Understanding the Results

### Performance Analysis

1. **Encryption/Decryption Overhead**: Compare encryption and decryption times to understand the computational overhead of FHE operations.

2. **Network Overhead**: Send and receive times show network latency and throughput limitations.

3. **Server Performance**: Inference time shows the actual computational time on the server for FHE operations.

4. **Data Size Impact**: Input and output sizes help understand memory and bandwidth requirements.

5. **FHE vs Non-FHE Trade-offs**: Compare FHE pipeline time with PyTorch inference time to understand the security vs performance trade-off.

### Optimization Insights

- **High encryption/decryption times**: Consider optimizing FHE parameters or reducing precision
- **High send/receive times**: Consider data compression or network optimization
- **High inference times**: May indicate server computational bottlenecks
- **Large data sizes**: Consider model compression or layer fusion techniques

## Configuration

### Server Setup

Make sure you have a SuperSayan server running:

```bash
python -m supersayan.remote.server --host 127.0.0.1 --port 8000
```

### Custom FHE Modules

You can specify which layer types should run in FHE:

```python
# Run only Linear layers in FHE
client = SupersayanClient(..., fhe_modules=[nn.Linear])

# Run Conv2d and Linear layers in FHE
client = SupersayanClient(..., fhe_modules=[nn.Conv2d, nn.Linear])

# Run specific layer instances
client = SupersayanClient(..., fhe_modules=['layer1.conv', 'classifier.linear'])
```

## Output Files

Benchmark results are saved to `benchmark_results.json` with complete timing data for further analysis:

```json
{
  "benchmarks": [
    {
      "model": "HousePriceRegressor",
      "num_samples": 10,
      "detailed_timing": {
        "fhe_layers": {...},
        "non_fhe_layers": {...},
        "totals": {...}
      }
    }
  ]
}
```

This comprehensive timing system allows you to:

- Identify performance bottlenecks
- Optimize FHE/non-FHE layer allocation
- Understand the cost of privacy-preserving inference
- Make data-driven decisions about model architecture and deployment
