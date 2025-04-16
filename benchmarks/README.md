# SuperSayan Benchmarks

This directory contains benchmarks for the SuperSayan library, focusing on performance evaluation of different neural network layer implementations with Fully Homomorphic Encryption (FHE).

## Overview

SuperSayan benchmarks measure:
- Execution time of different FHE layer implementations
- Comparison between standard and optimized implementations
- Scaling behavior with different input sizes, batch sizes, and layer configurations
- Correctness verification against PyTorch's native operations

## Available Benchmarks

| Benchmark File | Description |
|----------------|-------------|
| `test_benchmark_linear.py` | Benchmarks for the Linear layer with different input and output sizes |
| `test_benchmark_conv2d.py` | Benchmarks for the standard Conv2d layer with various configurations |
| `test_benchmark_conv2d_large.py` | Extended benchmarks for Conv2d with larger input sizes |
| `test_benchmark_conv2d_orion.py` | Benchmarks for the optimized Conv2dOrion layer using Orion techniques |
| `test_benchmark_comparison.py` | Direct comparison between Conv2d and Conv2dOrion with identical parameters |

## Benchmark Parameters

The benchmarks test various configurations:

- **Batch sizes**: 1, 8
- **Input channels**: 1, 3
- **Output channels**: 1, 4
- **Input sizes**: 8×8, 16×16
- **Kernel sizes**: 3×3
- **Stride**: 1, 2
- **Padding**: 1

## Running Benchmarks

### Basic Usage

```bash
# Run all benchmarks
pytest benchmarks/ -v

# Run specific benchmark
pytest benchmarks/test_benchmark_linear.py -v

# Run benchmarks and generate a detailed report
pytest benchmarks/ --benchmark-autosave --benchmark-save=mysession

# Compare results from different runs
pytest-benchmark compare last mysession
```

### Specific Comparison Tests

```bash
# Compare standard Conv2d vs Conv2dOrion
pytest benchmarks/test_benchmark_comparison.py -v

# Run verification tests against PyTorch
pytest benchmarks/test_verify_correctness.py -v
```

### Listing Available Tests

```bash
# List all available tests with their parameters
pytest benchmarks/ --collect-only -v
```

## Test Naming Conventions

All benchmarks follow a consistent naming pattern:

- **test_benchmark_***: Performance benchmarks for individual layer types
  - Example: `test_benchmark_linear[1-10-10]` - Linear layer with batch=1, in=10, out=10

- **test_benchmark_*_comparison**: Comparison benchmarks from test_benchmark_comparison.py
  - Example: `test_benchmark_conv2d_comparison[3-4-8-3-1-1]` - Regular Conv2d
  - Example: `test_benchmark_conv2d_orion_comparison[3-4-8-3-1-1]` - Optimized Conv2dOrion

## Optimization Techniques

The Conv2dOrion implementation uses several key optimizations from the [Orion FHE paper](https://arxiv.org/pdf/2311.03470):

1. **Toeplitz-based encoding**: Converting convolution to matrix-vector product
2. **Single-shot multiplexing**: Handling stride > 1 in a single multiplicative depth
3. **BSGS (Baby-Step Giant-Step)**: Reduces rotations from O(n) to O(sqrt(n))
4. **Double-hoisting**: Reuses expensive parts of key-switching across multiple rotations (in development)

## Interpreting Results

Benchmark results show:

- **Mean**: Average execution time
- **Min/Max**: Minimum/maximum execution time across runs
- **StdDev**: Standard deviation of execution times
- **Median**: Median execution time
- **IQR**: Interquartile range (spread of the middle 50% of execution times)
- **Outliers**: Number of outlier measurements

When comparing Conv2d vs Conv2dOrion, expect to see significant performance improvements with the optimized implementation, especially for larger input sizes and configurations with stride > 1.

## Adding New Benchmarks

When adding new benchmarks:

1. Follow the existing naming conventions
2. Use `@pytest.mark.parametrize` for testing multiple configurations
3. Include verification against PyTorch when appropriate
4. Document any new optimization techniques implemented