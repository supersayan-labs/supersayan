# Supersayan Benchmarks

This directory contains benchmarks for the Supersayan library, focusing on performance evaluation of different neural network layer implementations.

## Available Benchmarks

- `test_benchmark_linear.py`: Benchmarks for the Linear layer with different input and output sizes
- `test_benchmark_conv2d.py`: Benchmarks for the standard Conv2d layer with various configurations
- `test_benchmark_conv2d_orion.py`: Benchmarks for the optimized Conv2dOrion layer
- `test_benchmark_comparison.py`: Separate benchmarks for Conv2d and Conv2dOrion with same parameters for comparison
- `test_verify_correctness.py`: Tests that verify numerical correctness against PyTorch's native operations

## Test Naming Conventions

All tests follow a consistent naming pattern:

- `test_benchmark_*`: Performance benchmarks for individual layer types
  - Example: `test_benchmark_linear[1-10-10]` - Linear layer with batch=1, in=10, out=10

- `test_benchmark_*_comparison`: Comparison benchmarks from test_benchmark_comparison.py
  - Example: `test_benchmark_conv2d_comparison[3-4-8-3-1-1]` - Regular Conv2d
  - Example: `test_benchmark_conv2d_orion_comparison[3-4-8-3-1-1]` - Optimized Conv2dOrion

- `test_verify_*`: Correctness verification tests comparing against PyTorch
  - Example: `test_verify_conv2d` - Verifies Conv2d matches PyTorch results
  - Example: `test_verify_conv2d_orion` - Verifies Conv2dOrion matches PyTorch results

## Running Benchmarks

You can run the benchmarks using pytest with the following command:

```bash
# Run all benchmarks
pytest benchmarks/ -v

# Run specific benchmark
pytest benchmarks/test_benchmark_linear.py -v

# Run benchmarks and generate a detailed report
pytest benchmarks/ --benchmark-autosave --benchmark-save=mysession

# Compare results from different runs
pytest-benchmark compare last mysession

# Compare standard Conv2d vs Conv2dOrion
pytest benchmarks/test_benchmark_comparison.py -v

# Run correctness verification against PyTorch
pytest benchmarks/test_verify_correctness.py -v

# List all available tests to see standardized names
pytest benchmarks/ --collect-only -v
```

## Parameters

The benchmarks test various configurations:

- Batch sizes: 1, 8
- Input channels: 1, 3
- Output channels: 1, 4
- Input sizes: 8x8, 16x16
- Kernel sizes: 3x3
- Stride: 1, 2
- Padding: 1

## Optimization Notes

The Conv2dOrion implementation uses several optimizations:
1. Toeplitz-based encoding: Converting convolution to matrix-vector product
2. Single-shot multiplexing: Handling stride > 1 in a single multiplicative depth
3. BSGS (Baby-Step Giant-Step): Reduces rotations from O(n) to O(sqrt(n))
4. Double-hoisting: Reuses expensive parts of key-switching across multiple rotations # FIXME: This is not yet implemented.