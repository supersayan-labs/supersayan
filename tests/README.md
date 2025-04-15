# SuperSayan Test Suite

This directory contains the test suite for the SuperSayan library, a Fully Homomorphic Encryption (FHE) framework for neural networks that allows secure computation on encrypted data.

## Test Directory Contents

The tests directory contains unit tests, integration tests, and verification tests for the SuperSayan library. These tests ensure that:

1. Neural network layers implemented with FHE produce results comparable to regular PyTorch layers
2. Client-server architecture works correctly for remote computation
3. Different implementations of the same layer perform correctly with various configurations

## Test Files Overview

### Core Verification Tests

- **test_verify_correctness.py**: Verifies numerical correctness of FHE implementations compared to PyTorch native operations. This test focuses on ensuring that the encrypted computations produce results that are close to the non-encrypted versions.

### Layer Tests

- **test_conv2d.py**: Tests the standard 2D convolutional layer implementation with FHE operations, verifying correct shape and functionality with different parameters (e.g., stride, padding).

- **test_conv2d_orion.py**: Tests the optimized Orion-style convolutional layer implementation that uses several advanced optimization techniques:
  - Toeplitz-based encoding (converting convolution to matrix-vector product)
  - Single-shot multiplexing for handling stride > 1
  - BSGS (Baby-Step Giant-Step) algorithm to reduce rotations
  - Double-hoisting to reuse expensive parts of key-switching

### Client-Server Architecture Tests

- **test_client_server.py**: Tests the distributed computation capabilities where:
  - A server process runs the FHE operations
  - A client encrypts data, sends it to the server, and receives the encrypted results
  - Includes tests for both a simple regression model and a small CNN

## How to Run the Tests

SuperSayan uses Python's pytest. You can run the tests using the following commands:

### Run All Tests

```bash
# Using pytest (recommended for better output)
pytest tests/
```

### Run a Specific Test File

```bash
# Using pytest
pytest tests/test_verify_correctness.py -v
```

### Run a Specific Test Function

```bash
# Using unittest
python -m unittest tests.test_verify_correctness.test_verify_linear

# Using pytest
pytest tests/test_verify_correctness.py::test_verify_linear
```

## What Each Test Verifies

### test_verify_correctness.py

- **test_verify_linear**: Verifies that the FHE Linear layer produces results close to PyTorch's nn.Linear.
- **test_verify_conv2d**: Verifies that the FHE Conv2d layer produces results close to PyTorch's nn.Conv2d.
- **test_verify_conv2d_orion**: Verifies that the optimized FHE Conv2dOrion layer produces results close to PyTorch's nn.Conv2d.
- **test_verify_conv2d_with_stride**: Verifies that the FHE Conv2d layer with stride > 1 works correctly.
- **test_verify_conv2d_orion_with_stride**: Verifies that the optimized FHE Conv2dOrion layer with stride > 1 works correctly.

### test_conv2d.py

- **test_conv2d_layer**: Tests the basic functionality of the Conv2d layer with encrypted data.
- **test_conv2d_with_stride**: Tests the Conv2d layer with stride=2 to ensure correct downsampling behavior.

### test_conv2d_orion.py

- **test_conv2d_layer**: Tests the basic functionality of the optimized Conv2dOrion layer with encrypted data.
- **test_conv2d_with_stride**: Tests the Conv2dOrion layer with stride=2 to ensure correct downsampling behavior with optimizations.

### test_client_server.py

- **test_hybrid_house_price_regression**: Tests a hybrid model where only Linear layers run in FHE on the server, and other operations run locally.
- **test_hybrid_small_cnn**: Tests a hybrid CNN model where only Conv2d layers run in FHE on the server, and other operations run locally.

## Testing Framework

SuperSayan uses a combination of:

1. **unittest**: Python's built-in testing framework, providing the basic structure for the tests.
2. **pytest**: A more powerful testing framework that offers rich features and better output formatting.
3. **logging**: Extensive logging is used throughout the tests to provide visibility into the test process.
4. **Assertions and Tolerances**: Since FHE computations involve approximations, the tests use tolerance thresholds to determine if results are "close enough" to the non-encrypted versions.

### Key Testing Features

- **Secret Key Generation**: Each test generates a new secret key for encryption/decryption.
- **Comparison Metrics**: Tests calculate both max difference and mean difference between FHE and standard computations.
- **Shape Verification**: Tests verify that output shapes match expectations.
- **Client-Server Testing**: Uses temporary servers, free port discovery, and proper cleanup processes.

## Adding New Tests

When adding new tests, please follow these guidelines:

1. Group related tests in a single file
2. Use descriptive function names with the `test_` prefix
3. Include meaningful docstrings explaining what the test verifies
4. Set up appropriate logging
5. Structure tests with a clear setup, execution, and verification phases
6. Clean up any resources created during testing

## Common Issues and Debugging

- If a test is failing with numerical differences, check the tolerance thresholds in the assertions
- Client-server tests may fail due to networking issues; check the server logs for details
- For Julia backend tests, ensure the Julia environment is properly set up