# 🚀 SuperSayan

SuperSayan is a Python library for privacy-preserving neural networks using Fully Homomorphic Encryption (FHE). It integrates a PyTorch-style API with an optimized Julia backend ([Julia SupersayanTFHE](https://github.com/bonsainoodle/supersayan)) to enable secure machine learning on encrypted data.

## 📋 Table of Contents

- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Architecture](#-architecture)
- [Benchmarks](#-benchmarks)
- [Testing](#-testing)
- [Development Roadmap](#-development-roadmap)

## ✨ Features

- **PyTorch-Style API**: Familiar interface for building neural networks
- **Fully Homomorphic Encryption**: Perform computations directly on encrypted data
- **Optimized FHE Operations**: Implementation of state-of-the-art FHE techniques including BSGS
- **Client-Server Architecture**: Distributed computation with end-to-end encryption
- **Hybrid Models**: Mix encrypted and unencrypted operations for optimal performance
- **Orion Implementation**: Optimized convolutions based on the [Orion FHE paper](https://eprint.iacr.org/2023/1314)

## 🧱 Prerequisites

To use SuperSayan, you'll need:

- Python **3.9+**
- Julia **1.9+** (latest stable version recommended)

### 📦 Install Julia

- **macOS** (Homebrew):
  ```bash
  brew install julia
  ```

- **Linux** (APT):
  ```bash
  sudo apt install julia
  ```

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/supersayan.git
cd supersayan
```

### 2. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

## 🚀 Usage

### Basic Example

```python
import torch
import torch.nn as nn
from supersayan.core.keygen import generate_secret_key
from supersayan.core.encryption import encrypt, decrypt
from supersayan.nn.layers.linear import Linear as FHELinear

# Create a secret key for encryption
key = generate_secret_key()

# Prepare input data
input_data = torch.randn(1, 10)  # Batch size 1, 10 features

# Encrypt the input
encrypted_input = encrypt(input_data, key)

# Create a FHE-compatible linear layer
fhe_linear = FHELinear(in_features=10, out_features=5)

# Perform computation on encrypted data
encrypted_output = fhe_linear(encrypted_input)

# Decrypt the result
output = decrypt(encrypted_output, key)
print(output)
```

### Client-Server Example

```python
import torch
import torch.nn as nn
from supersayan.remote.client import SupersayanClient
from supersayan.nn.convert import ModelType

# Define a PyTorch model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(5, 1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# Create a hybrid client where only Linear layers run in FHE
client = SupersayanClient(
    server_url="http://localhost:8000",
    torch_model=SimpleModel(),
    model_type=ModelType.HYBRID,
    fhe_modules=[nn.Linear]  # Only Linear layers use FHE
)

# Run inference - Linear layers run on server with encryption
# ReLU runs locally on the client
output = client(torch.randn(1, 10))
print(output)

# Clean up
client.close()
```

## 🏗️ Architecture

SuperSayan consists of several key components:

1. **Python Frontend**: Provides a PyTorch-compatible API
2. **Julia Backend**: Implements optimized FHE operations
3. **Client-Server System**: Enables distributed computation with encryption

For more details on the client-server architecture, see [Client-Server Architecture](docs/client_server_architecture.md).

## 🚀 Benchmarks

The project includes benchmarks to measure performance of different layer implementations. For details, see the [Benchmarks README](benchmarks/README.md).

To run benchmarks:

```bash
pytest benchmarks/ -v
```

Compare benchmark results using:

```bash
pytest-benchmark compare last mysession
```

## 🧪 Testing

SuperSayan includes comprehensive tests for verifying correctness and functionality. For details, see the [Tests README](tests/README.md).

Run tests using:

```bash
# Run all tests
python -m unittest discover tests

# Using pytest with verbose output
pytest tests/ -v
```

## 📅 Development Roadmap

### 🔜 Next Steps

- [ ] Build **hybrid server-side architecture**
- [ ] Run **initial performance benchmarks**
- [ ] Implement **2D convolution** following concepts from the [Orion FHE paper](https://eprint.iacr.org/2023/1314)
- [ ] Implement double hoisting for Conv2D as described in the Orion paper

### 📌 Future Features

- [ ] Add **sparsity support**
- [ ] Apply **compression techniques** (e.g., pruning)
- [ ] Add **GPU acceleration** support
