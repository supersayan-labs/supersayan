# SuperSayan Client-Server Architecture

This document explains how to use the client-server architecture in SuperSayan for distributed execution of models with fully homomorphic encryption (FHE).

## Overview

The SuperSayan client-server architecture consists of:

1. **Client**: Extends SupersayanModel to transparently handle remote execution of FHE layers
2. **Server**: Hosts SuperSayan FHE layers and executes them on encrypted data

## How It Works

### Key Features

- **SupersayanClient inherits from SupersayanModel**: The client is a true extension of SupersayanModel, maintaining all its behavior while adding remote execution capabilities
- **Transparent Remote Execution**: The client looks and behaves like a regular SupersayanModel
- **Two Execution Modes**: Pure mode (all layers in FHE) and hybrid mode (selective FHE)
- **Data Security**: Data is always encrypted during transmission and server-side processing

### Workflow Example

Consider a network with the following layers:
```
Linear → ReLU → Linear
```

The execution flow works as follows:

1. In PURE mode:
   - All layers are converted to FHE and run remotely on the server
   - Client generates a single encryption key
   - Client encrypts input data once
   - All layers are processed sequentially on the server with encrypted data
   - Server returns the final encrypted result
   - Client decrypts the result with the same key

2. In HYBRID mode:
   - Only specified layers are converted to FHE and run on the server
   - For FHE layers (e.g., Linear layers): 
     - Client generates a key for each FHE layer
     - Client encrypts input data
     - Client sends encrypted data to server
     - Server executes the FHE layer
     - Server returns encrypted result
     - Client decrypts the result
   - For non-FHE layers (e.g., ReLU):
     - Client executes the layer locally with unencrypted data

## Implementation Details

### 1. Creating a SupersayanClient

The client is created just like a SupersayanModel, but with a server URL:

```python
from supersayan.remote.client import SupersayanClient
from supersayan.nn.convert import ModelType

# Create a client for pure model (all layers run in FHE remotely)
client_pure = SupersayanClient(
    server_url="http://localhost:8000",
    torch_model=torch_model,
    model_type=ModelType.PURE
)

# Or create a client for hybrid model (specified layers run in FHE on server)
client_hybrid = SupersayanClient(
    server_url="http://localhost:8000",
    torch_model=torch_model,
    model_type=ModelType.HYBRID,
    fhe_module_names=["linear1", "linear2"]  # These layers run in FHE on the server
)
```

### 2. Using the Client

The client is used just like any PyTorch module:

```python
# For pure models: runs all layers in FHE remotely
# Automatically handles:
# 1. Single encryption key generation
# 2. One-time input encryption
# 3. Complete remote execution in FHE
# 4. Final result decryption
output_pure = client_pure(input_tensor)

# For hybrid models: FHE layers run remotely, others run locally
# Automatically handles:
# 1. FHE model upload (if needed)
# 2. Per-layer key generation and encryption (for FHE layers)
# 3. Remote execution (for FHE layers)
# 4. Local execution (for non-FHE layers)
# 5. Per-layer result decryption
output_hybrid = client_hybrid(input_tensor)

# Clean up when done
client_pure.close()
client_hybrid.close()
```

### 3. Server Setup

The server can be deployed as a local process or in a Docker container:

```bash
# Using Docker
docker build -t supersayan-server .
docker run -p 8000:8000 supersayan-server

# Or running locally
python scripts/run_server.py
```

## Security Considerations

- **Client-side Key Generation**: Encryption keys are generated and kept only on the client
- **Encrypted Data Transfer**: Only encrypted data is transmitted between client and server
- **No Server Access to Plaintext**: The server never sees unencrypted data or encryption keys
- **Session Management**: Each client session is isolated

## Usage Example

The following example demonstrates a complete workflow:

```python
import torch
import torch.nn as nn
from supersayan.remote.client import SupersayanClient
from supersayan.nn.convert import ModelType

# Define a PyTorch model with named modules
class MyModel(nn.Module):
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

# Create model
model = MyModel()

# Create a pure client (all layers in FHE on server)
client_pure = SupersayanClient(
    server_url="http://localhost:8000",
    torch_model=model,
    model_type=ModelType.PURE
)

# Run inference - this happens with end-to-end encryption:
# 1. Single encryption of input
# 2. Sequential processing of all layers on server
# 3. Single decryption of output
output_pure = client_pure(input_data)

# Create a hybrid client (linear layers in FHE on server, relu locally)
client_hybrid = SupersayanClient(
    server_url="http://localhost:8000",
    torch_model=model,
    model_type=ModelType.HYBRID,
    fhe_module_names=["linear1", "linear2"]
)

# Run inference - this happens with distributed execution:
# 1. linear1 runs remotely on encrypted data
# 2. relu runs locally on decrypted data
# 3. linear2 runs remotely on re-encrypted data
output_hybrid = client_hybrid(input_data)

# Clean up
client_pure.close()
client_hybrid.close()
```

## Testing

To test the client-server architecture, see the instructions in `docs/client_server_testing.md`.