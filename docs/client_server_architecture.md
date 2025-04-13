# SuperSayan Client-Server Architecture

This document explains how to use the client-server architecture in SuperSayan for distributed execution of hybrid models, where some layers run in FHE on the server while others run as native PyTorch layers on the client.

## Overview

The SuperSayan client-server architecture consists of:

1. **Client**: Handles key generation, encryption, decryption, communication with the server, and execution of native PyTorch layers
2. **Server**: Hosts SuperSayan models and executes FHE layers
3. **Deployment**: Manages Docker containers for easy deployment and scaling

## How It Works

### Workflow Example

Consider a network with the following layers:
```
SupersayanLinear → ReLU → Flatten → SupersayanLinear
```

The execution flow works as follows:

1. Client generates a key for encryption
2. Client encrypts the input data
3. Client sends encrypted data to the server
4. Server executes the first SupersayanLinear layer
5. Server sends encrypted output back to the client
6. Client decrypts the output
7. Client executes ReLU and Flatten locally (non-FHE layers)
8. Client generates a new key (or reuses existing key)
9. Client encrypts the processed data
10. Client sends encrypted data to the server
11. Server executes the second SupersayanLinear layer
12. Server sends encrypted output back to the client
13. Client decrypts the final result

This approach allows for secure computation where sensitive data is always encrypted when transmitted and processed by the server, while taking advantage of the client's ability to execute non-FHE operations efficiently.

## Implementation Details

### 1. Model Conversion

Models are converted to SuperSayan models using the `convert_model` function:

```python
from supersayan.nn.convert import convert_model, ModelType

# Convert a PyTorch model to a hybrid SuperSayan model
supersayan_model = convert_model(
    torch_model,
    model_type=ModelType.HYBRID,
    fhe_module_names=["linear1", "linear3"]  # Only these layers run in FHE
)
```

### 2. Server Setup

The server can be deployed as a local process or in a Docker container:

```python
# Local server setup
from supersayan.remote.server import SupersayanServer

# Initialize server
server = SupersayanServer(storage_dir="/path/to/models")

# For Docker deployment
from supersayan.remote.deployment import DockerDeployment

# Create and run a Docker container
deployment = DockerDeployment()
container_id = deployment.run_container()
```

### 3. Client Usage

The client communicates with the server to perform FHE operations:

```python
from supersayan.remote.client import SupersayanClient

# Connect to server
client = SupersayanClient(server_url="http://localhost:8000")

# Upload a model
model_id = client.upload_model("/path/to/model.pt")

# Run inference with automatic distribution of computation
result = client.run_inference(
    model_id=model_id,
    input_data=input_tensor,
    fhe_layers=["linear1", "linear3"]
)

# Close the session when done
client.close()
```

## Testing the Architecture

The test file `/tests/test_client_server_house_price.py` demonstrates how to use the client-server architecture with a house price regression model:

1. Trains a PyTorch model for house price prediction
2. Converts the model to a hybrid SuperSayan model
3. Starts a local server for testing
4. Uploads the model to the server
5. Performs inference with the client-server architecture
6. Compares results with the original PyTorch model

To run the test:

```bash
python -m tests.test_client_server_house_price
```

## Security Considerations

- Each client session uses unique encryption keys
- Data is always encrypted during transmission and server-side processing
- The server never has access to unencrypted data or encryption keys
- Session management ensures proper isolation between different clients
- Automatic cleanup of unused resources to prevent resource exhaustion

## Production Deployment

For production deployment:

1. Build a Docker image: `deployment.build_image()`
2. Deploy containers as needed: `deployment.run_container()`
3. Configure load balancing for horizontal scaling
4. Set up appropriate security measures (TLS, authentication, etc.)
5. Implement proper monitoring and logging

## Limitations and Future Work

- Currently, each FHE operation requires a separate round-trip to the server
- Future versions may support executing multiple FHE layers in sequence on the server
- Support for more layer types is under development
- GPU acceleration for FHE operations is planned for future releases