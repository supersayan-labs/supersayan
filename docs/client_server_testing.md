# Testing the SuperSayan Client-Server Architecture

This document explains how to test the client-server architecture using the provided scripts.

## Prerequisites

Make sure you have the following dependencies installed:

```bash
pip install fastapi uvicorn
```

## Step 1: Start the Server

First, start the SuperSayan server in a separate terminal:

```bash
# Navigate to the project root
cd /path/to/supersayan

# Create a directory for model storage
mkdir -p /tmp/supersayan/models

# Run the server
python scripts/run_server.py --host 127.0.0.1 --port 8000
```

You should see output similar to:

```
INFO - Starting server on 127.0.0.1:8000
INFO - Using models directory: /tmp/supersayan/models
INFO - Uvicorn running on http://127.0.0.1:8000
```

The server will now be listening for client connections.

## Step 2: Run the Client Test

Once the server is running, open a new terminal and run the client test:

```bash
# Navigate to the project root
cd /path/to/supersayan

# Run the client test
python scripts/test_client.py --server-url http://127.0.0.1:8000
```

The client test will:

1. Train a simple neural network for house price prediction
2. Convert the model to a SuperSayan hybrid model
3. Upload the model to the server
4. Run inference using the client-server architecture
5. Compare the results with the original PyTorch model

## How It Works

### Server

The server script (`run_server.py`) does the following:

1. Creates a FastAPI web server that exposes several endpoints:
   - `/models/upload`: Uploads a serialized model to the server
   - `/models/{model_id}/structure`: Gets the structure of a model
   - `/inference/{model_id}/{layer_name}`: Executes a specific layer on encrypted data
   - `/sessions/{session_id}/close`: Closes a client session
   - `/health`: Health check endpoint

2. Initializes the SuperSayan server with a directory for model storage

3. Handles requests from clients, executing FHE operations on encrypted data

### Client

The client test script (`test_client.py`) does the following:

1. Creates and trains a simple neural network for house price prediction

2. Converts the model to a SuperSayan hybrid model, specifying which layers should be executed in FHE

3. Connects to the server and uploads the model

4. Runs inference with the client-server architecture:
   - Client performs key generation
   - Client encrypts the input data
   - Client sends encrypted data to the server
   - Server executes FHE layers
   - Server returns encrypted results to the client
   - Client decrypts the results
   - Client performs non-FHE operations locally
   - Process repeats for subsequent FHE layers

5. Compares the results with the original PyTorch model

## Next Steps

- For a more realistic deployment, you can use the Docker deployment functionality
- Implement additional layer types (currently only Linear is supported)
- Add support for executing multiple FHE layers sequentially on the server
- Add authentication and TLS encryption for secure communication

## Troubleshooting

- If the client can't connect to the server, check that the server is running and the URL is correct
- If the model upload fails, check the model file path and the server's model directory
- If inference fails, check that the model ID and layer names are correct