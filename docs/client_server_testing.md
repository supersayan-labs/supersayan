# Testing the SuperSayan Client-Server Architecture

This document explains how to test the client-server architecture using the provided scripts.

## Prerequisites

Make sure you have the following dependencies installed:

```bash
pip install fastapi uvicorn
```

## Option 1: Running Server Locally

### Step 1: Start the Server

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

## Option 2: Running Server in Docker

### Step 1: Build the Docker Image

```bash
# Navigate to the project root
cd /path/to/supersayan

# Build the Docker image
docker build -t supersayan-server .
```

### Step 2: Run the Docker Container

```bash
# Run the container
docker run -p 8000:8000 --name supersayan-server-instance -d supersayan-server
```

This will start the server and expose it on port 8000.

## Step 3: Run the Client Test

Once the server is running (either locally or in Docker), open a new terminal and run the client test:

```bash
# Navigate to the project root
cd /path/to/supersayan

# Run the client test
python scripts/test_client.py --server-url http://127.0.0.1:8000
```

The client test will:

1. Train a simple neural network for house price prediction
2. Convert the 
model to a SuperSayan pure model
3. Upload the model to the server
4. Run inference layer by layer:
   - Client encrypts input data
   - Client sends encrypted data to server
   - Server performs FHE operations
   - Server returns encrypted results
   - Client decrypts results
5. Compare the results with the original PyTorch model

## How It Works

### Server

The server script (`run_server.py`) does the following:

1. Initializes the Julia backend for FHE operations

2. Creates a FastAPI web server that exposes several endpoints:
   - `/models/upload`: Uploads a serialized model to the server
   - `/models/{model_id}/structure`: Gets the structure of a model
   - `/inference/{model_id}/{layer_name}`: Executes a specific layer on encrypted data
   - `/sessions/{session_id}/close`: Closes a client session
   - `/health`: Health check endpoint

3. Initializes the SuperSayan server with a directory for model storage

4. Handles requests from clients, executing FHE operations on encrypted data

### Client

The client test script (`test_client.py`) does the following:

1. Creates and trains a simple neural network for house price prediction

2. Converts the model to a SuperSayan pure model

3. Connects to the server and uploads the model

4. Runs inference with the client-server architecture:
   - For each layer in the model:
     - Client generates a key
     - Client encrypts the input data
     - Client sends encrypted data to the server
     - Server executes the FHE layer
     - Server returns encrypted results to the client
     - Client decrypts the results
     - Client passes the results to the next layer

5. Compares the results with the original PyTorch model

## Important Notes

- **All FHE operations occur on the server side**: The server processes encrypted data without ever having access to the encryption keys or unencrypted data.

- **Keys remain on the client**: The client generates encryption keys and handles encryption/decryption locally.

- **Data is always encrypted during transmission**: The server never sees or processes unencrypted data.

## Troubleshooting

- If the client can't connect to the server, check that the server is running and the URL is correct
- If the model upload fails, check the model file path and the server's model directory
- If inference fails, check that the model ID and layer names are correct
- For Docker issues, check the Docker logs: `docker logs supersayan-server-instance`