#!/usr/bin/env python3
"""
Script to run a Supersayan server.

This script starts a FastAPI server that provides endpoints for uploading models,
retrieving model structure, and performing inference with FHE layers.
"""

import os
import sys
import logging
import argparse
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import the server module
from supersayan.remote.server import SupersayanServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Supersayan FHE Server")

# Initialize server
server = None

# Request models
class UploadModelRequest(BaseModel):
    model_data: str

class InferenceRequest(BaseModel):
    encrypted_input: str

# API routes
@app.post("/models/upload")
async def upload_model(request: UploadModelRequest):
    response = server.handle_upload_model(
        request.model_data
    )
    
    if isinstance(response, tuple):
        response_data, status_code = response
        if status_code != 200:
            raise HTTPException(status_code=status_code, detail=response_data["error"])
        return response_data
    return response

@app.get("/models/{model_id}/structure")
async def get_model_structure(model_id: str):
    response = server.handle_get_model_structure(model_id)
    
    if isinstance(response, tuple):
        response_data, status_code = response
        if status_code != 200:
            raise HTTPException(status_code=status_code, detail=response_data["error"])
        return response_data
    return response

@app.post("/inference/{model_id}/{layer_name}")
async def inference(model_id: str, layer_name: str, request: InferenceRequest):
    # Debug print for encrypted input
    print("\nEncrypted Server Input Debug Info:")
    print(f"Type: {type(request.encrypted_input)}")
    print(f"Length: {len(request.encrypted_input)}")
    print(f"Sample (first 100 chars): {request.encrypted_input[:100]}...")
    print(f"Is Base64: {all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in request.encrypted_input)}")
    
    response = server.handle_inference(
        model_id,
        layer_name,
        request.encrypted_input
    )
    
    if isinstance(response, tuple):
        response_data, status_code = response
        if status_code != 200:
            raise HTTPException(status_code=status_code, detail=response_data["error"])
        return response_data
    return response

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

def main():
    parser = argparse.ArgumentParser(description="Run a Supersayan server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--models-dir", type=str, default="/tmp/supersayan/models", help="Directory for storing models")
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs(args.models_dir, exist_ok=True)
    
    # Initialize the server
    global server
    server = SupersayanServer(storage_dir=args.models_dir)
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"Using models directory: {args.models_dir}")
    
    # Start the server
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()