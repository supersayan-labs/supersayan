#!/usr/bin/env python3
"""
Script to run a Supersayan server.

This script starts a Socket.IO server that provides endpoints for uploading models,
retrieving model structure, and performing inference with FHE layers.
"""

import os
import sys
import logging
import argparse
import socketio
import uvicorn
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

# Create Socket.IO server with longer timeout
sio = socketio.AsyncServer(
    async_mode='asgi',
    ping_timeout=120,  # Increased from 60 to 120 seconds
    ping_interval=25,
    cors_allowed_origins='*',
    max_http_buffer_size=10_000_000_000 # 5GB buffer for large data
)
app = socketio.ASGIApp(sio)

# Initialize server
server = None

# Socket.IO event handlers
@sio.event
async def connect(sid, environ):
    logger.info(f"Client connected: {sid}")

@sio.event
async def disconnect(sid):
    logger.info(f"Client disconnected: {sid}")

@sio.event
async def upload_model(sid, data):
    """Handle model upload request"""
    logger.info(f"Processing model upload from client: {sid}")
    model_data = data.get('model_data', '')
    
    if not model_data:
        logger.error("Empty model data received")
        return {"error": "No model data provided"}, 400
        
    response = server.handle_upload_model(model_data)
    logger.info(f"Model upload processed")
    return response

@sio.event
async def get_model_structure(sid, data):
    """Handle get model structure request"""
    logger.info(f"Getting model structure for client: {sid}")
    model_id = data.get('model_id', '')
    
    if not model_id:
        logger.error("No model ID provided")
        return {"error": "No model ID provided"}, 400
        
    return server.handle_get_model_structure(model_id)

@sio.event
async def inference(sid, data):
    """Handle inference request"""
    model_id = data.get('model_id', '')
    layer_name = data.get('layer_name', '')
    encrypted_input = data.get('encrypted_input', '')
    
    if not model_id or not layer_name or not encrypted_input:
        missing = []
        if not model_id: missing.append("model_id")
        if not layer_name: missing.append("layer_name")
        if not encrypted_input: missing.append("encrypted_input")
        error_msg = f"Missing required parameters: {', '.join(missing)}"
        logger.error(error_msg)
        return {"error": error_msg}, 400
    
    logger.info(f"Processing inference for client {sid}, model {model_id}, layer {layer_name}")
    logger.debug(f"Encrypted input size: {len(encrypted_input)} bytes")
    
    response = server.handle_inference(model_id, layer_name, encrypted_input)
    logger.info(f"Inference processed for layer {layer_name}")
    return response

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
    
    logger.info(f"Starting Socket.IO server on {args.host}:{args.port}")
    logger.info(f"Using models directory: {args.models_dir}")
    
    # Start the server with proper uvicorn config for handling large requests
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port,
        timeout_keep_alive=120,  # Increase keep-alive timeout
        limit_concurrency=10,    # Limit concurrent connections for stability with large data
        limit_max_requests=100,    # No limit on max requests
        ws_max_size=10_000_000_000  # allow up to ~10 GB per message
    )

if __name__ == "__main__":
    main()