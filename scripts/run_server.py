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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now import the server module
from supersayan.remote.server import SupersayanServer
from supersayan.remote.chunking import ChunkManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create Socket.IO server with optimized settings for large data transfers
sio = socketio.AsyncServer(
    async_mode="asgi",
    ping_timeout=300,  # 5 minute timeout
    ping_interval=60,  # 1 minute ping interval
    cors_allowed_origins="*",
    max_http_buffer_size=1024 * 1024 * 1024,  # 1GB buffer
)
app = socketio.ASGIApp(sio)

# Initialize server
server = None

# Store chunk managers for each client
chunk_managers = {}


# Socket.IO event handlers
@sio.event
async def connect(sid, environ):
    """Handle client connection."""
    logger.info(f"Client connected: {sid}")
    chunk_managers[sid] = ChunkManager()


@sio.event
async def disconnect(sid):
    """Handle client disconnection."""
    logger.info(f"Client disconnected: {sid}")
    if sid in chunk_managers:
        del chunk_managers[sid]


@sio.event
async def upload_model(sid, data):
    """Handle model upload request."""
    logger.info(f"Processing model upload from client: {sid}")
    model_data = data.get("model_data", "")

    if not model_data:
        logger.error("Empty model data received")
        return {"error": "No model data provided"}, 400

    response = server.handle_upload_model(model_data)
    logger.info(f"Model upload processed")
    return response


@sio.event
async def get_model_structure(sid, data):
    """Handle get model structure request."""
    logger.info(f"Getting model structure for client: {sid}")
    model_id = data.get("model_id", "")

    if not model_id:
        logger.error("No model ID provided")
        return {"error": "No model ID provided"}, 400

    return server.handle_get_model_structure(model_id)


@sio.event
async def inference(sid, data):
    """Handle inference request."""
    try:
        model_id = data.get("model_id", "")
        layer_name = data.get("layer_name", "")
        encrypted_input = data.get("encrypted_input", "")

        if not model_id or not layer_name or not encrypted_input:
            missing = []
            if not model_id:
                missing.append("model_id")
            if not layer_name:
                missing.append("layer_name")
            if not encrypted_input:
                missing.append("encrypted_input")
            error_msg = f"Missing required parameters: {', '.join(missing)}"
            logger.error(error_msg)
            return {"error": error_msg}, 400

        logger.info(
            f"Processing inference for client {sid}, model {model_id}, layer {layer_name}"
        )

        response = server.handle_inference(model_id, layer_name, encrypted_input)
        logger.info(f"Inference processed for layer {layer_name}")

        response_data, status_code = response

        if status_code == 200 and "encrypted_output" in response_data:
            output_data = response_data["encrypted_output"]

            if sid not in chunk_managers:
                chunk_managers[sid] = ChunkManager()

            chunk_manager = chunk_managers[sid]
            if chunk_manager.needs_chunking(output_data):
                logger.info(f"Output is large, sending as chunked response")

                transfer_id, chunks = chunk_manager.create_transfer(output_data)

                # Store the chunks in the chunk manager for retrieval
                chunk_manager.register_transfer(transfer_id, len(chunks))

                # Add each chunk to the transfer
                for chunk in chunks:
                    chunk_manager.add_chunk(chunk)

                return {
                    "chunked": True,
                    "transfer_id": transfer_id,
                    "total_chunks": len(chunks),
                }, 200

        return response_data, status_code
    except Exception as e:
        logger.exception(f"Error processing inference: {e}")
        return {"error": f"Failed to process inference: {str(e)}"}, 500


# Chunked data handlers
@sio.event
async def inference_start(sid, data):
    """Handle start of a chunked inference request."""
    try:
        model_id = data.get("model_id", "")
        layer_name = data.get("layer_name", "")
        transfer_id = data.get("transfer_id", "")
        total_chunks = data.get("total_chunks", 0)

        if not model_id or not layer_name or not transfer_id or not total_chunks:
            missing = []
            if not model_id:
                missing.append("model_id")
            if not layer_name:
                missing.append("layer_name")
            if not transfer_id:
                missing.append("transfer_id")
            if not total_chunks:
                missing.append("total_chunks")
            error_msg = f"Missing required parameters: {', '.join(missing)}"
            logger.error(error_msg)
            return {"error": error_msg}, 400

        logger.info(
            f"Beginning chunked inference, model {model_id}, layer {layer_name}"
        )

        if sid not in chunk_managers:
            chunk_managers[sid] = ChunkManager()

        chunk_managers[sid].register_transfer(
            transfer_id, total_chunks, {"model_id": model_id, "layer_name": layer_name}
        )

        return {"status": "ready", "transfer_id": transfer_id}, 200
    except Exception as e:
        logger.exception(f"Error starting chunked inference: {e}")
        return {"error": f"Failed to start chunked inference: {str(e)}"}, 500


@sio.event
async def chunk(sid, data):
    """Handle a chunk of data for an ongoing transfer."""
    try:
        transfer_id = data.get("transfer_id", "")
        chunk_index = data.get("chunk_index", -1)
        total_chunks = data.get("total_chunks", 0)
        chunk_data = data.get("data", "")

        if not transfer_id or chunk_index < 0 or not total_chunks or not chunk_data:
            missing = []
            if not transfer_id:
                missing.append("transfer_id")
            if chunk_index < 0:
                missing.append("chunk_index")
            if not total_chunks:
                missing.append("total_chunks")
            if not chunk_data:
                missing.append("data")
            error_msg = f"Missing required parameters: {', '.join(missing)}"
            logger.error(error_msg)
            return {"error": error_msg}, 400

        if sid not in chunk_managers:
            error_msg = f"No chunk manager for client {sid}"
            logger.error(error_msg)
            return {"error": error_msg}, 400

        chunk_manager = chunk_managers[sid]
        all_received = chunk_manager.add_chunk(data)

        return {
            "status": "received",
            "transfer_id": transfer_id,
            "chunk_index": chunk_index,
            "complete": all_received,
        }, 200
    except Exception as e:
        logger.exception(f"Error processing chunk: {e}")
        return {"error": f"Failed to process chunk: {str(e)}"}, 500


@sio.event
async def inference_complete(sid, data):
    """Handle completion of a chunked inference request"""
    try:
        transfer_id = data.get("transfer_id", "")

        if not transfer_id:
            error_msg = "Missing transfer_id"
            logger.error(error_msg)
            return {"error": error_msg}, 400

        logger.info(f"Completing chunked inference for transfer {transfer_id[:10]}...")

        if sid not in chunk_managers:
            error_msg = f"No chunk manager for client {sid}"
            logger.error(error_msg)
            return {"error": error_msg}, 400

        chunk_manager = chunk_managers[sid]

        assembled_data = chunk_manager.get_assembled_data(transfer_id)
        if not assembled_data:
            error_msg = f"No assembled data for transfer {transfer_id}"
            logger.error(error_msg)
            return {"error": error_msg}, 400

        metadata = chunk_manager.get_metadata(transfer_id)
        model_id = metadata.get("model_id", "")
        layer_name = metadata.get("layer_name", "")

        if not model_id or not layer_name:
            error_msg = f"Missing metadata for transfer {transfer_id}"
            logger.error(error_msg)
            return {"error": error_msg}, 400

        logger.info(f"Processing inference with assembled data")
        response = server.handle_inference(model_id, layer_name, assembled_data)

        chunk_manager.cleanup_transfer(transfer_id)
        logger.info(f"Chunked inference completed")

        return response
    except Exception as e:
        logger.exception(f"Error completing chunked inference: {e}")
        return {"error": f"Failed to complete chunked inference: {str(e)}"}, 500


@sio.event
async def get_response_chunk(sid, data):
    """Handle request for a chunk of a chunked response"""
    try:
        transfer_id = data.get("transfer_id", "")
        chunk_index = data.get("chunk_index", -1)

        if not transfer_id or chunk_index < 0:
            missing = []
            if not transfer_id:
                missing.append("transfer_id")
            if chunk_index < 0:
                missing.append("chunk_index")
            error_msg = f"Missing required parameters: {', '.join(missing)}"
            logger.error(error_msg)
            return {"error": error_msg}, 400

        if sid not in chunk_managers:
            error_msg = f"No chunk manager for client {sid}"
            logger.error(error_msg)
            return {"error": error_msg}, 404

        chunk_manager = chunk_managers[sid]

        if transfer_id not in chunk_manager.transfers:
            error_msg = f"No transfer found for {transfer_id}"
            logger.error(error_msg)
            return {"error": error_msg}, 404

        transfer = chunk_manager.transfers[transfer_id]
        chunks = transfer.get("chunks", {})
        total_chunks = transfer.get("total_chunks", 0)

        if chunk_index >= total_chunks or chunk_index not in chunks:
            error_msg = f"Chunk {chunk_index} not available"
            logger.error(error_msg)
            return {"error": error_msg}, 400

        return {
            "transfer_id": transfer_id,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "data": chunks[chunk_index],
        }, 200
    except Exception as e:
        logger.exception(f"Error retrieving response chunk: {e}")
        return {"error": f"Failed to retrieve response chunk: {str(e)}"}, 500


@sio.event
async def cleanup_response_chunks(sid, data):
    """Cleanup chunks for a completed response transfer"""
    try:
        transfer_id = data.get("transfer_id", "")

        if not transfer_id:
            error_msg = "Missing transfer_id"
            logger.error(error_msg)
            return {"error": error_msg}, 400

        if sid not in chunk_managers:
            return {"status": "not_found"}, 404

        chunk_manager = chunk_managers[sid]
        if chunk_manager.cleanup_transfer(transfer_id):
            logger.info(f"Cleaned up transfer {transfer_id[:10]}")
            return {"status": "cleaned"}, 200

        return {"status": "not_found"}, 404
    except Exception as e:
        logger.exception(f"Error cleaning up chunks: {e}")
        return {"error": f"Failed to clean up chunks: {str(e)}"}, 500


def main():
    parser = argparse.ArgumentParser(description="Run a Supersayan server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="/tmp/supersayan/models",
        help="Directory for storing models",
    )
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)

    global server
    server = SupersayanServer(storage_dir=args.models_dir)

    logger.info(f"Starting Socket.IO server on {args.host}:{args.port}")
    logger.info(f"Using models directory: {args.models_dir}")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        timeout_keep_alive=300,  # 5 minute keep-alive timeout (increased)
        limit_concurrency=5,  # Limit concurrent connections to avoid memory issues
        limit_max_requests=None,  # No limit on max requests
        ws_max_size=1024 * 1024 * 1024,  # 1GB max websocket message size
        ws_ping_interval=60,  # 1 minute ping interval
        ws_ping_timeout=300,  # 5 minute ping timeout
    )


if __name__ == "__main__":
    main()
