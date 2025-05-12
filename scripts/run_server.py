#!/usr/bin/env python3
"""Lightweight TCP front‑end for **SupersayanServer**.

Usage
-----
$ python scripts/run_server.py --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import argparse
import hashlib
import io
import logging
import pickle
import socket
import struct
import threading
import multiprocessing as mp
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)


from supersayan.remote.server import SupersayanServer

# -----------------------------------------------------------------------------
# TCP helpers – identical to those used by the client
# -----------------------------------------------------------------------------
_HEADER_FMT = "!Q"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

# Counter to track connections
_connection_counter = 0

def _recv_exact(sock: socket.socket, n: int) -> bytes:
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("client disconnected")
        data.extend(chunk)
    return bytes(data)


def _recv_obj(sock: socket.socket) -> Tuple[Any, int]:
    size_bytes = _recv_exact(sock, _HEADER_SIZE)
    size = struct.unpack(_HEADER_FMT, size_bytes)[0]

    payload = _recv_exact(sock, size)
    actual_length = len(payload)
    if actual_length != size:
        logger.error(f"Length mismatch: Expected {size} bytes but received {actual_length} bytes")
        raise ValueError(f"Length mismatch: Expected {size} bytes but received {actual_length} bytes")

    received_hash = hashlib.md5(payload).hexdigest()
    obj = pickle.loads(payload)

    conn_id = 0
    if isinstance(obj, dict) and "_hash" in obj:
        sent_hash = obj["_hash"]                     # ① keep it
        calc_hash = hashlib.md5(
            pickle.dumps({k: v for k, v in obj.items() if k != "_hash"})
        ).hexdigest()                               # ② hash still includes _conn_id
        conn_id = obj.get("_conn_id", 0)

        logger.info(
            f"[CONN:{conn_id}] Received object: size={actual_length} "
            f"bytes, sent_hash={sent_hash}, calc_hash={calc_hash}"
        )
        if sent_hash != calc_hash:
            logger.warning(f"[CONN:{conn_id}] Hash mismatch!")

        # only now strip the helper keys
        obj.pop("_hash", None)
        obj.pop("_conn_id", None)
    else:
        logger.info(
            f"[CONN:unknown] Received non-dict object: "
            f"size={actual_length} bytes, "
            f"recv_hash={received_hash}"
        )

    return obj, conn_id


def _send_obj(sock: socket.socket, obj: Any, conn_id: int = None) -> None:
    global _connection_counter

    if conn_id is None:
        _connection_counter += 1
        conn_id = _connection_counter

    if isinstance(obj, dict):
        obj["_conn_id"] = conn_id
        
    payload_content = pickle.dumps(obj)
    content_hash = hashlib.md5(payload_content).hexdigest()

    # 2) Inject hash
    if isinstance(obj, dict):
        obj["_hash"] = content_hash

    # 3) Final serialize
    payload = pickle.dumps(obj)
    payload_size = len(payload)

    logger.info(
        f"[CONN:{conn_id}] Sending object: "
        f"size={payload_size} bytes, "
        f"content_hash={content_hash}"
    )

    sock.sendall(struct.pack(_HEADER_FMT, payload_size))
    sock.sendall(payload)

# -----------------------------------------------------------------------------
# Client handler
# -----------------------------------------------------------------------------

def _handle_client(conn: socket.socket, addr: tuple[str, int], server: SupersayanServer) -> None:  # noqa: D401
    logger.info("connection from %s:%s", *addr)
    try:
        while True:
            try:
                request, conn_id = _recv_obj(conn)
            except ConnectionError:
                break  # clean disconnect

            if not isinstance(request, dict) or "command" not in request:
                _send_obj(conn, {"status": False, "error": "invalid request"}, conn_id)
                continue

            cmd: str = request["command"]
            logger.info(f"[CONN:{conn_id}] Processing command: {cmd}")
            
            try:
                if cmd == "upload_model":
                    model_id = server.handle_upload_model_bytes(request["model_data"])
                    _send_obj(conn, {"status": True, "model_id": model_id}, conn_id)
                elif cmd == "get_model_structure":
                    structure = server.handle_get_model_structure_simple(request["model_id"])
                    _send_obj(conn, {"status": True, "structure": structure}, conn_id)
                elif cmd == "inference":
                    output = server.handle_inference_simple(
                        request["model_id"], request["layer_name"], request["encrypted_input"]
                    )
                    logger.info("calling _send_obj")
                    _send_obj(conn, {"status": True, "encrypted_output": output}, conn_id)
                    logger.info("sent encrypted_output")
                else:
                    _send_obj(conn, {"status": False, "error": f"unknown command: {cmd}"}, conn_id)
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"[CONN:{conn_id}] Error handling command {cmd}")
                _send_obj(conn, {"status": False, "error": str(exc)}, conn_id)
    finally:
        conn.close()
        logger.info("closed connection %s:%s", *addr)


# -----------------------------------------------------------------------------
# Helper: accept-loop wrapper
# -----------------------------------------------------------------------------
def _serve_forever(host: str, port: int, models_dir: str) -> None:
    """Listen on *host:port* and serve *one* client at a time, inline.

    Every request is handled on the **main thread**; no threads, no
    processes, therefore no Julia thread-safety problems.
    """
    srv = SupersayanServer(storage_dir=models_dir)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen()
        logger.info("Supersayan server listening on %s:%s", host, port)

        while True:
            conn, addr = s.accept()
            # Handle the whole client session right here, blocking.
            _handle_client(conn, addr, srv)
            # When the client disconnects, _handle_client() returns and we
            # immediately accept the next connection.

# -----------------------------------------------------------------------------
# Main entry-point  (single-threaded)
# -----------------------------------------------------------------------------
def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Run Supersayan TCP server")
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind to")
    parser.add_argument("--port", type=int, default=8000, help="TCP port")
    parser.add_argument(
        "--models-dir",
        default="/tmp/supersayan/models",
        help="directory where uploaded models are stored",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # One-client-at-a-time, on the main thread
    _serve_forever(args.host, args.port, args.models_dir)


if __name__ == "__main__":
    main()
