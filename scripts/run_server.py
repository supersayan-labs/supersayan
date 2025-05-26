#!/usr/bin/env python3
# run_server.py
"""Lightweight TCP front-end for **SupersayanServer**.

Usage
-----
$ python scripts/run_server.py --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import pickle
import socket
import struct
from typing import Any, Tuple

logger = logging.getLogger(__name__)

from supersayan.remote.server import SupersayanServer

# Constants for chunking
_CHUNK_SIZE = 500 * 1024 * 1024  # 50 MiB
_HEADER_FMT = "!Q"  # unsigned long long – 8-byte integer
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_COUNT_FMT = "!I"  # unsigned int – 4-byte integer for packet count
_COUNT_SIZE = struct.calcsize(_COUNT_FMT)

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
    # Read total payload size
    total_size_bytes = _recv_exact(sock, _HEADER_SIZE)
    total_size = struct.unpack(_HEADER_FMT, total_size_bytes)[0]
    # Read packet count
    count_bytes = _recv_exact(sock, _COUNT_SIZE)
    packet_count = struct.unpack(_COUNT_FMT, count_bytes)[0]
    # Read protocol-level conn_id
    conn_id_bytes = _recv_exact(sock, _HEADER_SIZE)
    conn_id = struct.unpack(_HEADER_FMT, conn_id_bytes)[0]

    logger.info(
        f"[CONN:{conn_id}] Receiving object: {total_size} bytes in {packet_count} packets"
    )

    # Collect chunks
    payload = bytearray()
    for idx in range(1, packet_count + 1):
        size_bytes = _recv_exact(sock, _HEADER_SIZE)
        chunk_size = struct.unpack(_HEADER_FMT, size_bytes)[0]
        logger.info(
            f"[CONN:{conn_id}] Receiving packet {idx}/{packet_count}: {chunk_size} bytes"
        )
        payload.extend(_recv_exact(sock, chunk_size))

    # Verify length
    if len(payload) != total_size:
        raise ValueError(
            f"[CONN:{conn_id}] Length mismatch: expected {total_size} bytes, got {len(payload)}"
        )

    # Read 32-byte ASCII MD5 trailer
    expected_hash = _recv_exact(sock, 32).decode("ascii")
    calc_hash = hashlib.md5(payload).hexdigest()
    logger.info(f"[CONN:{conn_id}] Trailer hash: sent={expected_hash} calc={calc_hash}")
    if expected_hash != calc_hash:
        raise ValueError(
            f"[CONN:{conn_id}] Hash mismatch: sent={expected_hash} vs calc={calc_hash}"
        )

    obj = pickle.loads(payload)
    return obj, conn_id


def _send_obj(sock: socket.socket, obj: Any, conn_id: int = None) -> None:
    global _connection_counter

    if conn_id is None:
        _connection_counter += 1
        conn_id = _connection_counter

    # Attach metadata only for logging, not included in the hash
    if isinstance(obj, dict):
        obj["_conn_id"] = conn_id

    # Final payload
    payload = pickle.dumps(obj)
    total_size = len(payload)
    chunks = [payload[i : i + _CHUNK_SIZE] for i in range(0, total_size, _CHUNK_SIZE)]
    total_packets = len(chunks)

    # Compute MD5 over the raw payload
    content_hash = hashlib.md5(payload).hexdigest()
    logger.info(
        f"[CONN:{conn_id}] Sending {total_size} bytes in {total_packets} packets, MD5={content_hash}"
    )

    # Send protocol headers
    sock.sendall(struct.pack(_HEADER_FMT, total_size))
    sock.sendall(struct.pack(_COUNT_FMT, total_packets))
    sock.sendall(struct.pack(_HEADER_FMT, conn_id))

    # Send each chunk
    for idx, chunk in enumerate(chunks, start=1):
        chunk_size = len(chunk)
        logger.info(
            f"[CONN:{conn_id}] Sending packet {idx}/{total_packets}: {chunk_size} bytes"
        )
        sock.sendall(struct.pack(_HEADER_FMT, chunk_size))
        sock.sendall(chunk)

    # Send 32-byte ASCII hex MD5 trailer
    sock.sendall(content_hash.encode("ascii"))


def _handle_client(
    conn: socket.socket, addr: tuple[str, int], server: SupersayanServer
) -> None:
    logger.info("connection from %s:%s", *addr)
    try:
        while True:
            try:
                request, conn_id = _recv_obj(conn)
            except ConnectionError:
                break

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
                    structure = server.handle_get_model_structure_simple(
                        request["model_id"]
                    )
                    _send_obj(conn, {"status": True, "structure": structure}, conn_id)
                elif cmd == "inference":
                    output = server.handle_inference_simple(
                        request["model_id"],
                        request["layer_name"],
                        request["encrypted_input"],
                    )
                    _send_obj(
                        conn, {"status": True, "encrypted_output": output}, conn_id
                    )
                else:
                    _send_obj(
                        conn,
                        {"status": False, "error": f"unknown command: {cmd}"},
                        conn_id,
                    )
            except Exception as exc:
                logger.exception(f"[CONN:{conn_id}] Error handling command {cmd}")
                _send_obj(conn, {"status": False, "error": str(exc)}, conn_id)
    finally:
        conn.close()
        logger.info("closed connection %s:%s", *addr)


def _serve_forever(host: str, port: int, models_dir: str) -> None:
    srv = SupersayanServer(storage_dir=models_dir)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen()
        logger.info("Supersayan server listening on %s:%s", host, port)
        while True:
            conn, addr = s.accept()
            _handle_client(conn, addr, srv)


def main() -> None:
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
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    _serve_forever(args.host, args.port, args.models_dir)


if __name__ == "__main__":
    main()
