from __future__ import annotations

import hashlib
import pickle
import socket
import struct
from typing import Any, Tuple

from supersayan.logging_config import get_logger

logger = get_logger(__name__)

# Global connection counter
_connection_counter = 0

# Constants for chunking
CHUNK_SIZE = 500 * 1024 * 1024  # 500MB per packet
HEADER_FMT = "!Q"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
COUNT_FMT = "!I"
COUNT_SIZE = struct.calcsize(COUNT_FMT)
REQUEST_TIMEOUT = 600  # seconds


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """
    Receive exactly n bytes from the socket.

    Args:
        sock: The socket to receive from
        n: The number of bytes to receive

    Returns:
        bytes: The bytes received
    """
    data = bytearray()

    while len(data) < n:
        chunk = sock.recv(n - len(data))

        if not chunk:
            raise ConnectionError("connection closed while receiving data")
        data.extend(chunk)

    return bytes(data)


def recv_obj(sock: socket.socket) -> Tuple[Any, int]:
    """
    Receive an object from the socket.

    Args:
        sock: The socket to receive from

    Returns:
        Tuple[Any, int]: The object received and the connection ID
    """
    # Read total payload size
    total_size_bytes = recv_exact(sock, HEADER_SIZE)
    total_size = struct.unpack(HEADER_FMT, total_size_bytes)[0]

    # Read packet count
    count_bytes = recv_exact(sock, COUNT_SIZE)
    packet_count = struct.unpack(COUNT_FMT, count_bytes)[0]

    # Read protocol-level conn_id
    conn_id_bytes = recv_exact(sock, HEADER_SIZE)
    conn_id = struct.unpack(HEADER_FMT, conn_id_bytes)[0]

    logger.info(
        f"[CONN:{conn_id}] Receiving object: {total_size} bytes in {packet_count} packets"
    )

    # Collect chunks
    payload = bytearray()
    for idx in range(1, packet_count + 1):
        size_bytes = recv_exact(sock, HEADER_SIZE)
        chunk_size = struct.unpack(HEADER_FMT, size_bytes)[0]
        logger.info(
            f"[CONN:{conn_id}] Receiving packet {idx}/{packet_count}: {chunk_size} bytes"
        )
        payload.extend(recv_exact(sock, chunk_size))

    # Verify length
    if len(payload) != total_size:
        raise ValueError(
            f"[CONN:{conn_id}] Length mismatch: expected {total_size} bytes, got {len(payload)}"
        )

    # Read 32-byte ASCII MD5 trailer
    expected_hash = recv_exact(sock, 32).decode("ascii")
    calc_hash = hashlib.md5(payload).hexdigest()
    logger.info(f"[CONN:{conn_id}] Trailer hash: sent={expected_hash} calc={calc_hash}")
    if expected_hash != calc_hash:
        raise ValueError(
            f"[CONN:{conn_id}] Hash mismatch: sent={expected_hash} vs calc={calc_hash}"
        )

    obj = pickle.loads(payload)

    # Extract conn_id if it exists in the object
    if isinstance(obj, dict) and "_conn_id" in obj:
        conn_id = obj.pop("_conn_id")

    return obj, conn_id


def send_obj(sock: socket.socket, obj: Any, conn_id: int = None) -> int:
    """
    Send an object to the socket.

    Args:
        sock: The socket to send to
        obj: The object to send
        conn_id: Optional connection ID to use (if None, generates a new one)

    Returns:
        int: The connection ID
    """
    global _connection_counter

    if conn_id is None:
        _connection_counter += 1
        conn_id = _connection_counter

    # Make a copy if obj is a dict to avoid modifying the original
    if isinstance(obj, dict):
        obj = obj.copy()
        obj["_conn_id"] = conn_id

    obj_bytes = pickle.dumps(obj)

    total_size = len(obj_bytes)
    chunks = [obj_bytes[i : i + CHUNK_SIZE] for i in range(0, total_size, CHUNK_SIZE)]
    total_packets = len(chunks)

    # Compute MD5
    content_hash = hashlib.md5(obj_bytes).hexdigest()
    logger.info(
        f"[CONN:{conn_id}] Sending {total_size} bytes in {total_packets} packets, MD5={content_hash}"
    )

    # Send headers
    sock.sendall(struct.pack(HEADER_FMT, total_size))
    sock.sendall(struct.pack(COUNT_FMT, total_packets))
    sock.sendall(struct.pack(HEADER_FMT, conn_id))

    # Send chunks
    for idx, chunk in enumerate(chunks, start=1):
        chunk_size = len(chunk)
        logger.info(
            f"[CONN:{conn_id}] Sending packet {idx}/{total_packets}: {chunk_size} bytes"
        )
        sock.sendall(struct.pack(HEADER_FMT, chunk_size))
        sock.sendall(chunk)

    # Send 32-byte ASCII hex MD5 trailer
    sock.sendall(content_hash.encode("ascii"))

    return conn_id
