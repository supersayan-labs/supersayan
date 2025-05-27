from __future__ import annotations
import argparse
import socket
from supersayan import SupersayanServer
from supersayan.logging_config import get_logger, configure_logging
from supersayan.remote.socket_utils import recv_obj, send_obj

logger = get_logger(__name__)


def handle_client(
    conn: socket.socket, addr: tuple[str, int], server: SupersayanServer
) -> None:
    """
    Handle a client connection.

    Args:
        conn: The socket connection
        addr: The address of the client
        server: The server instance
    """
    logger.info("Connection from %s:%s", *addr)

    try:
        while True:
            try:
                request, conn_id = recv_obj(conn)
            except ConnectionError:
                break

            if not isinstance(request, dict) or "command" not in request:
                send_obj(conn, {"status": False, "error": "invalid request"}, conn_id)
                continue

            cmd = request["command"]
            logger.info(f"[CONN:{conn_id}] Processing command: {cmd}")

            try:
                if cmd == "upload_model":
                    model_id = server.handle_upload_model_bytes(request["model_data"])
                    send_obj(conn, {"status": True, "model_id": model_id}, conn_id)
                elif cmd == "get_model_structure":
                    structure = server.handle_get_model_structure(request["model_id"])
                    send_obj(conn, {"status": True, "structure": structure}, conn_id)
                elif cmd == "inference":
                    output = server.handle_inference(
                        request["model_id"],
                        request["layer_name"],
                        request["encrypted_input"],
                    )
                    send_obj(
                        conn, {"status": True, "encrypted_output": output}, conn_id
                    )
                else:
                    send_obj(
                        conn,
                        {"status": False, "error": f"unknown command: {cmd}"},
                        conn_id,
                    )
            except Exception as exc:
                logger.exception(f"[CONN:{conn_id}] Error handling command {cmd}")
                send_obj(conn, {"status": False, "error": str(exc)}, conn_id)
    finally:
        conn.close()
        logger.info("closed connection %s:%s", *addr)


def serve_forever(host: str, port: int, models_dir: str) -> None:
    """
    Serve forever.

    Args:
        host: The host to bind to
        port: The port to bind to
        models_dir: The directory to store models
    """
    server = SupersayanServer(storage_dir=models_dir)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen()

        logger.info("Supersayan server listening on %s:%s", host, port)

        while True:
            conn, addr = s.accept()
            handle_client(conn, addr, server)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Supersayan TCP server")
    parser.add_argument("--host", default="127.0.0.1", help="TCP interface to bind to")
    parser.add_argument("--port", type=int, default=8000, help="TCP port to bind to")
    parser.add_argument(
        "--models-dir",
        default="/tmp/supersayan/models",
        help="Directory where uploaded models are stored",
    )

    args = parser.parse_args()

    configure_logging()

    serve_forever(args.host, args.port, args.models_dir)


if __name__ == "__main__":
    main()
