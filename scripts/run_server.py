from __future__ import annotations

import argparse
import socket

from supersayan import SupersayanServer
from supersayan.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


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
            server.handle_client(conn, addr)


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