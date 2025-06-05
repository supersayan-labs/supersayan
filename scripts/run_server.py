from __future__ import annotations

import argparse

from supersayan.logging_config import configure_logging, get_logger
from supersayan.remote.server import SupersayanServer

# Configure logging
configure_logging(
    level="INFO", console_format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = get_logger(__name__)


def main():
    """Run the Supersayan server."""
    parser = argparse.ArgumentParser(description="Run Supersayan FHE Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--storage-dir", default="server_db/models", help="Model storage directory"
    )

    args = parser.parse_args()

    logger.info("Starting Supersayan server...")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Storage directory: {args.storage_dir}")

    server = SupersayanServer(storage_dir=args.storage_dir)

    try:
        server.listen(args.host, args.port)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()