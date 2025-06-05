from __future__ import annotations

import argparse

from supersayan import SupersayanServer
from supersayan.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


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

    server = SupersayanServer(storage_dir=args.models_dir)

    server.listen(args.host, args.port)


if __name__ == "__main__":
    main()