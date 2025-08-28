import json
import logging
import logging.config
from pathlib import Path
from typing import Any, Dict, Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""

    grey = "\x1b[38;21m"
    blue = "\x1b[34m"
    green = "\x1b[32m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    COLORS = {
        logging.DEBUG: grey,
        logging.INFO: blue,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: bold_red,
    }

    def format(self, record) -> str:
        """
        Format the log record.

        Args:
            record: The log record to format

        Returns:
            str: The formatted log record
        """
        try:
            log_color = self.COLORS.get(record.levelno, self.grey)
            record.levelname = f"{log_color}{record.levelname}{self.reset}"
            return super().format(record)
        except Exception as e:
            return f"{record.asctime} - {record.name} - {record.levelname} - {record.getMessage()}"


# Default logging configuration
DEFAULT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "colored": {
            "()": ColoredFormatter,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "colored",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "supersayan.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "supersayan": {
            "level": "DEBUG",
            "handlers": ["console"],
            "propagate": False,
        }
    },
    "root": {"level": "WARNING", "handlers": ["console"]},
}


class SupersayanLogger:
    """
    Centralized logger for the Supersayan project.

    This class provides a unified logging interface with easy configuration
    and consistent behavior across the entire project.
    """

    _instance = None
    _initialized = False

    def __new__(cls) -> str:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not self._initialized:
            self._config = DEFAULT_CONFIG.copy()
            self._log_dir = None
            self._initialized = True

    def configure(
        self,
        level: Optional[str] = None,
        log_dir: Optional[str] = None,
        console_format: Optional[str] = None,
        file_format: Optional[str] = None,
        disable_file_logging: bool = True,
        disable_colors: bool = False,
        config_dict: Optional[Dict[str, Any]] = None,
        config_file: Optional[str] = None,
    ) -> None:
        """
        Configure the logging system.

        Args:
            level: Global logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files (default: current directory)
            console_format: Format string for console output
            file_format: Format string for file output
            disable_file_logging: If True, disable file logging
            disable_colors: If True, disable colored console output
            config_dict: Complete configuration dictionary (overrides other options)
            config_file: Path to JSON configuration file (overrides other options)
        """
        if config_file:
            with open(config_file, "r") as f:
                self._config = json.load(f)
        elif config_dict:
            self._config = config_dict
        else:
            # Update configuration based on parameters
            if level:
                self._config["loggers"]["supersayan"]["level"] = level
                self._config["handlers"]["console"]["level"] = level

            if log_dir:
                self._log_dir = Path(log_dir)
                self._log_dir.mkdir(parents=True, exist_ok=True)
                log_file = self._log_dir / "supersayan.log"
                self._config["handlers"]["file"]["filename"] = str(log_file)

            if console_format:
                self._config["formatters"]["colored"]["format"] = console_format
                self._config["formatters"]["standard"]["format"] = console_format

            if file_format:
                self._config["formatters"]["detailed"]["format"] = file_format

            if disable_file_logging:
                self._config["loggers"]["supersayan"]["handlers"] = ["console"]
                if "file" in self._config["handlers"]:
                    del self._config["handlers"]["file"]
            else:
                if "file" not in self._config["loggers"]["supersayan"]["handlers"]:
                    self._config["loggers"]["supersayan"]["handlers"].append("file")

            if disable_colors:
                self._config["handlers"]["console"]["formatter"] = "standard"

        # Apply the configuration
        logging.config.dictConfig(self._config)

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger instance for the given name.

        Args:
            name: Logger name (typically __name__ from the calling module)

        Returns:
            logging.Logger: Configured logger instance
        """
        if not name.startswith("supersayan"):
            if name == "__main__" or not name.startswith("supersayan"):
                name = f"supersayan.{name}"

        return logging.getLogger(name)

    def set_level(self, level: str, logger_name: Optional[str] = None) -> None:
        """
        Set logging level for a specific logger or all supersayan loggers.

        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            logger_name: Specific logger name, or None for all supersayan loggers
        """
        if logger_name:
            logging.getLogger(logger_name).setLevel(getattr(logging, level))
        else:
            logging.getLogger("supersayan").setLevel(getattr(logging, level))

    def add_file_handler(
        self, filename: str, level: str = "DEBUG", formatter: str = "detailed"
    ) -> None:
        """
        Add an additional file handler.

        Args:
            filename: Path to the log file
            level: Logging level for this handler
            formatter: Formatter name to use
        """
        handler = logging.handlers.RotatingFileHandler(
            filename, maxBytes=10485760, backupCount=5, encoding="utf-8"
        )
        handler.setLevel(getattr(logging, level))
        handler.setFormatter(
            logging.Formatter(self._config["formatters"][formatter]["format"])
        )
        logging.getLogger("supersayan").addHandler(handler)


# Create a singleton instance
_logger_config = SupersayanLogger()


# Convenience functions
def configure_logging(**kwargs) -> None:
    """
    Configure the Supersayan logging system.

    Args:
        **kwargs: Keyword arguments for the configuration
    """
    _logger_config.configure(**kwargs)


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (typically __name__ from the calling module)

    Returns:
        logging.Logger: Configured logger instance
    """
    return _logger_config.get_logger(name)


def set_log_level(level: str, logger_name: Optional[str] = None) -> None:
    """
    Set logging level for specific or all loggers.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        logger_name: Specific logger name, or None for all supersayan loggers
    """
    _logger_config.set_level(level, logger_name)


# Initialize with default configuration
_logger_config.configure()
