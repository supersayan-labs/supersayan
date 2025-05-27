# Supersayan Logging System

The Supersayan project now includes a centralized, configurable logging system that provides consistent logging behavior across all modules.

## Features

- **Centralized Configuration**: All logging is configured from a single place
- **Colored Console Output**: Different log levels are displayed in different colors for better visibility
- **Rotating File Logs**: Automatic log rotation to prevent disk space issues
- **Flexible Configuration**: Easy to customize for different environments
- **Hierarchical Logger Names**: Automatic namespace management under `supersayan.*`

## Basic Usage

### In Your Code

Instead of using Python's standard logging, import and use the Supersayan logger:

```python
# Old way (don't use this anymore)
# import logging
# logger = logging.getLogger(__name__)

# New way
from supersayan.logging_config import get_logger

logger = get_logger(__name__)

# Use the logger as normal
logger.info("This is an info message")
logger.debug("Debug information")
logger.warning("Warning message")
logger.error("Error occurred")
```

### Configuration

The logging system can be configured at startup or runtime:

```python
from supersayan import configure_logging

# Basic configuration
configure_logging(level="DEBUG")

# Disable file logging (useful for tests)
configure_logging(level="INFO")

# Custom log directory
configure_logging(log_dir="./logs")

# Disable colors (e.g., for CI/CD environments)
configure_logging(disable_colors=True)

# Custom formats
configure_logging(
    console_format="%(levelname)s - %(message)s",
    file_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
```

### Advanced Configuration

You can provide a complete configuration dictionary or JSON file:

```python
# Using a configuration dictionary
config = {
    "version": 1,
    "formatters": {
        "simple": {
            "format": "%(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple"
        }
    },
    "loggers": {
        "supersayan": {
            "handlers": ["console"],
            "level": "DEBUG"
        }
    }
}

configure_logging(config_dict=config)

# Or load from a JSON file
configure_logging(config_file="logging_config.json")
```

### Runtime Level Changes

You can change log levels at runtime:

```python
from supersayan import set_log_level

# Change all supersayan loggers to DEBUG
set_log_level("DEBUG")

# Change a specific logger
set_log_level("WARNING", "supersayan.nn.layers")
```

## Default Configuration

By default, the logging system:

- Outputs INFO and above to console with colors
- Outputs DEBUG and above to a rotating file (supersayan.log)
- Uses detailed formatting for file output
- Limits log files to 10MB with 5 backups
- Automatically namespaces all loggers under `supersayan.*`

## Log Levels

- **DEBUG**: Detailed information for diagnosing problems
- **INFO**: General informational messages
- **WARNING**: Warning messages for potentially harmful situations
- **ERROR**: Error messages for serious problems
- **CRITICAL**: Critical messages for very serious errors

## Examples

### Script Usage

```python
#!/usr/bin/env python
from supersayan import configure_logging, get_logger

# Configure logging at the start of your script
configure_logging(
    level="INFO",
    log_dir="./logs",
    console_format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = get_logger(__name__)

def main():
    logger.info("Starting application")
    # Your code here
    logger.info("Application finished")

if __name__ == "__main__":
    main()
```

### Test Usage

```python
import pytest
from supersayan import configure_logging, get_logger

# Disable file logging for tests
configure_logging(level="DEBUG")

logger = get_logger(__name__)

def test_something():
    logger.debug("Running test")
    # Test code
    assert True
```

### Production Usage

```python
from supersayan import configure_logging

# Production configuration
configure_logging(
    level="WARNING",  # Only warnings and above
    log_dir="/var/log/supersayan",
    disable_colors=True,  # No colors in production logs
    file_format="%(asctime)s - %(hostname)s - %(name)s - %(levelname)s - %(message)s"
)
```

## Migration Guide

If you're updating existing code:

1. Replace all `import logging` with keeping the import but adding `from supersayan.logging_config import get_logger`
2. Replace all `logger = logging.getLogger(__name__)` with `logger = get_logger(__name__)`
3. Remove any `logging.basicConfig()` calls and replace with `configure_logging()`
4. The rest of your logging code remains the same

## Troubleshooting

### No Log Output

- Check that you've called `configure_logging()` or that the default configuration is appropriate
- Verify the log level is set correctly
- For file output, check file permissions in the log directory

### Colors Not Working

- Colors are automatically disabled on non-TTY outputs
- You can manually disable with `configure_logging(disable_colors=True)`

### Too Many Log Files

- Adjust the rotation settings in a custom configuration
- Change `maxBytes` and `backupCount` in the file handler configuration
