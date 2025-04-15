# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands
- Install: `pip install -e .`
- Run tests: `pytest tests -v`
- Run benchmarks: `pytest benchmarks -v`

## Code Style Guidelines
- **Imports**: Group imports (stdlib, third-party, local) with stdlib first
- **Type hints**: Use type hints with `Optional`, `Union`, and `@overload` for polymorphic functions
- **Documentation**: Docstrings using """triple quotes""" with Args/Returns/Raises sections
- **Error handling**: Use specific exceptions with descriptive messages and logging
- **Logging**: Use Python's logging module with appropriate log levels
- **Naming**: snake_case for functions/variables, CamelCase for classes
- **Architecture**: Follows PyTorch-style NN module pattern
- **Julia interface**: Use bindings.py for Julia-Python integration