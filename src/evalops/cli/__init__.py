"""Typer CLI for EvalOps.

This module will provide command-line interface for:
- Running evaluations
- Managing datasets
- Viewing results
"""

from evalops.cli import regression, main

__all__ = ["regression", "main"]
