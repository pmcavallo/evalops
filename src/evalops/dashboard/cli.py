"""CLI entry point for running the EvalOps dashboard.

This module provides a command-line interface to launch the Streamlit dashboard.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Launch the EvalOps Streamlit dashboard."""
    # Get the path to the app.py file
    app_path = Path(__file__).parent / "app.py"

    # Build the streamlit command
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=true",
    ]

    # Add any additional arguments passed to the command
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])

    # Run streamlit
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
