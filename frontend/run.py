"""
Frontend Runner for Streamlit Application.
Provides programmatic and CLI entrypoints to launch the Streamlit Web UI.
"""

import os
import sys
import subprocess
from pathlib import Path


def start_frontend(host: str = "0.0.0.0", port: int = 8501):
    """Launch Streamlit UI server as a subprocess or via streamlit CLI."""
    ui_path = Path(__file__).parent / "ui.py"
    
    # Read environment overrides if available
    port = int(os.getenv("STREAMLIT_PORT", str(port)))
    host = os.getenv("STREAMLIT_HOST", host)
    
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ui_path),
        "--server.port",
        str(port),
        "--server.address",
        str(host),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    
    print(f"Starting Streamlit frontend on http://{host}:{port}...")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nShutting down frontend server.")
    except Exception as e:
        print(f"Error starting frontend: {e}")
        sys.exit(1)


def main():
    """CLI entrypoint."""
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Agent Knowledge Base Streamlit Frontend")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind")
    parser.add_argument("--port", type=int, default=8501, help="Port to bind")
    args = parser.parse_args()
    start_frontend(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
