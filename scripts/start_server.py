#!/usr/bin/env python
"""
Script to start the Mirix REST API server.

Usage:
    python scripts/start_server.py [OPTIONS]

Options:
    --host HOST         Host to bind to (default: 0.0.0.0)
    --port PORT         Port to bind to (default: 8531)
    --reload            Enable auto-reload for development
    --workers N         Number of worker processes (default: 1)
    --log-level LEVEL   Log level (default: info)
"""

import argparse
import os
import sys

# Ensure the project root (one level up from this file) is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="Start the Mirix REST API server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8531,
        help="Port to bind to (default: 8531)",
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )

    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Log level (default: info)",
    )

    parser.add_argument(
        "--production",
        action="store_true",
        help="Run in production mode with multiple workers (requires gunicorn)",
    )

    args = parser.parse_args()

    # Check if running in production mode
    if args.production:
        try:
            import gunicorn.app.base
        except ImportError:
            print("Error: gunicorn is required for production mode")
            print("Install it with: pip install gunicorn")
            sys.exit(1)

        print("Starting Mirix server in PRODUCTION mode...")
        print(f"  Host: {args.host}")
        print(f"  Port: {args.port}")
        print(f"  Workers: {args.workers}")
        print(f"  Log level: {args.log_level}")
        print()

        # Run with gunicorn
        os.system(
            f"gunicorn mirix.server.rest_api:app "
            f"--workers {args.workers} "
            f"--worker-class uvicorn.workers.UvicornWorker "
            f"--bind {args.host}:{args.port} "
            f"--timeout 120 "
            f"--log-level {args.log_level}"
        )
    else:
        # Development mode with uvicorn
        try:
            import uvicorn
        except ImportError:
            print("Error: uvicorn is required")
            print("Install it with: pip install uvicorn")
            sys.exit(1)

        print("Starting Mirix server in DEVELOPMENT mode...")
        print(f"  Host: {args.host}")
        print(f"  Port: {args.port}")
        print(f"  Log level: {args.log_level}")
        print(f"  Auto-reload: {args.reload}")
        print()
        print("Server will be available at:")
        print(f"  http://{args.host}:{args.port}")
        print(f"  http://localhost:{args.port} (if host is 0.0.0.0)")
        print()
        print("API documentation available at:")
        print(f"  http://localhost:{args.port}/docs (Swagger UI)")
        print(f"  http://localhost:{args.port}/redoc (ReDoc)")
        print()
        print("Press CTRL+C to stop the server")
        print()

        uvicorn.run(
            "mirix.server.rest_api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
        )


if __name__ == "__main__":
    main()
