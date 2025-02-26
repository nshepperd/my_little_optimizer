import argparse
import uvicorn
from .api import create_app
from .config import settings

def main():
    parser = argparse.ArgumentParser(description="My Little Optimizer Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind server to")
    parser.add_argument("--db-path", default=settings.db_path, help="SQLite database path")
    parser.add_argument("--dev", action="store_true", help="Enable auto-reload (development mode)")
    
    args = parser.parse_args()
    
    # Update settings
    settings.db_path = args.db_path
    
    if args.dev:
        # For development mode with reload
        uvicorn.run(
            "my_little_optimizer_server.api:create_app",
            host=args.host,
            port=args.port,
            reload=True,
        )
    else:
        # For production mode
        app = create_app(settings)
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
        )

if __name__ == "__main__":
    main()