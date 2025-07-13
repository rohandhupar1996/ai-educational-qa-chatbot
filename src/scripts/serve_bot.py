#!/usr/bin/env python3
"""
Production server script for AI Education Q&A Bot with OpenAI
"""
import os
import sys
import logging
import argparse
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import uvicorn
from dotenv import load_dotenv

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/app.log', mode='a')
        ]
    )

def create_directories():
    """Create necessary directories."""
    directories = ['outputs', 'logs', 'models']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logging.info(f"Ensured directory exists: {directory}")

def validate_environment():
    """Validate required environment variables."""
    required_vars = ['OPENAI_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logging.error(f"Missing required environment variables: {missing_vars}")
        logging.error("Please check your .env file or environment configuration")
        logging.error("Required variables:")
        logging.error("  OPENAI_API_KEY=your_openai_api_key_here")
        sys.exit(1)
    
    logging.info("Environment validation passed")

def main():
    """Main function to start the server."""
    parser = argparse.ArgumentParser(description="AI Education Q&A Bot Server (OpenAI)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--log-level", default="info", help="Log level")
    parser.add_argument("--env-file", default=".env", help="Environment file path")
    
    args = parser.parse_args()
    
    # Load environment variables
    if os.path.exists(args.env_file):
        load_dotenv(args.env_file)
        logging.info(f"Loaded environment from {args.env_file}")
    else:
        logging.warning(f"Environment file {args.env_file} not found")
    
    # Override with environment variables if set
    host = os.getenv("HOST", args.host)
    port = int(os.getenv("PORT", args.port))
    workers = int(os.getenv("WORKERS", args.workers))
    log_level = os.getenv("LOG_LEVEL", args.log_level)
    
    # Setup
    create_directories()
    setup_logging(log_level)
    validate_environment()
    
    # Server configuration
    config = {
        "app": "ai_edu_qa_bot.main:app",
        "host": host,
        "port": port,
        "log_level": log_level.lower(),
        "access_log": True,
        "use_colors": True,
    }
    
    # Development vs Production settings
    if args.reload or os.getenv("ENVIRONMENT") == "development":
        config.update({
            "reload": True,
            "reload_dirs": ["src"],
        })
        logging.info("Running in development mode with auto-reload")
    else:
        config.update({
            "workers": workers,
        })
        logging.info(f"Running in production mode with {workers} workers")
    
    logging.info(f"Starting AI Education Q&A Bot server (OpenAI) on {host}:{port}")
    
    try:
        uvicorn.run(**config)
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
    except Exception as e:
        logging.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()