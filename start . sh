#!/bin/sh

# Use the PORT environment variable provided by Render, or default to 8000
UVICORN_PORT=${PORT:-10000}

# Start uvicorn

exec uvicorn app:app --host 0.0.0.0 --port $UVICORN_PORT --workers 1 --timeout-keep-alive 30
