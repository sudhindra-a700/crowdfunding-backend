from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
import os
import logging
import sys
import secrets
import time # Import time for basic logging

# --- Minimal Logging Configuration ---
# This basic setup should work even if more complex ones fail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("Starting app.py execution - Global scope.")

# Create FastAPI app
app = FastAPI(
    title="HAVEN Backend Service - Minimal Debug",
    description="Minimal FastAPI app for debugging startup issues",
    version="0.0.1"
)

logger.info("FastAPI app instance created.")

# Add session middleware (minimal config)
# Ensure SECRET_KEY is set in Render environment, fallback for local testing
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", "a_very_secret_key_for_session_middleware_fallback")
)
logger.info("Session middleware added.")

# Add CORS middleware (minimal config)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("CORS middleware added.")

BASE_DIR = Path(__file__).resolve().parent
logger.info(f"BASE_DIR set to: {BASE_DIR}")

@app.get("/")
async def read_root():
    logger.info("Root endpoint hit.")
    return {"message": "Hello from HAVEN Backend (Minimal Debug Version)!"}

@app.get("/health")
async def health_check():
    logger.info("Health check endpoint hit.")
    return {"status": "healthy", "message": "Minimal backend is running."}

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup event triggered (Minimal).")
    # No complex initialization here for now
    logger.info("Minimal startup event completed.")

if __name__ == "__main__":
    logger.info("Running uvicorn directly.")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    logger.info("Uvicorn finished.")

