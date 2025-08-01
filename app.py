from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from starlette.middleware.sessions import SessionMiddleware
import os
import pandas as pd
import joblib
import json
import random
import sys
import logging
import time
import psutil
import base64
import secrets

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field

# Firebase Admin SDK imports
import firebase_admin
from firebase_admin import credentials, auth, firestore, messaging
from firebase_admin.exceptions import FirebaseAppError

# OAuth imports
from oauth_routes import get_oauth_router
from oauth_config import get_oauth_config
from jwt_utils import get_jwt_manager

# Import fraud detection module
from fraud_detection import predict_fraud, load_ngo_darpan_data, load_fraud_detection_model, fine_tune_model # Added load_fraud_detection_model, fine_tune_model

# Algolia Search Client
try:
    from algoliasearch.search_client import SearchClient
    ALGOLIA_AVAILABLE = True
except ImportError:
    ALGOLIA_AVAILABLE = False
    SearchClient = None

# --- Enhanced Logging Configuration ---
def setup_logging():
    """Configure enhanced logging for production"""
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_format = os.environ.get("LOG_FORMAT", "json")

    # Clear existing handlers to prevent duplicate logs
    logging.basicConfig(handlers=[], level=logging.INFO)

    # Use rich for local development, fallback to default for production
    try:
        from rich.logging import RichHandler
        if log_format == "rich":
            handler = RichHandler(rich_tracebacks=True,
                                  tracebacks_show_locals=False,
                                  markup=True)
            logging.basicConfig(level=log_level, format="%(message)s", datefmt="[%X]", handlers=[handler])
        else:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logging.basicConfig(level=log_level, handlers=[handler])
    except ImportError:
        logging.basicConfig(level=log_level)
        logging.warning("rich library not found. Using default logging.")

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# --- FastAPI Application Setup ---
app = FastAPI(
    title="Haven Crowdfunding Backend",
    description="A backend for a crowdfunding platform with fraud detection and OAuth.",
    version="1.0.0",
)

# CORS Middleware for all origins (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session Middleware for OAuth
app.add_middleware(SessionMiddleware, secret_key=secrets.token_urlsafe(32))

# OAuth2PasswordBearer for dependency injection
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

# Include the OAuth router
oauth_router = get_oauth_router()
app.include_router(oauth_router)

# --- Application Lifecycle Events ---
@app.on_event("startup")
async def startup_event():
    """
    Application startup event handler.
    Initializes Firebase, Algolia, and fraud detection models.
    """
    logger.info("Application startup event triggered.")
    try:
        # 1. Initialize Firebase Admin SDK
        try:
            # Check if the app is already initialized, a common issue with Gunicorn workers
            firebase_admin.get_app()
            logger.warning("Firebase app already initialized.")
        except ValueError:
            logger.info("Initializing Firebase app...")
            cred_json_base64 = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY_JSON_BASE64")
            if cred_json_base64:
                try:
                    cred_json = json.loads(base64.b64decode(cred_json_base64).decode('utf-8'))
                    cred = credentials.Certificate(cred_json)
                    firebase_admin.initialize_app(cred)
                    logger.info("Firebase app initialized successfully from base64 env var.")
                except Exception as e:
                    logger.error(f"Failed to initialize Firebase from base64 env var: {e}", exc_info=True)
                    # This will re-raise the exception to be caught by the outer block
                    raise
            else:
                logger.warning("FIREBASE_SERVICE_ACCOUNT_KEY_JSON_BASE64 not found. Attempting to use default credentials.")
                firebase_admin.initialize_app()
                logger.info("Firebase app initialized successfully with default credentials.")
        
        # 2. Initialize Algolia Client
        if ALGOLIA_AVAILABLE:
            algolia_client_id = os.getenv("ALGOLIA_APP_ID")
            algolia_api_key = os.getenv("ALGOLIA_API_KEY")
            if algolia_client_id and algolia_api_key:
                algolia_client = SearchClient.create(algolia_client_id, algolia_api_key)
                algolia_index = algolia_client.init_index('campaigns')
                logger.info("Algolia client initialized for index: campaigns")
            else:
                logger.warning("Algolia API keys not configured. Search functionality will be limited.")
                algolia_client = None
                algolia_index = None
        else:
            logger.warning("Algolia library not available. Search functionality will be limited.")
            algolia_client = None
            algolia_index = None

        STATIC_DIR = Path("static")
        if not STATIC_DIR.exists():
            STATIC_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created static directory: {STATIC_DIR}")

        for subdir in ["icons", "shap_plots"]:
            subdir_path = STATIC_DIR / subdir
            if not subdir_path.exists():
                subdir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created {subdir} directory: {subdir_path}")

        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
        logger.info(f"Mounted static files from: {STATIC_DIR}")

    except Exception as e:
        logger.error(f"Fatal error during application startup: {e}", exc_info=True)
        # Re-raise the exception to let Gunicorn handle it and fail the worker gracefully.
        raise

    logger.info("Application startup event completed. Ready to serve with OAuth support.")

if __name__ == "__main__":
    import uvicorn
    # A simple way to run the app. In production, use Gunicorn with the `app` object directly.
    uvicorn.run(app, host="0.0.0.0", port=8000)

