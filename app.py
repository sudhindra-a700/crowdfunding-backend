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
    log_format = os.environ.get("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.basicConfig(level=log_level, format=log_format, stream=sys.stdout)
    # Set a higher level for libraries that are too verbose
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("algoliasearch").setLevel(logging.WARNING)
    return logging.getLogger("app")

logger = setup_logging()

# --- Firebase Initialization ---
def initialize_firebase_app():
    """Initializes the Firebase Admin SDK using a base64 encoded service account key."""
    # The key is passed as a base64 encoded JSON string to handle it safely in an environment variable.
    firebase_key_base64 = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY_JSON_BASE64")
    if not firebase_key_base64:
        logger.error("FIREBASE_SERVICE_ACCOUNT_KEY_JSON_BASE64 environment variable not set.")
        return None

    try:
        # Decode the base64 string back to a JSON string
        key_json_str = base64.b64decode(firebase_key_base64).decode('utf-8')
        cred = credentials.Certificate(json.loads(key_json_str))
        
        # Check if an app is already initialized to avoid re-initialization errors
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
            logger.info("Firebase Admin SDK initialized successfully.")
        else:
            logger.info("Firebase Admin SDK already initialized.")
        return firebase_admin.get_app()
    except FirebaseAppError as e:
        logger.error(f"Failed to initialize Firebase Admin SDK: {e}")
        return None
    except (json.JSONDecodeError, base64.binascii.Error) as e:
        logger.error(f"Failed to decode or parse Firebase service account key: {e}")
        return None

# FastAPI app setup
app = FastAPI(
    title="NGO Verification & Fraud Detection API",
    description="An API to verify NGOs and detect potential fraudulent campaigns using a fine-tuned NLP model.",
    version="1.0.0",
)

# --- Middleware ---
# Session Middleware for OAuth flow
app.add_middleware(SessionMiddleware, secret_key=secrets.token_urlsafe(32))

# CORS Middleware for allowing cross-origin requests
origins = os.getenv("CORS_ORIGINS", "*").split(',')
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OAuth2PasswordBearer for token-based authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- Event Handlers ---
@app.on_event("startup")
async def startup_event():
    """
    Handles application startup events:
    - Initialize Firebase Admin SDK.
    - Load NGO Darpan data and fraud detection models.
    - Setup Algolia search client.
    - Create static directories.
    """
    logger.info("Application startup event started.")
    
    # Initialize Firebase Admin SDK
    firebase_app = initialize_firebase_app()
    if not firebase_app:
        logger.error("Firebase application could not be initialized. Some features may not work.")
        # Re-raise the exception to let Gunicorn handle it and fail the worker gracefully.
        raise RuntimeError("Firebase initialization failed.")

    try:
        logger.info("Loading fraud detection model and NGO Darpan data...")
        load_ngo_darpan_data()
        load_fraud_detection_model()
        logger.info("Models and data loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading models or data: {e}", exc_info=True)
        # Re-raise the exception to let Gunicorn handle it and fail the worker gracefully.
        raise

    try:
        if ALGOLIA_AVAILABLE:
            algolia_app_id = os.getenv("ALGOLIA_APP_ID")
            algolia_api_key = os.getenv("ALGOLIA_API_KEY")
            if algolia_app_id and algolia_api_key:
                algolia_client = SearchClient.create(algolia_app_id, algolia_api_key)
                algolia_index = algolia_client.init_index("ngo_campaigns")
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
