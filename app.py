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
try:
    import firebase_admin
    from firebase_admin import credentials, auth, firestore, messaging
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    firebase_admin = None
    credentials = None
    auth = None
    firestore = None
    messaging = None

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
    logging.getLogger().handlers.clear()

    if log_format == "json":
        # Production-ready JSON logging
        logging.basicConfig(
            level=log_level,
            format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "logger": "%(name)s"}',
            datefmt="%Y-%m-%dT%H:%M:%S%z"
        )
    else:
        # Development-friendly colored logging
        try:
            from rich.logging import RichHandler
            logging.basicConfig(
                level=log_level,
                format="%(message)s",
                datefmt="[%X]",
                handlers=[RichHandler()]
            )
        except ImportError:
            logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logging.getLogger("uvicorn").handlers.clear()
    logging.getLogger("uvicorn.access").handlers.clear()
    logging.getLogger("uvicorn.error").handlers.clear()

# Setup logging on module load
setup_logging()
logger = logging.getLogger(__name__)

# --- Application Setup ---
BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(
    title="HAVEN Crowdfunding Platform Backend",
    description="Backend API for managing campaigns, user authentication, and fraud detection.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session middleware for OAuth state
SECRET_KEY = os.getenv("SESSION_SECRET_KEY", secrets.token_urlsafe(32))
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# OAuth2PasswordBearer for dependency injection
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

# --- Dependency Functions ---
def get_current_user_from_token(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """Dependency to get current user from a JWT token."""
    jwt_manager = get_jwt_manager()
    user_data = jwt_manager.get_user_from_token(token)
    return user_data

# --- Event Handlers ---
@app.on_event("startup")
async def startup_event():
    """Application startup event handler."""
    logger.info("Application startup event triggered.")

    # --- Firebase Initialization (FIX) ---
    if FIREBASE_AVAILABLE:
        try:
            # Check if Firebase is already initialized
            if not firebase_admin._apps:
                firebase_service_account_base64 = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY_JSON_BASE64")
                if firebase_service_account_base64:
                    # Decode the base64 string to get the JSON content
                    service_account_json = base64.b64decode(firebase_service_account_base64).decode('utf-8')
                    cred = credentials.Certificate(json.loads(service_account_json))
                    firebase_admin.initialize_app(cred)
                    logger.info("Firebase app initialized successfully from environment variable.")
                else:
                    logger.warning("FIREBASE_SERVICE_ACCOUNT_KEY_JSON_BASE64 not found. Firebase features will be disabled.")
            else:
                logger.info("Firebase app already initialized.")
        except Exception as e:
            logger.error(f"Error initializing Firebase app: {e}", exc_info=True)
            global FIREBASE_AVAILABLE
            FIREBASE_AVAILABLE = False
    else:
        logger.warning("Firebase Admin SDK not available. Please install 'firebase-admin' to enable Firebase features.")

    # --- Fraud Detection Model Loading ---
    try:
        load_fraud_detection_model()
        logger.info("Fraud detection model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading fraud detection model: {e}", exc_info=True)

    # --- NGO Darpan Data Loading ---
    try:
        load_ngo_darpan_data()
        logger.info("NGO Darpan data loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading NGO Darpan data: {e}", exc_info=True)

    # --- Algolia Search Initialization ---
    try:
        if ALGOLIA_AVAILABLE:
            algolia_app_id = os.getenv("ALGOLIA_APP_ID")
            algolia_api_key = os.getenv("ALGOLIA_API_KEY")
            if algolia_app_id and algolia_api_key:
                algolia_client = SearchClient.create(algolia_app_id, algolia_api_key)
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

    except Exception as e:
        logger.error(f"Error initializing Algolia client: {e}", exc_info=True)
        algolia_client = None
        algolia_index = None

    STATIC_DIR = BASE_DIR / "static"
    try:
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
        logger.error(f"Error setting up static file serving: {e}", exc_info=True)

    logger.info("Application startup event completed. Ready to serve with OAuth support.")

# --- API Routes ---
app.include_router(get_oauth_router())

@app.get("/health", response_class=HTMLResponse)
async def health_check():
    """Health check endpoint."""
    return "OK"

@app.get("/")
async def home_redirect():
    """Redirects to the documentation."""
    return RedirectResponse(url="/docs")

# Main entry point (for local development)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
