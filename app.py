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
# Removed the problematic import below as it was causing a boot-up error.
# from firebase_admin.exceptions import FirebaseAppError

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
    # Silence verbose loggers if needed
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

# Call logging setup
setup_logging()
logger = logging.getLogger(__name__)

# Load the .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Loaded environment variables from .env file.")
except ImportError:
    logger.warning("python-dotenv not installed. Skipping .env file loading.")

# Check for JWT secret key
if not os.getenv("JWT_SECRET_KEY"):
    logger.warning("JWT_SECRET_KEY not set. Generating a temporary key. This is NOT secure for production.")
    os.environ["JWT_SECRET_KEY"] = secrets.token_urlsafe(32)

# Global variables
app = FastAPI()
db = None # Will be initialized in the lifespan event handler
oauth_router = get_oauth_router()
firebase_admin_app = None # Will be initialized in the lifespan event handler
algolia_client = None
algolia_index = None

# Add Session Middleware
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET_KEY", secrets.token_urlsafe(32)))

# Configure CORS
# In a production environment, you should restrict this to your frontend's domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# Pydantic models for request bodies
class FraudPredictionRequest(BaseModel):
    org_name: str
    bio: str
    follower_count: int = 0
    post_count: int = 0
    account_age_days: int = 0
    engagement_rate: float = 0.0
    recent_posts: Optional[str] = None
    pan: Optional[str] = None
    registration_type: Optional[str] = None
    registration_number: Optional[str] = None
    ngo_darpan_id: Optional[str] = None
    fcra_number: Optional[str] = None

class FeedbackRequest(BaseModel):
    prediction_id: str
    user_feedback: int # 1 for correct, 0 for incorrect

class OAuthStatusRequest(BaseModel):
    code: str = Field(..., description="The authorization code from OAuth provider")
    state: str = Field(..., description="The state token for CSRF protection")

# Helper function to get Firebase admin app instance
def get_firebase_app():
    if firebase_admin_app is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Firebase is not initialized"
        )
    return firebase_admin_app

# Include OAuth routes
app.include_router(oauth_router)


# FastAPI lifespan events
# This is a better way to handle startup/shutdown tasks than the old `@app.on_event`
@app.on_event("startup")
async def startup_event():
    """Application startup event handler"""
    global db, firebase_admin_app, algolia_client, algolia_index
    logger.info("Starting application startup event...")

    try:
        # Initialize Firebase Admin SDK
        firebase_service_account_key_base64 = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY_JSON_BASE64")
        if firebase_service_account_key_base64:
            service_account_info = json.loads(base64.b64decode(firebase_service_account_key_base64).decode('utf-8'))
            cred = credentials.Certificate(service_account_info)
            firebase_admin_app = firebase_admin.initialize_app(cred)
            db = firestore.client(app=firebase_admin_app)
            logger.info("Successfully initialized Firebase Admin SDK.")
        else:
            logger.warning("FIREBASE_SERVICE_ACCOUNT_KEY_JSON_BASE64 not found. Firebase Admin SDK will not be initialized.")

        # Lazy load NLP model and NGO data
        load_fraud_detection_model()
        load_ngo_darpan_data()

        # Initialize Algolia client if keys are available
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

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event handler"""
    # Clean up resources if necessary
    logger.info("Application shutdown event initiated.")

@app.get("/")
async def read_root():
    """Health check endpoint"""
    return {"message": "HAVEN backend is running!"}

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Check system health
        cpu_usage = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory()
        mem_usage = mem.percent

        # Check Firebase connection (simplified, as it's initialized on startup)
        firebase_status = "OK" if firebase_admin_app else "Uninitialized"

        # Check Algolia connection (simplified)
        algolia_status = "OK" if algolia_client else "Uninitialized or Config Error"

        # Check model loading status
        model_status = "Loaded" if load_fraud_detection_model() is not None else "Failed to Load"

        return {
            "status": "OK",
            "message": "Service is healthy",
            "system_metrics": {
                "cpu_usage_percent": cpu_usage,
                "memory_usage_percent": mem_usage
            },
            "dependencies": {
                "firebase_admin": firebase_status,
                "algolia_search": algolia_status,
                "ml_model": model_status
            },
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/predict-fraud")
async def predict_fraud_endpoint(request: FraudPredictionRequest):
    """API endpoint to predict fraud for a given organization."""
    try:
        # Pass the request data directly to the prediction function
        fraud_score, explanation, plot_path, verification = predict_fraud(request.dict())

        # Clean up old plots to prevent accumulation
        if plot_path:
            old_plots = list(Path("static/shap_plots").glob("shap_plot_*.png"))
            if len(old_plots) > 20: # Keep a reasonable number of plots
                for old_plot in sorted(old_plots, key=os.path.getmtime)[:-20]:
                    os.remove(old_plot)

        return {
            "fraud_score": fraud_score,
            "explanation": explanation,
            "shap_plot_url": f"/static/{plot_path}" if plot_path else None,
            "verification_details": verification
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """API endpoint to receive user feedback on a prediction."""
    # In a real app, you would save this to a database
    logger.info(f"Received feedback for prediction {request.prediction_id}: {request.user_feedback}")
    return {"message": "Feedback received successfully"}
