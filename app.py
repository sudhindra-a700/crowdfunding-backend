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

# OAuth imports
from oauth_routes import get_oauth_router
from oauth_config import get_oauth_config
from jwt_utils import get_jwt_manager

# Import fraud detection module (only import, no direct calls here)
from fraud_detection import predict_fraud, load_ngo_darpan_data, load_fraud_detection_model, fine_tune_model

# Algolia Search Client (only import, initialization moved to startup)
try:
    from algoliasearch.search_client import SearchClient
    ALGOLIA_AVAILABLE = True
except ImportError:
    ALGOLIA_AVAILABLE = False
    SearchClient = None

# Define BASE_DIR at the very top, as it's a fundamental path
BASE_DIR = Path(__file__).resolve().parent

# --- Enhanced Logging Configuration ---
def setup_logging():
    """Configure enhanced logging for production"""
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_format = os.environ.get("LOG_FORMAT", "json")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    for handler in logging.getLogger(__name__).handlers[:]:
        logging.getLogger(__name__).removeHandler(handler)

    if log_format.lower() == "json":
        import json
        import datetime

        class JSONFormatter(logging.Formatter):
            def format(self, record):
                try:
                    log_entry = {
                        "timestamp": datetime.datetime.utcnow().isoformat(),
                        "level": record.levelname,
                        "logger": record.name,
                        "message": record.getMessage(),
                        "module": record.module,
                        "function": record.funcName,
                        "line": record.lineno
                    }
                    if record.exc_info:
                        log_entry["exception"] = self.formatException(record.exc_info)
                    return json.dumps(log_entry)
                except Exception as e:
                    return f"ERROR: Could not format log record to JSON: {e} - Original message: {record.getMessage()}"

            def handleError(self, record):
                pass

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
    else:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        handler.setFormatter(formatter)

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        handlers=[handler],
        force=True,
        disable_existing_loggers=False
    )

    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("firebase_admin").setLevel(logging.INFO)
    logging.getLogger("authlib").setLevel(logging.INFO)

    return logging.getLogger(__name__)

logger = setup_logging()

class EnvironmentConfig:
    def __init__(self):
        self.required_vars = {}
        self.optional_vars = {
            "FIREBASE_SERVICE_ACCOUNT_KEY_JSON_BASE64": "Firebase authentication (fallback)",
            "ALGOLIA_API_KEY": "Search functionality",
            "ALGOLIA_APP_ID": "Search functionality",
            "BREVO_API_KEY": "Email notifications",
            "INSTAMOJO_API_KEY": "Payment processing",
            "INSTAMOJO_AUTH_TOKEN": "Payment processing",
            "LOG_LEVEL": "Logging configuration",
            "LOG_FORMAT": "Logging format",
            "ENVIRONMENT": "Environment identification",
            "GOOGLE_CLIENT_ID": "Google OAuth authentication",
            "GOOGLE_CLIENT_SECRET": "Google OAuth authentication",
            "FACEBOOK_CLIENT_ID": "Facebook OAuth authentication",
            "FACEBOOK_CLIENT_SECRET": "Facebook OAuth authentication",
            "JWT_SECRET_KEY": "JWT token signing",
            "SECRET_KEY": "Session management",
            "GOOGLE_REDIRECT_URI": "Google OAuth redirect URI",
            "FACEBOOK_REDIRECT_URI": "Facebook OAuth redirect URI",
            "FRONTEND_BASE_URL": "Frontend base URL for redirects"
        }
        self.config = {}
        self.missing_required = []
        self.missing_optional = []
        self._load_and_validate()

    def _load_and_validate(self):
        logger.info("Loading and validating environment variables...")
        for var_name, description in self.required_vars.items():
            value = os.environ.get(var_name)
            if not value:
                self.missing_required.append((var_name, description))
                logger.error(f"Missing required environment variable: {var_name} ({description})")
            else:
                self.config[var_name] = value
                logger.info(f"✓ Required variable loaded: {var_name}")

        for var_name, description in self.optional_vars.items():
            value = os.environ.get(var_name)
            if not value:
                self.missing_optional.append((var_name, description))
                logger.warning(
                    f"Missing optional environment variable: {var_name} ({description}) - Feature may be limited")
            else:
                self.config[var_name] = value
                logger.info(f"✓ Optional variable loaded: {var_name}")

        self.config.setdefault("LOG_LEVEL", "INFO")
        self.config.setdefault("LOG_FORMAT", "standard")
        self.config.setdefault("ENVIRONMENT", "production")

    def get(self, key: str, default: str = None) -> str:
        return self.config.get(key, default)

    def is_required_missing(self) -> bool:
        return len(self.missing_required) > 0

    def get_missing_required(self) -> List[tuple]:
        return self.missing_required

    def get_missing_optional(self) -> List[tuple]:
        return self.missing_optional

env_config = EnvironmentConfig()

# Pydantic models
class UserInfo(BaseModel):
    uid: str
    email: Optional[str] = None
    name: Optional[str] = None
    picture: Optional[str] = None
    user_type: str = "individual"
    phone: Optional[str] = None
    address: Optional[str] = None
    organization_name: Optional[str] = None
    organization_type: Optional[str] = None
    description: Optional[str] = None
    ngo_darpan_id: Optional[str] = None
    pan: Optional[str] = None
    fcra_number: Optional[str] = None
    is_fraudulent: Optional[bool] = None
    fraud_score: Optional[float] = None
    fraud_explanation: Optional[str] = None
    verification_details: Optional[Dict[str, Any]] = None


class LoginRequest(BaseModel):
    id_token: str

class IndividualProfileData(BaseModel):
    full_name: str
    phone: str
    address: str

class OrganizationProfileData(BaseModel):
    contact_full_name: str
    contact_phone: str
    organization_name: str
    organization_type: str
    description: str
    address: str
    ngo_darpan_id: Optional[str] = None
    pan: Optional[str] = None
    fcra_number: Optional[str] = None

class RegisterRequest(BaseModel):
    id_token: str
    user_type: str
    individual_data: Optional[IndividualProfileData] = None
    organization_data: Optional[OrganizationProfileData] = None

class UserProfileUpdateRequest(BaseModel):
    user_type: Optional[str] = None
    email: Optional[str] = None
    name: Optional[str] = None
    picture: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    organization_name: Optional[str] = None
    organization_type: Optional[str] = None
    description: Optional[str] = None
    ngo_darpan_id: Optional[str] = None
    pan: Optional[str] = None
    fcra_number: Optional[str] = None

class CampaignCreateRequest(BaseModel):
    campaign_name: str
    description: str
    goal: float
    category: str
    image_base64: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int = Field(default=3600, description="Expires in seconds")
    refresh_token: Optional[str] = None
    user_info: UserInfo

# Create FastAPI app instance at the very top
app = FastAPI(
    title="HAVEN Backend Service with OAuth and Firebase",
    description="Crowdfunding platform backend with Google/Facebook OAuth and Firebase Authentication/Firestore",
    version="2.0.0"
)

# Add middleware directly after app creation
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Firebase instances - initialized in startup_event
db: Optional[firestore.Client] = None
firebase_auth: Optional[auth.Auth] = None
algolia_client = None
algolia_index = None


# Dependency to get Firestore client
def get_firestore_client() -> firestore.Client:
    if db is None:
        raise HTTPException(status_code=500, detail="Firestore client not initialized.")
    return db

# Dependency to get Firebase Auth client
def get_firebase_auth() -> auth.Auth:
    # FIX: Changed '===' to 'is' for Python comparison
    if firebase_auth is None:
        raise HTTPException(status_code=500, detail="Firebase Auth client not initialized.")
    return firebase_auth

# Dependency to get current user from our custom JWT
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
) -> Dict[str, Any]:
    """Get current user from our custom JWT token"""
    jwt_manager = get_jwt_manager()
    try:
        user_data = jwt_manager.get_user_from_token(credentials.credentials)
        if "id" not in user_data:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User ID not found in token.")
        return user_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get current user from JWT: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )

# --- Firebase initialization function ---
def initialize_firebase():
    """Ultra robust Firebase initialization with multiple fallback strategies"""
    global db, firebase_auth

    logger.info("Starting ultra robust Firebase initialization...")

    firebase_key_file_paths = [
        BASE_DIR / "firebase-service-account-key.json",
        "/app/firebase-service-account-key.json",
        "firebase-service-account-key.json",
        "/opt/render/project/src/firebase-service-account-key.json",
        "./firebase-service-account-key.json"
    ]

    for key_file_path in firebase_key_file_paths:
        try:
            key_path = Path(key_file_path)
            if key_path.exists():
                logger.info(f"Found Firebase service account key file at: {key_file_path}")
                with open(key_path, 'r', encoding='utf-8') as f:
                    json_content = json.load(f)

                required_fields = ['type', 'project_id', 'private_key', 'client_email']
                if all(field in json_content for field in required_fields):
                    cred = credentials.Certificate(str(key_file_path))
                    if not firebase_admin._apps:
                        firebase_admin.initialize_app(cred)
                    db = firestore.client()
                    firebase_auth = auth.get_auth()
                    logger.info("Firebase Admin SDK initialized successfully from file.")
                    return True
                else:
                    logger.warning(f"Firebase key file at {key_file_path} is missing required fields")

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in Firebase key file {key_file_path}: {e}")
            continue
        except Exception as e:
            logger.warning(f"Failed to initialize Firebase from file {key_file_path}: {e}")
            continue

    firebase_key_base64 = env_config.get("FIREBASE_SERVICE_ACCOUNT_KEY_JSON_BASE64")
    if firebase_key_base64:
        try:
            firebase_key_base64 = firebase_key_base64.strip()
            decoded_bytes = base64.b64decode(firebase_key_base64)
            decoded_string = decoded_bytes.decode("utf-8")
            service_account_info = json.loads(decoded_string)

            required_fields = ['type', 'project_id', 'private_key', 'client_email']
            if all(field in service_account_info for field in required_fields):
                cred = credentials.Certificate(service_account_info)
                if not firebase_admin._apps:
                    firebase_admin.initialize_app(cred)
                db = firestore.client()
                firebase_auth = auth.get_auth()
                logger.info("Firebase Admin SDK initialized successfully from environment variable.")
                return True
            else:
                logger.error("Firebase service account info from environment variable is missing required fields")

        except base64.binascii.Error as e:
            logger.error(f"Invalid base64 encoding in FIREBASE_SERVICE_ACCOUNT_KEY_JSON_BASE64: {e}")
        except UnicodeDecodeError as e:
            logger.error(f"Invalid UTF-8 encoding in Firebase service account key: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in Firebase service account key: {e}")
        except Exception as e:
            logger.error(f"Unexpected error with environment variable Firebase key: {e}")

    # Attempt 3: Application Default Credentials (for Google Cloud environments)
    try:
        logger.info("Attempting ApplicationDefault credentials...")
        cred = credentials.ApplicationDefault()
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        firebase_auth = auth.get_auth()
        logger.info("Firebase Admin SDK initialized successfully with ApplicationDefault credentials.")
        return True
    except Exception as e:
        logger.warning(f"ApplicationDefault credentials failed: {e}")

    logger.critical("All Firebase initialization strategies failed. Running in degraded mode without Firebase.")
    db = None
    firebase_auth = None
    return False


@app.on_event("startup")
async def startup_event():
    global db, firebase_auth, algolia_client, algolia_index

    logger.info("Application startup event triggered.")

    if env_config.is_required_missing():
        missing_vars = env_config.get_missing_required()
        error_msg = f"Missing required environment variables: {', '.join([var for var, _ in missing_vars])}"
        logger.critical(error_msg)
        sys.exit(1)

    if env_config.get_missing_optional():
        missing_optional = env_config.get_missing_optional()
        for var, description in missing_optional:
            logger.warning(f"Optional variable {var} not set - {description} may be limited")

    # Initialize Firebase
    logger.info("Attempting Firebase initialization...")
    firebase_initialized = initialize_firebase()
    if not firebase_initialized:
        logger.critical("Firebase not fully initialized. Exiting application.")
        sys.exit(1) # Exit if Firebase initialization fails

    # Initialize OAuth configuration (now uses env_config to get redirect URIs)
    oauth_config = get_oauth_config()
    if oauth_config.is_configured:
        logger.info("OAuth configuration loaded successfully")
        if oauth_config.is_google_configured:
            logger.info("✓ Google OAuth configured")
        if oauth_config.is_facebook_configured:
            logger.info("✓ Facebook OAuth configured")
    else:
        logger.warning("No OAuth providers configured - OAuth functionality will be disabled")

    # Algolia client initialization (moved into startup_event)
    try:
        logger.info("Attempting Algolia client initialization...")
        if ALGOLIA_AVAILABLE:
            algolia_app_id = env_config.get("ALGOLIA_APP_ID")
            algolia_api_key = env_config.get("ALGOLIA_API_KEY")

            if algolia_app_id and algolia_api_key:
                algolia_client = SearchClient(algolia_app_id, algolia_api_key)
                algolia_index = algolia_client.init_index("campaigns")
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

    # Static file setup (MOVED INTO startup_event)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
