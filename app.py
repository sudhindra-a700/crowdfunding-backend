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

from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import urllib.parse

# Firebase Admin SDK imports
import firebase_admin
from firebase_admin import credentials, auth, firestore, messaging

# OAuth imports
from oauth_routes import get_oauth_router
from oauth_config import get_oauth_config
from jwt_utils import get_jwt_manager

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

    if log_format.lower() == "json":
        import json
        import datetime

        class JSONFormatter(logging.Formatter):
            def format(self, record):
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

        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
    else:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        handler.setFormatter(formatter)

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        handlers=[handler],
        force=True
    )

    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)

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
            # OAuth variables
            "GOOGLE_CLIENT_ID": "Google OAuth authentication",
            "GOOGLE_CLIENT_SECRET": "Google OAuth authentication",
            "FACEBOOK_CLIENT_ID": "Facebook OAuth authentication",
            "FACEBOOK_CLIENT_SECRET": "Facebook OAuth authentication",
            "JWT_SECRET_KEY": "JWT token signing",
            "SECRET_KEY": "Session management"
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
class UserLogin(BaseModel):
    id_token: str

class UserInfo(BaseModel):
    uid: str
    email: Optional[str] = None
    role: str = "user"

class Token(BaseModel):
    access_token: str
    token_type: str

class SearchQuery(BaseModel):
    query: str

class LoginRequest(BaseModel):
    email: str
    password: str

class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None
    phone_number: Optional[str] = None
    organization_name: Optional[str] = None
    organization_phone: Optional[str] = None
    organization_type: Optional[str] = None
    brief_description: Optional[str] = None
    type: str  # "individual" or "organization"

# Create FastAPI app
app = FastAPI(
    title="HAVEN Backend Service with OAuth",
    description="Crowdfunding platform backend with Google and Facebook OAuth authentication",
    version="2.0.0"
)

# Add session middleware for OAuth state management
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include OAuth router
app.include_router(get_oauth_router())

BASE_DIR = Path(__file__).resolve().parent

# Global variables
indictrans2_model = None
indictrans2_tokenizer = None
indictrans2_processor = None
DEVICE = "cpu"

db = None
algolia_client = None
algolia_index = None

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/verify-token")

# Enhanced health check with OAuth status
@app.get("/health")
async def health_check():
    try:
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "2.0.0",
            "environment": env_config.get("ENVIRONMENT", "unknown"),
            "services": {},
            "system": {},
            "oauth": {}
        }

        # Check OAuth configuration
        oauth_config = get_oauth_config()
        health_status["oauth"] = {
            "google_configured": oauth_config.is_google_configured,
            "facebook_configured": oauth_config.is_facebook_configured,
            "at_least_one_configured": oauth_config.is_configured
        }

        # Firebase health check
        try:
            if db:
                test_doc = db.collection("health_check").document("test")
                test_doc.set({"timestamp": time.time()}, merge=True)
                health_status["services"]["firebase"] = "connected"
            else:
                health_status["services"]["firebase"] = "not_initialized"
        except Exception as e:
            health_status["services"]["firebase"] = f"error: {str(e)}"
            logger.warning(f"Firebase health check failed: {e}")

        # Algolia health check
        try:
            if algolia_index:
                algolia_index.search("", {"hitsPerPage": 1})
                health_status["services"]["algolia"] = "connected"
            else:
                health_status["services"]["algolia"] = "not_configured"
        except Exception as e:
            health_status["services"]["algolia"] = f"error: {str(e)}"
            logger.warning(f"Algolia health check failed: {e}")

        # System metrics
        try:
            health_status["system"] = {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage("/").percent,
                "uptime_seconds": time.time() - psutil.boot_time()
            }
        except Exception as e:
            logger.warning(f"System metrics collection failed: {e}")
            health_status["system"] = {"error": str(e)}

        # Check for critical issues
        critical_issues = []
        if env_config.is_required_missing():
            critical_issues.extend([f"Missing required env var: {var}" for var, _ in env_config.get_missing_required()])

        if not db:
            critical_issues.append("Firebase not initialized")

        if critical_issues:
            health_status["status"] = "degraded"
            health_status["issues"] = critical_issues
            return health_status, 503

        return health_status

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }, 500

@app.get("/ready")
async def readiness_check():
    try:
        if env_config.is_required_missing():
            return {"ready": False, "reason": "Missing required environment variables"}, 503

        return {"ready": True, "timestamp": time.time(), "firebase_status": "connected" if db else "degraded"}

    except Exception as e:
        logger.error(f"Readiness check failed: {e}", exc_info=True)
        return {"ready": False, "reason": str(e)}, 503

@app.get("/live")
async def liveness_check():
    return {"alive": True, "timestamp": time.time()}

# Authentication functions
async def get_current_user(id_token: str = Depends(oauth2_scheme)):
    if not firebase_admin._apps or not db:
        logger.error("Firebase not initialized in get_current_user.")
        raise HTTPException(status_code=500, detail="Firebase not initialized.")
    try:
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token["uid"]
        email = decoded_token.get("email")
        role = decoded_token.get("role", "user")
        return UserInfo(uid=uid, email=email, role=role)
    except Exception as e:
        logger.error(f"Firebase ID token verification failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_admin_user(current_user: UserInfo = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

# Authentication endpoints (keeping existing for backward compatibility)
@app.post("/login")
async def login_user(request: LoginRequest):
    """Login endpoint for email/password authentication."""
    try:
        if not request.email or not request.password:
            raise HTTPException(status_code=400, detail="Email and password are required")

        # Mock successful login - in production, verify against Firebase Auth
        mock_token = f"mock_token_{request.email}_{int(time.time())}"

        return {
            "access_token": mock_token,
            "token_type": "bearer",
            "role": "user",
            "email": request.email
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /login endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during login.")

@app.post("/register")
async def register_user(request: RegisterRequest):
    """Register endpoint for new user registration."""
    try:
        if not request.email or not request.type:
            raise HTTPException(status_code=400, detail="Email and type are required")

        logger.info(f"Registering new user: {request.email} as {request.type}")

        return {
            "message": "Registration successful",
            "email": request.email,
            "type": request.type,
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /register endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during registration.")

@app.get("/campaigns")
async def get_campaigns():
    """Get all campaigns."""
    try:
        # Mock campaigns data
        mock_campaigns = [
            {
                "id": "1",
                "name": "Sustainable Farming Initiative",
                "description": "Support local farmers in adopting sustainable practices.",
                "author": "Green Earth Foundation",
                "funded": 75000,
                "goal": 100000,
                "days_left": 30,
                "category": "Environment",
                "verification_status": "Verified",
                "image_url": "https://via.placeholder.com/600x400"
            },
            {
                "id": "2",
                "name": "Clean Water Project",
                "description": "Provide access to clean and safe drinking water.",
                "author": "Water for All",
                "funded": 50000,
                "goal": 80000,
                "days_left": 45,
                "category": "Health",
                "verification_status": "Verified",
                "image_url": "https://via.placeholder.com/600x400"
            },
            {
                "id": "3",
                "name": "Education for All",
                "description": "Fund educational resources for underprivileged children.",
                "author": "Education First",
                "funded": 30000,
                "goal": 60000,
                "days_left": 60,
                "category": "Education",
                "verification_status": "Verified",
                "image_url": "https://via.placeholder.com/600x400"
            }
        ]

        return mock_campaigns
    except Exception as e:
        logger.error(f"Error in /campaigns endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error fetching campaigns.")

# Firebase initialization
def initialize_firebase():
    """Ultra robust Firebase initialization with multiple fallback strategies"""
    global db

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

    firebase_key = env_config.get("FIREBASE_SERVICE_ACCOUNT_KEY_JSON_BASE64")
    if firebase_key:
        try:
            firebase_key = firebase_key.strip()
            decoded_bytes = base64.b64decode(firebase_key)
            decoded_string = decoded_bytes.decode("utf-8")
            service_account_info = json.loads(decoded_string)

            required_fields = ['type', 'project_id', 'private_key', 'client_email']
            if all(field in service_account_info for field in required_fields):
                cred = credentials.Certificate(service_account_info)
                if not firebase_admin._apps:
                    firebase_admin.initialize_app(cred)
                db = firestore.client()
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

    try:
        logger.info("Attempting ApplicationDefault credentials...")
        cred = credentials.ApplicationDefault()
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        logger.info("Firebase Admin SDK initialized successfully with ApplicationDefault credentials.")
        return True
    except Exception as e:
        logger.warning(f"ApplicationDefault credentials failed: {e}")

    logger.warning("All Firebase initialization strategies failed. Running in degraded mode.")
    db = None
    return False

@app.on_event("startup")
async def startup_event():
    global db, algolia_client, algolia_index

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

    # Initialize OAuth configuration
    oauth_config = get_oauth_config()
    if oauth_config.is_configured:
        logger.info("OAuth configuration loaded successfully")
        if oauth_config.is_google_configured:
            logger.info("✓ Google OAuth configured")
        if oauth_config.is_facebook_configured:
            logger.info("✓ Facebook OAuth configured")
    else:
        logger.warning("No OAuth providers configured - OAuth functionality will be disabled")

    try:
        firebase_success = initialize_firebase()
        if not firebase_success:
            logger.warning("Firebase initialization failed, but continuing with degraded functionality")
    except Exception as e:
        logger.error(f"Unexpected error during Firebase initialization: {e}", exc_info=True)
        db = None

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

