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

    # Clear existing handlers to prevent re-adding them on reload/rerun
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
                    # Fallback to a simpler format if JSON formatting fails
                    return f"ERROR: Could not format log record to JSON: {e} - Original message: {record.getMessage()}"

            # Override handleError to prevent reentrant calls
            def handleError(self, record):
                """
                Do not call sys.stderr.write directly to avoid reentrant calls.
                Instead, let the default logging system handle it, or just pass.
                """
                pass # Suppress default error handling for this formatter


        handler = logging.StreamHandler(sys.stdout) # Use stdout for JSON logs
        handler.setFormatter(JSONFormatter())
    else:
        handler = logging.StreamHandler(sys.stdout) # Use stdout for standard logs
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        handler.setFormatter(formatter)

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        handlers=[handler],
        force=True, # Ensure configuration is applied
        disable_existing_loggers=False # Do not disable existing loggers (like Gunicorn's)
    )

    # Set specific log levels for common libraries to reduce verbosity
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING) # Reduce noise from HTTP requests
    logging.getLogger("httpcore").setLevel(logging.WARNING) # Reduce noise from HTTP requests
    logging.getLogger("firebase_admin").setLevel(logging.INFO) # Keep Firebase info
    logging.getLogger("authlib").setLevel(logging.INFO) # Keep Authlib info

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
            "SECRET_KEY": "Session management",
            "GOOGLE_REDIRECT_URI": "Google OAuth redirect URI", # Added
            "FACEBOOK_REDIRECT_URI": "Facebook OAuth redirect URI", # Added
            "FRONTEND_BASE_URL": "Frontend base URL for redirects" # Added
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
    # Fields for NGO fraud data
    ngo_darpan_id: Optional[str] = None
    pan: Optional[str] = None
    fcra_number: Optional[str] = None
    is_fraudulent: Optional[bool] = None
    fraud_score: Optional[float] = None
    fraud_explanation: Optional[str] = None
    verification_details: Optional[Dict[str, Any]] = None


class LoginRequest(BaseModel):
    id_token: str # Firebase ID token from client-side authentication

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

# New Pydantic models for RegisterRequest and UserProfileUpdateRequest
class RegisterRequest(BaseModel):
    id_token: str
    user_type: str
    individual_data: Optional[IndividualProfileData] = None
    organization_data: Optional[OrganizationProfileData] = None

class UserProfileUpdateRequest(BaseModel):
    # Allow partial updates by making all fields optional
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
    image_base64: Optional[str] = None # Base64 encoded image string

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int = Field(default=3600, description="Expires in seconds") # Default 1 hour
    refresh_token: Optional[str] = None
    user_info: UserInfo # Include user info directly

# Create FastAPI app
app = FastAPI(
    title="HAVEN Backend Service with OAuth and Firebase",
    description="Crowdfunding platform backend with Google/Facebook OAuth and Firebase Authentication/Firestore",
    version="2.0.0"
)

# Add session middleware for OAuth state management
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", secrets.token_urlsafe(32)) # Use env var, fallback to generated
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Firebase instances
db: Optional[firestore.Client] = None
firebase_auth: Optional[auth.Auth] = None

# Include OAuth router
# Ensure oauth_routes has access to db and firebase_auth
oauth_router = get_oauth_router()
app.include_router(oauth_router)

BASE_DIR = Path(__file__).resolve().parent

# Global variables for Algolia, etc.
algolia_client = None
algolia_index = None

# Dependency to get Firestore client
def get_firestore_client() -> firestore.Client:
    if db is None:
        raise HTTPException(status_code=500, detail="Firestore client not initialized.")
    return db

# Dependency to get Firebase Auth client
def get_firebase_auth() -> auth.Auth:
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
        # Ensure the UID is present in the user_data from our JWT
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
            if db and firebase_auth:
                # Test Firestore connection
                test_doc_ref = db.collection("health_check").document("test")
                test_doc_ref.set({"timestamp": time.time()}, merge=True)
                # Test Firebase Auth (e.g., get a dummy user, though not ideal for health check)
                # For simplicity, we'll just check if the client objects exist
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

        # If Firebase is not initialized, it's a critical issue for this app
        if not db or not firebase_auth:
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

        # Check Firebase initialization
        if not db or not firebase_auth:
            return {"ready": False, "reason": "Firebase not initialized"}, 503

        return {"ready": True, "timestamp": time.time(), "firebase_status": "connected"}

    except Exception as e:
        logger.error(f"Readiness check failed: {e}", exc_info=True)
        return {"ready": False, "reason": str(e)}, 503

@app.get("/live")
async def liveness_check():
    return {"alive": True, "timestamp": time.time()}

# --- Authentication and Registration Endpoints ---

@app.post("/login", response_model=Token)
async def login_user(
    request: LoginRequest,
    firestore_db: firestore.Client = Depends(get_firestore_client),
    firebase_auth_client: auth.Auth = Depends(get_firebase_auth)
):
    """
    Login endpoint for Firebase ID token authentication.
    Expects Firebase ID token from client-side authentication.
    """
    try:
        # Verify the Firebase ID token
        decoded_token = firebase_auth_client.verify_id_token(request.id_token)
        uid = decoded_token["uid"]
        email = decoded_token.get("email")

        # Fetch user data from Firestore
        user_doc_ref = firestore_db.collection("users").document(uid)
        user_doc = user_doc_ref.get()

        if not user_doc.exists:
            logger.warning(f"User profile not found in Firestore for UID: {uid}. Email: {email}")
            # This case might happen if user registered via Firebase Auth but profile creation failed.
            # Or if it's a new OAuth user completing profile.
            # For now, we'll return basic info and let frontend guide to profile completion.
            user_data = {
                "uid": uid,
                "email": email,
                "name": decoded_token.get("name", email),
                "picture": decoded_token.get("picture"),
                "user_type": "individual" # Default to individual if no profile
            }
            # Create a basic profile if it doesn't exist, to avoid breaking frontend
            user_doc_ref.set(user_data, merge=True)
            logger.info(f"Created basic Firestore profile for new login UID: {uid}")
        else:
            user_data = user_doc.to_dict()
            user_data['uid'] = uid # Ensure uid is in the dict

        # Create our custom JWT
        jwt_manager = get_jwt_manager()
        access_token = jwt_manager.create_access_token(user_data)
        refresh_token = jwt_manager.create_refresh_token(uid) # Use UID for refresh token subject

        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=jwt_manager.expiration_hours * 3600,
            refresh_token=refresh_token,
            user_info=UserInfo(**user_data)
        )

    except auth.InvalidIdTokenError as e:
        logger.error(f"Invalid Firebase ID token: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Firebase ID token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Error during login: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during login.")


@app.post("/register", response_model=Token)
async def register_user(
    request: RegisterRequest,
    firestore_db: firestore.Client = Depends(get_firestore_client),
    firebase_auth_client: auth.Auth = Depends(get_firebase_auth)
):
    """
    Register endpoint for new user registration.
    Expects Firebase ID token from client-side authentication
    and additional profile data.
    """
    try:
        # Verify the Firebase ID token
        decoded_token = firebase_auth_client.verify_id_token(request.id_token)
        uid = decoded_token["uid"]
        email = decoded_token.get("email")
        name = decoded_token.get("name")

        # Prepare user data for Firestore
        user_data = {
            "uid": uid,
            "email": email,
            "name": name,
            "user_type": request.user_type,
            "registered_at": firestore.SERVER_TIMESTAMP
        }

        if request.user_type == "individual" and request.individual_data:
            user_data.update({
                "name": request.individual_data.full_name, # Use provided full name if available
                "phone": request.individual_data.phone,
                "address": request.individual_data.address
            })
        elif request.user_type == "organization" and request.organization_data:
            user_data.update({
                "contact_full_name": request.organization_data.contact_full_name,
                "contact_phone": request.organization_data.contact_phone,
                "organization_name": request.organization_data.organization_name,
                "organization_type": request.organization_data.organization_type,
                "description": request.organization_data.description,
                "address": request.organization_data.address,
                "ngo_darpan_id": request.organization_data.ngo_darpan_id,
                "pan": request.organization_data.pan,
                "fcra_number": request.organization_data.fcra_number
            })

            # Integrate NGO fraud data if organization
            org_data_for_fraud_check = {
                'org_name': request.organization_data.organization_name,
                'bio': request.organization_data.description,
                'pan': request.organization_data.pan,
                'ngo_darpan_id': request.organization_data.ngo_darpan_id,
                'fcra_number': request.organization_data.fcra_number,
                # Add other relevant fields for fraud detection if available
                'recent_posts': '', # Placeholder
                'follower_count': 0, # Placeholder
                'post_count': 0, # Placeholder
                'account_age_days': 0, # Placeholder
                'engagement_rate': 0.0 # Placeholder
            }
            try:
                # Path to NGO Darpan CSV
                ngo_darpan_csv_path = BASE_DIR / "DEhli.csv"
                # Load NGO Darpan data (will be cached)
                load_ngo_darpan_data(str(ngo_darpan_csv_path))

                fraud_score, explanation, plot_path, verification_details = predict_fraud(
                    org_data_for_fraud_check, api_key_trustcheckr=os.getenv("TRUSTCHECKR_API_KEY", "mock_key")
                )
                user_data["is_fraudulent"] = fraud_score > 0.5 # Simple threshold
                user_data["fraud_score"] = fraud_score
                user_data["fraud_explanation"] = explanation
                user_data["verification_details"] = verification_details # Store full verification details
                logger.info(f"Fraud detection run for new organization {uid}. Score: {fraud_score:.2f}")

            except Exception as e:
                logger.error(f"Fraud detection failed for organization {uid}: {e}", exc_info=True)
                user_data["is_fraudulent"] = None
                user_data["fraud_score"] = None
                user_data["fraud_explanation"] = "Fraud detection service unavailable."
                user_data["verification_details"] = {"error": "Fraud detection failed."}

        # Store user profile in Firestore
        user_doc_ref = firestore_db.collection("users").document(uid)
        user_doc_ref.set(user_data, merge=True) # Use merge to avoid overwriting existing fields if any

        # Create our custom JWT
        jwt_manager = get_jwt_manager()
        access_token = jwt_manager.create_access_token(user_data)
        refresh_token = jwt_manager.create_refresh_token(uid)

        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=jwt_manager.expiration_hours * 3600,
            refresh_token=refresh_token,
            user_info=UserInfo(**user_data)
        )

    except auth.InvalidIdTokenError as e:
        logger.error(f"Invalid Firebase ID token during registration: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Firebase ID token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except auth.UserNotFoundError:
        logger.error(f"Firebase user not found for provided ID token during registration.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Firebase user not found. Please ensure you've signed up with Firebase Auth first."
        )
    except Exception as e:
        logger.error(f"Error during registration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during registration.")

@app.post("/update_profile", response_model=UserInfo)
async def update_user_profile(
    request: UserProfileUpdateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user), # Our custom JWT provides user info
    firestore_db: firestore.Client = Depends(get_firestore_client)
):
    """Update user profile in Firestore."""
    uid = current_user.get("id") # 'id' field in our custom JWT payload is the Firebase UID
    if not uid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User ID not found in token.")

    user_doc_ref = firestore_db.collection("users").document(uid)
    user_doc = user_doc_ref.get()

    if not user_doc.exists:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User profile not found.")

    update_data = request.dict(exclude_unset=True) # Get only fields that were explicitly set in the request

    # Special handling for organization data if user_type is organization
    if request.user_type == "organization":
        org_data_for_fraud_check = {
            'org_name': update_data.get('organization_name', current_user.get('organization_name')),
            'bio': update_data.get('description', current_user.get('description')),
            'pan': update_data.get('pan', current_user.get('pan')),
            'ngo_darpan_id': update_data.get('ngo_darpan_id', current_user.get('ngo_darpan_id')),
            'fcra_number': update_data.get('fcra_number', current_user.get('fcra_number')),
            'recent_posts': '', # Placeholder
            'follower_count': 0, # Placeholder
            'post_count': 0, # Placeholder
            'account_age_days': 0, # Placeholder
            'engagement_rate': 0.0 # Placeholder
        }
        try:
            ngo_darpan_csv_path = BASE_DIR / "DEhli.csv"
            load_ngo_darpan_data(ngo_darpan_csv_path) # Ensure data is loaded
            fraud_score, explanation, plot_path, verification_details = predict_fraud(
                org_data_for_fraud_check, api_key_trustcheckr=os.getenv("TRUSTCHECKR_API_KEY", "mock_key")
            )
            update_data["is_fraudulent"] = fraud_score > 0.5
            update_data["fraud_score"] = fraud_score
            update_data["fraud_explanation"] = explanation
            update_data["verification_details"] = verification_details
            logger.info(f"Fraud detection re-run for organization {uid} during profile update. Score: {fraud_score:.2f}")
        except Exception as e:
            logger.error(f"Fraud detection failed during profile update for organization {uid}: {e}", exc_info=True)
            update_data["is_fraudulent"] = None
            update_data["fraud_score"] = None
            update_data["fraud_explanation"] = "Fraud detection service unavailable."
            update_data["verification_details"] = {"error": "Fraud detection failed."}

    try:
        user_doc_ref.update(update_data)
        updated_user_doc = user_doc_ref.get()
        return UserInfo(**updated_user_doc.to_dict())
    except Exception as e:
        logger.error(f"Error updating user profile for UID {uid}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update user profile.")


@app.post("/create_campaign")
async def create_campaign(
    request: CampaignCreateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    firestore_db: firestore.Client = Depends(get_firestore_client)
):
    """Create a new campaign."""
    user_uid = current_user.get("id")
    user_type = current_user.get("user_type")

    if user_type != "organization":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only organization accounts can create campaigns."
        )

    # Fetch organization name to use as author
    organization_name = current_user.get("organization_name", "N/A")
    if organization_name == "N/A":
        # Try to fetch from Firestore if not in current_user (e.g., if token is old)
        org_doc = firestore_db.collection("users").document(user_uid).get()
        if org_doc.exists:
            organization_name = org_doc.to_dict().get("organization_name", "N/A")

    campaign_data = {
        "user_uid": user_uid,
        "campaign_name": request.campaign_name,
        "description": request.description,
        "goal": request.goal,
        "funded": 0.0, # Start with 0 funded
        "category": request.category,
        "image_base64": request.image_base64, # Store base64 for now, ideally upload to storage
        "author": organization_name,
        "created_at": firestore.SERVER_TIMESTAMP,
        "last_updated": firestore.SERVER_TIMESTAMP,
        "status": "active",
        "days_left": 60, # Default days left for new campaigns
        "verification_status": "Pending" # New campaigns are pending verification
    }

    try:
        doc_ref = await firestore_db.collection("campaigns").add(campaign_data)
        logger.info(f"Campaign '{request.campaign_name}' created by {user_uid} with ID: {doc_ref.id}")
        return {"message": "Campaign created successfully!", "campaign_id": doc_ref.id}
    except Exception as e:
        logger.error(f"Error creating campaign for user {user_uid}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create campaign.")


@app.get("/campaigns")
async def get_campaigns(firestore_db: firestore.Client = Depends(get_firestore_client)):
    """Get all campaigns from Firestore."""
    try:
        campaigns_ref = firestore_db.collection("campaigns")
        docs = campaigns_ref.stream()
        campaigns_list = []
        for doc in docs:
            campaign_data = doc.to_dict()
            campaign_data["id"] = doc.id
            # Generate a placeholder image URL if image_base664 is not present
            if 'image_base64' in campaign_data and campaign_data['image_base64']:
                # In a real app, you'd upload this to a storage service (e.g., Firebase Storage)
                # and store the URL. For now, we'll use a data URL for direct display.
                # However, data URLs can be very long and are not ideal for large images or production.
                # For this demo, we'll just return a placeholder.
                campaign_data['image_url'] = "https://placehold.co/600x400/4CAF50/ffffff?text=Campaign+Image"
            else:
                campaign_data['image_url'] = "https://placehold.co/600x400/4CAF50/ffffff?text=No+Image"

            campaigns_list.append(campaign_data)
        return campaigns_list
    except Exception as e:
        logger.error(f"Error in /campaigns endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error fetching campaigns.")


# Firebase initialization
def initialize_firebase():
    """Ultra robust Firebase initialization with multiple fallback strategies"""
    global db, firebase_auth

    logger.info("Starting ultra robust Firebase initialization...")

    # Attempt 1: From service account JSON file
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

    # Attempt 2: From base64 encoded environment variable
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

    # Removed explicit loading of fraud detection model and NGO Darpan data from startup.
    # These are now expected to be loaded on demand by the fraud_detection module
    # when predict_fraud is called. This reduces startup memory footprint.
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
