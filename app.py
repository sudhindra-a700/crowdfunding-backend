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
# import csv # Removed csv module as we are moving to Firestore

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, model_validator  # Import model_validator

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
    user_type: Optional[str] = None  # Added user_type to UserInfo


class Token(BaseModel):
    access_token: str
    token_type: str


class SearchQuery(BaseModel):
    query: str


class LoginRequest(BaseModel):
    email: str
    password: str


# --- New Pydantic Models for Registration and Profile Update ---

class IndividualDetails(BaseModel):
    full_name: str
    phone: str
    address: str


class OrganizationContactPersonDetails(BaseModel):
    contact_full_name: str
    contact_phone: str


class OrganizationDetails(BaseModel):
    organization_name: str
    organization_type: str
    description: str
    address: str


class RegisterRequest(BaseModel):
    email: str

    # Password is now optional for OAuth registrations where email/password might not be set directly
    # However, for direct email/password registration, it should be required.
    # The frontend should ensure it's provided when necessary.
    password: Optional[str] = None

    user_type: str = Field(..., pattern="^(individual|organization)$")  # Enforce type
    individual_data: Optional[IndividualDetails] = None
    organization_contact_data: Optional[OrganizationContactPersonDetails] = None
    organization_data: Optional[OrganizationDetails] = None

    # Validator to ensure correct data based on user_type
    @model_validator(mode='after')
    def validate_data_based_on_type(self):
        if self.user_type == "individual":
            if not self.individual_data:
                raise ValueError("individual_data is required for individual user_type")
            if self.organization_contact_data or self.organization_data:
                raise ValueError("organization data should not be provided for individual user_type")
        elif self.user_type == "organization":
            if not self.organization_contact_data or not self.organization_data:
                raise ValueError(
                    "organization_contact_data and organization_data are required for organization user_type")
            if self.individual_data:
                raise ValueError("individual_data should not be provided for organization user_type")
        return self


class UpdateProfileRequest(BaseModel):
    user_type: str = Field(..., pattern="^(individual|organization)$")
    # For individual updates
    full_name: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    # For organization updates
    contact_full_name: Optional[str] = None
    contact_phone: Optional[str] = None
    organization_name: Optional[str] = None
    organization_type: Optional[str] = None
    description: Optional[str] = None
    # Note: Email is not updated via this endpoint as it's the primary identifier


class CreateCampaignRequest(BaseModel):
    campaign_name: str
    description: str
    goal: float
    category: str
    image_base64: Optional[str] = None  # Base64 encoded image string


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
    # In a real application, you'd verify the token against Firebase Auth or your JWT secret
    # For now, we'll mock the user info based on the token content (if it's a mock token)
    # or rely on the JWTManager for OAuth tokens.
    jwt_manager = get_jwt_manager()
    try:
        user_data = jwt_manager.get_user_from_token(id_token)
        # If the token is from OAuth, user_data will have provider info.
        # If it's a mock login token, we'll need to infer user_type

        # Mock user_type for demonstration if not from OAuth
        if user_data.get("provider") not in ["google", "facebook"]:
            # For simplicity, let's assume mock users are individuals unless specified
            # In a real app, you'd fetch this from your DB based on user_data.get("id") or email

            # Attempt to retrieve user_type from Firestore if available
            if db:
                user_doc = await db.collection("users").document(user_data.get("email")).get()
                if user_doc.exists:
                    user_data["user_type"] = user_doc.to_dict().get("user_type", "individual")
                else:
                    user_data["user_type"] = "individual"  # Default if not found
            else:
                user_data["user_type"] = "individual"  # Default if Firestore not initialized
                if "org" in user_data.get("email", ""):  # Simple heuristic for testing
                    user_data["user_type"] = "organization"

        return UserInfo(
            uid=user_data.get("id"),
            email=user_data.get("email"),
            role="user",  # Default role
            user_type=user_data.get("user_type", "individual")  # Pass user_type
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting current user from token: {e}", exc_info=True)
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

        # In a real application, you would verify the email/password against Firebase Auth
        # and then fetch the user's full profile from Firestore or your database.
        # For now, we'll simulate a login and return a mock token and user info.

        simulated_user_type = "individual"
        mock_user_name = request.email.split('@')[0].capitalize()

        if db:
            user_doc = await db.collection("users").document(request.email).get()
            if user_doc.exists:
                user_data_from_db = user_doc.to_dict()
                simulated_user_type = user_data_from_db.get("user_type", "individual")
                if simulated_user_type == "individual":
                    mock_user_name = user_data_from_db.get("full_name", mock_user_name)
                elif simulated_user_type == "organization":
                    mock_user_name = user_data_from_db.get("contact_full_name", mock_user_name)
            else:
                # If user not found in DB, default to individual for mock
                simulated_user_type = "individual"
        else:
            # Fallback for when Firebase is not initialized
            if "org" in request.email.lower():  # Simple heuristic for testing organization login
                simulated_user_type = "organization"

        mock_token = f"mock_token_{request.email}_{int(time.time())}"

        mock_user_info = {
            "id": f"mock_user_{request.email}",
            "email": request.email,
            "name": mock_user_name,
            "provider": "email/password",
            "provider_id": "N/A",
            "user_type": simulated_user_type  # Include user_type here
        }

        return {
            "access_token": mock_token,
            "token_type": "bearer",
            "user_info": mock_user_info  # Include user_info in login response
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
        if not db:
            raise HTTPException(status_code=503, detail="Database service not available.")

        logger.info(f"Registering new user: {request.email} as {request.user_type}")

        user_ref = db.collection("users").document(request.email)

        # Check if user already exists
        existing_user = await user_ref.get()
        if existing_user.exists:
            raise HTTPException(status_code=409, detail="User with this email already exists.")

        user_data_to_save = {
            "email": request.email,
            "password_hash": request.password,  # In a real app, hash and store password securely
            "user_type": request.user_type,
            "created_at": firestore.SERVER_TIMESTAMP
        }

        if request.user_type == "individual":
            user_data_to_save.update(request.individual_data.model_dump())

        elif request.user_type == "organization":
            user_data_to_save.update(request.organization_contact_data.model_dump())
            user_data_to_save.update(request.organization_data.model_dump())

        await user_ref.set(user_data_to_save)
        logger.info(f"User {request.email} registered and saved to Firestore as {request.user_type}.")

        return {
            "message": "Registration successful",
            "email": request.email,
            "user_type": request.user_type,
            "status": "success"
        }
    except ValueError as e:
        logger.error(f"Validation error in /register endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail=str(e))  # Unprocessable Entity for validation errors
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /register endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during registration.")


@app.post("/update_profile")
async def update_profile(request: UpdateProfileRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Update user profile with additional details."""
    try:
        if not db:
            raise HTTPException(status_code=503, detail="Database service not available.")

        user_id = current_user.get("id")
        user_email = current_user.get("email")
        logger.info(f"Updating profile for user: {user_id} ({user_email}) with type: {request.user_type}")

        user_ref = db.collection("users").document(user_email)

        update_data = request.model_dump(exclude_unset=True)  # Only include fields that were set in the request
        # Remove user_type from update_data as it's not meant to be changed directly via update
        update_data.pop('user_type', None)

        if not update_data:
            raise HTTPException(status_code=400, detail="No data provided for update.")

        await user_ref.update(update_data)
        logger.info(f"User {user_email} profile updated in Firestore with: {update_data}")

        return {
            "message": "Profile updated successfully",
            "user_id": user_id,
            "updated_fields": update_data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /update_profile endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during profile update.")


@app.post("/create_campaign")
async def create_campaign(request: CreateCampaignRequest, current_user: UserInfo = Depends(get_current_user)):
    """Endpoint to create a new campaign."""
    try:
        if not db:
            raise HTTPException(status_code=503, detail="Database service not available.")

        if current_user.user_type != "organization":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organization users can create campaigns."
            )

        logger.info(f"Organization user {current_user.email} is creating a campaign.")

        campaign_data = {
            "organization_email": current_user.email,
            "campaign_name": request.campaign_name,
            "description": request.description,
            "goal": request.goal,
            "category": request.category,
            "funded": 0,
            "status": "pending",
            "created_at": firestore.SERVER_TIMESTAMP
        }

        campaign_image_url = None
        if request.image_base64:
            # In a real application, you would save this image to cloud storage (e.g., Google Cloud Storage, AWS S3)
            # and store the URL in your database. For this example, we'll just acknowledge its presence.
            logger.info(f"Received image for campaign (size: {len(request.image_base64)} bytes). "
                        "Saving image to cloud storage is recommended for production.")
            # Placeholder for image URL, in a real scenario this would be the URL from cloud storage
            campaign_image_url = f"https://placehold.co/600x400/000000/FFFFFF?text={request.campaign_name.replace(' ', '+')}"

        campaign_data["image_url"] = campaign_image_url

        campaign_ref = db.collection("campaigns").document()  # Auto-generate document ID
        await campaign_ref.set(campaign_data)

        logger.info(f"Campaign '{request.campaign_name}' created by {current_user.email} and saved to Firestore.")

        return {
            "message": "Campaign created successfully!",
            "campaign_id": campaign_ref.id,
            "campaign_name": request.campaign_name,
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating campaign: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during campaign creation.")


@app.get("/campaigns")
async def get_campaigns():
    """Get all campaigns."""
    try:
        if not db:
            return []  # Return empty list if DB not available

        campaigns_ref = db.collection("campaigns")
        docs = await campaigns_ref.stream()

        campaign_list = []
        # Use a list comprehension to await all documents
        # Note: 'await' inside a list comprehension requires Python 3.9+ or specific async libraries.
        # For broader compatibility, an explicit loop is safer.
        for doc in docs:
            campaign_data = doc.to_dict()
            campaign_list.append({
                "id": doc.id,
                "name": campaign_data.get("campaign_name", "N/A"),
                "description": campaign_data.get("description", "N/A"),
                "author": campaign_data.get("organization_email", "N/A"),  # Assuming author is the organization email
                "funded": campaign_data.get("funded", 0),
                "goal": campaign_data.get("goal", 1),  # Avoid division by zero
                # Calculate days_left based on created_at (if available) or mock it
                "days_left": round((campaign_data.get("created_at").timestamp() + (30 * 24 * 3600) - time.time()) / (
                            24 * 3600)) if campaign_data.get("created_at") else 30,
                "category": campaign_data.get("category", "N/A"),
                "verification_status": campaign_data.get("status", "pending"),
                "image_url": campaign_data.get("image_url", "https://via.placeholder.com/600x400")
            })

        # Add mock campaigns if no real campaigns or for demonstration
        if not campaign_list:
            mock_campaigns = [
                {
                    "id": "mock1",
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
                    "id": "mock2",
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
                    "id": "mock3",
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
            campaign_list.extend(mock_campaigns)

        return campaign_list
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

