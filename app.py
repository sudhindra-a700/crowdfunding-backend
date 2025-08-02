# Corrected app.py with all fixes for deployment issues
# This version fixes Firebase imports, logging configuration, cache directory issues, and adds SessionMiddleware

import os
import sys

# CRITICAL: Configure cache directories FIRST, before any ML library imports
def configure_cache_directories():
    """
    Configure cache directories for containerized environment
    This must be called before importing any ML libraries
    """
    # Set Matplotlib cache directory
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
    
    # Set HuggingFace cache directories
    os.environ.setdefault('TRANSFORMERS_CACHE', '/tmp/huggingface')
    os.environ.setdefault('HF_HOME', '/tmp/huggingface')
    os.environ.setdefault('HUGGINGFACE_HUB_CACHE', '/tmp/huggingface')
    
    # Set general cache directory
    os.environ.setdefault('XDG_CACHE_HOME', '/tmp/cache')
    
    # Create directories if they don't exist
    cache_dirs = ['/tmp/matplotlib', '/tmp/huggingface', '/tmp/cache']
    for cache_dir in cache_dirs:
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create cache directory {cache_dir}: {e}")

# Call cache configuration BEFORE any other imports
configure_cache_directories()

# Now import other libraries
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware  # ✅ ADDED: SessionMiddleware import
import uvicorn
import json
import logging
from typing import Optional, Dict, Any

# Firebase Admin SDK imports - CORRECTED VERSION
import firebase_admin
from firebase_admin import credentials, auth, firestore, messaging

# CORRECTED: Use proper exception imports from firebase_admin.exceptions
from firebase_admin.exceptions import (
    FirebaseError,           # Base exception class
    InvalidArgumentError,    # For invalid arguments
    NotFoundError,          # For missing resources
    PermissionDeniedError,  # For permission issues
    UnauthenticatedError,   # For authentication failures
    AlreadyExistsError,     # For duplicate resources
    InternalError,          # For internal server errors
    FailedPreconditionError, # For state validation errors
    ResourceExhaustedError  # For rate limiting
)

# FIXED: Logging setup function
def setup_logging():
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=log_format,
        stream=sys.stdout,
        force=True
    )

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

# Safe Firebase functions with comprehensive error handling
def get_user_safely(uid: str):
    """Get user by UID with proper error handling"""
    try:
        user_record = auth.get_user(uid)
        return user_record
    except NotFoundError:
        logger.warning(f"User not found: {uid}")
        return None
    except InvalidArgumentError as e:
        logger.error(f"Invalid UID format: {uid}, error: {e}")
        return None
    except FirebaseError as e:
        logger.error(f"Firebase error getting user {uid}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting user {uid}: {e}")
        return None

def create_user_safely(email: str, password: str, display_name: str = None):
    """Create user with proper error handling"""
    try:
        user_record = auth.create_user(
            email=email,
            password=password,
            display_name=display_name
        )
        logger.info(f"Successfully created user: {user_record.uid}")
        return user_record
    except AlreadyExistsError:
        logger.warning(f"User already exists: {email}")
        return None
    except InvalidArgumentError as e:
        logger.error(f"Invalid user data: {e}")
        return None
    except FirebaseError as e:
        logger.error(f"Firebase error creating user: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error creating user: {e}")
        return None

def send_notification_safely(token: str, title: str, body: str):
    """Send notification with proper error handling"""
    try:
        message = messaging.Message(
            notification=messaging.Notification(title=title, body=body),
            token=token
        )
        response = messaging.send(message)
        logger.info(f"Successfully sent message: {response}")
        return response
    except InvalidArgumentError as e:
        logger.error(f"Invalid message data: {e}")
        return None
    except FirebaseError as e:
        logger.error(f"Firebase error sending notification: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error sending notification: {e}")
        return None

def initialize_firebase():
    """Initialize Firebase with multiple credential sources and comprehensive error handling"""
    try:
        # Check if Firebase is already initialized
        try:
            firebase_admin.get_app()
            logger.info("✅ Firebase Admin SDK already initialized")
            return firestore.client()
        except ValueError:
            # Firebase not initialized, proceed with initialization
            pass
        
        # Method 1: Environment variables (recommended for production)
        if all(os.getenv(key) for key in ['FIREBASE_PROJECT_ID', 'FIREBASE_PRIVATE_KEY', 'FIREBASE_CLIENT_EMAIL']):
            logger.info("Initializing Firebase with environment variables...")
            
            # Get credentials from environment variables
            project_id = os.getenv('FIREBASE_PROJECT_ID')
            private_key = os.getenv('FIREBASE_PRIVATE_KEY').replace('\\n', '\n')
            client_email = os.getenv('FIREBASE_CLIENT_EMAIL')
            
            # Create credentials dictionary
            cred_dict = {
                "type": "service_account",
                "project_id": project_id,
                "private_key_id": os.getenv('FIREBASE_PRIVATE_KEY_ID', ''),
                "private_key": private_key,
                "client_email": client_email,
                "client_id": os.getenv('FIREBASE_CLIENT_ID', ''),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{client_email}"
            }
            
            # Validate required fields
            required_fields = ['project_id', 'private_key', 'client_email']
            missing_fields = [field for field in required_fields if not cred_dict.get(field)]
            if missing_fields:
                raise ValueError(f"Missing required Firebase credentials: {missing_fields}")
            
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            logger.info("✅ Firebase Admin SDK initialized successfully with environment variables")
        
        # Method 2: Base64 encoded credentials
        elif os.getenv('FIREBASE_CREDENTIALS_BASE64'):
            logger.info("Initializing Firebase with base64 credentials...")
            import base64
            
            encoded_creds = os.getenv('FIREBASE_CREDENTIALS_BASE64')
            decoded_creds = base64.b64decode(encoded_creds).decode('utf-8')
            cred_dict = json.loads(decoded_creds)
            
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            logger.info("✅ Firebase Admin SDK initialized successfully with base64 credentials")
        
        # Method 3: Service account key from environment variable (JSON string)
        elif os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY'):
            logger.info("Initializing Firebase with service account key...")
            
            service_account_info = json.loads(os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY'))
            cred = credentials.Certificate(service_account_info)
            firebase_admin.initialize_app(cred)
            logger.info("✅ Firebase Admin SDK initialized successfully with service account key")
        
        # Method 4: Local file (Development only)
        elif os.path.exists('serviceAccountKey.json'):
            logger.info("Initializing Firebase with local service account file...")
            
            cred = credentials.Certificate('serviceAccountKey.json')
            firebase_admin.initialize_app(cred)
            logger.info("✅ Firebase Admin SDK initialized successfully with local file")
        
        else:
            # No credentials found - provide helpful error message
            error_msg = """
❌ No Firebase credentials found. Please configure one of the following:

Option 1 (Recommended): Individual environment variables
- FIREBASE_PROJECT_ID
- FIREBASE_PRIVATE_KEY_ID
- FIREBASE_PRIVATE_KEY
- FIREBASE_CLIENT_EMAIL
- FIREBASE_CLIENT_ID
- FIREBASE_CLIENT_X509_CERT_URL

Option 2: Base64 encoded credentials
- FIREBASE_CREDENTIALS_BASE64

Option 3: JSON string
- FIREBASE_SERVICE_ACCOUNT_KEY

Option 4: Local file (development only)
- serviceAccountKey.json in project root
"""
            logger.error(error_msg)
            raise ValueError("No Firebase credentials configured. See logs for configuration options.")
        
        return firestore.client()
    
    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to parse Firebase credentials JSON: {str(e)}")
        raise ValueError(f"Invalid Firebase credentials format: {str(e)}")
    
    except FirebaseError as e:
        logger.error(f"❌ Firebase initialization failed: {e.code} - {e.message}")
        raise e
    
    except Exception as e:
        logger.error(f"❌ Unexpected error initializing Firebase: {str(e)}")
        raise e

# Call initialization
initialize_firebase()

# Your FastAPI app setup continues here...
app = FastAPI(title="Crowdfunding Backend", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ ADDED: SessionMiddleware for OAuth state management
app.add_middleware(
    SessionMiddleware, 
    secret_key=os.getenv("SESSION_SECRET_KEY", "UpLvYLpd2ZdXEVyiGSQmXS9DMQVGLB2dLvc6twgyJX8"),
    max_age=3600,  # Session timeout in seconds (1 hour)
    same_site="lax",  # CSRF protection
    https_only=False  # Set to True in production with HTTPS (Render handles HTTPS termination)
)

# Import and include OAuth router
from oauth_routes import oauth_router
app.include_router(oauth_router)

# Health check endpoint with Firebase connectivity test
@app.get("/health")
async def health_check():
    """
    Health check endpoint that verifies Firebase connectivity
    """
    try:
        # Test Firebase connectivity by attempting to get a non-existent user
        # This will throw NotFoundError if Firebase is working correctly
        auth.get_user('test-connectivity-check')
        return {"status": "healthy", "firebase": "connected"}
    except NotFoundError:
        # This is expected - Firebase is working correctly
        return {"status": "healthy", "firebase": "connected"}
    except FirebaseError as e:
        return {
            "status": "unhealthy",
            "firebase": f"error: {e.message}",
            "error_code": e.code
        }

# Example API endpoints using the safe Firebase functions
@app.get("/user/{uid}")
async def get_user(uid: str):
    """
    Get user by UID with proper error handling
    """
    user_data = get_user_safely(uid)
    if user_data:
        return user_data
    else:
        raise HTTPException(status_code=404, detail="User not found")

@app.post("/user")
async def create_user(email: str, password: str, display_name: str = None):
    """
    Create a new user with proper error handling
    """
    user_data = create_user_safely(email, password, display_name)
    if user_data:
        return {"uid": user_data.uid, "email": user_data.email}
    else:
        raise HTTPException(status_code=400, detail="Failed to create user")

@app.post("/notification")
async def send_notification(token: str, title: str, body: str):
    """
    Send a push notification with proper error handling
    """
    result = send_notification_safely(token, title, body)
    if result:
        return {"message": "Notification sent successfully", "message_id": result}
    else:
        raise HTTPException(status_code=400, detail="Failed to send notification")

# Global exception handler for Firebase errors
@app.exception_handler(FirebaseError)
async def firebase_exception_handler(request: Request, exc: FirebaseError):
    logger.error(f"Firebase error: {exc.code} - {exc.message}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Firebase error: {exc.message}", "error_code": exc.code}
    )

# Global exception handler for general exceptions
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Crowdfunding Backend API...")
    logger.info("✅ Firebase Admin SDK initialized successfully")
    logger.info("✅ SessionMiddleware configured for OAuth")
    logger.info("✅ Application startup complete")
    logger.info("🎉 FastAPI application ready")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Shutting down Crowdfunding Backend API...")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

