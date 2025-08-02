# Corrected app.py with all fixes for deployment issues
# This version fixes Firebase imports, logging configuration, and cache directory issues

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
import uvicorn
import json
import logging
from typing import Optional, Dict, Any

# Firebase Admin SDK imports - CORRECTED VERSION
import firebase_admin
from firebase_admin import credentials, auth, firestore, messaging

# CORRECTED: Use proper exception imports from firebase_admin.exceptions
from firebase_admin.exceptions import (
    FirebaseError,              # Base exception class
    InvalidArgumentError,       # For invalid arguments
    NotFoundError,             # For missing resources
    PermissionDeniedError,     # For permission issues
    UnauthenticatedError,      # For authentication failures
    AlreadyExistsError,        # For duplicate resources
    InternalError,             # For internal server errors
    FailedPreconditionError,   # For state validation errors
    ResourceExhaustedError     # For rate limiting
)

# FIXED: Logging setup function
def setup_logging():
    """
    Fixed logging setup function that handles the format properly
    """
    # Get log level from environment variable, default to INFO
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # Fix the logging format issue - use proper format string
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure logging with proper parameters
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=log_format,
        stream=sys.stdout,
        force=True  # This ensures the configuration is applied even if logging was already configured
    )
    
    # Set specific loggers to appropriate levels
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('uvicorn.access').setLevel(logging.INFO)
    logging.getLogger('gunicorn').setLevel(logging.INFO)
    
    logging.info("Logging configuration completed successfully")

# Set up logging
setup_logging()

# Example of proper exception handling in your Firebase operations
def get_user_safely(uid: str) -> Optional[Dict[str, Any]]:
    """
    Example function showing proper Firebase exception handling
    """
    try:
        user_record = auth.get_user(uid)
        return {
            "uid": user_record.uid,
            "email": user_record.email,
            "display_name": user_record.display_name
        }
    except NotFoundError:
        # Handle user not found specifically
        logging.warning(f"User with UID {uid} not found")
        return None
    except PermissionDeniedError as e:
        # Handle permission issues
        logging.error(f"Permission denied for user {uid}: {e.message}")
        raise HTTPException(status_code=403, detail="Access denied")
    except UnauthenticatedError as e:
        # Handle authentication issues
        logging.error(f"Authentication failed: {e.message}")
        raise HTTPException(status_code=401, detail="Authentication required")
    except FirebaseError as e:
        # Handle any other Firebase errors
        logging.error(f"Firebase error: {e.code} - {e.message}")
        raise HTTPException(status_code=500, detail="Internal server error")

def create_user_safely(email: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Example function for creating users with proper error handling
    """
    try:
        user_record = auth.create_user(
            email=email,
            password=password
        )
        return {
            "uid": user_record.uid,
            "email": user_record.email
        }
    except AlreadyExistsError:
        # Handle user already exists
        logging.warning(f"User with email {email} already exists")
        raise HTTPException(status_code=409, detail="User already exists")
    except InvalidArgumentError as e:
        # Handle invalid input
        logging.error(f"Invalid argument: {e.message}")
        raise HTTPException(status_code=400, detail="Invalid input data")
    except FirebaseError as e:
        # Handle any other Firebase errors
        logging.error(f"Firebase error creating user: {e.code} - {e.message}")
        raise HTTPException(status_code=500, detail="Failed to create user")

def send_notification_safely(token: str, title: str, body: str) -> bool:
    """
    Example function for sending notifications with proper error handling
    """
    try:
        message = messaging.Message(
            notification=messaging.Notification(
                title=title,
                body=body
            ),
            token=token
        )
        response = messaging.send(message)
        logging.info(f"Successfully sent message: {response}")
        return True
    except InvalidArgumentError as e:
        # Handle invalid token or message format
        logging.error(f"Invalid message format: {e.message}")
        return False
    except ResourceExhaustedError as e:
        # Handle rate limiting
        logging.error(f"Rate limit exceeded: {e.message}")
        return False
    except FirebaseError as e:
        # Handle any other Firebase errors
        logging.error(f"Firebase messaging error: {e.code} - {e.message}")
        return False

# Initialize Firebase Admin SDK
def initialize_firebase():
    """
    Initialize Firebase Admin SDK with proper error handling
    Supports multiple credential sources for different environments
    """
    
    if firebase_admin._apps:
        # Firebase already initialized
        logging.info("Firebase Admin SDK already initialized")
        return firestore.client()
    
    try:
        # Method 1: Individual environment variables (Recommended for production)
        if os.getenv('FIREBASE_PROJECT_ID'):
            logging.info("Initializing Firebase with environment variables...")
            
            # Construct credentials dictionary from environment variables
            cred_dict = {
                "type": os.getenv('FIREBASE_TYPE', 'service_account'),
                "project_id": os.getenv('FIREBASE_PROJECT_ID'),
                "private_key_id": os.getenv('FIREBASE_PRIVATE_KEY_ID'),
                "private_key": os.getenv('FIREBASE_PRIVATE_KEY', '').replace('\\n', '\n'),
                "client_email": os.getenv('FIREBASE_CLIENT_EMAIL'),
                "client_id": os.getenv('FIREBASE_CLIENT_ID'),
                "auth_uri": os.getenv('FIREBASE_AUTH_URI', 'https://accounts.google.com/o/oauth2/auth'),
                "token_uri": os.getenv('FIREBASE_TOKEN_URI', 'https://oauth2.googleapis.com/token'),
                "auth_provider_x509_cert_url": os.getenv('FIREBASE_AUTH_PROVIDER_X509_CERT_URL', 'https://www.googleapis.com/oauth2/v1/certs'),
                "client_x509_cert_url": os.getenv('FIREBASE_CLIENT_X509_CERT_URL')
            }
            
            # Validate required fields
            required_fields = ['project_id', 'private_key', 'client_email']
            missing_fields = [field for field in required_fields if not cred_dict.get(field)]
            
            if missing_fields:
                raise ValueError(f"Missing required Firebase environment variables: {missing_fields}")
            
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            logging.info("✅ Firebase Admin SDK initialized successfully with environment variables")
            
        # Method 2: Base64 encoded JSON (Alternative for production)
        elif os.getenv('FIREBASE_CREDENTIALS_BASE64'):
            logging.info("Initializing Firebase with base64 encoded credentials...")
            
            import base64
            encoded_creds = os.getenv('FIREBASE_CREDENTIALS_BASE64')
            decoded_creds = base64.b64decode(encoded_creds).decode('utf-8')
            cred_dict = json.loads(decoded_creds)
            
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            logging.info("✅ Firebase Admin SDK initialized successfully with base64 credentials")
            
        # Method 3: Service account key from environment variable (JSON string)
        elif os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY'):
            logging.info("Initializing Firebase with service account key...")
            
            service_account_info = json.loads(os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY'))
            cred = credentials.Certificate(service_account_info)
            firebase_admin.initialize_app(cred)
            logging.info("✅ Firebase Admin SDK initialized successfully with service account key")
            
        # Method 4: Local file (Development only)
        elif os.path.exists('serviceAccountKey.json'):
            logging.info("Initializing Firebase with local service account file...")
            
            cred = credentials.Certificate('serviceAccountKey.json')
            firebase_admin.initialize_app(cred)
            logging.info("✅ Firebase Admin SDK initialized successfully with local file")
            
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
            
            logging.error(error_msg)
            raise ValueError("No Firebase credentials configured. See logs for configuration options.")
            
        return firestore.client()
        
    except json.JSONDecodeError as e:
        logging.error(f"❌ Failed to parse Firebase credentials JSON: {str(e)}")
        raise ValueError(f"Invalid Firebase credentials format: {str(e)}")
        
    except FirebaseError as e:
        logging.error(f"❌ Firebase initialization failed: {e.code} - {e.message}")
        raise e
        
    except Exception as e:
        logging.error(f"❌ Unexpected error initializing Firebase: {str(e)}")
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
        auth.get_user("test-connectivity-check")
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
async def create_user(user_data: dict):
    """
    Create new user with proper error handling
    """
    email = user_data.get("email")
    password = user_data.get("password")
    
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")
    
    return create_user_safely(email, password)

# Add any additional API endpoints here...

# Optional: Add startup event to verify everything is working
@app.on_event("startup")
async def startup_event():
    """
    Startup event to verify all systems are working
    """
    logging.info("Starting Crowdfunding Backend API...")
    logging.info("Cache directories configured")
    logging.info("Firebase Admin SDK initialized")
    logging.info("FastAPI application ready")

# Optional: Add shutdown event for cleanup
@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown event for cleanup
    """
    logging.info("Shutting down Crowdfunding Backend API...")

# For local development
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

