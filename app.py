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
    """
    try:
        if not firebase_admin._apps:
            # Check if running in production with service account key
            if os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY'):
                service_account_info = json.loads(os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY'))
                cred = credentials.Certificate(service_account_info)
            else:
                # Development mode - use service account file
                cred = credentials.Certificate('path/to/serviceAccountKey.json')
            
            firebase_admin.initialize_app(cred)
            logging.info("Firebase Admin SDK initialized successfully")
    except FirebaseError as e:
        logging.error(f"Failed to initialize Firebase: {e.code} - {e.message}")
        raise e
    except Exception as e:
        logging.error(f"Unexpected error initializing Firebase: {str(e)}")
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

