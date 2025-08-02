# Corrected Firebase imports for app.py
# Replace the problematic import section with this corrected version

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
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

# REMOVED: The following import that was causing the error
# from firebase_admin.exceptions import FirebaseAppError  # THIS DOES NOT EXIST!

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

# Continue with your other API endpoints...

