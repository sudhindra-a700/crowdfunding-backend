# app.py
# This is a comprehensive FastAPI backend that serves as the core
# for the HAVEN crowdfunding platform. It integrates a fraud detection
# model, handles environment variables, configures CORS, and provides
# a full set of API endpoints for campaigns and user authentication.

import os
import json
import uuid
import hashlib
import secrets
from enum import Enum
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import requests
import jwt
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
import uvicorn

import firebase_admin
from firebase_admin import credentials, firestore

# --- Environment Variable Loading ---
# This loads variables from a .env file, which is essential for configuration.
load_dotenv()

# --- Configuration Variables ---
FRONTEND_URL = os.getenv("FRONTEND_BASE_URI", "http://localhost:8501")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# OAuth
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", f"{BACKEND_URL}/auth/google/callback")

FACEBOOK_CLIENT_ID = os.getenv("FACEBOOK_CLIENT_ID")
FACEBOOK_CLIENT_SECRET = os.getenv("FACEBOOK_CLIENT_SECRET")
FACEBOOK_REDIRECT_URI = os.getenv("FACEBOOK_REDIRECT_URI", f"{BACKEND_URL}/auth/facebook/callback")

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secret-key")
TRUSTCHECKR_API_KEY = os.getenv("TRUSTCHECKR_API_KEY", "your-trustcheckr-api-key")

# --- Firebase Initialization ---
# This sets up the Firebase Admin SDK using credentials from your environment.
try:
    firebase_credentials_json = {
        "type": "service_account",
        "project_id": os.getenv("FIREBASE_PROJECT_ID"),
        "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
        "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace("\\n", "\n"),
        "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
        "client_id": os.getenv("FIREBASE_CLIENT_ID"),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_X509_CERT_URL"),
        "universe_domain": "googleapis.com"
    }

    # Initialize Firebase if it hasn't been already
    if not firebase_admin._apps:
        cred = credentials.Certificate(firebase_credentials_json)
        firebase_admin.initialize_app(cred)
    
    db = firestore.client()
    print("Successfully initialized Firebase.")

except Exception as e:
    print(f"Failed to initialize Firebase: {e}. Running without Firestore.")
    db = None

# --- Fraud Detection Model Import ---
# This block attempts to import the fraud detection model.
# A mock function is provided for development or if the model file is not available.
try:
    from fraud_detection import predict_fraud
except ImportError:
    print("Warning: `fraud_detection.py` not found. Using mock function.")
    def predict_fraud(organization_data, api_key_trustcheckr=None):
        """Mock fraud detection function for local development."""
        # Simple mock logic
        is_fraudulent = "crypto" in organization_data.get("bio", "").lower()
        fraud_score = 0.9 if is_fraudulent else 0.1
        explanation = "Mock explanation: High fraud score due to keyword 'crypto'." if is_fraudulent else "Mock explanation: Low fraud score."
        verification = {"pan": "Verified"}
        return fraud_score, explanation, "mock_plot.png", verification

# --- App Initialization ---
app = FastAPI()

# --- CORS Middleware Configuration ---
# This is critical for allowing the frontend to communicate with the backend.
# It explicitly lists the allowed origins, methods, and headers.
origins = [
    FRONTEND_URL,
    "https://haven-streamlit-frontend.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class CampaignCreate(BaseModel):
    """Data model for creating a new campaign."""
    title: str
    description: str
    organization: str
    category: str
    ngo_darpan_id: Optional[str] = None
    pan_number: Optional[str] = None
    has_certificate: bool
    donors_count: int = 0
    created_at: datetime = datetime.now()

class Campaign(CampaignCreate):
    """Full data model for a campaign, including moderation results."""
    id: str
    status: str
    fraud_score: float
    explanation: str
    verification_details: Dict[str, Any]

# --- In-memory database (for demonstration) ---
campaigns_db: Dict[str, Campaign] = {}

# --- API Endpoints ---
@app.get("/")
async def root():
    """A simple welcome message to confirm the backend is running."""
    return {"message": "Welcome to the HAVEN Crowdfunding Backend"}

@app.get("/health")
async def health_check():
    """Endpoint for health checks."""
    return {"status": "ok"}

@app.get("/api/campaigns")
async def get_all_campaigns():
    """Retrieves all campaigns from the in-memory database."""
    # In a production environment, this would query a database like Firestore
    return {"campaigns": list(campaigns_db.values())}

@app.get("/api/trending")
async def get_trending_campaigns():
    """
    Retrieves a hardcoded list of trending campaigns.
    This would be replaced with real logic in a production app.
    """
    trending_data = [
        {
            "id": "trend-1",
            "title": "Clean Water Initiative",
            "organization": "Aqua Aid",
            "category": "Health",
            "description": "Provide clean drinking water to remote villages...",
            "current_amount": 95000,
            "target_amount": 100000,
            "donors_count": 500,
            "status": "active",
            "fraud_score": 0.05,
            "explanation": "Verified campaign.",
            "verification_details": {"pan": "Verified", "ngo_darpan": "Verified"},
            "has_certificate": True,
            "created_at": datetime.now().isoformat()
        }
    ]
    return {"trending_campaigns": trending_data}

@app.post("/api/campaigns/submit", response_model=Campaign)
async def submit_campaign_for_moderation(campaign_data: CampaignCreate):
    """
    Submits a new campaign for review and fraud detection using the
    `predict_fraud` function.
    """
    try:
        # Create a dictionary for the organization data to be passed to the model
        organization_data = {
            "org_name": campaign_data.organization,
            "bio": campaign_data.description,
            "pan": campaign_data.pan_number,
            "ngo_darpan_id": campaign_data.ngo_darpan_id,
        }
        
        # Call the fraud detection model
        fraud_score, explanation, _, verification = predict_fraud(organization_data, api_key_trustcheckr=TRUSTCHECKR_API_KEY)

        # Determine campaign status based on fraud score
        status_text = "pending_review"
        if fraud_score > 0.8:
            status_text = "flagged_for_review"
        
        # Create a new campaign object and store it
        campaign_id = str(uuid.uuid4())
        new_campaign = Campaign(
            id=campaign_id,
            status=status_text,
            fraud_score=fraud_score,
            explanation=explanation,
            verification_details=verification,
            **campaign_data.dict()
        )
        
        campaigns_db[campaign_id] = new_campaign
        return new_campaign
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during campaign submission: {e}"
        )

# --- OAuth Handlers ---
@app.get("/auth/google/callback")
async def google_auth_callback(code: str, request: Request):
    """Handles the callback from Google OAuth."""
    if not all([GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT_URI]):
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Missing Google OAuth credentials.")
    
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "code": code,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code",
    }
    
    try:
        response = requests.post(token_url, data=data)
        response.raise_for_status()
        token_info = response.json()
        
        # Here you would process the token, create a user in your DB,
        # generate a JWT, and redirect to the frontend.
        return {"message": "Google authentication successful", "token_info": token_info}
    
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get Google token: {e}")

@app.get("/auth/facebook/callback")
async def facebook_auth_callback(code: str, request: Request):
    """Handles the callback from Facebook OAuth."""
    if not all([FACEBOOK_CLIENT_ID, FACEBOOK_CLIENT_SECRET, FACEBOOK_REDIRECT_URI]):
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Missing Facebook OAuth credentials.")
    
    token_url = "https://graph.facebook.com/v19.0/oauth/access_token"
    data = { 
        "client_id": FACEBOOK_CLIENT_ID, 
        "client_secret": FACEBOOK_CLIENT_SECRET, 
        "code": code, 
        "redirect_uri": FACEBOOK_REDIRECT_URI, 
    }
    try:
        response = requests.get(token_url, params=data)
        response.raise_for_status()
        token_info = response.json()
        return {"message": "Facebook authentication successful", "token_info": token_info}
    except requests.exceptions.RequestException as e: 
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get Facebook token: {e}")

# Custom 404 handler (unchanged)
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=404, content={"error": "Endpoint not found", "message": f"The requested endpoint {request.url.path} was not found"})

# --- Main Entry Point ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

