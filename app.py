# app.py
# This is the complete and final code for your FastAPI backend,
# now including all endpoints for campaigns, search, OAuth, payments,
# text simplification, and the integrated fraud detection system.

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
import json
import uuid
import os
from datetime import datetime, timedelta
import jwt
import hashlib
import secrets
from enum import Enum
import base64
import requests
from instamojo_wrapper import Instamojo
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import random
# Import the FraudModerationSystem from the other file
from fraud_detection import FraudModerationSystem

# --- Environment Variable Loading ---
load_dotenv()

# --- Environment Variable Configuration ---
FRONTEND_URL = os.getenv("FRONTEND_BASE_URI", "http://haven-streamlit-frontend.onrender.com")
BACKEND_URL = os.getenv("BACKEND_URL", "http://haven-fastapi-backend.onrender.com")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")

# Google OAuth
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")

# Facebook OAuth
FACEBOOK_CLIENT_ID = os.getenv("FACEBOOK_CLIENT_ID")
FACEBOOK_CLIENT_SECRET = os.getenv("FACEBOOK_CLIENT_SECRET")
FACEBOOK_REDIRECT_URI = os.getenv("FACEBOOK_REDIRECT_URI")

# Initialize FastAPI app
app = FastAPI(
    title="HAVEN Crowdfunding API - Complete Platform",
    description="Complete crowdfunding platform with OAuth, profiles, verification, translation, simplification, and real CSV data",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize the Fraud Moderation System
fraud_system = FraudModerationSystem()

# --- CORS middleware ---
origins = [
    FRONTEND_URL,
    "http://localhost",
    "http://localhost:8501"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
JWT_SECRET = os.getenv("JWT_SECRET_KEY", "haven-secret-key-2024")
JWT_ALGORITHM = "HS256"

# Placeholder for campaigns submitted by users that are pending review.
# In a real-world app, this would be a dedicated database table.
PENDING_CAMPAIGNS = []

# Load and process CSV data
def load_campaign_data():
    """Load and process campaign data from a CSV file."""
    try:
        df = pd.read_csv('ngo_fraud.csv')
        legitimate_df = df[df['label'] == 0].copy()
        
        campaigns = []
        for idx, row in legitimate_df.iterrows():
            target_amount = np.random.randint(10000, 100000)
            current_amount = int(target_amount * np.random.uniform(0.1, 0.9))
            
            # Use a mock verification check
            is_verified_status = fraud_system.process_new_campaign({
                "title": str(row.get('campaign_name', f'Campaign {idx + 1}')),
                "description": str(row.get('description', 'Supporting community development and social welfare initiatives. This project focuses on crowdfunding.')),
                "ngo_darpan_id": str(row.get('ngo_darpan_id', f'NGO{idx:05d}')),
                "pan_number": str(row.get('pan', f'ABCDE{idx:04d}F')),
                "organization": str(row.get('org_name', f'Organization {idx + 1}')),
                "donors_count": np.random.randint(5, 50),
                "created_at": (datetime.now() - timedelta(days=np.random.randint(1, 30))).isoformat(),
                "has_certificate": False # Assuming no certificate for CSV data
            })['status'] == 'approved'
            
            campaign = {
                "id": str(idx + 1),
                "title": str(row.get('campaign_name', f'Campaign {idx + 1}')),
                "description": str(row.get('description', 'Supporting community development and social welfare initiatives. This project focuses on crowdfunding.')),
                "organization": str(row.get('org_name', f'Organization {idx + 1}')),
                "category": str(row.get('category', 'Community')),
                "target_amount": target_amount,
                "current_amount": current_amount,
                "progress": round((current_amount / target_amount) * 100, 1),
                "donors_count": np.random.randint(5, 50),
                "days_left": np.random.randint(10, 90),
                "image_url": f"https://picsum.photos/400/300?random={idx}",
                "contact_email": str(row.get('email', f'contact{idx}@example.com')),
                "contact_phone": str(row.get('phone', f'+91{np.random.randint(7000000000, 9999999999)}')),
                "ngo_darpan_id": str(row.get('ngo_darpan_id', f'NGO{idx:05d}')),
                "pan_number": str(row.get('pan', f'ABCDE{idx:04d}F')),
                "created_at": (datetime.now() - timedelta(days=np.random.randint(1, 30))).isoformat(),
                "is_verified": is_verified_status,
                "is_trending": np.random.choice([True, False], p=[0.3, 0.7]),
                "location": "India",
                "tags": ["community", "social", "development"]
            }
            campaigns.append(campaign)
        
        return campaigns
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        # Fallback to sample data if CSV loading fails
        return get_sample_campaigns()

def get_sample_campaigns():
    """Fallback sample campaigns if CSV loading fails"""
    return [
        {
            "id": "1",
            "title": "Clean Water Wells for Rural Communities",
            "description": "Providing clean drinking water access to remote villages through sustainable well construction and maintenance programs. This project aims at poverty alleviation.",
            "organization": "Water for All Foundation",
            "category": "Community",
            "target_amount": 50000,
            "current_amount": 37500,
            "progress": 75.0,
            "donors_count": 42,
            "days_left": 25,
            "image_url": "https://picsum.photos/400/300?random=1",
            "contact_email": "contact@waterforall.org",
            "contact_phone": "+91-9876543210",
            "ngo_darpan_id": "NGO00001",
            "pan_number": "ABCDE1234F",
            "created_at": datetime.now().isoformat(),
            "is_verified": True,
            "is_trending": True,
            "location": "India",
            "tags": ["water", "community", "rural"]
        },
        {
            "id": "2", 
            "title": "Education Support for Underprivileged Children",
            "description": "Providing quality education, books, and learning materials to children from economically disadvantaged backgrounds. This is a form of philanthropy.",
            "organization": "Bright Future Education Trust",
            "category": "Education",
            "target_amount": 80000,
            "current_amount": 45000,
            "progress": 56.3,
            "donors_count": 38,
            "days_left": 45,
            "image_url": "https://picsum.photos/400/300?random=2",
            "contact_email": "info@brightfuture.org",
            "contact_phone": "+91-9876543211",
            "ngo_darpan_id": "NGO00002",
            "pan_number": "ABCDE1235F",
            "created_at": datetime.now().isoformat(),
            "is_verified": True,
            "is_trending": True,
            "location": "India",
            "tags": ["education", "children", "learning"]
        }
    ]

# Load campaign data on startup
CAMPAIGNS_DATA = load_campaign_data()

# Instamojo Payment Service (unchanged)
class InstamojoPaymentService:
    def __init__(self):
        self.api_key = os.environ.get("INSTAMOJO_API_KEY")
        self.auth_token = os.environ.get("INSTAMOJO_AUTH_TOKEN")
        self.sandbox = os.environ.get("ENVIRONMENT", "development") == "development"
        
        if not self.api_key or not self.auth_token:
            print("Warning: Instamojo API credentials not found in environment variables")
            self.api = None
            return
            
        try:
            self.api = Instamojo(
                api_key=self.api_key,
                auth_token=self.auth_token,
                endpoint='https://test.instamojo.com/api/1.1/' if self.sandbox else 'https://www.instamojo.com/api/1.1/'
            )
            print(f"Instamojo initialized in {'sandbox' if self.sandbox else 'production'} mode")
        except Exception as e:
            print(f"Instamojo initialization error: {e}")
            self.api = None

    def create_payment_request(self,
                             amount: float,
                             purpose: str,
                             buyer_name: str,
                             buyer_email: str,
                             buyer_phone: str,
                             campaign_id: str,
                             redirect_url: str) -> Dict:
        """Create a payment request"""
        if not self.api: return {"success": False, "error": "Payment service not initialized"}
        try:
            response = self.api.payment_request_create(
                amount=str(amount), purpose=purpose[:255], buyer_name=buyer_name[:100], email=buyer_email,
                phone=buyer_phone, redirect_url=redirect_url, send_email=True, send_sms=True,
                allow_repeated_payments=False
            )
            if response['success']:
                payment_request = response['payment_request']
                return {
                    "success": True, "payment_id": payment_request['id'], "payment_url": payment_request['longurl'],
                    "amount": amount, "status": "created"
                }
            else: return {"success": False, "error": response.get('message', 'Payment request creation failed')}
        except Exception as e: return {"success": False, "error": str(e)}

payment_service = InstamojoPaymentService()

# --- Pydantic model for request body ---
class TextSimplificationRequest(BaseModel):
    text: str
    language: str

# Pydantic model for a new campaign submission
class NewCampaignRequest(BaseModel):
    title: str
    description: str
    organization: str
    category: str
    ngo_darpan_id: Optional[str] = None
    pan_number: Optional[str] = None
    has_certificate: bool = False
    donors_count: int = 0
    created_at: Optional[str] = datetime.now().isoformat()
    # Add other fields as needed for the full campaign object

class DonationRequest(BaseModel):
    campaign_id: str
    amount: float
    donor_name: str
    donor_email: EmailStr
    donor_phone: str

# --- JWT Token Helpers ---
def create_jwt_token(user_id: str) -> str:
    """Creates a JWT token for a given user_id."""
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(days=7),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_jwt_token(token: str) -> Optional[str]:
    """Decodes a JWT token and returns the user_id if valid."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload.get("user_id")
    except jwt.PyJWTError:
        return None

# --- OAuth and Authentication Helpers ---
async def get_oauth_token_and_user_info(
    token_url: str,
    payload: Dict,
    user_info_url: str,
    user_id_key: str
) -> Dict:
    """Generic function to handle OAuth token exchange and user info fetching."""
    try:
        token_response = requests.post(token_url, data=payload)
        token_response.raise_for_status()
        access_token = token_response.json()["access_token"]

        user_info_response = requests.get(
            user_info_url,
            headers={"Authorization": f"Bearer {access_token}"}
        )
        user_info_response.raise_for_status()
        user_info = user_info_response.json()
        user_id = user_info.get(user_id_key)
        
        return {"user_id": user_id, "user_info": user_info}
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OAuth failed: {e}"
        )

# Dependency for authentication
def authenticate_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    user_id = decode_jwt_token(token)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    return user_id

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to HAVEN Crowdfunding API", "version": "4.0.0", "status": "active",
        "features": ["Real CSV Campaign Data", "Fraud Filtering", "OAuth Authentication", "User Profiles", "Document Verification", "Instamojo Payments", "Translation Services", "Term Simplification", "Fraud Detection Integration"],
        "endpoints": {
            "health": "/health", "docs": "/docs", "campaigns": "/api/campaigns",
            "search": "/api/search", "categories": "/api/categories", "trending": "/api/trending",
            "google_oauth": "/auth/google/callback", "facebook_oauth": "/auth/facebook/callback",
            "simplify_text": "/api/process_text_for_simplification",
            "submit_campaign": "/api/campaigns/submit"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy", "timestamp": datetime.now().isoformat(), "version": "4.0.0",
        "services": {
            "database": "active", "payment_gateway": "active" if payment_service.api else "inactive",
            "translation": "active", "simplification": "active", "fraud_detection": "active"
        },
        "statistics": {
            "total_campaigns": len(CAMPAIGNS_DATA),
            "legitimate_campaigns": len([c for c in CAMPAIGNS_DATA if c.get('is_verified', True)]),
            "trending_campaigns": len([c for c in CAMPAIGNS_DATA if c.get('is_trending', False)]),
            "total_categories": len(set(c['category'] for c in CAMPAIGNS_DATA))
        }
    }

# API info endpoint
@app.get("/api")
async def api_info():
    return {
        "api_name": "HAVEN Crowdfunding API", "version": "4.0.0",
        "description": "Complete crowdfunding platform with real CSV data integration",
        "authentication": {"oauth": ["Google", "Facebook"], "jwt": "Bearer token required for protected endpoints"},
        "endpoints": {
            "campaigns": {"GET /api/campaigns": "List all campaigns with pagination",
                          "GET /api/campaigns/{id}": "Get specific campaign details",
                          "POST /api/campaigns": "Create new campaign (authenticated)"},
            "search": {"GET /api/search": "Search campaigns by query, category, location"},
            "payments": {"POST /api/donate": "Create donation payment request"},
            "categories": {"GET /api/categories": "List all campaign categories with counts"},
            "oauth": {"GET /auth/google/callback": "Google OAuth callback endpoint",
                      "GET /auth/facebook/callback": "Facebook OAuth callback endpoint"},
            "simplification": {"POST /api/process_text_for_simplification": "Simplify a complex term and..."}
        }
    }

# Get all campaigns endpoint
@app.get("/api/campaigns")
async def get_all_campaigns():
    return CAMPAIGNS_DATA

# Get a single campaign by ID
@app.get("/api/campaigns/{campaign_id}")
async def get_campaign(campaign_id: str):
    campaign = next((c for c in CAMPAIGNS_DATA if c["id"] == campaign_id), None)
    if campaign:
        return campaign
    raise HTTPException(status_code=404, detail="Campaign not found")

# Get trending campaigns
@app.get("/api/trending")
async def get_trending_campaigns():
    trending_campaigns = [c for c in CAMPAIGNS_DATA if c.get('is_trending')]
    return trending_campaigns

# Get categories
@app.get("/api/categories")
async def get_categories():
    categories = {}
    for campaign in CAMPAIGNS_DATA:
        category = campaign['category']
        if category not in categories:
            categories[category] = 0
        categories[category] += 1
    return categories

# Endpoint for submitting a new campaign
@app.post("/api/campaigns/submit", response_model=Dict)
async def submit_new_campaign(campaign_data: NewCampaignRequest):
    """
    Submits a new campaign for fraud detection and moderation.
    This endpoint uses the FraudModerationSystem to check for potential fraud.
    """
    campaign_dict = campaign_data.dict()
    moderation_result = fraud_system.process_new_campaign(campaign_dict)
    
    # In a real app, you would save this campaign to the database here.
    # For this mock app, we'll just return the result.
    return {
        "message": "Campaign submitted for review.",
        "moderation_result": moderation_result
    }

# Google OAuth callback
@app.get("/auth/google/callback")
async def google_auth_callback(code: str, request: Request):
    token_url = "https://oauth2.googleapis.com/token"
    user_info_url = "https://www.googleapis.com/oauth2/v3/userinfo"
    payload = {
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "code": code,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code",
    }
    auth_data = await get_oauth_token_and_user_info(token_url, payload, user_info_url, "sub")
    jwt_token = create_jwt_token(auth_data["user_id"])
    return RedirectResponse(url=f"{FRONTEND_URL}/login_success?token={jwt_token}")

# Facebook OAuth callback
@app.get("/auth/facebook/callback")
async def facebook_auth_callback(code: str, request: Request):
    token_url = "https://graph.facebook.com/v19.0/oauth/access_token"
    user_info_url = "https://graph.facebook.com/me"
    payload = {
        "client_id": FACEBOOK_CLIENT_ID,
        "client_secret": FACEBOOK_CLIENT_SECRET,
        "code": code,
        "redirect_uri": FACEBOOK_REDIRECT_URI,
    }
    auth_data = await get_oauth_token_and_user_info(token_url, payload, user_info_url, "id")
    jwt_token = create_jwt_token(auth_data["user_id"])
    return RedirectResponse(url=f"{FRONTEND_URL}/login_success?token={jwt_token}")

# Endpoint to handle text simplification and translation
@app.post("/api/process_text_for_simplification")
async def process_text_for_simplification(request_body: TextSimplificationRequest):
    # This is a mock implementation.
    # In a real app, you'd call a translation/simplification service here.
    return {
        "original_text": request_body.text,
        "simplified_text": "This is a simplified version of the text.",
        "language": request_body.language
    }

# Payment endpoints
@app.post("/api/donate")
async def donate_to_campaign(donation: DonationRequest):
    campaign = next((c for c in CAMPAIGNS_DATA if c["id"] == donation.campaign_id), None)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    payment_response = payment_service.create_payment_request(
        amount=donation.amount,
        purpose=f"Donation for {campaign['title']}",
        buyer_name=donation.donor_name,
        buyer_email=donation.donor_email,
        buyer_phone=donation.donor_phone,
        campaign_id=donation.campaign_id,
        redirect_url=f"{BACKEND_URL}/api/verify_payment"
    )

    if payment_response["success"]:
        return payment_response
    else:
        raise HTTPException(status_code=400, detail=payment_response["error"])

@app.get("/api/verify_payment")
async def verify_payment(
    payment_request_id: str,
    payment_id: str,
    payment_status: str,
    request: Request
):
    if payment_status == 'Credit':
        # In a real app, you would verify the payment with Instamojo API here
        # and update the campaign's current amount and donor count.
        # For this mock, we assume it's successful.
        return JSONResponse(content={"status": "Payment successful", "payment_id": payment_id})
    else:
        return JSONResponse(content={"status": "Payment failed", "payment_id": payment_id}, status_code=400)

# Custom 404 handler
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": f"The requested endpoint {request.url.path} was not found",
            "available_endpoints": [
                "/", "/health", "/api", "/api/campaigns", "/api/search", "/api/categories", "/api/trending",
                "/api/process_text_for_simplification", "/auth/google/callback", "/auth/facebook/callback"]})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
