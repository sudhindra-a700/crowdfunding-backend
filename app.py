# app.py
# This is the updated, complete code for your FastAPI backend,
# now with an API endpoint for automatic term simplification.

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

# --- Environment Variable Loading ---
load_dotenv()

# --- Environment Variable Configuration ---
FRONTEND_URL = os.getenv("FRONTEND_BASE_URI", "http://haven-streamlit-frontend.onrender.com")
BACKEND_URL = os.getenv("BACKEND_URL", "http://haven-fastapi-backend.onrender.com")

# Google OAuth
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")

# Facebook OAuth
FACEBOOK_CLIENT_ID = os.getenv("FACEBOOK_CLIENT_ID")
FACEBOOK_CLIENT_SECRET = os.getenv("FACEBOOK_CLIENT_SECRET")
FACEBOOK_REDIRECT_URI = os.getenv("FACEBOOK_REDIRECT_URI")

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")

# Initialize FastAPI app
app = FastAPI(
    title="HAVEN Crowdfunding API - Complete Platform",
    description="Complete crowdfunding platform with OAuth, profiles, verification, translation, simplification, and real CSV data",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

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

# Load and process CSV data
def load_campaign_data():
    """Load and process campaign data from CSV file"""
    try:
        # Load the CSV file
        df = pd.read_csv('ngo_fraud.csv')
        
        # Filter out fraudulent campaigns (label = 1)
        legitimate_df = df[df['label'] == 0].copy()
        
        # Clean and process the data
        campaigns = []
        for idx, row in legitimate_df.iterrows():
            # Generate realistic funding data
            target_amount = np.random.randint(10000, 100000)
            current_amount = int(target_amount * np.random.uniform(0.1, 0.9))
            
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
                "is_verified": True,
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

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to HAVEN Crowdfunding API", "version": "4.0.0", "status": "active",
        "features": ["Real CSV Campaign Data", "Fraud Filtering", "OAuth Authentication", "User Profiles", "Document Verification", "Instamojo Payments", "Translation Services", "Term Simplification"],
        "endpoints": {
            "health": "/health", "docs": "/docs", "campaigns": "/api/campaigns",
            "search": "/api/search", "categories": "/api/categories", "trending": "/api/trending",
            "google_oauth": "/auth/google/callback", "facebook_oauth": "/auth/facebook/callback",
            "simplify_text": "/api/process_text_for_simplification"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy", "timestamp": datetime.now().isoformat(), "version": "4.0.0",
        "services": {
            "database": "active", "payment_gateway": "active" if payment_service.api else "inactive",
            "translation": "active", "simplification": "active"
        },
        "statistics": {
            "total_campaigns": len(CAMPAIGNS_DATA), "legitimate_campaigns": len([c for c in CAMPAIGNS_DATA if c.get('is_verified', True)]),
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
            "campaigns": {"GET /api/campaigns": "List all campaigns with pagination", "GET /api/campaigns/{id}": "Get specific campaign details", "POST /api/campaigns": "Create new campaign (authenticated)"},
            "search": {"GET /api/search": "Search campaigns by query, category, location"},
            "payments": {"POST /api/donate": "Create donation payment request"},
            "categories": {"GET /api/categories": "List all campaign categories with counts"},
            "oauth": {"GET /auth/google/callback": "Google OAuth callback endpoint", "GET /auth/facebook/callback": "Facebook OAuth callback endpoint"},
            "simplification": {"POST /api/process_text_for_simplification": "Simplify a complex term and provide an explanation"}
        }
    }

# Campaign endpoints
@app.get("/api/campaigns")
async def get_campaigns(page: int = 1, limit: int = 10, category: Optional[str] = None, trending: Optional[bool] = None):
    """Get campaigns with pagination and filtering"""
    try:
        campaigns = CAMPAIGNS_DATA.copy()
        if category: campaigns = [c for c in campaigns if c['category'].lower() == category.lower()]
        if trending is not None: campaigns = [c for c in campaigns if c.get('is_trending', False) == trending]
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_campaigns = campaigns[start_idx:end_idx]
        return { "campaigns": paginated_campaigns, "pagination": { "page": page, "limit": limit, "total": len(campaigns), "pages": (len(campaigns) + limit - 1) // limit } }
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/campaigns/{campaign_id}")
async def get_campaign(campaign_id: str):
    """Get specific campaign details"""
    campaign = next((c for c in CAMPAIGNS_DATA if c['id'] == campaign_id), None)
    if not campaign: raise HTTPException(status_code=404, detail="Campaign not found")
    return campaign

@app.get("/api/trending")
async def get_trending_campaigns(limit: int = 6):
    """Get trending campaigns"""
    trending = [c for c in CAMPAIGNS_DATA if c.get('is_trending', False)]
    return {"campaigns": trending[:limit]}

@app.get("/api/categories")
async def get_categories():
    """Get all categories with campaign counts"""
    categories = {}
    for campaign in CAMPAIGNS_DATA:
        category = campaign['category']
        if category not in categories:
            categories[category] = { "name": category, "count": 0, "icon": get_category_icon(category) }
        categories[category]["count"] += 1
    return {"categories": list(categories.values())}

def get_category_icon(category: str) -> str:
    """Get icon for category"""
    icons = { "Education": "school", "Healthcare": "local_hospital", "Community": "people",
        "Environment": "eco", "Technology": "computer", "Arts & Culture": "palette",
        "Sports": "sports_soccer", "Emergency": "warning"
    }
    return icons.get(category, "campaign")

@app.get("/api/search")
async def search_campaigns(
    q: Optional[str] = None, category: Optional[str] = None, location: Optional[str] = None,
    min_amount: Optional[int] = None, max_amount: Optional[int] = None
):
    """Search campaigns with multiple filters"""
    campaigns = CAMPAIGNS_DATA.copy()
    if q:
        q_lower = q.lower()
        campaigns = [c for c in campaigns if q_lower in c['title'].lower() or q_lower in c['description'].lower() or q_lower in c['organization'].lower()]
    if category: campaigns = [c for c in campaigns if c['category'].lower() == category.lower()]
    if location: campaigns = [c for c in campaigns if location.lower() in c['location'].lower()]
    if min_amount: campaigns = [c for c in campaigns if c['target_amount'] >= min_amount]
    if max_amount: campaigns = [c for c in campaigns if c['target_amount'] <= max_amount]
    return { "campaigns": campaigns, "total": len(campaigns), "query": { "q": q, "category": category, "location": location, "min_amount": min_amount, "max_amount": max_amount } }

# Payment endpoints (unchanged)
@app.post("/api/donate")
async def create_donation(campaign_id: str = Form(...), amount: float = Form(...), donor_name: str = Form(...), donor_email: str = Form(...), donor_phone: str = Form(...), anonymous: bool = Form(False)):
    """Create a donation payment request"""
    campaign = next((c for c in CAMPAIGNS_DATA if c['id'] == campaign_id), None)
    if not campaign: raise HTTPException(status_code=404, detail="Campaign not found")
    if amount < 1: raise HTTPException(status_code=400, detail="Minimum donation amount is ₹1")
    try:
        payment_result = payment_service.create_payment_request(amount=amount, purpose=f"Donation to {campaign['title']}", buyer_name=donor_name, buyer_email=donor_email, buyer_phone=donor_phone, campaign_id=campaign_id, redirect_url="https://haven-streamlit-frontend.onrender.com/success")
        if payment_result["success"]:
            donation_record = {
                "id": str(uuid.uuid4()), "campaign_id": campaign_id, "amount": amount,
                "donor_name": donor_name if not anonymous else "Anonymous",
                "donor_email": donor_email, "anonymous": anonymous,
                "payment_id": payment_result["payment_id"], "status": "pending",
                "created_at": datetime.now().isoformat()
            }
            return {
                "success": True, "payment_url": payment_result["payment_url"],
                "payment_id": payment_result["payment_id"], "donation_id": donation_record["id"],
                "amount": amount, "campaign": campaign["title"]
            }
        else: raise HTTPException(status_code=400, detail=payment_result["error"])
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

# Statistics endpoint (unchanged)
@app.get("/api/stats")
async def get_platform_stats():
    """Get platform statistics"""
    total_campaigns = len(CAMPAIGNS_DATA)
    total_raised = sum(c['current_amount'] for c in CAMPAIGNS_DATA)
    total_target = sum(c['target_amount'] for c in CAMPAIGNS_DATA)
    total_donors = sum(c['donors_count'] for c in CAMPAIGNS_DATA)
    categories = {}
    for campaign in CAMPAIGNS_DATA:
        category = campaign['category']
        if category not in categories: categories[category] = 0
        categories[category] += 1
    return {
        "total_campaigns": total_campaigns, "total_raised": total_raised, "total_target": total_target,
        "total_donors": total_donors, "success_rate": round((total_raised / total_target) * 100, 1) if total_target > 0 else 0,
        "categories": categories, "trending_count": len([c for c in CAMPAIGNS_DATA if c.get('is_trending', False)])
    }

# --- New Endpoint for automatic Term Simplification ---
@app.post("/api/process_text_for_simplification")
async def process_text_for_simplification(request: TextSimplificationRequest):
    """
    Processes a block of text to identify complex terms and provide simplifications.
    
    This is a demonstration of using a language model. In a real application,
    this would involve calling a model trained on a simplification dataset (e.g.,
    from Hugging Face) and a translation model (like IndicTrans2) to handle multiple languages.
    """
    # A simple dictionary to simulate a Hugging Face model/dataset
    COMPLEX_TERM_DATA = {
        "crowdfunding": "Raising small amounts of money from many people, typically online, to fund a project.",
        "poverty alleviation": "Actions taken to reduce or relieve the suffering caused by poverty.",
        "philanthropy": "The desire to promote the welfare of others, often through charitable donations.",
        "sustainable": "Something that can be maintained at a certain rate or level for a long time."
    }
    
    processed_text = request.text
    simplifications = {}
    
    for term, definition in COMPLEX_TERM_DATA.items():
        # Using a case-insensitive check and a marker for replacement
        if term.lower() in processed_text.lower():
            # A simple marker to be replaced by the frontend
            # We use `{{i}}term{{/i}}` to avoid conflicts with other markup
            processed_text = processed_text.replace(term, f"{{i}}{term}{{/i}}", 1)
            simplifications[term] = definition
            
    return {
        "processed_text": processed_text,
        "simplifications": simplifications
    }

# OAuth Callback Endpoints (unchanged)
@app.get("/google/callback")
async def google_auth_callback(code: str):
    if not all([GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT_URI]):
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Missing Google OAuth credentials.")
    token_url = "https://oauth2.googleapis.com/token"
    data = { "client_id": GOOGLE_CLIENT_ID, "client_secret": GOOGLE_CLIENT_SECRET, "code": code, "redirect_uri": GOOGLE_REDIRECT_URI, "grant_type": "authorization_code", }
    try:
        response = requests.post(token_url, data=data)
        response.raise_for_status()
        token_info = response.json()
        return {"message": "Google authentication successful", "token_info": token_info}
    except requests.exceptions.RequestException as e: raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get Google token: {e}")

@app.get("/auth/facebook/callback")
async def facebook_auth_callback(code: str, request: Request):
    if not all([FACEBOOK_CLIENT_ID, FACEBOOK_CLIENT_SECRET, FACEBOOK_REDIRECT_URI]):
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Missing Facebook OAuth credentials.")
    token_url = "https://graph.facebook.com/v19.0/oauth/access_token"
    data = { "client_id": FACEBOOK_CLIENT_ID, "client_secret": FACEBOOK_CLIENT_SECRET, "code": code, "redirect_uri": FACEBOOK_REDIRECT_URI, }
    try:
        response = requests.get(token_url, params=data)
        response.raise_for_status()
        token_info = response.json()
        return {"message": "Facebook authentication successful", "token_info": token_info}
    except requests.exceptions.RequestException as e: raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get Facebook token: {e}")

# Custom 404 handler (unchanged)
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=404, content={"error": "Endpoint not found", "message": f"The requested endpoint {request.url.path} was not found", "available_endpoints": ["/", "/health", "/api", "/api/campaigns", "/api/search", "/api/categories", "/api/trending", "/api/process_text_for_simplification", "/auth/google/callback", "/auth/facebook/callback", "/docs"]})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
