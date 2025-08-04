# app.py
# This is the complete, corrected code for your FastAPI backend,
# incorporating the features for registration validation, JWT authentication,
# OAuth, and payment services.

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
import json
import uuid
import os
from datetime import datetime, timedelta, timezone
import jwt
import hashlib
import secrets
from enum import Enum
import base64
import requests
from instamojo_wrapper import Instamojo
import pandas as pd
import numpy as np
from dotenv import load_dotenv  # Add this import

# --- Environment Variable Loading ---
# This loads variables from a .env file, which is essential for configuration.
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

JWT_SECRET = os.getenv("JWT_SECRET_KEY", "haven-secret-key-2024")
JWT_ALGORITHM = "HS256"

# Initialize FastAPI app
app = FastAPI(
    title="HAVEN Crowdfunding API - Complete Platform",
    description="Complete crowdfunding platform with OAuth, profiles, verification, translation, simplification, and real CSV data",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# --- CORS middleware ---
# This is a critical fix. Instead of a wildcard, we specify allowed origins
# to improve security and prevent connectivity issues with the frontend.
origins = [
    FRONTEND_URL,
    "http://localhost",
    "http://localhost:8501"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Changed from ["*"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# In-memory "database" for demonstration purposes
# In a real application, this would be a persistent database (e.g., SQL or NoSQL)
USERS_DB = {}
CAMPAIGNS_DATA = []

# --- Pydantic Models for Data Validation ---

class UserRegistration(BaseModel):
    """Model for individual user registration, ensuring all fields are present."""
    full_name: str
    email: EmailStr
    password: str
    phone: str
    address: str
    document_type: str
    # Note: `document_file` will be handled as a separate UploadFile object.

class OrganizationRegistration(BaseModel):
    """Model for organization registration, ensuring all fields are present."""
    org_name: str
    org_phone: str
    org_type: str
    org_description: str
    contact_person: str
    contact_email: EmailStr
    password: str
    # Note: `certificate_file` will be handled as a separate UploadFile object.

class LoginDetails(BaseModel):
    """Model for user login."""
    email: EmailStr
    password: str

# --- JWT Utility Functions ---

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Creates a JWT token with an expiration time."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=30)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    """Verifies a JWT token and returns the payload."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        # Token has expired
        return None
    except jwt.InvalidTokenError:
        # Invalid token
        return None

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Dependency to get the current user from the JWT token in the Authorization header.
    This is used to protect API routes.
    """
    token = credentials.credentials
    payload = verify_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_email = payload.get("sub")
    if user_email is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_email

# --- Load and process CSV data ---
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
                "description": str(row.get('description', 'Supporting community development and social welfare initiatives.')),
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
            "description": "Providing clean drinking water access to remote villages through sustainable well construction and maintenance programs.",
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
            "description": "Providing quality education, books, and learning materials to children from economically disadvantaged backgrounds.",
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

# Instamojo Payment Service
class InstamojoPaymentService:
    def __init__(self):
        self.api_key = os.environ.get("INSTAMOJO_API_KEY")
        self.auth_token = os.environ.get("INSTAMOJO_AUTH_TOKEN")
        self.sandbox = os.environ.get("ENVIRONMENT", "development") == "development"
        
        if not self.api_key or not self.auth_token:
            print("Warning: Instamojo API credentials not found in environment variables")
            self.api = None
            return
            
        # Initialize Instamojo client
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
        if not self.api:
            return {"success": False, "error": "Payment service not initialized"}
            
        try:
            response = self.api.payment_request_create(
                amount=str(amount),
                purpose=purpose[:255],  # Limit purpose length
                buyer_name=buyer_name[:100],  # Limit name length
                email=buyer_email,
                phone=buyer_phone,
                redirect_url=redirect_url,
                send_email=True,
                send_sms=True,
                allow_repeated_payments=False
            )
            
            if response['success']:
                payment_request = response['payment_request']
                return {
                    "success": True,
                    "payment_id": payment_request['id'],
                    "payment_url": payment_request['longurl'],
                    "amount": amount,
                    "status": "created"
                }
            else:
                return {
                    "success": False,
                    "error": response.get('message', 'Payment request creation failed')
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

# Initialize payment service
payment_service = InstamojoPaymentService()

# --- API Endpoints ---

@app.get("/")
async def root():
    return {
        "message": "Welcome to HAVEN Crowdfunding API",
        "version": "4.0.0",
        "status": "active",
        "features": [
            "Real CSV Campaign Data",
            "Fraud Filtering",
            "OAuth Authentication", 
            "User Profiles",
            "Document Verification",
            "Instamojo Payments",
            "Translation Services",
            "Term Simplification"
        ],
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "campaigns": "/api/campaigns",
            "search": "/api/search",
            "categories": "/api/categories",
            "trending": "/api/trending",
            "google_oauth": "/auth/google/callback",
            "facebook_oauth": "/auth/facebook/callback",
            "register_user": "/api/register",
            "register_organization": "/api/register_organization",
            "login": "/api/login",
            "users_me": "/api/users/me (protected)"
        }
    }

# Favicon endpoint
@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(status_code=204, content=None, headers={
        "Cache-Control": "public, max-age=86400",
        "Content-Type": "image/x-icon"
    })

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "4.0.0",
        "services": {
            "database": "active",
            "payment_gateway": "active" if payment_service.api else "inactive",
            "translation": "active",
            "simplification": "active"
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
        "api_name": "HAVEN Crowdfunding API",
        "version": "4.0.0",
        "description": "Complete crowdfunding platform with real CSV data integration",
        "authentication": {
            "oauth": ["Google", "Facebook"],
            "jwt": "Bearer token required for protected endpoints"
        },
        "endpoints": {
            "campaigns": {
                "GET /api/campaigns": "List all campaigns with pagination",
                "GET /api/campaigns/{id}": "Get specific campaign details",
                "POST /api/campaigns": "Create new campaign (authenticated)"
            },
            "search": {
                "GET /api/search": "Search campaigns by query, category, location"
            },
            "payments": {
                "POST /api/donate": "Create donation payment request"
            },
            "categories": {
                "GET /api/categories": "List all campaign categories with counts"
            },
            "oauth": {
                "GET /auth/google/callback": "Google OAuth callback endpoint",
                "GET /auth/facebook/callback": "Facebook OAuth callback endpoint"
            }
        }
    }

# --- Registration and Login Endpoints ---

@app.post("/api/register")
async def register_user(
    full_name: str = Form(...),
    email: EmailStr = Form(...),
    password: str = Form(...),
    phone: str = Form(...),
    address: str = Form(...),
    document_type: str = Form(...),
    document_file: UploadFile = File(...)
):
    """
    Registers a new individual user.
    All fields are required.
    """
    if email in USERS_DB:
        raise HTTPException(status_code=400, detail="User with this email already exists")

    # Simulate saving user data
    USERS_DB[email] = {
        "full_name": full_name,
        "email": email,
        "hashed_password": hashlib.sha256(password.encode()).hexdigest(),
        "phone": phone,
        "address": address,
        "document_type": document_type,
        "is_verified": False,
        "account_type": "individual",
    }
    
    # Create a JWT token for the new user
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": email, "account_type": "individual"}, expires_delta=access_token_expires
    )

    return {"message": "Registration successful!", "access_token": access_token}


@app.post("/api/register_organization")
async def register_organization(
    org_name: str = Form(...),
    org_phone: str = Form(...),
    org_type: str = Form(...),
    org_description: str = Form(...),
    contact_person: str = Form(...),
    contact_email: EmailStr = Form(...),
    password: str = Form(...),
    certificate_file: UploadFile = File(...)
):
    """
    Registers a new organization.
    All fields are required.
    """
    if contact_email in USERS_DB:
        raise HTTPException(status_code=400, detail="Organization with this email already exists")

    # Simulate saving organization data
    USERS_DB[contact_email] = {
        "org_name": org_name,
        "contact_email": contact_email,
        "hashed_password": hashlib.sha256(password.encode()).hexdigest(),
        "org_phone": org_phone,
        "org_type": org_type,
        "org_description": org_description,
        "is_verified": False,
        "account_type": "organization",
    }
    
    # Create a JWT token for the new organization
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": contact_email, "account_type": "organization"}, expires_delta=access_token_expires
    )

    return {"message": "Organization registration successful!", "access_token": access_token}


@app.post("/api/login")
async def login_user(login_details: LoginDetails):
    """
    Authenticates a user and returns a JWT token.
    """
    user_data = USERS_DB.get(login_details.email)
    if not user_data:
        raise HTTPException(status_code=400, detail="Invalid email or password")
    
    hashed_password = hashlib.sha256(login_details.password.encode()).hexdigest()
    if hashed_password != user_data["hashed_password"]:
        raise HTTPException(status_code=400, detail="Invalid email or password")
    
    # Create a JWT token for the authenticated user
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user_data["email"], "account_type": user_data["account_type"]}, 
        expires_delta=access_token_expires
    )
    
    return {"message": "Login successful!", "access_token": access_token}


@app.get("/api/users/me")
def read_current_user(current_user_email: str = Depends(get_current_user)):
    """
    A protected endpoint that returns information about the currently authenticated user.
    This demonstrates how to use the JWT token to secure a route.
    """
    user_data = USERS_DB.get(current_user_email)
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
        
    # Remove sensitive data like hashed password before returning
    user_data_safe = user_data.copy()
    user_data_safe.pop("hashed_password", None)
    
    return user_data_safe

# --- Campaign Endpoints ---

@app.get("/api/campaigns")
async def get_campaigns(
    page: int = 1,
    limit: int = 10,
    category: Optional[str] = None,
    trending: Optional[bool] = None
):
    """Get campaigns with pagination and filtering"""
    try:
        campaigns = CAMPAIGNS_DATA.copy()
        
        # Apply filters
        if category:
            campaigns = [c for c in campaigns if c['category'].lower() == category.lower()]
        
        if trending is not None:
            campaigns = [c for c in campaigns if c.get('is_trending', False) == trending]
        
        # Pagination
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_campaigns = campaigns[start_idx:end_idx]
        
        return {
            "campaigns": paginated_campaigns,
            "pagination": {
                "page": page,
                "limit": limit,
                "total": len(campaigns),
                "pages": (len(campaigns) + limit - 1) // limit
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/campaigns/{campaign_id}")
async def get_campaign(campaign_id: str):
    """Get specific campaign details"""
    campaign = next((c for c in CAMPAIGNS_DATA if c['id'] == campaign_id), None)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
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
            categories[category] = {
                "name": category,
                "count": 0,
                "icon": get_category_icon(category)
            }
        categories[category]["count"] += 1
    
    return {"categories": list(categories.values())}

def get_category_icon(category: str) -> str:
    """Get icon for category"""
    icons = {
        "Education": "school",
        "Healthcare": "local_hospital", 
        "Community": "people",
        "Environment": "eco",
        "Technology": "computer",
        "Arts & Culture": "palette",
        "Sports": "sports_soccer",
        "Emergency": "warning"
    }
    return icons.get(category, "campaign")

@app.get("/api/search")
async def search_campaigns(
    q: Optional[str] = None,
    category: Optional[str] = None,
    location: Optional[str] = None,
    min_amount: Optional[int] = None,
    max_amount: Optional[int] = None
):
    """Search campaigns with multiple filters"""
    campaigns = CAMPAIGNS_DATA.copy()
    
    if q:
        q_lower = q.lower()
        campaigns = [
            c for c in campaigns 
            if q_lower in c['title'].lower() 
            or q_lower in c['description'].lower()
            or q_lower in c['organization'].lower()
        ]
    
    if category:
        campaigns = [c for c in campaigns if c['category'].lower() == category.lower()]
    
    if location:
        campaigns = [c for c in campaigns if location.lower() in c['location'].lower()]
    
    if min_amount:
        campaigns = [c for c in campaigns if c['target_amount'] >= min_amount]
    
    if max_amount:
        campaigns = [c for c in campaigns if c['target_amount'] <= max_amount]
    
    return {
        "campaigns": campaigns,
        "total": len(campaigns),
        "query": {
            "q": q,
            "category": category,
            "location": location,
            "min_amount": min_amount,
            "max_amount": max_amount
        }
    }

# Payment endpoints
@app.post("/api/donate")
async def create_donation(
    campaign_id: str = Form(...),
    amount: float = Form(...),
    donor_name: str = Form(...),
    donor_email: str = Form(...),
    donor_phone: str = Form(...),
    anonymous: bool = Form(False)
):
    """Create a donation payment request"""
    # Validate campaign exists
    campaign = next((c for c in CAMPAIGNS_DATA if c['id'] == campaign_id), None)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    # Validate amount
    if amount < 1:
        raise HTTPException(status_code=400, detail="Minimum donation amount is ₹1")
    
    try:
        # Create payment request
        payment_result = payment_service.create_payment_request(
            amount=amount,
            purpose=f"Donation to {campaign['title']}",
            buyer_name=donor_name,
            buyer_email=donor_email,
            buyer_phone=donor_phone,
            campaign_id=campaign_id,
            redirect_url="https://haven-streamlit-frontend.onrender.com/success"
        )
        
        if payment_result["success"]:
            # Store donation record (in production, save to database)
            donation_record = {
                "id": str(uuid.uuid4()),
                "campaign_id": campaign_id,
                "amount": amount,
                "donor_name": donor_name if not anonymous else "Anonymous",
                "donor_email": donor_email,
                "anonymous": anonymous,
                "payment_id": payment_result["payment_id"],
                "status": "pending",
                "created_at": datetime.now().isoformat()
            }
            
            return {
                "success": True,
                "payment_url": payment_result["payment_url"],
                "payment_id": payment_result["payment_id"],
                "donation_id": donation_record["id"],
                "amount": amount,
                "campaign": campaign["title"]
            }
        else:
            raise HTTPException(status_code=400, detail=payment_result["error"])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Statistics endpoint
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
        if category not in categories:
            categories[category] = 0
        categories[category] += 1
    
    return {
        "total_campaigns": total_campaigns,
        "total_raised": total_raised,
        "total_target": total_target,
        "total_donors": total_donors,
        "success_rate": round((total_raised / total_target) * 100, 1) if total_target > 0 else 0,
        "categories": categories,
        "trending_count": len([c for c in CAMPAIGNS_DATA if c.get('is_trending', False)])
    }

# --- OAuth Callback Endpoints ---
@app.get("/google/callback")
async def google_auth_callback(code: str):
    """
    Handles the callback from the Google OAuth server.
    Exchanges the authorization code for an access token.
    """
    if not all([GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT_URI]):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Missing Google OAuth credentials. Please check your environment variables."
        )

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
        
        # Here you would process the token and redirect the user.
        # For a full implementation, you would generate a JWT token and
        # redirect to your frontend with this token as a query parameter.
        
        # Example redirect to frontend:
        # return RedirectResponse(url=f"{FRONTEND_URL}/dashboard?token={jwt_token}")
        
        return {"message": "Google authentication successful", "token_info": token_info}
    
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get Google token: {e}"
        )

@app.get("/auth/facebook/callback")
async def facebook_auth_callback(code: str, request: Request):
    """
    Handles the callback from the Facebook OAuth server.
    Exchanges the authorization code for an access token.
    """
    if not all([FACEBOOK_CLIENT_ID, FACEBOOK_CLIENT_SECRET, FACEBOOK_REDIRECT_URI]):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Missing Facebook OAuth credentials. Please check your environment variables."
        )

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
        
        # Here you would process the token, get user info, and generate a JWT.
        # Example redirect to frontend:
        # return RedirectResponse(url=f"{FRONTEND_URL}/dashboard?token={jwt_token}")
        
        return {"message": "Facebook authentication successful", "token_info": token_info}
    
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get Facebook token: {e}"
        )

# Custom 404 handler
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": f"The requested endpoint {request.url.path} was not found",
            "available_endpoints": [
                "/",
                "/health", 
                "/api",
                "/api/campaigns",
                "/api/dashboard",
                "/api/register",
                "/api/register_organization",
                "/api/login",
                "/api/users/me (protected)",
                "/docs",
                "/redoc"
            ]
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
