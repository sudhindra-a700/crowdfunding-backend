from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
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

# Initialize FastAPI app
app = FastAPI(
    title="HAVEN Crowdfunding API",
    description="Complete crowdfunding platform with OAuth, profiles, and verification",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
JWT_SECRET = os.environ.get("JWT_SECRET", "haven-secret-key-2024")
JWT_ALGORITHM = "HS256"

# Enums
class UserType(str, Enum):
    INDIVIDUAL = "individual"
    ORGANIZATION = "organization"
    NGO = "ngo"

class VerificationStatus(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"

class DocumentType(str, Enum):
    # Individual documents
    AADHAR = "aadhar"
    PAN = "pan"
    PASSPORT = "passport"
    DRIVING_LICENSE = "driving_license"
    VOTER_ID = "voter_id"
    
    # Organization documents
    INCORPORATION_CERTIFICATE = "incorporation_certificate"
    GST_CERTIFICATE = "gst_certificate"
    NGO_REGISTRATION = "ngo_registration"
    TRUST_DEED = "trust_deed"
    SOCIETY_REGISTRATION = "society_registration"

# Pydantic models
class UserBase(BaseModel):
    email: EmailStr
    first_name: str
    last_name: str
    phone: Optional[str] = None
    user_type: UserType
    address: Optional[str] = None

class UserCreate(UserBase):
    password: Optional[str] = None
    oauth_provider: Optional[str] = None
    oauth_id: Optional[str] = None

class UserProfile(UserBase):
    id: str
    created_at: datetime
    verification_status: VerificationStatus
    profile_image: Optional[str] = None
    bio: Optional[str] = None
    website: Optional[str] = None
    
    # Organization specific fields
    organization_name: Optional[str] = None
    organization_type: Optional[str] = None
    registration_number: Optional[str] = None
    
    # Statistics
    total_donations: Optional[float] = 0.0
    total_campaigns: Optional[int] = 0
    total_raised: Optional[float] = 0.0

class DocumentUpload(BaseModel):
    document_type: DocumentType
    document_number: Optional[str] = None
    description: Optional[str] = None

class Campaign(BaseModel):
    id: str
    title: str
    description: str
    goal: float
    raised: float
    creator_id: str
    creator_name: str
    category: str
    location: str
    image_url: Optional[str] = None
    created_at: datetime
    end_date: datetime
    status: str = "active"

class Donation(BaseModel):
    id: str
    campaign_id: str
    campaign_title: str
    donor_id: str
    donor_name: str
    amount: float
    message: Optional[str] = None
    created_at: datetime
    is_anonymous: bool = False

# In-memory storage (replace with database in production)
users_db = {}
campaigns_db = {}
donations_db = {}
documents_db = {}
oauth_states = {}

# Sample data initialization
def init_sample_data():
    # Sample users
    sample_users = [
        {
            "id": "user_001",
            "email": "john.doe@example.com",
            "first_name": "John",
            "last_name": "Doe",
            "phone": "+91-9876543210",
            "user_type": "individual",
            "address": "123 Main St, Mumbai, Maharashtra",
            "created_at": datetime.now() - timedelta(days=30),
            "verification_status": "verified",
            "bio": "Passionate about helping communities and supporting education initiatives.",
            "total_donations": 25000.0
        },
        {
            "id": "user_002",
            "email": "contact@helpinghands.org",
            "first_name": "Helping",
            "last_name": "Hands",
            "phone": "+91-9876543211",
            "user_type": "organization",
            "address": "456 NGO Street, Delhi, Delhi",
            "created_at": datetime.now() - timedelta(days=60),
            "verification_status": "verified",
            "organization_name": "Helping Hands Foundation",
            "organization_type": "NGO",
            "registration_number": "NGO/2020/001",
            "bio": "Dedicated to improving lives through education, healthcare, and community development.",
            "website": "https://helpinghands.org",
            "total_campaigns": 5,
            "total_raised": 150000.0
        }
    ]
    
    for user in sample_users:
        users_db[user["id"]] = user
    
    # Sample campaigns
    sample_campaigns = [
        {
            "id": "camp_001",
            "title": "Clean Water for Rural Villages",
            "description": "Providing clean drinking water access to 500 families in rural Maharashtra through sustainable well construction and water purification systems.",
            "goal": 500000.0,
            "raised": 325000.0,
            "creator_id": "user_002",
            "creator_name": "Helping Hands Foundation",
            "category": "community",
            "location": "Maharashtra",
            "image_url": "https://images.unsplash.com/photo-1541919329513-35f7af297129?w=600&h=400&fit=crop",
            "created_at": datetime.now() - timedelta(days=20),
            "end_date": datetime.now() + timedelta(days=15),
            "status": "active"
        },
        {
            "id": "camp_002",
            "title": "Education for Underprivileged Children",
            "description": "Building a school and providing education materials for 200 children in urban slums.",
            "goal": 750000.0,
            "raised": 450000.0,
            "creator_id": "user_002",
            "creator_name": "Helping Hands Foundation",
            "category": "education",
            "location": "Delhi",
            "image_url": "https://images.unsplash.com/photo-1497486751825-1233686d5d80?w=600&h=400&fit=crop",
            "created_at": datetime.now() - timedelta(days=25),
            "end_date": datetime.now() + timedelta(days=25),
            "status": "active"
        }
    ]
    
    for campaign in sample_campaigns:
        campaigns_db[campaign["id"]] = campaign
    
    # Sample donations
    sample_donations = [
        {
            "id": "don_001",
            "campaign_id": "camp_001",
            "campaign_title": "Clean Water for Rural Villages",
            "donor_id": "user_001",
            "donor_name": "John Doe",
            "amount": 5000.0,
            "message": "Great cause! Happy to contribute.",
            "created_at": datetime.now() - timedelta(days=5),
            "is_anonymous": False
        },
        {
            "id": "don_002",
            "campaign_id": "camp_002",
            "campaign_title": "Education for Underprivileged Children",
            "donor_id": "user_001",
            "donor_name": "John Doe",
            "amount": 10000.0,
            "message": "Education is the key to a better future.",
            "created_at": datetime.now() - timedelta(days=3),
            "is_anonymous": False
        }
    ]
    
    for donation in sample_donations:
        donations_db[donation["id"]] = donation

# Initialize sample data
init_sample_data()

# Utility functions
def create_jwt_token(user_id: str) -> str:
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(days=7)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload.get("user_id")
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    user_id = verify_jwt_token(credentials.credentials)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user_id

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# Root endpoint - Fix for 404 on /
@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Welcome to HAVEN Crowdfunding API",
        "version": "2.0.0",
        "status": "active",
        "features": [
            "OAuth Authentication",
            "User Profiles",
            "Document Verification", 
            "Campaign Management",
            "Donation Tracking"
        ],
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "redoc": "/redoc",
            "campaigns": "/api/campaigns",
            "auth": "/api/auth/*",
            "profiles": "/api/profile/*"
        },
        "timestamp": datetime.now().isoformat()
    }

# Favicon endpoint - Fix for 404 on /favicon.ico
@app.get("/favicon.ico")
async def favicon():
    """Favicon endpoint"""
    # Return a simple response or redirect to a favicon URL
    return JSONResponse(
        status_code=204,
        content=None,
        headers={"Cache-Control": "public, max-age=86400"}
    )

# API Info endpoint
@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "api": "HAVEN Crowdfunding Platform",
        "version": "2.0.0",
        "description": "Complete crowdfunding platform with OAuth, profiles, and verification",
        "status": "operational",
        "endpoints": {
            "authentication": {
                "oauth_url": "/api/auth/oauth/{provider}/url",
                "oauth_callback": "/api/auth/oauth/callback", 
                "login": "/api/auth/login",
                "register": "/api/auth/register"
            },
            "profiles": {
                "my_profile": "/api/profile/me",
                "public_profile": "/api/profile/{user_id}",
                "update_profile": "/api/profile/me [PUT]"
            },
            "verification": {
                "upload_document": "/api/verification/upload",
                "get_documents": "/api/verification/documents",
                "requirements": "/api/verification/requirements"
            },
            "campaigns": {
                "list_campaigns": "/api/campaigns",
                "create_campaign": "/api/campaigns [POST]",
                "search_campaigns": "/api/search"
            }
        },
        "timestamp": datetime.now().isoformat()
    }

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "authentication": "active",
            "campaigns": "active",
            "profiles": "active",
            "verification": "active",
            "oauth": "active"
        },
        "database": {
            "users": len(users_db),
            "campaigns": len(campaigns_db),
            "donations": len(donations_db),
            "documents": len(documents_db)
        }
    }

# OAuth endpoints
@app.get("/api/auth/oauth/{provider}/url")
async def get_oauth_url(provider: str):
    """Generate OAuth URL for popup authentication"""
    state = secrets.token_urlsafe(32)
    oauth_states[state] = {"provider": provider, "created_at": datetime.now()}
    
    if provider == "google":
        client_id = os.environ.get("GOOGLE_CLIENT_ID", "your-google-client-id")
        redirect_uri = os.environ.get("GOOGLE_REDIRECT_URI", "https://haven-streamlit-frontend.onrender.com/auth/callback")
        
        oauth_url = (
            f"https://accounts.google.com/o/oauth2/v2/auth?"
            f"client_id={client_id}&"
            f"redirect_uri={redirect_uri}&"
            f"scope=openid email profile&"
            f"response_type=code&"
            f"state={state}"
        )
    elif provider == "facebook":
        app_id = os.environ.get("FACEBOOK_APP_ID", "your-facebook-app-id")
        redirect_uri = os.environ.get("FACEBOOK_REDIRECT_URI", "https://haven-streamlit-frontend.onrender.com/auth/callback")
        
        oauth_url = (
            f"https://www.facebook.com/v18.0/dialog/oauth?"
            f"client_id={app_id}&"
            f"redirect_uri={redirect_uri}&"
            f"scope=email,public_profile&"
            f"response_type=code&"
            f"state={state}"
        )
    else:
        raise HTTPException(status_code=400, detail="Unsupported OAuth provider")
    
    return {"oauth_url": oauth_url, "state": state}

@app.post("/api/auth/oauth/callback")
async def oauth_callback(code: str, state: str, provider: str):
    """Handle OAuth callback and create/login user"""
    if state not in oauth_states:
        raise HTTPException(status_code=400, detail="Invalid state parameter")
    
    # In a real implementation, you would:
    # 1. Exchange code for access token
    # 2. Get user info from OAuth provider
    # 3. Create or update user in database
    
    # Mock OAuth user data
    if provider == "google":
        oauth_user = {
            "id": f"google_{secrets.token_hex(8)}",
            "email": "user@gmail.com",
            "first_name": "Google",
            "last_name": "User",
            "profile_image": "https://via.placeholder.com/150"
        }
    else:  # facebook
        oauth_user = {
            "id": f"facebook_{secrets.token_hex(8)}",
            "email": "user@facebook.com",
            "first_name": "Facebook",
            "last_name": "User",
            "profile_image": "https://via.placeholder.com/150"
        }
    
    # Check if user exists
    existing_user = None
    for user_id, user in users_db.items():
        if user.get("email") == oauth_user["email"]:
            existing_user = user
            break
    
    if existing_user:
        user_id = existing_user["id"]
    else:
        # Create new user
        user_id = f"user_{secrets.token_hex(8)}"
        users_db[user_id] = {
            "id": user_id,
            "email": oauth_user["email"],
            "first_name": oauth_user["first_name"],
            "last_name": oauth_user["last_name"],
            "user_type": "individual",
            "oauth_provider": provider,
            "oauth_id": oauth_user["id"],
            "profile_image": oauth_user.get("profile_image"),
            "created_at": datetime.now(),
            "verification_status": "pending",
            "total_donations": 0.0
        }
    
    # Clean up state
    del oauth_states[state]
    
    # Generate JWT token
    token = create_jwt_token(user_id)
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": users_db[user_id]
    }

# Authentication endpoints
@app.post("/api/auth/register")
async def register_user(user_data: UserCreate):
    """Register a new user"""
    # Check if user already exists
    for existing_user in users_db.values():
        if existing_user["email"] == user_data.email:
            raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    user_id = f"user_{secrets.token_hex(8)}"
    new_user = {
        "id": user_id,
        **user_data.dict(),
        "created_at": datetime.now(),
        "verification_status": "pending",
        "total_donations": 0.0,
        "total_campaigns": 0,
        "total_raised": 0.0
    }
    
    if user_data.password:
        new_user["password_hash"] = hash_password(user_data.password)
    
    users_db[user_id] = new_user
    
    # Generate JWT token
    token = create_jwt_token(user_id)
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {k: v for k, v in new_user.items() if k != "password_hash"}
    }

@app.post("/api/auth/login")
async def login_user(email: EmailStr, password: str):
    """Login user with email and password"""
    user = None
    user_id = None
    
    for uid, u in users_db.items():
        if u["email"] == email:
            user = u
            user_id = uid
            break
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    if "password_hash" not in user:
        raise HTTPException(status_code=401, detail="Please use OAuth login")
    
    if user["password_hash"] != hash_password(password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    # Generate JWT token
    token = create_jwt_token(user_id)
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {k: v for k, v in user.items() if k != "password_hash"}
    }

# Profile endpoints
@app.get("/api/profile/{user_id}")
async def get_user_profile(user_id: str):
    """Get user profile (public view)"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = users_db[user_id]
    profile = {k: v for k, v in user.items() if k != "password_hash"}
    
    # Add user's campaigns if organization
    if user["user_type"] in ["organization", "ngo"]:
        user_campaigns = []
        for campaign in campaigns_db.values():
            if campaign["creator_id"] == user_id:
                user_campaigns.append(campaign)
        profile["campaigns"] = user_campaigns
    
    # Add user's donations if individual
    if user["user_type"] == "individual":
        user_donations = []
        for donation in donations_db.values():
            if donation["donor_id"] == user_id and not donation["is_anonymous"]:
                user_donations.append(donation)
        profile["donations"] = user_donations
    
    return {"profile": profile}

@app.get("/api/profile/me")
async def get_my_profile(current_user: str = Depends(get_current_user)):
    """Get current user's profile (private view)"""
    if current_user not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = users_db[current_user]
    profile = {k: v for k, v in user.items() if k != "password_hash"}
    
    # Add all campaigns for organizations
    if user["user_type"] in ["organization", "ngo"]:
        user_campaigns = []
        for campaign in campaigns_db.values():
            if campaign["creator_id"] == current_user:
                user_campaigns.append(campaign)
        profile["campaigns"] = user_campaigns
    
    # Add all donations for individuals (including anonymous)
    if user["user_type"] == "individual":
        user_donations = []
        for donation in donations_db.values():
            if donation["donor_id"] == current_user:
                user_donations.append(donation)
        profile["donations"] = user_donations
    
    # Add verification documents
    user_documents = []
    for doc in documents_db.values():
        if doc["user_id"] == current_user:
            user_documents.append(doc)
    profile["documents"] = user_documents
    
    return {"profile": profile}

@app.put("/api/profile/me")
async def update_my_profile(
    first_name: Optional[str] = Form(None),
    last_name: Optional[str] = Form(None),
    phone: Optional[str] = Form(None),
    address: Optional[str] = Form(None),
    bio: Optional[str] = Form(None),
    website: Optional[str] = Form(None),
    organization_name: Optional[str] = Form(None),
    organization_type: Optional[str] = Form(None),
    profile_image: Optional[UploadFile] = File(None),
    current_user: str = Depends(get_current_user)
):
    """Update current user's profile"""
    if current_user not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = users_db[current_user]
    
    # Update fields
    if first_name:
        user["first_name"] = first_name
    if last_name:
        user["last_name"] = last_name
    if phone:
        user["phone"] = phone
    if address:
        user["address"] = address
    if bio:
        user["bio"] = bio
    if website:
        user["website"] = website
    if organization_name:
        user["organization_name"] = organization_name
    if organization_type:
        user["organization_type"] = organization_type
    
    # Handle profile image upload
    if profile_image:
        # In production, save to cloud storage
        image_url = f"https://storage.example.com/profiles/{current_user}_{profile_image.filename}"
        user["profile_image"] = image_url
    
    users_db[current_user] = user
    
    return {"message": "Profile updated successfully", "profile": {k: v for k, v in user.items() if k != "password_hash"}}

# Document verification endpoints
@app.post("/api/verification/upload")
async def upload_verification_document(
    document_type: DocumentType = Form(...),
    document_number: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    document_file: UploadFile = File(...),
    current_user: str = Depends(get_current_user)
):
    """Upload verification document"""
    if current_user not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = users_db[current_user]
    
    # Validate document type based on user type
    individual_docs = [DocumentType.AADHAR, DocumentType.PAN, DocumentType.PASSPORT, 
                      DocumentType.DRIVING_LICENSE, DocumentType.VOTER_ID]
    org_docs = [DocumentType.INCORPORATION_CERTIFICATE, DocumentType.GST_CERTIFICATE,
               DocumentType.NGO_REGISTRATION, DocumentType.TRUST_DEED, DocumentType.SOCIETY_REGISTRATION]
    
    if user["user_type"] == "individual" and document_type not in individual_docs:
        raise HTTPException(status_code=400, detail="Invalid document type for individual")
    
    if user["user_type"] in ["organization", "ngo"] and document_type not in org_docs:
        raise HTTPException(status_code=400, detail="Invalid document type for organization")
    
    # Create document record
    doc_id = f"doc_{secrets.token_hex(8)}"
    document = {
        "id": doc_id,
        "user_id": current_user,
        "document_type": document_type,
        "document_number": document_number,
        "description": description,
        "file_name": document_file.filename,
        "file_url": f"https://storage.example.com/documents/{doc_id}_{document_file.filename}",
        "uploaded_at": datetime.now(),
        "verification_status": "pending"
    }
    
    documents_db[doc_id] = document
    
    # Update user verification status to pending if not already verified
    if user["verification_status"] != "verified":
        user["verification_status"] = "pending"
        users_db[current_user] = user
    
    return {"message": "Document uploaded successfully", "document": document}

@app.get("/api/verification/documents")
async def get_my_documents(current_user: str = Depends(get_current_user)):
    """Get current user's verification documents"""
    user_documents = []
    for doc in documents_db.values():
        if doc["user_id"] == current_user:
            user_documents.append(doc)
    
    return {"documents": user_documents}

@app.get("/api/verification/requirements")
async def get_verification_requirements(current_user: str = Depends(get_current_user)):
    """Get verification requirements for current user"""
    if current_user not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = users_db[current_user]
    
    if user["user_type"] == "individual":
        requirements = {
            "required_documents": 1,
            "document_types": [
                {"type": "aadhar", "name": "Aadhar Card", "description": "Government issued identity card"},
                {"type": "pan", "name": "PAN Card", "description": "Permanent Account Number card"},
                {"type": "passport", "name": "Passport", "description": "Valid passport"},
                {"type": "driving_license", "name": "Driving License", "description": "Valid driving license"},
                {"type": "voter_id", "name": "Voter ID", "description": "Voter identification card"}
            ],
            "instructions": "Upload any one of the above documents to verify your identity."
        }
    else:
        requirements = {
            "required_documents": 1,
            "document_types": [
                {"type": "incorporation_certificate", "name": "Certificate of Incorporation", "description": "Company registration certificate"},
                {"type": "gst_certificate", "name": "GST Certificate", "description": "Goods and Services Tax registration"},
                {"type": "ngo_registration", "name": "NGO Registration", "description": "Non-profit organization registration"},
                {"type": "trust_deed", "name": "Trust Deed", "description": "Trust registration document"},
                {"type": "society_registration", "name": "Society Registration", "description": "Society registration certificate"}
            ],
            "instructions": "Upload your organization's registration certificate to verify authenticity."
        }
    
    return requirements

# Campaign endpoints
@app.get("/api/campaigns")
async def get_campaigns():
    """Get all campaigns"""
    campaigns_list = list(campaigns_db.values())
    return {"campaigns": campaigns_list}

@app.post("/api/campaigns")
async def create_campaign(
    title: str = Form(...),
    description: str = Form(...),
    goal: float = Form(...),
    category: str = Form(...),
    location: str = Form(...),
    duration_days: int = Form(...),
    current_user: str = Depends(get_current_user)
):
    """Create a new campaign"""
    if current_user not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = users_db[current_user]
    
    # Check if user is verified for creating campaigns
    if user["verification_status"] != "verified":
        raise HTTPException(status_code=403, detail="Account must be verified to create campaigns")
    
    campaign_id = f"camp_{secrets.token_hex(8)}"
    campaign = {
        "id": campaign_id,
        "title": title,
        "description": description,
        "goal": goal,
        "raised": 0.0,
        "creator_id": current_user,
        "creator_name": f"{user['first_name']} {user['last_name']}",
        "category": category,
        "location": location,
        "created_at": datetime.now(),
        "end_date": datetime.now() + timedelta(days=duration_days),
        "status": "active"
    }
    
    campaigns_db[campaign_id] = campaign
    
    # Update user's campaign count
    user["total_campaigns"] = user.get("total_campaigns", 0) + 1
    users_db[current_user] = user
    
    return {"message": "Campaign created successfully", "campaign": campaign}

@app.post("/api/search")
async def search_campaigns(query: str, category: Optional[str] = None, limit: int = 20):
    """Search campaigns"""
    results = []
    query_lower = query.lower()
    
    for campaign in campaigns_db.values():
        if (query_lower in campaign["title"].lower() or 
            query_lower in campaign["description"].lower() or
            query_lower in campaign["category"].lower()):
            
            if category and campaign["category"] != category:
                continue
                
            results.append(campaign)
    
    return {"campaigns": results[:limit]}

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"The requested endpoint {request.url.path} was not found",
            "available_endpoints": {
                "root": "/",
                "api_info": "/api", 
                "health": "/health",
                "docs": "/docs",
                "campaigns": "/api/campaigns",
                "auth": "/api/auth/*",
                "profiles": "/api/profile/*"
            }
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

