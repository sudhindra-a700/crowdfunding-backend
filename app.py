from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Request
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

# Initialize FastAPI app
app = FastAPI(
    title="HAVEN Crowdfunding API - Complete Platform",
    description="Complete crowdfunding platform with OAuth, profiles, verification, translation, simplification, and Instamojo payments",
    version="4.0.0",
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
JWT_SECRET = os.environ.get("JWT_SECRET_KEY", "haven-secret-key-2024")
JWT_ALGORITHM = "HS256"

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
    
    def verify_payment(self, payment_id: str, payment_request_id: str) -> Dict:
        """Verify payment status"""
        if not self.api:
            return {"success": False, "error": "Payment service not initialized"}
            
        try:
            response = self.api.payment_request_payment_status(
                id=payment_request_id,
                payment_id=payment_id
            )
            
            if response['success']:
                payment = response['payment_request']['payment']
                return {
                    "success": True,
                    "status": payment['status'],
                    "amount": float(payment['amount']),
                    "buyer_name": payment['buyer_name'],
                    "buyer_email": payment['buyer_email'],
                    "payment_id": payment['payment_id'],
                    "transaction_id": payment.get('transaction_id'),
                    "created_at": payment['created_at']
                }
            else:
                return {
                    "success": False,
                    "error": response.get('message', 'Payment verification failed')
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_payment_status(self, payment_request_id: str) -> Dict:
        """Get payment request status"""
        if not self.api:
            return {"success": False, "error": "Payment service not initialized"}
            
        try:
            response = self.api.payment_request_status(payment_request_id)
            
            if response['success']:
                payment_request = response['payment_request']
                return {
                    "success": True,
                    "status": payment_request['status'],
                    "amount": float(payment_request['amount']),
                    "payments": payment_request.get('payments', [])
                }
            else:
                return {
                    "success": False,
                    "error": response.get('message', 'Status check failed')
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

# Initialize payment service
payment_service = InstamojoPaymentService()

# Translation Service
class TranslationService:
    def __init__(self):
        self.enabled = os.environ.get("TRANSLATION_ENABLED", "true").lower() == "true"
        self.default_language = os.environ.get("TRANSLATION_DEFAULT_LANGUAGE", "en")
        self.supported_languages = {
            "en": "English",
            "hi": "Hindi (हिन्दी)",
            "ta": "Tamil (தமிழ்)",
            "te": "Telugu (తెలుగు)"
        }
    
    def translate_text(self, text: str, target_language: str) -> Dict:
        """Translate text to target language"""
        if not self.enabled:
            return {"success": False, "error": "Translation service disabled"}
        
        if target_language not in self.supported_languages:
            return {"success": False, "error": "Unsupported language"}
        
        if target_language == "en":
            return {"success": True, "translated_text": text, "source_language": "en"}
        
        # Mock translation for demo (replace with actual translation service)
        translations = {
            "hi": {
                "Welcome to HAVEN": "हेवन में आपका स्वागत है",
                "Crowdfunding Platform": "क्राउडफंडिंग प्लेटफॉर्म",
                "Donate Now": "अभी दान करें",
                "Create Campaign": "अभियान बनाएं"
            },
            "ta": {
                "Welcome to HAVEN": "ஹேவனுக்கு வரவேற்கிறோம்",
                "Crowdfunding Platform": "கூட்டு நிதியளிப்பு தளம்",
                "Donate Now": "இப்போது நன்கொடை",
                "Create Campaign": "பிரச்சாரம் உருவாக்கவும்"
            },
            "te": {
                "Welcome to HAVEN": "హేవన్‌కు స్వాగతం",
                "Crowdfunding Platform": "క్రౌడ్‌ఫండింగ్ ప్లాట్‌ఫారమ్",
                "Donate Now": "ఇప్పుడే దానం చేయండి",
                "Create Campaign": "ప్రచారాన్ని సృష్టించండి"
            }
        }
        
        translated = translations.get(target_language, {}).get(text, text)
        return {
            "success": True,
            "translated_text": translated,
            "source_language": "en",
            "target_language": target_language
        }

# Term Simplification Service
class SimplificationService:
    def __init__(self):
        self.enabled = os.environ.get("SIMPLIFICATION_ENABLED", "true").lower() == "true"
        self.terms_database = {
            "crowdfunding": "raising money from many people for a project",
            "campaign": "a project that needs funding",
            "donation": "money given to help a cause",
            "fundraising": "collecting money for a purpose",
            "pledge": "promise to give money",
            "backer": "person who supports a project",
            "goal": "target amount of money to raise",
            "equity": "ownership share in a company",
            "venture capital": "money invested in new businesses",
            "angel investor": "wealthy person who invests in startups",
            "IPO": "Initial Public Offering - when a company sells shares publicly",
            "ROI": "Return on Investment - profit made from an investment",
            "valuation": "estimated worth of a company",
            "startup": "new business company",
            "entrepreneur": "person who starts a business"
        }
    
    def simplify_term(self, term: str) -> Dict:
        """Get simplified explanation of a term"""
        if not self.enabled:
            return {"success": False, "error": "Simplification service disabled"}
        
        term_lower = term.lower()
        if term_lower in self.terms_database:
            return {
                "success": True,
                "term": term,
                "simplified": self.terms_database[term_lower],
                "complexity_level": "simple"
            }
        else:
            return {
                "success": False,
                "error": "Term not found in database"
            }
    
    def get_all_terms(self) -> Dict:
        """Get all available terms"""
        return {
            "success": True,
            "terms": list(self.terms_database.keys()),
            "count": len(self.terms_database)
        }

# Initialize services
translation_service = TranslationService()
simplification_service = SimplificationService()

# Enums
class UserType(str, Enum):
    INDIVIDUAL = "individual"
    ORGANIZATION = "organization"
    NGO = "ngo"

class VerificationStatus(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"

class PaymentStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"

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
class DonationRequest(BaseModel):
    campaign_id: str
    amount: float
    donor_name: str
    donor_email: EmailStr
    donor_phone: str
    message: Optional[str] = None
    is_anonymous: bool = False

class PaymentVerification(BaseModel):
    payment_id: str
    payment_request_id: str
    campaign_id: str

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

class CampaignCreate(BaseModel):
    title: str
    description: str
    goal: float
    category: str
    location: str
    duration_days: int
    image_url: Optional[str] = None

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
    payment_status: PaymentStatus = PaymentStatus.PENDING
    transaction_id: Optional[str] = None

class TranslationRequest(BaseModel):
    text: str
    target_language: str

class SimplificationRequest(BaseModel):
    term: str

# In-memory storage (replace with database in production)
users_db = {}
campaigns_db = {}
donations_db = {}
pending_donations = {}
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
            "description": "Providing clean drinking water access to 500 families in rural Maharashtra through sustainable well construction and water purification systems. This project will install 10 community wells with solar-powered pumps and water purification units.",
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
            "description": "Building a school and providing education materials for 200 children in urban slums. The project includes construction of 8 classrooms, library, computer lab, and providing books, uniforms, and meals for students.",
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
        },
        {
            "id": "camp_003",
            "title": "Cancer Treatment Support",
            "description": "Supporting cancer patients with treatment costs and medical care. This fund helps cover chemotherapy, radiation therapy, medications, and hospital expenses for patients who cannot afford treatment.",
            "goal": 150000.0,
            "raised": 120000.0,
            "creator_id": "user_002",
            "creator_name": "Helping Hands Foundation",
            "category": "medical",
            "location": "Bangalore",
            "image_url": "https://images.unsplash.com/photo-1559757148-5c350d0d3c56?w=600&h=400&fit=crop",
            "created_at": datetime.now() - timedelta(days=30),
            "end_date": datetime.now() + timedelta(days=10),
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
            "message": "Great cause! Happy to contribute to clean water access.",
            "created_at": datetime.now() - timedelta(days=5),
            "is_anonymous": False,
            "payment_status": "completed",
            "transaction_id": "TXN001"
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
            "is_anonymous": False,
            "payment_status": "completed",
            "transaction_id": "TXN002"
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

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Welcome to HAVEN Crowdfunding API - Complete Platform",
        "version": "4.0.0",
        "status": "active",
        "features": [
            "OAuth Authentication (Google, Facebook)",
            "User Profiles & Verification",
            "Document Upload & Verification", 
            "Campaign Management",
            "Donation Tracking",
            "Instamojo Payment Integration",
            "Multi-language Translation",
            "Term Simplification",
            "Real-time Analytics"
        ],
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "campaigns": "/api/campaigns",
            "auth": "/api/auth/*",
            "profiles": "/api/profile/*",
            "payments": "/api/payments/*",
            "translation": "/api/translation/*",
            "simplification": "/api/simplification/*"
        },
        "payment_gateway": "Instamojo",
        "supported_languages": translation_service.supported_languages,
        "timestamp": datetime.now().isoformat()
    }

# Favicon endpoint
@app.get("/favicon.ico")
async def favicon():
    """Favicon endpoint"""
    return JSONResponse(
        status_code=204,
        content=None,
        headers={"Cache-Control": "public, max-age=86400"}
    )

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
            "oauth": "active",
            "payments": "active" if payment_service.api else "inactive",
            "translation": "active" if translation_service.enabled else "inactive",
            "simplification": "active" if simplification_service.enabled else "inactive"
        },
        "database": {
            "users": len(users_db),
            "campaigns": len(campaigns_db),
            "donations": len(donations_db),
            "pending_donations": len(pending_donations),
            "documents": len(documents_db)
        },
        "payment_gateway": {
            "provider": "Instamojo",
            "mode": "sandbox" if payment_service.sandbox else "production",
            "status": "active" if payment_service.api else "inactive"
        },
        "features": {
            "translation_enabled": translation_service.enabled,
            "simplification_enabled": simplification_service.enabled,
            "supported_languages": list(translation_service.supported_languages.keys()),
            "terms_count": len(simplification_service.terms_database)
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
        app_id = os.environ.get("FACEBOOK_CLIENT_ID", "your-facebook-app-id")
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
    
    # Mock OAuth user data (replace with actual OAuth implementation)
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
    
    # Add user's donations if individual (non-anonymous only)
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

# Campaign endpoints
@app.get("/api/campaigns")
async def get_campaigns():
    """Get all campaigns with donation statistics"""
    campaigns_list = []
    for campaign in campaigns_db.values():
        # Add donation statistics
        campaign_donations = [d for d in donations_db.values() if d["campaign_id"] == campaign["id"]]
        campaign_copy = campaign.copy()
        campaign_copy["donation_count"] = len(campaign_donations)
        campaign_copy["progress_percentage"] = (campaign["raised"] / campaign["goal"]) * 100
        campaign_copy["days_remaining"] = max(0, (campaign["end_date"] - datetime.now()).days)
        campaigns_list.append(campaign_copy)
    
    return {"campaigns": campaigns_list}

@app.post("/api/campaigns")
async def create_campaign(
    campaign_data: CampaignCreate,
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
        "title": campaign_data.title,
        "description": campaign_data.description,
        "goal": campaign_data.goal,
        "raised": 0.0,
        "creator_id": current_user,
        "creator_name": f"{user['first_name']} {user['last_name']}",
        "category": campaign_data.category,
        "location": campaign_data.location,
        "image_url": campaign_data.image_url,
        "created_at": datetime.now(),
        "end_date": datetime.now() + timedelta(days=campaign_data.duration_days),
        "status": "active"
    }
    
    campaigns_db[campaign_id] = campaign
    
    # Update user's campaign count
    user["total_campaigns"] = user.get("total_campaigns", 0) + 1
    users_db[current_user] = user
    
    return {"message": "Campaign created successfully", "campaign": campaign}

@app.get("/api/campaigns/{campaign_id}")
async def get_campaign(campaign_id: str):
    """Get specific campaign details"""
    if campaign_id not in campaigns_db:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    campaign = campaigns_db[campaign_id].copy()
    
    # Add donation statistics
    campaign_donations = [d for d in donations_db.values() if d["campaign_id"] == campaign_id]
    campaign["donation_count"] = len(campaign_donations)
    campaign["progress_percentage"] = (campaign["raised"] / campaign["goal"]) * 100
    campaign["days_remaining"] = max(0, (campaign["end_date"] - datetime.now()).days)
    
    # Add recent donations (non-anonymous)
    recent_donations = []
    for donation in sorted(campaign_donations, key=lambda x: x["created_at"], reverse=True)[:5]:
        if not donation["is_anonymous"]:
            recent_donations.append({
                "donor_name": donation["donor_name"],
                "amount": donation["amount"],
                "message": donation.get("message"),
                "created_at": donation["created_at"]
            })
    campaign["recent_donations"] = recent_donations
    
    return {"campaign": campaign}

# Payment endpoints
@app.post("/api/payments/create-donation")
async def create_donation_payment(
    donation: DonationRequest,
    current_user: str = Depends(get_current_user)
):
    """Create payment request for campaign donation"""
    
    # Validate campaign exists
    if donation.campaign_id not in campaigns_db:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    campaign = campaigns_db[donation.campaign_id]
    
    # Validate donation amount
    if donation.amount < 100:
        raise HTTPException(status_code=400, detail="Minimum donation amount is ₹100")
    
    if donation.amount > 100000:
        raise HTTPException(status_code=400, detail="Maximum donation amount is ₹1,00,000")
    
    # Create payment request
    frontend_url = os.environ.get("FRONTEND_BASE_URI", "https://haven-streamlit-frontend.onrender.com")
    redirect_url = f"{frontend_url}/payment/callback"
    
    payment_result = payment_service.create_payment_request(
        amount=donation.amount,
        purpose=f"Donation to {campaign['title']}",
        buyer_name=donation.donor_name,
        buyer_email=donation.donor_email,
        buyer_phone=donation.donor_phone,
        campaign_id=donation.campaign_id,
        redirect_url=redirect_url
    )
    
    if payment_result["success"]:
        # Store pending donation
        donation_id = f"don_{uuid.uuid4().hex[:8]}"
        pending_donations[donation_id] = {
            "id": donation_id,
            "campaign_id": donation.campaign_id,
            "campaign_title": campaign["title"],
            "donor_id": current_user,
            "amount": donation.amount,
            "donor_name": donation.donor_name,
            "donor_email": donation.donor_email,
            "donor_phone": donation.donor_phone,
            "message": donation.message,
            "is_anonymous": donation.is_anonymous,
            "payment_request_id": payment_result["payment_id"],
            "status": "pending",
            "created_at": datetime.now()
        }
        
        return {
            "success": True,
            "donation_id": donation_id,
            "payment_url": payment_result["payment_url"],
            "payment_request_id": payment_result["payment_id"],
            "amount": donation.amount,
            "campaign_title": campaign["title"]
        }
    else:
        raise HTTPException(status_code=400, detail=payment_result["error"])

@app.post("/api/payments/verify-payment")
async def verify_donation_payment(verification: PaymentVerification):
    """Verify payment and complete donation"""
    
    # Verify payment with Instamojo
    verification_result = payment_service.verify_payment(
        verification.payment_id,
        verification.payment_request_id
    )
    
    if verification_result["success"] and verification_result["status"] == "Credit":
        # Find pending donation
        donation = None
        donation_id = None
        
        for did, don in pending_donations.items():
            if don["payment_request_id"] == verification.payment_request_id:
                donation = don
                donation_id = did
                break
        
        if donation:
            # Complete the donation
            donation["status"] = "completed"
            donation["payment_status"] = "completed"
            donation["transaction_id"] = verification_result.get("transaction_id")
            donation["completed_at"] = datetime.now()
            
            # Add to completed donations
            donations_db[donation_id] = donation
            
            # Update campaign raised amount
            campaign = campaigns_db[verification.campaign_id]
            campaign["raised"] += donation["amount"]
            campaigns_db[verification.campaign_id] = campaign
            
            # Update user donation total
            if donation["donor_id"] in users_db:
                user = users_db[donation["donor_id"]]
                user["total_donations"] = user.get("total_donations", 0) + donation["amount"]
                users_db[donation["donor_id"]] = user
            
            # Remove from pending
            del pending_donations[donation_id]
            
            return {
                "success": True,
                "message": "Donation completed successfully",
                "donation": donation,
                "campaign_new_total": campaign["raised"],
                "campaign_progress": (campaign["raised"] / campaign["goal"]) * 100
            }
        else:
            raise HTTPException(status_code=404, detail="Donation not found")
    else:
        raise HTTPException(status_code=400, detail="Payment verification failed")

@app.get("/api/payments/donations/{user_id}")
async def get_user_donations(
    user_id: str,
    current_user: str = Depends(get_current_user)
):
    """Get user's donation history"""
    
    user_donations = []
    for donation in donations_db.values():
        if donation["donor_id"] == user_id:
            # Show all donations to the donor, only non-anonymous to others
            if current_user == user_id or not donation["is_anonymous"]:
                user_donations.append(donation)
    
    return {
        "donations": user_donations,
        "total_donated": sum(d["amount"] for d in user_donations),
        "donation_count": len(user_donations)
    }

# Translation endpoints
@app.post("/api/translation/translate")
async def translate_text(request: TranslationRequest):
    """Translate text to target language"""
    result = translation_service.translate_text(request.text, request.target_language)
    
    if result["success"]:
        return result
    else:
        raise HTTPException(status_code=400, detail=result["error"])

@app.get("/api/translation/languages")
async def get_supported_languages():
    """Get supported languages"""
    return {
        "languages": translation_service.supported_languages,
        "default": translation_service.default_language
    }

# Simplification endpoints
@app.post("/api/simplification/simplify")
async def simplify_term(request: SimplificationRequest):
    """Get simplified explanation of a term"""
    result = simplification_service.simplify_term(request.term)
    
    if result["success"]:
        return result
    else:
        raise HTTPException(status_code=404, detail=result["error"])

@app.get("/api/simplification/terms")
async def get_all_terms():
    """Get all available terms"""
    return simplification_service.get_all_terms()

# Search endpoint
@app.post("/api/search")
async def search_campaigns(query: str, category: Optional[str] = None, limit: int = 20):
    """Search campaigns"""
    results = []
    query_lower = query.lower()
    
    for campaign in campaigns_db.values():
        if (query_lower in campaign["title"].lower() or 
            query_lower in campaign["description"].lower() or
            query_lower in campaign["category"].lower() or
            query_lower in campaign["location"].lower()):
            
            if category and campaign["category"] != category:
                continue
            
            # Add campaign statistics
            campaign_copy = campaign.copy()
            campaign_donations = [d for d in donations_db.values() if d["campaign_id"] == campaign["id"]]
            campaign_copy["donation_count"] = len(campaign_donations)
            campaign_copy["progress_percentage"] = (campaign["raised"] / campaign["goal"]) * 100
            campaign_copy["days_remaining"] = max(0, (campaign["end_date"] - datetime.now()).days)
            
            results.append(campaign_copy)
    
    return {"campaigns": results[:limit], "total_found": len(results)}

# Analytics endpoint
@app.get("/api/analytics/dashboard")
async def get_dashboard_analytics(current_user: str = Depends(get_current_user)):
    """Get dashboard analytics"""
    
    total_campaigns = len(campaigns_db)
    total_donations = len(donations_db)
    total_users = len(users_db)
    total_raised = sum(campaign["raised"] for campaign in campaigns_db.values())
    
    # Recent activity
    recent_donations = sorted(donations_db.values(), key=lambda x: x["created_at"], reverse=True)[:5]
    recent_campaigns = sorted(campaigns_db.values(), key=lambda x: x["created_at"], reverse=True)[:5]
    
    # Category statistics
    category_stats = {}
    for campaign in campaigns_db.values():
        category = campaign["category"]
        if category not in category_stats:
            category_stats[category] = {"count": 0, "raised": 0}
        category_stats[category]["count"] += 1
        category_stats[category]["raised"] += campaign["raised"]
    
    return {
        "overview": {
            "total_campaigns": total_campaigns,
            "total_donations": total_donations,
            "total_users": total_users,
            "total_raised": total_raised
        },
        "recent_activity": {
            "donations": recent_donations,
            "campaigns": recent_campaigns
        },
        "category_statistics": category_stats,
        "payment_gateway_status": "active" if payment_service.api else "inactive"
    }

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
                "health": "/health",
                "docs": "/docs",
                "campaigns": "/api/campaigns",
                "auth": "/api/auth/*",
                "profiles": "/api/profile/*",
                "payments": "/api/payments/*",
                "translation": "/api/translation/*",
                "simplification": "/api/simplification/*",
                "analytics": "/api/analytics/*"
            }
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

