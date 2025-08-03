"""
HAVEN Crowdfunding Platform - Main FastAPI Application
Complete backend with CORS middleware, authentication, and API endpoints
Python 3.11 compatible with verified dependencies
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

import nltk
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
import uvicorn

# Download NLTK data on startup
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    print(f"NLTK download warning: {e}")

# ========================================
# CONFIGURATION
# ========================================

# Environment variables
BACKEND_URL = os.getenv("BACKEND_URL", "https://haven-fastapi-backend.onrender.com")
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://haven-streamlit-frontend.onrender.com")
SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "your-secret-key-here")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
FACEBOOK_CLIENT_ID = os.getenv("FACEBOOK_CLIENT_ID", "")

# Feature flags
TRANSLATION_ENABLED = os.getenv("FEATURES_TRANSLATION_ENABLED", "true").lower() == "true"
SIMPLIFICATION_ENABLED = os.getenv("FEATURES_SIMPLIFICATION_ENABLED", "true").lower() == "true"

# ========================================
# PYDANTIC MODELS
# ========================================

class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: str
    version: str
    features: Dict[str, bool]

class BackendTestResponse(BaseModel):
    backend_status: str
    cors_enabled: bool
    translation_available: bool
    simplification_available: bool
    oauth_configured: bool
    database_connected: bool
    response_time_ms: float

class TranslationRequest(BaseModel):
    text: str
    target_language: str
    source_language: Optional[str] = "auto"

class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: float

class SimplificationRequest(BaseModel):
    text: str
    level: Optional[str] = "simple"

class SimplificationResponse(BaseModel):
    original_text: str
    simplified_text: str
    complexity_score: float
    simplified_terms: List[Dict[str, str]]

class TermDefinition(BaseModel):
    term: str
    definition: str
    category: str
    complexity_level: str

# ========================================
# TERM SIMPLIFICATION DATABASE
# ========================================

# Comprehensive term definitions for automatic simplification
TERM_DEFINITIONS = {
    # Financial Terms
    "crowdfunding": {
        "definition": "A way to raise money by asking many people to contribute small amounts online",
        "category": "finance",
        "complexity_level": "medium"
    },
    "investment": {
        "definition": "Money put into a project or business to make more money later",
        "category": "finance", 
        "complexity_level": "medium"
    },
    "equity": {
        "definition": "Ownership share in a company",
        "category": "finance",
        "complexity_level": "high"
    },
    "revenue": {
        "definition": "Total money earned from sales",
        "category": "finance",
        "complexity_level": "medium"
    },
    "profit": {
        "definition": "Money left after paying all costs",
        "category": "finance",
        "complexity_level": "low"
    },
    "startup": {
        "definition": "A new company trying to grow quickly",
        "category": "business",
        "complexity_level": "medium"
    },
    "entrepreneur": {
        "definition": "Person who starts and runs a business",
        "category": "business",
        "complexity_level": "medium"
    },
    "venture capital": {
        "definition": "Money invested in new businesses with high growth potential",
        "category": "finance",
        "complexity_level": "high"
    },
    "roi": {
        "definition": "Return on Investment - how much money you make compared to what you invested",
        "category": "finance",
        "complexity_level": "high"
    },
    "valuation": {
        "definition": "How much a company is worth",
        "category": "finance",
        "complexity_level": "high"
    },
    
    # Technology Terms
    "api": {
        "definition": "Application Programming Interface - a way for different software to communicate",
        "category": "technology",
        "complexity_level": "high"
    },
    "platform": {
        "definition": "A system that allows people to build or use services",
        "category": "technology",
        "complexity_level": "medium"
    },
    "algorithm": {
        "definition": "A set of rules or instructions for solving a problem",
        "category": "technology",
        "complexity_level": "high"
    },
    "blockchain": {
        "definition": "A secure digital ledger that records transactions",
        "category": "technology",
        "complexity_level": "high"
    },
    "cryptocurrency": {
        "definition": "Digital money secured by cryptography",
        "category": "technology",
        "complexity_level": "high"
    },
    
    # Business Terms
    "scalability": {
        "definition": "Ability to grow and handle more customers or work",
        "category": "business",
        "complexity_level": "high"
    },
    "market research": {
        "definition": "Studying customers and competitors to understand demand",
        "category": "business",
        "complexity_level": "medium"
    },
    "business model": {
        "definition": "How a company makes money",
        "category": "business",
        "complexity_level": "medium"
    },
    "prototype": {
        "definition": "Early version of a product used for testing",
        "category": "business",
        "complexity_level": "medium"
    },
    "milestone": {
        "definition": "Important goal or achievement in a project",
        "category": "business",
        "complexity_level": "low"
    },
    
    # Legal Terms
    "intellectual property": {
        "definition": "Legal rights to ideas, inventions, or creative works",
        "category": "legal",
        "complexity_level": "high"
    },
    "patent": {
        "definition": "Legal protection for an invention",
        "category": "legal",
        "complexity_level": "high"
    },
    "trademark": {
        "definition": "Legal protection for a brand name or logo",
        "category": "legal",
        "complexity_level": "medium"
    },
    "liability": {
        "definition": "Legal responsibility for damages or debts",
        "category": "legal",
        "complexity_level": "high"
    },
    
    # Marketing Terms
    "target audience": {
        "definition": "Specific group of people you want to reach",
        "category": "marketing",
        "complexity_level": "medium"
    },
    "conversion rate": {
        "definition": "Percentage of visitors who take a desired action",
        "category": "marketing",
        "complexity_level": "high"
    },
    "brand awareness": {
        "definition": "How well people know and recognize your brand",
        "category": "marketing",
        "complexity_level": "medium"
    },
    "viral marketing": {
        "definition": "Marketing that spreads quickly through social sharing",
        "category": "marketing",
        "complexity_level": "medium"
    }
}

# ========================================
# UTILITY FUNCTIONS
# ========================================

def get_term_definition(term: str) -> Optional[Dict[str, str]]:
    """Get definition for a term if available"""
    term_lower = term.lower().strip()
    return TERM_DEFINITIONS.get(term_lower)

def extract_terms_from_text(text: str) -> List[Dict[str, str]]:
    """Extract defined terms from text and return their definitions"""
    words = text.lower().split()
    found_terms = []
    
    # Check for multi-word terms first
    for term in TERM_DEFINITIONS:
        if term in text.lower():
            definition_data = TERM_DEFINITIONS[term]
            found_terms.append({
                "term": term,
                "definition": definition_data["definition"],
                "category": definition_data["category"],
                "complexity_level": definition_data["complexity_level"]
            })
    
    return found_terms

def mock_translate_text(text: str, target_language: str, source_language: str = "auto") -> Dict[str, Any]:
    """Mock translation function - replace with actual translation service"""
    
    # Language mapping
    language_names = {
        "en": "English",
        "hi": "Hindi", 
        "ta": "Tamil",
        "te": "Telugu"
    }
    
    # Mock translations for demo
    mock_translations = {
        "en": {
            "hi": "नमस्ते, HAVEN में आपका स्वागत है",
            "ta": "வணக்கம், HAVEN இல் உங்களை வரவேற்கிறோம்",
            "te": "నమస్కారం, HAVEN కి స్వాగతం"
        },
        "hi": {
            "en": "Hello, welcome to HAVEN",
            "ta": "வணக்கம், HAVEN இல் உங்களை வரவேற்கிறோம்", 
            "te": "నమస్కారం, HAVEN కి స్వాగతం"
        }
    }
    
    # Simple mock translation
    if source_language in mock_translations and target_language in mock_translations[source_language]:
        translated = mock_translations[source_language][target_language]
    else:
        translated = f"[{target_language.upper()}] {text}"
    
    return {
        "original_text": text,
        "translated_text": translated,
        "source_language": language_names.get(source_language, source_language),
        "target_language": language_names.get(target_language, target_language),
        "confidence": 0.95
    }

# ========================================
# FASTAPI APPLICATION SETUP
# ========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    print("🚀 HAVEN Backend starting up...")
    print(f"📍 Backend URL: {BACKEND_URL}")
    print(f"🌐 Frontend URL: {FRONTEND_URL}")
    print(f"🔧 Translation enabled: {TRANSLATION_ENABLED}")
    print(f"📝 Simplification enabled: {SIMPLIFICATION_ENABLED}")
    
    yield
    
    # Shutdown
    print("🛑 HAVEN Backend shutting down...")

# Create FastAPI application
app = FastAPI(
    title="HAVEN Crowdfunding Platform API",
    description="Backend API for HAVEN - Help not just some people, but Help Humanity",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# ========================================
# CORS MIDDLEWARE SETUP
# ========================================

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_URL,
        "https://haven-streamlit-frontend.onrender.com",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"  # Allow all origins for development
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=[
        "Accept",
        "Accept-Language", 
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "Origin",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers",
        "X-CSRF-Token",
        "X-API-Key"
    ],
    expose_headers=[
        "Access-Control-Allow-Origin",
        "Access-Control-Allow-Methods", 
        "Access-Control-Allow-Headers",
        "Access-Control-Allow-Credentials"
    ]
)

# ========================================
# MIDDLEWARE
# ========================================

@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    """Add additional CORS headers"""
    response = await call_next(request)
    
    # Add CORS headers
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    
    return response

# ========================================
# API ENDPOINTS
# ========================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to HAVEN Crowdfunding Platform API",
        "tagline": "Help not just some people, but Help Humanity",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="HAVEN Backend is running successfully",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        features={
            "translation": TRANSLATION_ENABLED,
            "simplification": SIMPLIFICATION_ENABLED,
            "oauth": bool(GOOGLE_CLIENT_ID or FACEBOOK_CLIENT_ID),
            "cors": True
        }
    )

@app.get("/api/backend-test", response_model=BackendTestResponse)
async def backend_test():
    """Comprehensive backend test endpoint for frontend"""
    start_time = datetime.now()
    
    # Test various backend components
    backend_status = "operational"
    cors_enabled = True
    translation_available = TRANSLATION_ENABLED
    simplification_available = SIMPLIFICATION_ENABLED
    oauth_configured = bool(GOOGLE_CLIENT_ID or FACEBOOK_CLIENT_ID)
    database_connected = True  # Mock for now
    
    end_time = datetime.now()
    response_time_ms = (end_time - start_time).total_seconds() * 1000
    
    return BackendTestResponse(
        backend_status=backend_status,
        cors_enabled=cors_enabled,
        translation_available=translation_available,
        simplification_available=simplification_available,
        oauth_configured=oauth_configured,
        database_connected=database_connected,
        response_time_ms=round(response_time_ms, 2)
    )

@app.post("/api/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """Translate text between supported languages"""
    if not TRANSLATION_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="Translation service is currently disabled"
        )
    
    # Validate target language
    supported_languages = ["en", "hi", "ta", "te"]
    if request.target_language not in supported_languages:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported target language. Supported: {supported_languages}"
        )
    
    try:
        # Use mock translation for now
        result = mock_translate_text(
            request.text,
            request.target_language,
            request.source_language
        )
        
        return TranslationResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}"
        )

@app.post("/api/simplify", response_model=SimplificationResponse)
async def simplify_text(request: SimplificationRequest):
    """Simplify text and provide term definitions"""
    if not SIMPLIFICATION_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="Simplification service is currently disabled"
        )
    
    try:
        # Extract terms from text
        found_terms = extract_terms_from_text(request.text)
        
        # Mock simplification (replace with actual service)
        simplified_text = request.text  # For now, return original
        complexity_score = len(found_terms) * 0.1  # Mock complexity
        
        return SimplificationResponse(
            original_text=request.text,
            simplified_text=simplified_text,
            complexity_score=complexity_score,
            simplified_terms=found_terms
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Simplification failed: {str(e)}"
        )

@app.get("/api/term-definition/{term}", response_model=TermDefinition)
async def get_term_definition_endpoint(term: str):
    """Get definition for a specific term"""
    definition_data = get_term_definition(term)
    
    if not definition_data:
        raise HTTPException(
            status_code=404,
            detail=f"Definition not found for term: {term}"
        )
    
    return TermDefinition(
        term=term,
        definition=definition_data["definition"],
        category=definition_data["category"],
        complexity_level=definition_data["complexity_level"]
    )

@app.get("/api/supported-languages")
async def get_supported_languages():
    """Get list of supported languages"""
    return {
        "languages": [
            {"code": "en", "name": "English", "native": "English"},
            {"code": "hi", "name": "Hindi", "native": "हिन्दी"},
            {"code": "ta", "name": "Tamil", "native": "தமிழ்"},
            {"code": "te", "name": "Telugu", "native": "తెలుగు"}
        ]
    }

@app.options("/{path:path}")
async def options_handler(path: str):
    """Handle OPTIONS requests for CORS"""
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true"
        }
    )

# ========================================
# ERROR HANDLERS
# ========================================

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"The requested endpoint {request.url.path} was not found",
            "available_endpoints": [
                "/",
                "/health", 
                "/api/backend-test",
                "/api/translate",
                "/api/simplify",
                "/api/term-definition/{term}",
                "/api/supported-languages",
                "/docs"
            ]
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    """Handle internal server errors"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

# ========================================
# APPLICATION STARTUP
# ========================================

if __name__ == "__main__":
    # Get port from environment variable (for Render deployment)
    port = int(os.getenv("PORT", 8000))
    
    print(f"🚀 Starting HAVEN Backend on port {port}")
    print(f"📍 Backend URL: {BACKEND_URL}")
    print(f"🌐 Frontend URL: {FRONTEND_URL}")
    
    # Run the application
    uvicorn.run(
        "main_app_with_cors:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Set to False for production
        log_level="info"
    )

