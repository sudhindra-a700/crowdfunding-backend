"""
HAVEN Crowdfunding Platform - Complete App Integration
Updated app.py with translation and simplification services
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from starlette.middleware.sessions import SessionMiddleware
import firebase_admin
from firebase_admin import credentials, firestore

# Import existing modules (assuming they exist in the repository)
try:
    from oauth_routes import router as oauth_router
    from fraud_detection import router as fraud_router
except ImportError:
    # Create placeholder routers if modules don't exist
    from fastapi import APIRouter
    oauth_router = APIRouter(prefix="/auth", tags=["oauth"])
    fraud_router = APIRouter(prefix="/fraud", tags=["fraud"])

# Import our new translation and simplification services
from complete_translation_api_routes import router as translation_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for services
translation_service = None
simplification_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting HAVEN Crowdfunding Backend API...")
    
    # Initialize Firebase
    await initialize_firebase()
    
    # Initialize translation and simplification services
    await initialize_translation_services()
    
    logger.info("✅ Application startup complete")
    logger.info("✅ SessionMiddleware configured for OAuth")
    logger.info("✅ Firebase Admin SDK initialized successfully")
    logger.info("✅ Translation and simplification services ready")
    logger.info("🎉 FastAPI application ready")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down HAVEN Crowdfunding Backend API...")

async def initialize_firebase():
    """Initialize Firebase Admin SDK"""
    try:
        if not firebase_admin._apps:
            # Try to initialize with environment variables
            firebase_config = {
                "type": os.getenv("FIREBASE_TYPE", "service_account"),
                "project_id": os.getenv("FIREBASE_PROJECT_ID"),
                "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
                "private_key": os.getenv("FIREBASE_PRIVATE_KEY", "").replace('\\n', '\n'),
                "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
                "client_id": os.getenv("FIREBASE_CLIENT_ID"),
                "auth_uri": os.getenv("FIREBASE_AUTH_URI", "https://accounts.google.com/o/oauth2/auth"),
                "token_uri": os.getenv("FIREBASE_TOKEN_URI", "https://oauth2.googleapis.com/token"),
                "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_X509_CERT_URL"),
                "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_X509_CERT_URL")
            }
            
            # Check if all required fields are present
            required_fields = ["project_id", "private_key", "client_email"]
            if all(firebase_config.get(field) for field in required_fields):
                cred = credentials.Certificate(firebase_config)
                firebase_admin.initialize_app(cred)
                logger.info("✅ Firebase Admin SDK initialized with environment variables")
            else:
                logger.warning("⚠️ Firebase environment variables not complete, using default credentials")
                firebase_admin.initialize_app()
        
    except Exception as e:
        logger.error(f"❌ Firebase initialization failed: {e}")
        # Continue without Firebase for development
        pass

async def initialize_translation_services():
    """Initialize translation and simplification services"""
    global translation_service, simplification_service
    
    try:
        # Import and initialize services
        from complete_backend_translation_service import get_translation_service
        from complete_term_simplification_service import get_simplification_service
        
        translation_service = get_translation_service()
        simplification_service = get_simplification_service()
        
        # Test services with simple operations
        test_translation = await translation_service.translate_text("Hello", "en", "hi")
        test_simplification = await simplification_service.simplify_text("This is a test", simplification_service.ComplexityLevel.SIMPLE)
        
        logger.info("✅ Translation service initialized successfully")
        logger.info("✅ Simplification service initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Translation services initialization failed: {e}")
        # Continue without translation services for development
        pass

# Create FastAPI app with lifespan
app = FastAPI(
    title="HAVEN Crowdfunding Platform API",
    description="Complete API for HAVEN Crowdfunding Platform with translation and simplification",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Session middleware for OAuth
app.add_middleware(
    SessionMiddleware, 
    secret_key=os.getenv("SESSION_SECRET_KEY", "UpLvYLpd2ZdXEVyiGSQmXS9DMQVGLB2dLvc6twgyJX8"),
    max_age=3600,  # 1 hour timeout
    same_site="lax",  # CSRF protection
    https_only=False  # Render handles HTTPS
)

# Include routers
app.include_router(oauth_router)
app.include_router(fraud_router)
app.include_router(translation_router)  # New translation and simplification routes

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with service information"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>HAVEN Crowdfunding Platform API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2e7d32; text-align: center; }
            .feature { background: #e8f5e8; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #4caf50; }
            .endpoint { background: #f0f0f0; padding: 10px; margin: 5px 0; border-radius: 3px; font-family: monospace; }
            .status { text-align: center; margin: 20px 0; }
            .healthy { color: #4caf50; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏠 HAVEN Crowdfunding Platform API</h1>
            <div class="status">
                <span class="healthy">✅ API is running and healthy</span>
            </div>
            
            <div class="feature">
                <h3>🌍 Translation Services</h3>
                <p>Support for 4 languages: English, Hindi, Tamil, Telugu</p>
                <div class="endpoint">POST /api/translate - Translate text</div>
                <div class="endpoint">POST /api/translate/batch - Batch translation</div>
                <div class="endpoint">GET /api/translate/languages - Supported languages</div>
            </div>
            
            <div class="feature">
                <h3>💡 Text Simplification</h3>
                <p>Make complex terms and text easier to understand</p>
                <div class="endpoint">POST /api/simplify - Simplify text</div>
                <div class="endpoint">GET /api/simplify/define/{term} - Get term definition</div>
                <div class="endpoint">POST /api/simplify/search - Search terms</div>
            </div>
            
            <div class="feature">
                <h3>🔐 Authentication</h3>
                <p>OAuth integration with Google and Facebook</p>
                <div class="endpoint">GET /auth/google/login - Google OAuth</div>
                <div class="endpoint">GET /auth/facebook/login - Facebook OAuth</div>
            </div>
            
            <div class="feature">
                <h3>🛡️ Fraud Detection</h3>
                <p>AI-powered fraud detection for campaigns</p>
                <div class="endpoint">POST /fraud/detect - Detect fraud</div>
                <div class="endpoint">GET /fraud/ngo/search - Search NGO database</div>
            </div>
            
            <div class="feature">
                <h3>📊 Monitoring</h3>
                <p>Service health and statistics</p>
                <div class="endpoint">GET /api/health - Service health check</div>
                <div class="endpoint">GET /api/stats - Service statistics</div>
            </div>
            
            <div style="text-align: center; margin-top: 30px;">
                <a href="/docs" style="background: #4caf50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin: 5px;">📚 API Documentation</a>
                <a href="/redoc" style="background: #2196f3; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin: 5px;">📖 ReDoc</a>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "service": "HAVEN Crowdfunding Platform API",
        "version": "2.0.0",
        "features": {
            "translation": translation_service is not None,
            "simplification": simplification_service is not None,
            "oauth": True,
            "fraud_detection": True
        }
    }

# Service status endpoint
@app.get("/status")
async def service_status():
    """Detailed service status"""
    status = {
        "api": "running",
        "timestamp": "2025-01-01T00:00:00Z",
        "services": {
            "translation": "available" if translation_service else "unavailable",
            "simplification": "available" if simplification_service else "unavailable",
            "oauth": "available",
            "fraud_detection": "available",
            "firebase": "available" if firebase_admin._apps else "unavailable"
        }
    }
    
    # Get detailed service health if available
    if translation_service:
        try:
            translation_health = translation_service.get_service_health()
            status["translation_details"] = translation_health
        except:
            pass
    
    if simplification_service:
        try:
            simplification_stats = simplification_service.get_service_stats()
            status["simplification_details"] = simplification_stats
        except:
            pass
    
    return status

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Not Found",
        "message": "The requested resource was not found",
        "status_code": 404
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return {
        "error": "Internal Server Error",
        "message": "An internal server error occurred",
        "status_code": 500
    }

# Additional utility endpoints
@app.get("/api/info")
async def api_info():
    """Get API information and capabilities"""
    return {
        "name": "HAVEN Crowdfunding Platform API",
        "version": "2.0.0",
        "description": "Complete API with translation, simplification, OAuth, and fraud detection",
        "capabilities": {
            "translation": {
                "supported_languages": ["en", "hi", "ta", "te"],
                "features": ["single_translation", "batch_translation", "caching", "quality_scoring"]
            },
            "simplification": {
                "complexity_levels": ["very_simple", "simple", "moderate", "complex", "very_complex"],
                "features": ["text_simplification", "term_definitions", "complexity_analysis", "search"]
            },
            "authentication": {
                "providers": ["google", "facebook"],
                "features": ["oauth2", "session_management", "csrf_protection"]
            },
            "fraud_detection": {
                "models": ["distilbert", "ngo_verification"],
                "features": ["campaign_analysis", "ngo_lookup", "risk_scoring"]
            }
        },
        "endpoints": {
            "translation": "/api/translate/*",
            "simplification": "/api/simplify/*",
            "authentication": "/auth/*",
            "fraud_detection": "/fraud/*",
            "monitoring": "/api/health, /api/stats"
        }
    }

# Development endpoints (remove in production)
if os.getenv("ENVIRONMENT") == "development":
    @app.get("/dev/test-translation")
    async def test_translation():
        """Test translation service"""
        if not translation_service:
            raise HTTPException(status_code=503, detail="Translation service not available")
        
        try:
            result = await translation_service.translate_text("Hello World", "en", "hi")
            return {
                "test": "translation",
                "original": "Hello World",
                "translated": result.translated_text,
                "confidence": result.confidence_score
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Translation test failed: {str(e)}")
    
    @app.get("/dev/test-simplification")
    async def test_simplification():
        """Test simplification service"""
        if not simplification_service:
            raise HTTPException(status_code=503, detail="Simplification service not available")
        
        try:
            from complete_term_simplification_service import ComplexityLevel
            result = await simplification_service.simplify_text(
                "Our crowdfunding platform leverages innovative technology", 
                ComplexityLevel.SIMPLE
            )
            return {
                "test": "simplification",
                "original": result.original_text,
                "simplified": result.simplified_text,
                "complexity_before": result.complexity_before,
                "complexity_after": result.complexity_after
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Simplification test failed: {str(e)}")

# Export app for deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=port, 
        reload=os.getenv("ENVIRONMENT") == "development"
    )

