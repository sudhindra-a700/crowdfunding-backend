from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
import os
import pandas as pd  # Keep for type hinting/potential future use, but data loading is lazy
import joblib  # Keep for type hinting/potential future use
import json
import random
import sys
import logging
import time
import psutil  # For enhanced health checks

from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import urllib.parse

# Firebase Admin SDK imports
import firebase_admin
from firebase_admin import credentials, auth, firestore, messaging

# Algolia Search Client
from algoliasearch.search_client import SearchClient

# Suppress warnings for cleaner output
import warnings

warnings.filterwarnings("ignore")


# --- Enhanced Logging Configuration ---
def setup_logging():
    """Configure enhanced logging for production"""
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_format = os.environ.get("LOG_FORMAT", "json")

    if log_format.lower() == "json":
        # JSON logging for production
        import json
        import datetime

        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno
                }
                if record.exc_info:
                    log_entry["exception"] = self.formatException(record.exc_info)
                return json.dumps(log_entry)

        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
    else:
        # Standard logging for development
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        handler.setFormatter(formatter)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        handlers=[handler],
        force=True
    )

    # Set specific loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)

    return logging.getLogger(__name__)


# Initialize enhanced logging
logger = setup_logging()


# --- Environment Variable Validation ---
class EnvironmentConfig:
    """Centralized environment variable management with validation"""

    def __init__(self):
        self.required_vars = {
            "FIREBASE_SERVICE_ACCOUNT_KEY_JSON_BASE64": "Firebase authentication",
        }

        self.optional_vars = {
            "ALGOLIA_API_KEY": "Search functionality",
            "ALGOLIA_APP_ID": "Search functionality",
            "BREVO_API_KEY": "Email notifications",
            "INSTAMOJO_API_KEY": "Payment processing",
            "INSTAMOJO_AUTH_TOKEN": "Payment processing",
            "LOG_LEVEL": "Logging configuration",
            "LOG_FORMAT": "Logging format",
            "ENVIRONMENT": "Environment identification"
        }

        self.config = {}
        self.missing_required = []
        self.missing_optional = []

        self._load_and_validate()

    def _load_and_validate(self):
        """Load and validate all environment variables"""
        logger.info("Loading and validating environment variables...")

        # Check required variables
        for var_name, description in self.required_vars.items():
            value = os.environ.get(var_name)
            if not value:
                self.missing_required.append((var_name, description))
                logger.error(f"Missing required environment variable: {var_name} ({description})")
            else:
                self.config[var_name] = value
                logger.info(f"✓ Required variable loaded: {var_name}")

        # Check optional variables
        for var_name, description in self.optional_vars.items():
            value = os.environ.get(var_name)
            if not value:
                self.missing_optional.append((var_name, description))
                logger.warning(
                    f"Missing optional environment variable: {var_name} ({description}) - Feature may be limited")
            else:
                self.config[var_name] = value
                logger.info(f"✓ Optional variable loaded: {var_name}")

        # Set defaults for optional variables
        self.config.setdefault("LOG_LEVEL", "INFO")
        self.config.setdefault("LOG_FORMAT", "standard")
        self.config.setdefault("ENVIRONMENT", "production")

    def get(self, key: str, default: str = None) -> str:
        """Get environment variable value"""
        return self.config.get(key, default)

    def is_required_missing(self) -> bool:
        """Check if any required variables are missing"""
        return len(self.missing_required) > 0

    def get_missing_required(self) -> List[tuple]:
        """Get list of missing required variables"""
        return self.missing_required

    def get_missing_optional(self) -> List[tuple]:
        """Get list of missing optional variables"""
        return self.missing_optional


# Initialize environment configuration
env_config = EnvironmentConfig()

# --- Pydantic Models (MOVED TO EARLY SECTION TO FIX NameError) ---
class UserLogin(BaseModel):
    id_token: str


class UserInfo(BaseModel):
    uid: str
    email: Optional[str] = None
    role: str = "user"


class Token(BaseModel):
    access_token: str
    token_type: str


class SearchQuery(BaseModel):
    query: str


class NotificationRequest(BaseModel):
    campaign_id: str
    message: str
    recipient_email: Optional[str] = None
    device_token: Optional[str] = None


class FraudCheckRequest(BaseModel):
    org_name: str
    bio: Optional[str]
    follower_count: int
    post_count: int
    account_age_days: int
    engagement_rate: float
    recent_posts: Optional[str]
    pan: Optional[str] = None
    reg_number: Optional[str] = None
    registration_type: Optional[str] = None
    ngo_darpan_id: Optional[str] = None
    fcra_number: Optional[str] = None


# FraudCheckResponse model (keeping XAI fields)
class FraudCheckResponse(BaseModel):
    fraud_score: float
    explanation: Optional[str] = None
    shap_plot: Optional[str] = None  # Path to the SHAP plot image
    verification: Optional[Dict[str, Any]] = None
    verification_status: str


class CampaignCreateRequest(BaseModel):
    name: str
    description: str
    author: str
    goal: int
    category: str
    registration_type: Optional[str] = None
    registration_number: Optional[str] = None
    pan: Optional[str] = None
    ngo_darpan_id: Optional[str] = None
    fcra_number: Optional[str] = None


class Campaign(BaseModel):
    id: str
    name: str
    description: str
    author: str
    funded: int
    goal: int
    days_left: int
    category: str
    verification_status: str = "Pending"
    fraud_score: Optional[float] = None
    fraud_explanation: Optional[str] = None  # Keeping XAI field
    verification_details: Optional[Dict[str, Any]] = None
    image_url: Optional[str] = None


class InitiatePaymentRequest(BaseModel):
    campaign_id: str
    amount: int
    payment_method: str  # e.g., 'instamojo_gateway', 'upi'
    donor_name: Optional[str] = "Anonymous"
    donor_email: Optional[str] = "anonymous@example.com"
    donor_phone: Optional[str] = "9999999999"


class CampaignBulkUploadRequest(BaseModel):
    campaigns: List[CampaignCreateRequest]


class TranslationRequest(BaseModel):
    campaign_id: str
    field: str  # e.g., "name", "description"
    target_language: str  # e.g., "hi", "bn", "ta"


# --- Initialize FastAPI app ---
app = FastAPI(
    title="HAVEN Backend Service (Enhanced Cloud Ready)",
    description="Enhanced version with improved error handling, logging, and health checks",
    version="2.0.0"
)


# --- Enhanced Health Check ---
@app.get("/health")
async def health_check():
    """Enhanced health check with system metrics"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "2.0.0",
            "environment": env_config.get("ENVIRONMENT", "unknown"),
            "services": {},
            "system": {}
        }

        # Check Firebase connection
        try:
            if db:
                # Simple Firestore connectivity test
                test_doc = db.collection("health_check").document("test")
                test_doc.set({"timestamp": time.time()}, merge=True)
                health_status["services"]["firebase"] = "connected"
            else:
                health_status["services"]["firebase"] = "not_initialized"
        except Exception as e:
            health_status["services"]["firebase"] = f"error: {str(e)}"
            logger.warning(f"Firebase health check failed: {e}")

        # Check Algolia connection
        try:
            if algolia_index:
                # Simple Algolia connectivity test
                algolia_index.search("", {"hitsPerPage": 1})
                health_status["services"]["algolia"] = "connected"
            else:
                health_status["services"]["algolia"] = "not_configured"
        except Exception as e:
            health_status["services"]["algolia"] = f"error: {str(e)}"
            logger.warning(f"Algolia health check failed: {e}")

        # System metrics
        try:
            health_status["system"] = {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "uptime_seconds": time.time() - psutil.boot_time()
            }
        except Exception as e:
            logger.warning(f"System metrics collection failed: {e}")
            health_status["system"] = {"error": str(e)}

        # Check for any critical issues
        critical_issues = []
        if env_config.is_required_missing():
            critical_issues.extend([f"Missing required env var: {var}" for var, _ in env_config.get_missing_required()])

        if not db:
            critical_issues.append("Firebase not initialized")

        if critical_issues:
            health_status["status"] = "degraded"
            health_status["issues"] = critical_issues
            return health_status, 503

        return health_status

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }, 500


# --- Readiness Check ---
@app.get("/ready")
async def readiness_check():
    """Kubernetes-style readiness check"""
    try:
        # Check if all critical services are ready
        if env_config.is_required_missing():
            return {"ready": False, "reason": "Missing required environment variables"}, 503

        if not db:
            return {"ready": False, "reason": "Firebase not initialized"}, 503

        return {"ready": True, "timestamp": time.time()}

    except Exception as e:
        logger.error(f"Readiness check failed: {e}", exc_info=True)
        return {"ready": False, "reason": str(e)}, 503


# --- Liveness Check ---
@app.get("/live")
async def liveness_check():
    """Kubernetes-style liveness check"""
    return {"alive": True, "timestamp": time.time()}


# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Define Paths ---
BASE_DIR = Path(__file__).resolve().parent

# --- Global variables for IndicTrans2 model, tokenizer, and processor ---
# These will be loaded lazily on first request to /translate
indictrans2_model = None
indictrans2_tokenizer = None
indictrans2_processor = None
DEVICE = "cpu"  # Enforce CPU for Render free tier. Set to "cuda" if GPU is available.

# --- Global Firebase and Algolia clients ---
db = None
algolia_client = None
algolia_index = None

# --- Authentication ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/verify-token")


async def get_current_user(id_token: str = Depends(oauth2_scheme)):
    if not firebase_admin._apps:
        logger.error("Firebase not initialized in get_current_user.")
        raise HTTPException(status_code=500, detail="Firebase not initialized.")
    try:
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token["uid"]
        email = decoded_token.get("email")
        role = decoded_token.get("role", "user")
        return UserInfo(uid=uid, email=email, role=role)
    except Exception as e:
        logger.error(f"Firebase ID token verification failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_admin_user(current_user: UserInfo = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


# --- Lazy Loading for IndicTrans2 Translation Model ---
def load_indictrans2_model():
    global indictrans2_model, indictrans2_tokenizer, indictrans2_processor, DEVICE
    if indictrans2_model is None or indictrans2_tokenizer is None or indictrans2_processor is None:
        logger.info("Lazily loading IndicTrans2 model...")
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            from IndicTransToolkit.processor import IndicProcessor  # Assuming this is installed via requirements.txt

            if torch.cuda.is_available():
                DEVICE = "cuda"
                logger.info("CUDA (GPU) is available. Using GPU for IndicTrans2.")
            else:
                DEVICE = "cpu"
                logger.info("CUDA (GPU) is not available. Using CPU for IndicTrans2.")

            model_name = "ai4bharat/indictrans2-en-indic-1B"
            indictrans2_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            indictrans2_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True).to(DEVICE)
            indictrans2_processor = IndicProcessor(build_map_filename=True)
            logger.info("IndicTrans2 model loaded successfully (lazy).")
        except Exception as e:
            logger.error(f"Error lazy loading IndicTrans2 model: {e}", exc_info=True)
            indictrans2_model = None
            indictrans2_tokenizer = None
            indictrans2_processor = None
            raise RuntimeError("Translation service initialization failed.")  # Raise to propagate error


# --- Real IndicTrans2 Translation Function ---
def indictrans2_translate(text: str, source_lang: str, target_lang: str) -> str:
    # Ensure model is loaded before attempting translation
    load_indictrans2_model()  # This will load if not already loaded

    try:
        lang_map = {
            "en": "eng_Latn", "hi": "hin_Deva", "bn": "ben_Beng", "ta": "tam_Taml",
            "te": "tel_Telu", "mr": "mar_Deva", "gu": "guj_Gujr", "pa": "pan_Guru",
            "kn": "kan_Knda", "ml": "mal_Mlym", "or": "ori_Orya", "as": "asm_Beng",
            "ur": "urd_Arab", "ne": "nep_Deva", "si": "sin_Sinh", "my": "mya_Mymr",
        }

        src_lang_code = lang_map.get(source_lang, None)
        tgt_lang_code = lang_map.get(target_lang, None)

        if not src_lang_code or not tgt_lang_code:
            return f"{text} (Unsupported language code for IndicTrans2: {source_lang} or {target_lang})"

        batch = indictrans2_processor.preprocess_batch([text], src_lang=src_lang_code, tgt_lang=tgt_lang_code)

        inputs = indictrans2_tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)

        with torch.no_grad():
            generated_tokens = indictrans2_model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        translated_text = \
            indictrans2_tokenizer.batch_decode(generated_tokens.detach().cpu().tolist(), skip_special_tokens=True)[0]
        return translated_text

    except Exception as e:
        logger.error(f"Error during real IndicTrans2 translation: {e}", exc_info=True)
        return f"{text} (Translation Failed: {e})"


# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def serve_pwa_shell():
    """Serves the main PWA HTML shell (index.html)."""
    index_html_path = BASE_DIR / "static" / "index.html"
    if not index_html_path.exists():
        logger.error(f"index.html not found at {index_html_path}")
        raise HTTPException(status_code=404, detail="index.html not found in static directory.")
    return FileResponse(index_html_path)


@app.get("/manifest.json", response_class=FileResponse)
async def serve_manifest():
    """Serves the PWA manifest file."""
    manifest_path = BASE_DIR / "static" / "manifest.json"
    if not manifest_path.exists():
        logger.error(f"manifest.json not found at {manifest_path}")
        raise HTTPException(status_code=404, detail="manifest.json not found in static directory.")
    return FileResponse(manifest_path, media_type="application/manifest+json")


@app.get("/sw.js", response_class=FileResponse)
async def serve_service_worker():
    """Serves the PWA service worker file."""
    sw_path = BASE_DIR / "static" / "sw.js"
    if not sw_path.exists():
        logger.error(f"sw.js not found at {sw_path}")
        raise HTTPException(status_code=404, detail="sw.js not found in static directory.")
    return FileResponse(sw_path, media_type="application/javascript")


@app.post("/verify-token", response_model=UserInfo)
async def verify_firebase_id_token(user_login: UserLogin):
    if not firebase_admin._apps:
        logger.error("Firebase not initialized in /verify-token endpoint.")
        raise HTTPException(status_code=500, detail="Firebase not initialized.")
    try:
        decoded_token = auth.verify_id_token(user_login.id_token)
        uid = decoded_token["uid"]
        email = decoded_token.get("email")
        role = decoded_token.get("role", "user")
        return UserInfo(uid=uid, email=email, role=role)
    except Exception as e:
        logger.error(f"Firebase ID token verification failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.on_event("startup")
async def startup_event():
    global db, algolia_client, algolia_index

    logger.info("Application startup event triggered.")

    # Check for missing required environment variables
    if env_config.is_required_missing():
        missing_vars = env_config.get_missing_required()
        error_msg = f"Missing required environment variables: {', '.join([var for var, _ in missing_vars])}"
        logger.critical(error_msg)

        # In production, you might want to exit, but for development, we'll continue with warnings
        if env_config.get("ENVIRONMENT", "production").lower() == "production":
            logger.critical("Exiting due to missing required environment variables in production")
            sys.exit(1)
        else:
            logger.warning("Continuing startup despite missing required variables (development mode)")

    # Log missing optional variables
    if env_config.get_missing_optional():
        missing_optional = env_config.get_missing_optional()
        for var, description in missing_optional:
            logger.warning(f"Optional variable {var} not set - {description} may be limited")

    # --- Firebase Initialization with Enhanced Error Handling ---
    try:
        logger.info("Attempting Firebase Admin SDK initialization...")
        firebase_key = env_config.get("FIREBASE_SERVICE_ACCOUNT_KEY_JSON_BASE64")

        if firebase_key:
            import base64
            try:
                service_account_info = json.loads(base64.b64decode(firebase_key).decode("utf-8"))
                cred = credentials.Certificate(service_account_info)
                logger.info("Using Firebase service account from environment variable")
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Invalid Firebase service account key format: {e}")
                raise
        else:
            logger.warning(
                "FIREBASE_SERVICE_ACCOUNT_KEY_JSON_BASE64 not found. Attempting ApplicationDefault credentials.")
            cred = credentials.ApplicationDefault()

        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        logger.info("Firebase Admin SDK initialized successfully.")

    except Exception as e:
        logger.critical(f"FATAL ERROR: Firebase Admin SDK initialization failed: {e}", exc_info=True)
        if env_config.get("ENVIRONMENT", "production").lower() == "production":
            sys.exit(1)
        else:
            logger.warning("Continuing without Firebase (development mode)")
            db = None

    # --- Algolia Client Initialization with Enhanced Error Handling ---
    try:
        logger.info("Attempting Algolia client initialization...")
        algolia_app_id = env_config.get("ALGOLIA_APP_ID")
        algolia_api_key = env_config.get("ALGOLIA_API_KEY")

        if algolia_app_id and algolia_api_key:
            algolia_client = SearchClient(algolia_app_id, algolia_api_key)
            algolia_index = algolia_client.init_index("campaigns")
            logger.info("Algolia client initialized for index: campaigns")
        else:
            logger.warning(
                "Algolia API keys not configured. Search functionality will be limited to Firestore fallback.")
            algolia_client = None
            algolia_index = None

    except Exception as e:
        logger.error(f"Error initializing Algolia client: {e}", exc_info=True)
        algolia_client = None
        algolia_index = None

    # --- PWA Static Files Serving with Enhanced Error Handling ---
    STATIC_DIR = BASE_DIR / "static"
    try:
        if not STATIC_DIR.exists():
            STATIC_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created static directory: {STATIC_DIR}")

        # Ensure required subdirectories exist
        for subdir in ["icons", "shap_plots"]:
            subdir_path = STATIC_DIR / subdir
            if not subdir_path.exists():
                subdir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created {subdir} directory: {subdir_path}")

        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
        logger.info(f"Mounted static files from: {STATIC_DIR}")

    except Exception as e:
        logger.critical(f"FATAL ERROR: Could not set up static file serving: {e}", exc_info=True)
        if env_config.get("ENVIRONMENT", "production").lower() == "production":
            sys.exit(1)

    logger.info("Application startup event completed successfully. Ready to serve.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

