from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import pandas as pd
import joblib # Assuming you might load a model with joblib
import json # For Firebase config if it's JSON
import random # For dummy data/predictions
import sys # Import sys for sys.exit()
import logging # Import logging for detailed output

from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import urllib.parse

from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer

# Firebase Admin SDK imports
import firebase_admin
from firebase_admin import credentials, auth, firestore, messaging

# Algolia Search Client
from algoliasearch.search_client import SearchClient

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Initialize FastAPI app ---
app = FastAPI(title="HAVEN Backend Service (Cloud Ready with Algolia & Payments)")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

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
indictrans2_model = None
indictrans2_tokenizer = None
indictrans2_processor = None
DEVICE = "cpu" # Default to CPU for Render free tier. Change to "cuda" if GPU is available and configured.

# --- Global Firebase and Algolia clients ---
db = None
algolia_client = None
algolia_index = None

# --- Load Machine Learning Model and Data (Adjust as per your fraud_detection.py) ---
NGO_FRAUD_DATA_PATH = BASE_DIR / "ngo_fraud.csv"
ngo_fraud_df = pd.DataFrame() # Initialize as empty, will be loaded in startup

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

# --- Real IndicTrans2 Translation Function ---
def indictrans2_translate(text: str, source_lang: str, target_lang: str) -> str:
    global indictrans2_model, indictrans2_tokenizer, indictrans2_processor
    import torch # Import torch locally to ensure it's available for this function

    if not indictrans2_model or not indictrans2_tokenizer or not indictrans2_processor:
        logger.warning("IndicTrans2 model not initialized. Translation will not work.")
        return f"{text} (Translation Service Unavailable)"

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

        translated_text = indictrans2_tokenizer.batch_decode(generated_tokens.detach().cpu().tolist(), skip_special_tokens=True)[0]
        return translated_text

    except Exception as e:
        logger.error(f"Error during real IndicTrans2 translation: {e}", exc_info=True)
        return f"{text} (Translation Failed: {e})"

# --- Pydantic Models ---
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
    fraud_explanation: Optional[str] = None
    verification_details: Optional[Dict[str, Any]] = None
    image_url: Optional[str] = None

class InitiatePaymentRequest(BaseModel):
    campaign_id: str
    amount: int
    payment_method: str # e.g., 'instamojo_gateway', 'upi'
    donor_name: Optional[str] = "Anonymous"
    donor_email: Optional[str] = "anonymous@example.com"
    donor_phone: Optional[str] = "9999999999"

class CampaignBulkUploadRequest(BaseModel):
    campaigns: List[CampaignCreateRequest]


# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def serve_pwa_shell():
    """Serves the main PWA HTML shell (index.html)."""
    index_html_path = STATIC_DIR / "index.html"
    if not index_html_path.exists():
        logger.error(f"index.html not found at {index_html_path}")
        raise HTTPException(status_code=404, detail="index.html not found in static directory.")
    return FileResponse(index_html_path)

@app.get("/manifest.json", response_class=FileResponse)
async def serve_manifest():
    """Serves the PWA manifest file."""
    manifest_path = STATIC_DIR / "manifest.json"
    if not manifest_path.exists():
        logger.error(f"manifest.json not found at {manifest_path}")
        raise HTTPException(status_code=404, detail="manifest.json not found in static directory.")
    return FileResponse(manifest_path, media_type="application/manifest+json")

@app.get("/sw.js", response_class=FileResponse)
async def serve_service_worker():
    """Serves the PWA service worker file."""
    sw_path = STATIC_DIR / "sw.js"
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

@app.post("/translate")
async def translate_text_endpoint(
    request: TranslationRequest,
    current_user: UserInfo = Depends(get_current_user)
):
    if not db:
        logger.error("Firestore not initialized in /translate endpoint.")
        raise HTTPException(status_code=500, detail="Firestore not initialized.")
    if not indictrans2_model:
        logger.error("Translation model not initialized in /translate endpoint.")
        raise HTTPException(status_code=500, detail="Translation model not initialized.")

    campaign_ref = db.collection("campaigns").document(request.campaign_id)
    campaign_doc = campaign_ref.get()

    if not campaign_doc.exists:
        raise HTTPException(status_code=404, detail="Campaign not found")

    campaign_data = campaign_doc.to_dict()

    translations_ref = db.collection("translations")
    query = translations_ref.where("campaign_id", "==", request.campaign_id)\
                            .where("field", "==", request.field)\
                            .where("language", "==", request.target_language)

    cached_translation_docs = query.limit(1).get()

    if cached_translation_docs:
        for doc in cached_translation_docs:
            return {"translated_text": doc.to_dict().get("translated_text")}

    original_text = campaign_data.get(request.field)
    if not original_text:
        raise HTTPException(status_code=400, detail="Invalid field or field not found in campaign.")

    translated_text = indictrans2_translate(
        text=original_text,
        source_lang="en",
        target_lang=request.target_language
    )

    translations_ref.add({
        "campaign_id": request.campaign_id,
        "field": request.field,
        "language": request.target_language,
        "translated_text": translated_text,
        "timestamp": firestore.SERVER_TIMESTAMP
    })

    return {"translated_text": translated_text}

@app.get("/campaigns", response_model=List[Campaign])
async def get_campaigns(
    language: Optional[str] = "en",
    current_user: UserInfo = Depends(get_current_user)
):
    if not db:
        logger.error("Firestore not initialized in /campaigns endpoint.")
        raise HTTPException(status_code=500, detail="Firestore not initialized.")
    if language != "en" and not indictrans2_model:
        logger.error("Translation model not initialized for non-English content in /campaigns endpoint.")
        raise HTTPException(status_code=500, detail="Translation model not initialized for non-English content.")

    campaigns_ref = db.collection("campaigns")
    campaign_docs = campaigns_ref.stream()

    result = []
    for doc in campaign_docs:
        campaign_data = doc.to_dict()
        campaign_data["id"] = doc.id

        if campaign_data.get('verification_status') == 'Rejected':
            continue

        if "image_url" not in campaign_data or not campaign_data["image_url"]:
            campaign_data["image_url"] = f"https://placehold.co/600x400/E0E0E0/333333?text={campaign_data['name'].replace(' ', '+')}"

        if language != "en":
            for field in ["name", "description"]:
                translations_query = db.collection("translations")\
                                       .where("campaign_id", "==", doc.id)\
                                       .where("field", "==", field)\
                                       .where("language", "==", language)

                cached_translation_docs = translations_query.limit(1).get()

                if cached_translation_docs:
                    for t_doc in cached_translation_docs:
                        campaign_data[field] = t_doc.to_dict().get("translated_text")
                else:
                    original_text = campaign_data.get(field)
                    if original_text:
                        translated_text = indictrans2_translate(
                            text=original_text,
                            source_lang="en",
                            target_lang=language
                        )
                        db.collection("translations").add({
                            "campaign_id": doc.id,
                            "field": field,
                            "language": language,
                            "translated_text": translated_text,
                            "timestamp": firestore.SERVER_TIMESTAMP
                        })
                        campaign_data[field] = translated_text
        result.append(campaign_data)

    return result

@app.get("/campaign-stats")
async def get_campaign_stats(
    current_user: UserInfo = Depends(get_current_user)
):
    if not db:
        logger.error("Firestore not initialized in /campaign-stats endpoint.")
        raise HTTPException(status_code=500, detail="Firestore not initialized.")

    campaigns_ref = db.collection("campaigns")
    campaign_docs = campaigns_ref.stream()

    total_campaigns = 0
    total_funded = 0
    total_goal = 0
    category_counts = {}

    for doc in campaign_docs:
        campaign_data = doc.to_dict()
        if campaign_data.get('verification_status') == 'Rejected':
            continue

        total_campaigns += 1
        total_funded += campaign_data.get("funded", 0)
        total_goal += campaign_data.get("goal", 0)
        category = campaign_data.get("category", "Other")
        category_counts[category] = category_counts.get(category, 0) + 1

    funding_percentage = (total_funded / total_goal * 100) if total_goal > 0 else 0

    return {
        "total_campaigns": total_campaigns,
        "total_funded": total_funded,
        "total_goal": total_goal,
        "funding_percentage": round(funding_percentage, 2),
        "category_distribution": category_counts
    }

@app.post("/search")
async def search_campaigns(
    query: SearchQuery,
    current_user: UserInfo = Depends(get_current_user)
):
    if algolia_index:
        try:
            search_results = algolia_index.search(query.query)
            return [c for c in search_results.get("hits", []) if c.get('verification_status') != 'Rejected']
        except Exception as e:
            logger.warning(f"Algolia search error: {e}. Falling back to Firestore search.", exc_info=True)

    if not db:
        logger.error("Firestore not initialized in /search endpoint (fallback).")
        raise HTTPException(status_code=500, detail="Firestore not initialized.")

    search_term_lower = query.query.lower()
    campaigns_ref = db.collection("campaigns")

    all_campaign_docs = campaigns_ref.stream()
    filtered_campaigns = []
    for doc in all_campaign_docs:
        campaign_data = doc.to_dict()
        campaign_data["id"] = doc.id
        if campaign_data.get('verification_status') == 'Rejected':
            continue

        if search_term_lower in campaign_data.get("name", "").lower() or \
           search_term_lower in campaign_data.get("description", "").lower():
            filtered_campaigns.append(campaign_data)

    return filtered_campaigns

@app.post("/notify")
async def send_notification(
    request: NotificationRequest,
    current_user: UserInfo = Depends(get_current_user)
):
    results = {}

    if request.recipient_email and BREVO_API_KEY:
        try:
            # This is a mock. In real code, you'd use a Brevo SDK or requests.post to Brevo API
            # For example:
            # headers = {"api-key": BREVO_API_KEY, "Content-Type": "application/json"}
            # payload = {"sender": {"email": "no-reply@haven.org"}, "to": [{"email": request.recipient_email}], "subject": "HAVEN Update", "htmlContent": request.message}
            # brevo_response = requests.post("https://api.brevo.com/v3/smtp/email", headers=headers, json=payload)
            # brevo_response.raise_for_status()
            results["email"] = "Sent successfully (mocked via configured API)"
            logger.info(f"Mock email sent to {request.recipient_email}")
        except Exception as e:
            results["email"] = f"Email sending failed (Brevo API error): {e}"
            logger.error(f"Email sending failed to {request.recipient_email}: {e}", exc_info=True)
    elif request.recipient_email:
        results["email"] = f"Mock email sent: {request.message} (Brevo API key not configured)"
        logger.warning("Brevo API key not configured for email sending.")

    if request.device_token:
        try:
            message = messaging.Message(
                notification=messaging.Notification(
                    title="HAVEN Update",
                    body=request.message,
                ),
                token=request.device_token,
            )
            response = messaging.send(message)
            results["push"] = f"Sent successfully: {response}"
            logger.info(f"FCM push notification sent to device: {request.device_token}")
        except Exception as e:
            results["push"] = f"FCM push notification failed: {e}"
            logger.error(f"FCM push notification failed for device {request.device_token}: {e}", exc_info=True)
    else:
        results["push"] = f"Mock push notification not sent (no device token provided)."
        logger.warning("No device token provided for push notification.")

    return results

@app.post("/fraud-check")
async def check_fraud(
    request: FraudCheckRequest,
    current_user: UserInfo = Depends(get_current_user)
):
    from fraud_detection import predict_fraud # Local import
    org_data = request.dict()
    try:
        fraud_score, explanation, plot_path, verification_details = predict_fraud(
            organization_data=org_data,
            api_key_trustcheckr="test_key" # You might pass a real TrustCheckr key here if you have one
        )
    except Exception as e:
        logger.error(f"Error during fraud prediction: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Fraud prediction failed: {e}")

    verification_status = ""
    if fraud_score <= 0.20:
        verification_status = "Verified by AI"
    elif 0.21 <= fraud_score <= 0.50:
        verification_status = "Needs Manual Review"
    else: # fraud_score > 0.50
        verification_status = "Rejected"

    if db:
        verification_collection = db.collection("organization_verifications")
        doc_id = org_data.get("org_name", "unknown_org").replace(" ", "_").lower() + "_" + str(random.randint(1000, 9999))
        try:
            verification_collection.document(doc_id).set({
                "org_name": org_data.get("org_name"),
                "fraud_score": fraud_score,
                "explanation": explanation,
                "verification_details": verification_details,
                "verification_status": verification_status,
                "timestamp": firestore.SERVER_TIMESTAMP,
                "checked_by_uid": current_user.uid
            })
            logger.info(f"Fraud check result saved for {org_data.get('org_name')}")
        except Exception as e:
            logger.error(f"Error saving fraud check result to Firestore: {e}", exc_info=True)

    return {
        "fraud_score": fraud_score,
        "explanation": explanation,
        "shap_plot": plot_path,
        "verification": verification_details,
        "verification_status": verification_status
    }

@app.post("/create-campaign", response_model=dict, dependencies=[Depends(get_admin_user)])
async def create_campaign(
    request: CampaignCreateRequest,
    current_user: UserInfo = Depends(get_admin_user)
):
    if not db:
        logger.error("Firestore not initialized in /create-campaign endpoint.")
        raise HTTPException(status_code=500, detail="Firestore not initialized.")

    try:
        campaign_data = request.dict()
        campaign_data["funded"] = 0
        campaign_data["days_left"] = 30
        campaign_data["created_by_uid"] = current_user.uid
        campaign_data["created_at"] = firestore.SERVER_TIMESTAMP
        campaign_data["image_url"] = f"https://placehold.co/600x400/E0E0E0/333333?text={campaign_data['name'].replace(' ', '+')}"

        org_data_for_fraud_check = {
            "org_name": campaign_data["author"],
            "bio": campaign_data["description"],
            "follower_count": 0,
            "post_count": 0,
            "account_age_days": 0,
            "engagement_rate": 0.0,
            "recent_posts": campaign_data["description"],
            "pan": campaign_data.get("pan"),
            "reg_number": campaign_data.get("registration_number"),
            "registration_type": campaign_data.get("registration_type"),
            "ngo_darpan_id": campaign_data.get("ngo_darpan_id"),
            "fcra_number": campaign_data.get("fcra_number")
        }

        from fraud_detection import predict_fraud # Local import
        fraud_score, explanation, plot_path, verification_details = predict_fraud(
            organization_data=org_data_for_fraud_check
        )

        campaign_data["fraud_score"] = fraud_score
        campaign_data["fraud_explanation"] = explanation
        campaign_data["verification_details"] = verification_details

        if fraud_score <= 0.20:
            campaign_data["verification_status"] = "Verified by AI"
        elif 0.21 <= fraud_score <= 0.50:
            campaign_data["verification_status"] = "Needs Manual Review"
        else:
            campaign_data["verification_status"] = "Rejected"
            logger.info(f"Campaign '{campaign_data['name']}' rejected due to high fraud risk (score: {fraud_score:.2f}).")
            return {"campaign_id": None, "message": "Campaign rejected due to high fraud risk."}

        doc_ref = db.collection("campaigns").add(campaign_data)
        campaign_id = doc_ref[1].id

        if algolia_index:
            try:
                algolia_object = {**campaign_data, "objectID": campaign_id}
                algolia_index.save_object(algolia_object).wait()
                logger.info(f"Campaign {campaign_id} indexed in Algolia.")
            except Exception as e:
                logger.error(f"Error indexing campaign {campaign_id} in Algolia: {e}", exc_info=True)

        return {"campaign_id": campaign_id, "message": "Campaign created successfully with initial verification status."}
    except Exception as e:
        logger.error(f"Error creating campaign: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create campaign: {e}")


@app.post("/bulk-upload-campaigns", response_model=dict, dependencies=[Depends(get_admin_user)])
async def bulk_upload_campaigns(request: CampaignBulkUploadRequest, current_user: UserInfo = Depends(get_admin_user)):
    if not db:
        logger.error("Firestore not initialized in /bulk-upload-campaigns endpoint.")
        raise HTTPException(status_code=500, detail="Firestore not initialized.")

    uploaded_count = 0
    failed_count = 0
    errors = []
    batch = db.batch()
    algolia_objects_to_save = []

    for campaign_data_req in request.campaigns:
        try:
            campaign_data = campaign_data_req.dict()
            campaign_data["funded"] = campaign_data.get("funded", 0)
            campaign_data["days_left"] = campaign_data.get("days_left", 30)
            campaign_data["created_by_uid"] = current_user.uid
            campaign_data["created_at"] = firestore.SERVER_TIMESTAMP
            campaign_data["image_url"] = f"https://placehold.co/600x400/E0E0E0/333333?text={campaign_data['name'].replace(' ', '+')}"

            org_data_for_fraud_check = {
                "org_name": campaign_data.get("author"),
                "bio": campaign_data.get("description"),
                "follower_count": 0, "post_count": 0, "account_age_days": 0, "engagement_rate": 0.0,
                "recent_posts": campaign_data.get("description"),
                "pan": campaign_data.get("pan"),
                "reg_number": campaign_data.get("registration_number"),
                "registration_type": campaign_data.get("registration_type"),
                "ngo_darpan_id": campaign_data.get("ngo_darpan_id"),
                "fcra_number": campaign_data.get("fcra_number")
            }
            from fraud_detection import predict_fraud # Local import
            fraud_score, explanation, plot_path, verification_details = predict_fraud(
                organization_data=org_data_for_fraud_check
            )
            campaign_data["fraud_score"] = fraud_score
            campaign_data["fraud_explanation"] = explanation
            campaign_data["verification_details"] = verification_details

            if fraud_score <= 0.20:
                campaign_data["verification_status"] = "Verified by AI"
            elif 0.21 <= fraud_score <= 0.50:
                campaign_data["verification_status"] = "Needs Manual Review"
            else:
                campaign_data["verification_status"] = "Rejected"
                failed_count += 1
                errors.append(f"Campaign '{campaign_data_req.name}' rejected due to high fraud risk (score: {fraud_score:.2f}).")
                continue

            new_doc_ref = db.collection("campaigns").document()
            batch.set(new_doc_ref, campaign_data)

            algolia_object = {**campaign_data, "objectID": new_doc_ref.id}
            algolia_objects_to_save.append(algolia_object)

            uploaded_count += 1
        except Exception as e:
            failed_count += 1
            errors.append(f"Failed to upload campaign '{campaign_data_req.name}': {e}")
            logger.error(f"Error processing bulk upload campaign '{campaign_data_req.name}': {e}", exc_info=True)

    try:
        batch.commit()
        logger.info(f"Firestore batch commit for bulk upload completed. Uploaded: {uploaded_count}, Failed: {failed_count}")
    except Exception as e:
        logger.error(f"Error committing Firestore batch for bulk upload: {e}", exc_info=True)
        failed_count += uploaded_count # Mark all as failed if batch commit fails
        errors.append(f"Failed to commit all campaigns to Firestore: {e}")


    if algolia_index and algolia_objects_to_save:
        try:
            algolia_index.save_objects(algolia_objects_to_save).wait()
            logger.info(f"Successfully indexed {len(algolia_objects_to_save)} campaigns in Algolia batch.")
        except Exception as e:
                logger.error(f"Error indexing campaigns in Algolia batch: {e}", exc_info=True)
                errors.append(f"Failed to index campaigns in Algolia: {e}")

    if failed_count > 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded {uploaded_count} campaigns, failed {failed_count}. Errors: {errors}"
        )
    return {"message": f"Successfully uploaded {uploaded_count} campaigns.", "uploaded_count": uploaded_count}

@app.post("/initiate-payment")
async def initiate_payment(
    request: InitiatePaymentRequest,
    current_user: UserInfo = Depends(get_current_user)
):
    if not db:
        logger.error("Firestore not initialized in /initiate-payment endpoint.")
        raise HTTPException(status_code=500, detail="Firestore not initialized.")

    campaign_id = request.campaign_id
    amount = request.amount
    payment_method = request.payment_method
    user_uid = current_user.uid
    donor_name = request.donor_name
    donor_email = request.donor_email
    donor_phone = request.donor_phone

    campaign_ref = db.collection("campaigns").document(campaign_id)
    campaign_doc = campaign_ref.get()

    if not campaign_doc.exists:
        raise HTTPException(status_code=404, detail="Campaign not found.")

    campaign_data = campaign_doc.to_dict()
    if campaign_data.get('verification_status') == 'Rejected':
        raise HTTPException(status_code=403, detail="Payment cannot be initiated for a rejected campaign.")

    campaign_author = campaign_data.get("author", "Unknown Beneficiary")
    current_funded = campaign_data.get("funded", 0)

    payment_status = "pending"
    payment_url = None
    transaction_id = f"txn_{random.randint(100000, 999999)}"

    try:
        if INSTAMOJO_API_KEY and INSTAMOJO_AUTH_TOKEN:
            payment_url = f"https://mock.instamojo.com/payment/{transaction_id}/?amount={amount}&purpose={urllib.parse.quote(campaign_data['name'])}"
            payment_status = "initiated_instamojo_mock"
            logger.info(f"Mock Instamojo Payment URL generated: {payment_url}")

        elif payment_method == "upi":
            mock_vpa = "havenplatform@upi"
            mock_payee_name = urllib.parse.quote(f"HAVEN - {campaign_author}")
            mock_transaction_ref = f"HAVEN{random.randint(10000000, 99999999)}"

            payment_url = (
                f"upi://pay?pa={mock_vpa}&pn={mock_payee_name}"
                f"&am={amount:.2f}&cu=INR&tr={mock_transaction_ref}"
                f"&tid={transaction_id}&mc=8999"
            )
            payment_status = "initiated_upi_mock"
            logger.info(f"Mock UPI Deep Link generated: {payment_url}")

        elif payment_method == "card":
            payment_url = "https://mock-card-payment-gateway.com/checkout"
            payment_status = "initiated_card_mock"
            logger.info(f"Mock Card Payment URL generated: {payment_url}")

        else:
            raise ValueError("Unsupported payment method or Instamojo API keys not configured.")

        db.collection("payment_intents").document(transaction_id).set({
            "campaign_id": campaign_id,
            "user_uid": user_uid,
            "amount": amount,
            "payment_method": payment_method,
            "status": payment_status,
            "payment_gateway_ref": transaction_id,
            "payment_url": payment_url,
            "created_at": firestore.SERVER_TIMESTAMP,
            "donor_name": donor_name,
            "donor_email": donor_email,
            "donor_phone": donor_phone
        })
        logger.info(f"Payment intent {transaction_id} recorded in Firestore.")

        return {
            "message": "Payment initiated successfully.",
            "transaction_id": transaction_id,
            "payment_url": payment_url,
            "status": payment_status
        }

    except Exception as e:
        logger.error(f"Error initiating payment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initiate payment: {e}")

@app.on_event("startup")
async def startup_event():
    global indictrans2_model, indictrans2_tokenizer, indictrans2_processor, db, algolia_client, algolia_index, ngo_fraud_df, DEVICE

    logger.info("Application startup event triggered.")

    # --- Load Secrets from Environment Variables ---
    FIREBASE_SERVICE_ACCOUNT_KEY_JSON_BASE64 = os.environ.get("FIREBASE_SERVICE_ACCOUNT_KEY_JSON_BASE64")
    ALGOLIA_API_KEY = os.environ.get("ALGOLIA_API_KEY")
    ALGOLIA_APP_ID = os.environ.get("ALGOLIA_APP_ID")
    BREVO_API_KEY = os.environ.get("BREVO_API_KEY")
    INSTAMOJO_API_KEY = os.environ.get("INSTAMOJO_API_KEY")
    INSTAMOJO_AUTH_TOKEN = os.environ.get("INSTAMOJO_AUTH_TOKEN")

    logger.info(f"Secrets Loaded: Firebase: {'Yes' if FIREBASE_SERVICE_ACCOUNT_KEY_JSON_BASE64 else 'No'}, Algolia API Key: {'Yes' if ALGOLIA_API_KEY else 'No'}, Algolia App ID: {'Yes' if ALGOLIA_APP_ID else 'No'}, Brevo: {'Yes' if BREVO_API_KEY else 'No'}, Instamojo API Key: {'Yes' if INSTAMOJO_API_KEY else 'No'}, Instamojo Auth Token: {'Yes' if INSTAMOJO_AUTH_TOKEN else 'No'}")


    # --- Firebase Initialization ---
    try:
        logger.info("Attempting Firebase Admin SDK initialization...")
        if FIREBASE_SERVICE_ACCOUNT_KEY_JSON_BASE64:
            import base64
            service_account_info = json.loads(base64.b64decode(FIREBASE_SERVICE_ACCOUNT_KEY_JSON_BASE64).decode("utf-8"))
            cred = credentials.Certificate(service_account_info)
        else:
            logger.warning("FIREBASE_SERVICE_ACCOUNT_KEY_JSON_BASE64 not found. Attempting ApplicationDefault credentials.")
            cred = credentials.ApplicationDefault() # Fallback for local development or GCP default service account

        if not firebase_admin._apps: # Initialize only if not already initialized
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        logger.info("Firebase Admin SDK initialized successfully.")
    except Exception as e:
        logger.critical(f"FATAL ERROR: Firebase Admin SDK initialization failed: {e}", exc_info=True)
        sys.exit(1) # Critical error, exit application


    # --- Algolia Client Initialization ---
    try:
        logger.info("Attempting Algolia client initialization...")
        if ALGOLIA_APP_ID and ALGOLIA_API_KEY:
            algolia_client = SearchClient(ALGOLIA_APP_ID, ALGOLIA_API_KEY)
            algolia_index = algolia_client.init_index("campaigns")
            logger.info(f"Algolia client initialized for index: campaigns")
        else:
            logger.warning("Algolia API keys not configured. Search functionality will be limited to Firestore fallback.")
            algolia_client = None
            algolia_index = None
    except Exception as e:
        logger.error(f"Error initializing Algolia client: {e}", exc_info=True)
        algolia_client = None
        algolia_index = None


    # --- PWA Static Files Serving ---
    STATIC_DIR = BASE_DIR / "static"
    try:
        if not STATIC_DIR.exists():
            STATIC_DIR.mkdir() # This might not be needed in Docker as files are copied
            logger.info(f"Created static directory: {STATIC_DIR}")
        if not (STATIC_DIR / "icons").exists():
            (STATIC_DIR / "icons").mkdir() # This might not be needed in Docker
            logger.info(f"Created static/icons directory: {STATIC_DIR / 'icons'}")
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
        logger.info(f"Mounted static files from: {STATIC_DIR}")
    except Exception as e:
        logger.critical(f"FATAL ERROR: Could not set up static file serving: {e}", exc_info=True)
        sys.exit(1)


    # --- Load NGO Fraud Data ---
    try:
        logger.info(f"Attempting to load NGO fraud data from {NGO_FRAUD_DATA_PATH}...")
        if NGO_FRAUD_DATA_PATH.exists():
            ngo_fraud_df = pd.read_csv(NGO_FRAUD_DATA_PATH)
            logger.info(f"NGO fraud data loaded successfully. Shape: {ngo_fraud_df.shape}")
        else:
            logger.warning(f"NGO fraud data file not found at {NGO_FRAUD_DATA_PATH}. Fraud detection model fine-tuning may be affected.")
            ngo_fraud_df = pd.DataFrame() # Ensure it's an empty DataFrame if file is missing
    except Exception as e:
        logger.critical(f"FATAL ERROR: Could not load NGO fraud data: {e}", exc_info=True)
        sys.exit(1) # Critical error, exit application


    # --- Initialize IndicTrans2 Model ---
    try:
        logger.info("Attempting to load IndicTrans2 model...")
        import torch # Ensure torch is imported for this section
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        # Ensure IndicTransToolkit is correctly installed and accessible
        # If IndicTransToolkit is a local dependency, ensure it's copied correctly in Dockerfile
        from IndicTransToolkit.processor import IndicProcessor

        # Check if GPU is available and set DEVICE
        if torch.cuda.is_available():
            DEVICE = "cuda"
            logger.info("CUDA (GPU) is available. Using GPU for IndicTrans2.")
        else:
            DEVICE = "cpu"
            logger.info("CUDA (GPU) is not available. Using CPU for IndicTrans2.")

        model_name = "ai4bharat/indictrans2-en-indic-1B"
        logger.info(f"Loading IndicTrans2 model '{model_name}' on {DEVICE}...")
        indictrans2_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        indictrans2_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True).to(DEVICE)
        indictrans2_processor = IndicProcessor(build_map_filename=True)
        logger.info("IndicTrans2 model loaded successfully.")
    except Exception as e:
        logger.critical(f"FATAL ERROR: Could not load IndicTrans2 model: {e}", exc_info=True)
        logger.critical("Translation functionality will be unavailable. Ensure 'IndicTransToolkit' is correctly installed and torch/transformers are compatible.")
        indictrans2_model = None
        indictrans2_tokenizer = None
        indictrans2_processor = None
        sys.exit(1) # Critical error, exit application


    # --- Fine-tune Fraud Detection Model ---
    try:
        logger.info("Attempting to fine-tune fraud detection model...")
        # Local import of fraud_detection.py
        from fraud_detection import fine_tune_model, predict_fraud

        output_dir = "./distilbert-fraud-finetuned"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory for fine-tuned model: {output_dir}")

        # Only fine-tune if the directory is empty (i.e., model not already saved)
        if not os.listdir(output_dir):
            logger.info("Fine-tuning fraud detection model on startup (first run)...")
            # Make sure 'ngo_fraud.csv' is present in the backend folder for fine-tuning
            # Pass the loaded DataFrame if available, otherwise handle empty case in fine_tune_model
            fine_tune_model(dataset_path=NGO_FRAUD_DATA_PATH, k_folds=1)
            logger.info("Fraud detection model fine-tuned and saved.")
        else:
            logger.info("Fraud detection model already fine-tuned. Skipping fine-tuning.")
    except Exception as e:
        logger.critical(f"FATAL ERROR: Could not fine-tune fraud detection model: {e}", exc_info=True)
        sys.exit(1) # Critical error, exit application


    # --- Initial Data Population for Firestore ---
    try:
        logger.info("Checking Firestore for initial data population...")
        if not db:
            logger.warning("Firestore not initialized, skipping initial data population.")
        else:
            campaigns_ref = db.collection("campaigns")
            # Check if the collection is empty before populating
            if not campaigns_ref.limit(1).get():
                logger.info("Populating initial campaign data in Firestore...")
                sample_campaigns = [
                    {"name": "EcoDrone: AI for Reforestation", "description": "Help us build AI-powered drones that plant trees and monitor forest health. Revolutionizing conservation efforts.", "author": "GreenFuture Now", "funded": 25000, "goal": 50000, "days_left": 15, "category": "Technology", "image_url": "https://placehold.co/600x400/A8DADC/333333?text=EcoDrone"},
                    {"name": "Echoes of Tomorrow - Indie Sci-Fi Film", "description": "Support our ambitious independent science fiction film exploring themes of memory and identity in a dystopian future.", "author": "Nova Pictures", "funded": 75000, "goal": 100000, "days_left": 30, "category": "Arts & Culture", "image_url": "https://placehold.co/600x400/DAB887/333333?text=Sci-Fi+Film"},
                    {"name": "The Art Hive: Community Art Space", "description": "We are creating a vibrant, accessible art studio and gallery space for everyone in our community.", "author": "Local Artists Collective", "funded": 10000, "goal": 20000, "days_left": 7, "category": "Community", "image_url": "https://placehold.co/600x400/ADD8E6/333333?text=Art+Hive"},
                    {"name": "Melody Weaver - Debut Album", "description": "Help me record and release my debut folk-pop album, filled with heartfelt stories and enchanting melodies.", "author": "Seraphina Moon", "funded": 3000, "goal": 18000, "days_left": 60, "category": "Music", "image_url": "https://placehold.co/600x400/FFD700/333333?text=Music+Album"},
                    {"name": "ReThread: Sustainable Fashion Line", "description": "Launching a new line of clothing made entirely from recycled materials and ethical practices.", "author": "EarthWear Designs", "funded": 5000, "goal": 15000, "days_left": 20, "category": "Fashion", "image_url": "https://placehold.co/600x400/98FB98/333333?text=Sustainable+Fashion"},
                    {"name": "Future Farms: Hydroponic Innovation", "description": "Developing modular hydroponic systems for sustainable urban farming, reducing water usage by 90%.", "author": "AgroTech Solutions", "funded": 40000, "goal": 60000, "days_left": 25, "category": "Technology", "image_url": "https://placehold.co/600x400/B0E0E6/333333?text=Hydroponics"},
                    {"name": "Historic Landmark Restoration", "description": "Raise funds to restore the dilapidated town hall, preserving its historical integrity for future generations.", "author": "Heritage Keepers", "funded": 15000, "goal": 30000, "days_left": 10, "category": "Community", "image_url": "https://placehold.co/600x400/DDA0DD/333333?text=Restoration"},
                    {"name": "Interactive STEM Workshops for Kids", "description": "Providing hands-on STEM education to underprivileged children through engaging workshops and resources.", "author": "Bright Minds Initiative", "funded": 8000, "goal": 12000, "days_left": 18, "category": "Education", "image_url": "https://placehold.co/600x400/87CEEB/333333?text=STEM+Kids"},
                ]
                batch = db.batch()
                algolia_initial_objects = []
                for campaign in sample_campaigns:
                    org_data_for_fraud_check = {
                        "org_name": campaign["author"],
                        "bio": campaign["description"],
                        "follower_count": random.randint(100, 5000),
                        "post_count": random.randint(10, 100),
                        "account_age_days": random.randint(100, 1000),
                        "engagement_rate": random.uniform(0.01, 0.05),
                        "recent_posts": campaign["description"],
                        "pan": "ABCDE1234F" if random.random() > 0.1 else "INVALID",
                        "registration_type": random.choice(["Section 8 Company", "Society", "Trust", ""]),
                        "registration_number": "U12345ABCDE67890FGHIJ" if random.random() > 0.1 else "",
                        "ngo_darpan_id": "UP1234567890" if random.random() > 0.1 else "",
                        "fcra_number": "1234567890" if random.random() > 0.1 else ""
                    }
                    fraud_score, explanation, plot_path, verification_details = predict_fraud(
                        organization_data=org_data_for_fraud_check
                    )

                    verification_status = ""
                    if fraud_score <= 0.20:
                        verification_status = "Verified by AI"
                    elif 0.21 <= fraud_score <= 0.50:
                        verification_status = "Needs Manual Review"
                    else:
                        verification_status = "Rejected"

                    if verification_status != "Rejected":
                        campaign_with_defaults = {
                            **campaign,
                            "created_by_uid": "system_init",
                            "created_at": firestore.SERVER_TIMESTAMP,
                            "fraud_score": fraud_score,
                            "fraud_explanation": explanation,
                            "verification_details": verification_details,
                            "verification_status": verification_status
                        }
                        new_doc_ref = campaigns_ref.document()
                        batch.set(new_doc_ref, campaign_with_defaults)
                        algolia_initial_objects.append({**campaign_with_defaults, "objectID": new_doc_ref.id})
                    else:
                        logger.info(f"Campaign '{campaign['name']}' initially rejected due to high fraud risk (score: {fraud_score:.2f}). Skipping population.")

                batch.commit()
                logger.info("Initial campaign data populated in Firestore.")

                if algolia_index and algolia_initial_objects:
                    try:
                        algolia_index.save_objects(algolia_initial_objects).wait()
                        logger.info(f"Initial {len(algolia_initial_objects)} campaigns indexed in Algolia.")
                    except Exception as e:
                        logger.error(f"Error indexing initial campaigns in Algolia: {e}", exc_info=True)
            else:
                logger.info("Firestore 'campaigns' collection already has data. Skipping initial population.")
    except Exception as e:
        logger.critical(f"FATAL ERROR: Initial data population in Firestore failed: {e}", exc_info=True)
        sys.exit(1) # Critical error, exit application

    logger.info("Application startup event completed successfully. Ready to serve.")


if __name__ == "__main__":
    import uvicorn
    # For local testing, if you don't have a GPU, set DEVICE = "cpu" at the top.
    # Also, ensure IndicTrans2 model files are available locally or downloaded.
    uvicorn.run(app, host="0.0.0.0", port=8000)