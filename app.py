# app.py
# This is the updated, complete code for your FastAPI backend,
# now with API endpoints for user profile management and image upload.

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

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your_jwt_secret_key")
ALGORITHM = "HS256"

# --- Mock Data and Storage ---
# In-memory "database" for demonstration purposes
MOCK_DATABASE = {
    "campaigns": [
        {"id": "camp1", "title": "Clean Water for All", "organization": "WaterOrg", "category": "Environment", "description": "...", "target_amount": 10000, "current_amount": 7500},
        {"id": "camp2", "title": "Education for Rural Children", "organization": "EduFund", "category": "Education", "description": "...", "target_amount": 5000, "current_amount": 4000},
        {"id": "camp3", "title": "Medical Supplies for Hospitals", "organization": "HealthAid", "category": "Health", "description": "...", "target_amount": 20000, "current_amount": 18000},
        {"id": "camp4", "title": "Sustainable Farming Initiative", "organization": "GreenHands", "category": "Environment", "description": "...", "target_amount": 8000, "current_amount": 2500},
    ],
    "users": {
        "test@example.com": {
            "name": "John Doe",
            "email": "test@example.com",
            "password": "hashed_password", # This should be a real hash
            "user_type": "donor",
            "profile_image": None # Store base64 string here
        }
    }
}

# --- FastAPI App Initialization ---
app = FastAPI(title="HAVEN Crowdfunding Backend", description="API for the HAVEN crowdfunding platform.")

# --- CORS Middleware ---
origins = [
    FRONTEND_URL,
    "http://localhost:8501",  # Streamlit's default local address
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class UpdateProfile(BaseModel):
    name: str

class TextToSimplify(BaseModel):
    text: str
    target_language: str = "en"

# --- Mock Authentication Dependency ---
def get_current_user_email(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET_KEY, algorithms=[ALGORITHM])
        user_email: str = payload.get("sub")
        if user_email is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return user_email
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Welcome to the HAVEN Crowdfunding API!"}

# Mock user login endpoint
@app.post("/api/login")
async def login(login_data: LoginRequest):
    user = MOCK_DATABASE["users"].get(login_data.email)
    if user and user["password"] == "hashed_password": # Replace with real hash check
        token_data = {"sub": user["email"]}
        token = jwt.encode(token_data, JWT_SECRET_KEY, algorithm=ALGORITHM)
        return JSONResponse(content={"token": token, "user": user})
    raise HTTPException(status_code=401, detail="Invalid credentials")

# New endpoint to get the authenticated user's data
@app.get("/api/user/me")
async def get_user_me(user_email: str = Depends(get_current_user_email)):
    user = MOCK_DATABASE["users"].get(user_email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# New endpoint to update user profile details
@app.post("/api/user/profile")
async def update_user_profile(profile_data: UpdateProfile, user_email: str = Depends(get_current_user_email)):
    user = MOCK_DATABASE["users"].get(user_email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user["name"] = profile_data.name
    return {"message": "Profile updated successfully", "user": user}

# New endpoint to upload a profile image
@app.post("/api/user/profile/image")
async def upload_profile_image(file: UploadFile = File(...), user_email: str = Depends(get_current_user_email)):
    user = MOCK_DATABASE["users"].get(user_email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Read the file and encode to base64
    image_bytes = await file.read()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    user["profile_image"] = image_base64
    
    return {"message": "Profile image uploaded successfully"}

# Placeholder endpoint for creating a new campaign
@app.post("/api/campaigns")
async def create_campaign(user_email: str = Depends(get_current_user_email)):
    # Logic for creating a campaign would go here
    return {"message": "Campaign creation endpoint is a placeholder."}

# Existing endpoints (trending, search, etc.)
@app.get("/api/trending")
async def get_trending_campaigns():
    # Return the first 3 campaigns as trending for now
    trending = MOCK_DATABASE["campaigns"][:3]
    return {"trending_campaigns": trending}

@app.get("/api/search")
async def search_campaigns(query: str):
    # Mock search functionality
    results = [c for c in MOCK_DATABASE["campaigns"] if query.lower() in c["title"].lower()]
    return {"search_results": results}

@app.get("/api/categories")
async def get_categories():
    categories = sorted(list(set(c["category"] for c in MOCK_DATABASE["campaigns"])))
    return {"categories": categories}

@app.get("/api/campaigns/{campaign_id}")
async def get_campaign_details(campaign_id: str):
    campaign = next((c for c in MOCK_DATABASE["campaigns"] if c["id"] == campaign_id), None)
    if campaign is None:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return {"campaign": campaign}

@app.post("/api/process_text")
async def process_text(text_data: TextToSimplify):
    # This endpoint is a placeholder for a term simplification API call
    # Replace this with a real call to the Gemini API
    # For now, it will just return a placeholder response
    simplified_text = f"Simplified version of '{text_data.text}' in {text_data.target_language}."
    return {"simplified_text": simplified_text}

# --- OAuth Endpoints (unchanged) ---
@app.get("/auth/google/callback")
async def google_auth_callback(code: str, request: Request):
    if not all([GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT_URI]):
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Missing Google OAuth credentials.")
    token_url = "https://oauth2.googleapis.com/token"
    data = { "code": code, "client_id": GOOGLE_CLIENT_ID, "client_secret": GOOGLE_CLIENT_SECRET, "redirect_uri": GOOGLE_REDIRECT_URI, "grant_type": "authorization_code", }
    try:
        response = requests.post(token_url, data=data)
        response.raise_for_status()
        token_info = response.json()
        return RedirectResponse(url=f"{FRONTEND_URL}?token_info={token_info['access_token']}")
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
        return RedirectResponse(url=f"{FRONTEND_URL}?token_info={token_info['access_token']}")
    except requests.exceptions.RequestException as e: raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get Facebook token: {e}")

# Custom 404 handler (unchanged)
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=404, content={"error": "Endpoint not found", "message": f"The requested endpoint {request.url.path} was not found"})

# --- Main Entry Point ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

