"""
Google OAuth 2.0 Implementation
Handles Google OAuth authentication flow for FastAPI
"""

import logging
import requests  # Added missing import
from typing import Dict, Any, Optional
from fastapi import HTTPException, status, Request, Depends
from fastapi.responses import RedirectResponse
import json
from oauth_config import get_oauth_config, OAuthUser, OAuthProvider
from jwt_utils import get_jwt_manager, TokenResponse
from firebase_admin import auth, firestore  # Import Firebase Admin SDK
import os  # Import os for environment variables
from fraud_detection import predict_fraud, load_ngo_darpan_data  # Import for fraud detection

logger = logging.getLogger(__name__)

class GoogleOAuthHandler:
    """Google OAuth authentication handler"""
    
    def __init__(self):
        self.oauth_config = get_oauth_config()
        self.jwt_manager = get_jwt_manager()

    def initiate_google_login(self, request: Request) -> RedirectResponse:
        """Initiate Google OAuth login flow"""
        if not self.oauth_config.is_google_configured:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Google OAuth is not configured"
            )

        # Generate state token for security
        state = self.jwt_manager.create_state_token(OAuthProvider.GOOGLE)
        
        # Store state in session (you might want to use Redis or database in production)
        request.session["oauth_state"] = state

        # ✅ FIXED: Use oauth_config helper method instead of manual OAuth2Session
        authorization_url = self.oauth_config.get_google_auth_url(state)

        return RedirectResponse(url=authorization_url)

    async def handle_google_callback(
        self,
        request: Request,
        code: str,
        state: str,
        firebase_auth_client: auth.Client = Depends(lambda: auth),
        firestore_db: firestore.Client = Depends(lambda: firestore.client())
    ) -> TokenResponse:
        """Handle Google OAuth callback and get tokens"""
        
        # Validate state token
        if not self.jwt_manager.verify_state_token(state, OAuthProvider.GOOGLE):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired state token."
            )

        # ✅ FIXED: Use oauth_config helper method for token exchange
        try:
            token = self.oauth_config.exchange_google_code_for_token(code, state)
        except Exception as e:
            logger.error(f"Error fetching Google token: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not exchange code for token."
            )

        # Get user info from Google
        user_profile = await self.fetch_google_user_profile(token.get("access_token"))

        # --- Firebase Integration ---
        user_email = user_profile.get("email")
        user_name = user_profile.get("name")
        user_photo = user_profile.get("picture")

        try:
            # Check if a user with this email already exists
            firebase_user = firebase_auth_client.get_user_by_email(user_email)
            firebase_uid = firebase_user.uid
            logger.info(f"Existing Firebase user found with UID: {firebase_uid}")
        except auth.AuthError:
            # If not, create a new Firebase user
            firebase_user = firebase_auth_client.create_user(
                email=user_email,
                email_verified=True,
                display_name=user_name,
                photo_url=user_photo
            )
            firebase_uid = firebase_user.uid
            logger.info(f"New Firebase user created with UID: {firebase_uid}")

        # Generate a Firebase Custom Token for client-side authentication
        firebase_custom_token = firebase_auth_client.create_custom_token(firebase_uid)

        # Store or update user data in Firestore
        user_doc_ref = firestore_db.collection("users").document(firebase_uid)
        user_doc_data = {
            "email": user_email,
            "name": user_name,
            "photo_url": user_photo,
            "provider": OAuthProvider.GOOGLE.value,
            "last_login": firestore.SERVER_TIMESTAMP
        }

        return {
            "access_token": token.get("access_token"),
            "refresh_token": token.get("refresh_token"),
            "firebase_token": firebase_custom_token.decode('utf-8'),
            "user": {
                "email": user_email,
                "name": user_name,
                "photo_url": user_photo,
                "firebase_uid": firebase_uid
            }
        }

    async def fetch_google_user_profile(self, access_token: str) -> Dict[str, Any]:
        """Fetch user profile from Google API"""
        try:
            response = requests.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching Google user profile: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not fetch user profile from Google."
            )

# Create global instance
google_oauth_handler = GoogleOAuthHandler()

# ✅ Export the exact function names that oauth_routes.py expects
def get_google_oauth_handler():
    """Get the Google OAuth handler instance"""
    return google_oauth_handler

async def google_login_route(request: Request) -> RedirectResponse:
    """Initiate Google OAuth login flow - matches oauth_routes.py import"""
    return google_oauth_handler.initiate_google_login(request)

async def google_callback_route(
    request: Request,
    code: str,
    state: str,
    firebase_auth_client: auth.Client = Depends(lambda: auth),
    firestore_db: firestore.Client = Depends(lambda: firestore.client())
) -> TokenResponse:
    """Handle Google OAuth callback - matches oauth_routes.py import"""
    return await google_oauth_handler.handle_google_callback(
        request, code, state, firebase_auth_client, firestore_db
    )

