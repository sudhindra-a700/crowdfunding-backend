"""
Facebook OAuth 2.0 Implementation
Handles Facebook OAuth authentication flow for FastAPI
"""

import logging
import requests
from typing import Dict, Any, Optional
from fastapi import HTTPException, status, Request, Depends
from fastapi.responses import RedirectResponse
import json
from oauth_config import get_oauth_config, OAuthUser, OAuthProvider
from jwt_utils import get_jwt_manager, TokenResponse
from firebase_admin import auth, firestore
import os

logger = logging.getLogger(__name__)

class FacebookOAuthHandler:
    """Facebook OAuth authentication handler"""
    
    def __init__(self):
        self.oauth_config = get_oauth_config()
        self.jwt_manager = get_jwt_manager()

    def initiate_facebook_login(self, request: Request) -> RedirectResponse:
        """Initiate Facebook OAuth login flow"""
        if not self.oauth_config.is_facebook_configured:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Facebook OAuth is not configured"
            )
        state = self.jwt_manager.create_state_token(OAuthProvider.FACEBOOK.value)
        request.session["oauth_state"] = state
        authorization_url = self.oauth_config.get_facebook_auth_url(state)
        return RedirectResponse(url=authorization_url)

    async def handle_facebook_callback(
        self,
        request: Request,
        code: str,
        state: str,
        firebase_auth_client: auth.Client = Depends(lambda: auth),
        firestore_db: firestore.Client = Depends(lambda: firestore.client())
    ) -> TokenResponse:
        """Handle Facebook OAuth callback and get tokens"""
        if not self.jwt_manager.verify_state_token(state, OAuthProvider.FACEBOOK.value):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired state token."
            )
        try:
            token = self.oauth_config.exchange_facebook_code_for_token(code, state)
        except Exception as e:
            logger.error(f"Error fetching Facebook token: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not exchange code for token."
            )
        user_profile = await self.fetch_facebook_user_profile(token.get("access_token"))
        user_email = user_profile.get("email")
        user_name = user_profile.get("name")
        user_id = user_profile.get("id")
        user_photo = f"https://graph.facebook.com/{user_id}/picture?type=large"
        try:
            firebase_user = firebase_auth_client.get_user_by_email(user_email)
            firebase_uid = firebase_user.uid
            logger.info(f"Existing Firebase user found with UID: {firebase_uid}")
        except auth.AuthError:
            firebase_user = firebase_auth_client.create_user(
                email=user_email,
                email_verified=True,
                display_name=user_name,
                photo_url=user_photo
            )
            firebase_uid = firebase_user.uid
            logger.info(f"New Firebase user created with UID: {firebase_uid}")
        firebase_custom_token = firebase_auth_client.create_custom_token(firebase_uid)
        user_doc_ref = firestore_db.collection("users").document(firebase_uid)
        user_doc_data = {
            "email": user_email, "name": user_name, "photo_url": user_photo,
            "provider": OAuthProvider.FACEBOOK.value, "last_login": firestore.SERVER_TIMESTAMP
        }
        user_doc_ref.set(user_doc_data, merge=True)
        
        app_token = self.jwt_manager.create_access_token({"uid": firebase_uid, "email": user_email, "name": user_name, "user_photo": user_photo})

        return TokenResponse(
            access_token=app_token,
            firebase_custom_token=firebase_custom_token.decode('utf-8'),
            user_info=user_doc_data
        )

    async def fetch_facebook_user_profile(self, access_token: str) -> Dict[str, Any]:
        """Fetch user profile from Facebook API"""
        try:
            response = requests.get(
                "https://graph.facebook.com/me",
                params={"fields": "id,name,email,picture", "access_token": access_token}
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching Facebook user profile: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not fetch user profile from Facebook."
            )

facebook_oauth_handler = FacebookOAuthHandler()

def get_facebook_oauth_handler():
    return facebook_oauth_handler

async def facebook_login_route(request: Request) -> RedirectResponse:
    return facebook_oauth_handler.initiate_facebook_login(request)

async def facebook_callback_route(
    request: Request, code: str, state: str,
    firebase_auth_client: auth.Client = Depends(lambda: auth),
    firestore_db: firestore.Client = Depends(lambda: firestore.client())
) -> TokenResponse:
    return await facebook_oauth_handler.handle_facebook_callback(
        request, code, state, firebase_auth_client, firestore_db
    )
