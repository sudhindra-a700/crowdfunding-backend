"""
Google OAuth 2.0 Implementation
Handles Google OAuth authentication flow for FastAPI
"""

import logging
import requests  # Added missing import
from typing import Dict, Any, Optional
from fastapi import HTTPException, status, Request, Depends
from fastapi.responses import RedirectResponse
import secrets
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

        # Generate authorization URL
        authorization_url = self.oauth_config.google_client.get_authorization_url(
            redirect_uri=self.oauth_config.google_callback_uri,
            state=state,
            scope="openid email profile"
        )

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

        # Exchange authorization code for tokens
        try:
            token = await self.oauth_config.google_client.fetch_token(
                self.oauth_config.google_token_uri,
                authorization_response=str(request.url),
                client_id=self.oauth_config.google_client_id,
                client_secret=self.oauth_config.google_client_secret
            )
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
        user_doc_ref.set(user_doc_data, merge=True)
        logger.info(f"User data for {firebase_uid} updated in Firestore.")
        
        # Create a JWT for API access
        jwt_access_token = self.jwt_manager.create_access_token({
            "uid": firebase_uid,
            "email": user_email,
            "name": user_name,
            "user_photo": user_photo
        })
        
        # Create the TokenResponse object
        response = TokenResponse(
            access_token=jwt_access_token,
            refresh_token=token.get("refresh_token"),
            expires_in=token.get("expires_in", 3600),
            user_info={
                "uid": firebase_uid,
                "email": user_email,
                "name": user_name,
                "photo_url": user_photo
            },
            firebase_custom_token=firebase_custom_token
        )

        return response

    async def fetch_google_user_profile(self, access_token: str) -> Dict[str, Any]:
        """Fetch user profile information from Google"""
        profile_url = "https://www.googleapis.com/oauth2/v3/userinfo"
        headers = {"Authorization": f"Bearer {access_token}"}
        try:
            response = requests.get(profile_url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching Google user profile: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not fetch user profile from Google."
            )

# Global instance of GoogleOAuthHandler
google_oauth_handler = GoogleOAuthHandler()


def get_google_oauth_handler() -> GoogleOAuthHandler:
    """Dependency injector for GoogleOAuthHandler"""
    return google_oauth_handler

# FastAPI routes for Google OAuth
async def google_login_route(request: Request):
    """FastAPI route for initiating Google OAuth login"""
    handler = get_google_oauth_handler()
    return handler.initiate_google_login(request)


# Added Firebase dependencies
async def google_callback_route(
        request: Request,
        code: str,
        state: str,
        firebase_auth_client: Any = Depends(lambda: auth.get_auth()),  # Get auth client from app
        firestore_db: firestore.Client = Depends(lambda: firestore.client())  # Get firestore client from app
) -> Dict[str, Any]:
    """FastAPI route for Google OAuth callback"""
    handler = get_google_oauth_handler()
    token_response = await handler.handle_google_callback(request, code, state, firebase_auth_client, firestore_db)
    # The frontend expects 'access_token', 'token_type', 'expires_in', 'refresh_token' and 'user_info'
    # The TokenResponse object has a to_dict() method that provides these.
    # We need to add user_info to this dict.
    jwt_manager = get_jwt_manager()
    user_info_from_jwt = jwt_manager.get_user_from_token(token_response.access_token)
    response_dict = token_response.to_dict()
    response_dict['user_info'] = user_info_from_jwt  # Add user_info
    return response_dict


async def google_profile_route(access_token: str) -> Dict[str, Any]:
    """FastAPI route for fetching Google user profile"""
    handler = get_google_oauth_handler()
    return await handler.fetch_google_user_profile(access_token)
