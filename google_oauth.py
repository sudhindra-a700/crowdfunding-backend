"""
Google OAuth 2.0 Implementation
Handles Google OAuth authentication flow for FastAPI
"""

import logging
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
        # Firebase instances will be passed via dependencies in routes, but for direct use in handler:
        # self.firebase_auth = auth.get_auth() # This would require app initialization
        # self.firestore_db = firestore.client() # This would require app initialization

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
        auth_url = self.oauth_config.get_google_auth_url(state)

        logger.info("Initiating Google OAuth login")
        return RedirectResponse(url=auth_url, status_code=302)

    async def handle_google_callback(
            self,
            request: Request,
            code: str,
            state: str,
            firebase_auth_client: auth.Auth,  # Dependency injection for Firebase Auth
            firestore_db: firestore.Client  # Dependency injection for Firestore
    ) -> TokenResponse:
        """Handle Google OAuth callback"""
        if not self.oauth_config.is_google_configured:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Google OAuth is not configured"
            )

        # Verify state parameter
        stored_state = request.session.get("oauth_state")
        if not stored_state or not self.jwt_manager.verify_state_token(stored_state, OAuthProvider.GOOGLE):
            logger.warning("Invalid or missing OAuth state parameter")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid state parameter"
            )

        # Clear state from session
        request.session.pop("oauth_state", None)

        try:
            # Exchange authorization code for access token
            token_data = self.oauth_config.exchange_google_code_for_token(code, state)
            access_token = token_data.get("access_token")

            if not access_token:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to obtain access token"
                )

            # Get user information from Google
            user_info_from_google = self.oauth_config.get_google_user_info(access_token)

            # Create OAuth user object
            oauth_user = OAuthUser.from_google_data(user_info_from_google)

            # --- Firebase Authentication Integration ---
            uid = None
            try:
                # Try to get user by email first
                firebase_user = firebase_auth_client.get_user_by_email(oauth_user.email)
                uid = firebase_user.uid
                logger.info(f"Existing Firebase user found for Google OAuth: {oauth_user.email}")
            except auth.UserNotFoundError:
                # If user not found by email, create a new Firebase user
                try:
                    firebase_user = firebase_auth_client.create_user(
                        email=oauth_user.email,
                        display_name=oauth_user.name,
                        photo_url=oauth_user.picture,
                        email_verified=True  # Assume email verified by Google
                    )
                    uid = firebase_user.uid
                    logger.info(f"New Firebase user created for Google OAuth: {oauth_user.email}")
                except Exception as create_err:
                    logger.error(f"Failed to create Firebase user for Google OAuth {oauth_user.email}: {create_err}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Failed to create user account"
                    )

            # --- Store/Update User Profile in Firestore ---
            user_doc_ref = firestore_db.collection("users").document(uid)
            user_doc = user_doc_ref.get()

            user_profile_data = {
                "uid": uid,
                "email": oauth_user.email,
                "name": oauth_user.name,
                "picture": oauth_user.picture,
                "provider": oauth_user.provider,
                "provider_id": oauth_user.provider_id,
                "last_login": firestore.SERVER_TIMESTAMP
            }

            if not user_doc.exists:
                user_profile_data["user_type"] = "individual"  # Default for new OAuth users
                user_profile_data["registered_at"] = firestore.SERVER_TIMESTAMP
                logger.info(f"Creating new Firestore profile for Google OAuth user: {uid}")
            else:
                # If profile exists, update it with latest OAuth info and last login
                existing_data = user_doc.to_dict()
                user_profile_data["user_type"] = existing_data.get("user_type", "individual")  # Preserve existing type
                logger.info(f"Updating existing Firestore profile for Google OAuth user: {uid}")

            user_doc_ref.set(user_profile_data, merge=True)  # Use merge to update or create

            # --- Create our custom JWT tokens ---
            # Ensure the user_profile_data passed to create_access_token has all necessary fields
            # including uid, user_type, etc.
            final_user_data_for_jwt = user_doc_ref.get().to_dict()  # Fetch the latest state

            jwt_access_token = self.jwt_manager.create_access_token(final_user_data_for_jwt)
            refresh_token = self.jwt_manager.create_refresh_token(uid)

            logger.info(f"Google OAuth login successful for user: {oauth_user.email}. Firebase UID: {uid}")

            return TokenResponse(
                access_token=jwt_access_token,
                refresh_token=refresh_token,
                token_type="bearer",
                expires_in=self.jwt_manager.expiration_hours * 3600  # Pass expiration
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Google OAuth callback error: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OAuth authentication failed"
            )

    def get_google_user_profile(self, access_token: str) -> Dict[str, Any]:
        """Get Google user profile using access token (from our custom JWT)"""
        try:
            user_data = self.jwt_manager.get_user_from_token(access_token)

            # Verify this is a Google user or a user authenticated via Google
            if user_data.get("provider") != OAuthProvider.GOOGLE:
                logger.warning(f"Token is not for a Google user: {user_data.get('provider')}")
                # This could be a user logged in via email/password or Facebook, but trying to access Google profile
                # For this specific endpoint, we expect a Google-linked user.
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Token is not for a Google-linked user"
                )

            return {
                "id": user_data.get("id"),  # This is the Firebase UID
                "email": user_data.get("email"),
                "name": user_data.get("name"),
                "picture": user_data.get("picture"),
                "provider": user_data.get("provider"),
                "provider_id": user_data.get("provider_id")
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get Google user profile: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get user profile"
            )


# Global Google OAuth handler instance
google_oauth_handler = GoogleOAuthHandler()


def get_google_oauth_handler() -> GoogleOAuthHandler:
    """Get the global Google OAuth handler instance"""
    return google_oauth_handler


# FastAPI route handlers
async def google_login_route(request: Request) -> RedirectResponse:
    """FastAPI route for Google OAuth login"""
    handler = get_google_oauth_handler()
    return handler.initiate_google_login(request)


# The actual callback route, now with Firebase dependencies
async def google_callback_route(
        request: Request,
        code: str,
        state: str,
        firebase_auth_client: auth.Auth = Depends(lambda: auth.get_auth()),  # Get auth client from app
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
    """FastAPI route for getting Google user profile"""
    handler = get_google_oauth_handler()
    return handler.get_google_user_profile(access_token)

