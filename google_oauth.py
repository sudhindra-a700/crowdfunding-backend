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
        authorize_url = self.oauth_config.google_oauth_client.create_authorization_url(
            url=self.oauth_config.google_auth_url,
            state=state,
            scope="openid email profile",
            redirect_uri=self.oauth_config.google_redirect_uri
        )
        return RedirectResponse(authorize_url)


# The actual callback route, now with Firebase dependencies
async def google_callback_route(
        request: Request,
        code: str,
        state: str,
        # Removed auth.Auth type hint
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

        state = self.jwt_manager.create_state_token(OAuthProvider.GOOGLE)
        request.session["oauth_state"] = state

        authorize_url = self.oauth_config.google_oauth_client.create_authorization_url(
            url=self.oauth_config.google_auth_url,
            state=state,
            scope="openid email profile",
            redirect_uri=self.oauth_config.google_redirect_uri
        )
        return RedirectResponse(authorize_url)

    async def handle_google_callback(
            self,
            request: Request,
            code: str,
            state: str,
            firebase_auth_client: Any, # Removed auth.Auth type hint
            firestore_db: firestore.Client
    ) -> TokenResponse:
        """Handle Google OAuth callback and exchange code for token"""
        if state != request.session.pop("oauth_state", None):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid state parameter")

        try:
            token_response = await self.oauth_config.google_oauth_client.fetch_token(
                url=self.oauth_config.google_token_url,
                code=code,
                redirect_uri=self.oauth_config.google_redirect_uri,
                client_id=self.oauth_config.google_client_id,
                client_secret=self.oauth_config.google_client_secret
            )
            access_token = token_response["access_token"]
            user_info = await self.fetch_google_user_profile(access_token)
            oauth_user = OAuthUser.from_google_data(user_info)

            # Check for existing user by email or provider ID
            try:
                firebase_user = firebase_auth_client.get_user_by_email(oauth_user.email)
                uid = firebase_user.uid
                logger.info(f"Existing Firebase user found for email: {oauth_user.email}, UID: {uid}")
            except auth.UserNotFoundError:
                try:
                    # Create Firebase user if not exists
                    firebase_user = firebase_auth_client.create_user(
                        email=oauth_user.email,
                        email_verified=True,
                        display_name=oauth_user.name,
                        photo_url=oauth_user.picture,
                        uid=oauth_user.id # Use OAuth provider ID as Firebase UID
                    )
                    uid = firebase_user.uid
                    logger.info(f"New Firebase user created with UID: {uid} from Google OAuth.")
                except Exception as e:
                    logger.error(f"Error creating Firebase user from Google OAuth: {e}", exc_info=True)
                    # If UID from oauth_user.id already exists, try to get it
                    if "already exists" in str(e):
                        firebase_user = firebase_auth_client.get_user(oauth_user.id)
                        uid = firebase_user.uid
                        logger.info(f"Firebase user with UID {uid} already exists, proceeding with existing user.")
                    else:
                        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Firebase user creation failed.")

            # Create or update user profile in Firestore
            user_doc_ref = firestore_db.collection("users").document(uid)
            user_data = {
                "uid": uid,
                "email": oauth_user.email,
                "name": oauth_user.name,
                "picture": oauth_user.picture,
                "provider": oauth_user.provider.value,
                "provider_id": oauth_user.provider_id,
                "last_login": firestore.SERVER_TIMESTAMP
            }
            user_doc_ref.set(user_data, merge=True)
            logger.info(f"Firestore profile updated/created for UID: {uid}")

            # Create custom JWT token for frontend
            custom_token = self.jwt_manager.create_access_token(user_data)
            refresh_token = self.jwt_manager.create_refresh_token(uid)

            return TokenResponse(
                access_token=custom_token,
                token_type="bearer",
                expires_in=self.jwt_manager.expiration_hours * 3600,
                refresh_token=refresh_token,
                user_info=user_data # Pass the user_data dictionary
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Google token exchange failed: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to exchange Google token.")
        except Exception as e:
            logger.error(f"Error during Google OAuth callback: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Google OAuth callback failed.")

    async def fetch_google_user_profile(self, access_token: str) -> Dict[str, Any]:
        """Fetch user profile from Google API"""
        try:
            userinfo_response = requests.get(
                self.oauth_config.google_userinfo_url,
                headers={"Authorization": f"Bearer {access_token}"}
            )
            userinfo_response.raise_for_status()
            return userinfo_response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch Google user profile: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch Google user profile.")


# Global instance of GoogleOAuthHandler
google_oauth_handler = GoogleOAuthHandler()

def get_google_oauth_handler() -> GoogleOAuthHandler:
    """Get the global Google OAuth handler instance"""
    return google_oauth_handler
