"""
Google OAuth 2.0 Implementation
Handles Google OAuth authentication flow for FastAPI
"""

import logging
from typing import Dict, Any, Optional
from fastapi import HTTPException, status, Request
from fastapi.responses import RedirectResponse
import secrets

from oauth_config import get_oauth_config, OAuthUser, OAuthProvider
from jwt_utils import get_jwt_manager, TokenResponse

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
        auth_url = self.oauth_config.get_google_auth_url(state)
        
        logger.info("Initiating Google OAuth login")
        return RedirectResponse(url=auth_url, status_code=302)
    
    async def handle_google_callback(self, request: Request, code: str, state: str) -> TokenResponse:
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
            user_info = self.oauth_config.get_google_user_info(access_token)
            
            # Create OAuth user object
            oauth_user = OAuthUser.from_google_data(user_info)
            
            # Create JWT tokens
            user_data = oauth_user.to_dict()
            jwt_access_token = self.jwt_manager.create_access_token(user_data)
            refresh_token = self.jwt_manager.create_refresh_token(oauth_user.id)
            
            logger.info(f"Google OAuth login successful for user: {oauth_user.email}")
            
            return TokenResponse(
                access_token=jwt_access_token,
                refresh_token=refresh_token,
                token_type="bearer"
            )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Google OAuth callback error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OAuth authentication failed"
            )
    
    def get_google_user_profile(self, access_token: str) -> Dict[str, Any]:
        """Get Google user profile using access token"""
        try:
            user_data = self.jwt_manager.get_user_from_token(access_token)
            
            # Verify this is a Google user
            if user_data.get("provider") != OAuthProvider.GOOGLE:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Token is not for a Google user"
                )
            
            return {
                "id": user_data.get("id"),
                "email": user_data.get("email"),
                "name": user_data.get("name"),
                "picture": user_data.get("picture"),
                "provider": user_data.get("provider"),
                "provider_id": user_data.get("provider_id")
            }
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get Google user profile: {e}")
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


async def google_callback_route(request: Request, code: str, state: str) -> Dict[str, Any]:
    """FastAPI route for Google OAuth callback"""
    handler = get_google_oauth_handler()
    token_response = await handler.handle_google_callback(request, code, state)
    return token_response.to_dict()


async def google_profile_route(access_token: str) -> Dict[str, Any]:
    """FastAPI route for getting Google user profile"""
    handler = get_google_oauth_handler()
    return handler.get_google_user_profile(access_token)

