"""
Facebook OAuth 2.0 Implementation
Handles Facebook OAuth authentication flow for FastAPI
"""

import logging
from typing import Dict, Any, Optional
from fastapi import HTTPException, status, Request
from fastapi.responses import RedirectResponse
import secrets
from .oauth_config import get_oauth_config, OAuthUser, OAuthProvider
from .jwt_utils import get_jwt_manager, TokenResponse

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
        
        # Generate state token for security
        state = self.jwt_manager.create_state_token(OAuthProvider.FACEBOOK)
        
        # Store state in session (you might want to use Redis or database in production)
        request.session["oauth_state"] = state
        
        # Generate authorization URL
        auth_url = self.oauth_config.get_facebook_auth_url(state)
        
        logger.info("Initiating Facebook OAuth login")
        return RedirectResponse(url=auth_url, status_code=302)
    
    async def handle_facebook_callback(self, request: Request, code: str, state: str) -> TokenResponse:
        """Handle Facebook OAuth callback"""
        if not self.oauth_config.is_facebook_configured:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Facebook OAuth is not configured"
            )
        
        # Verify state parameter
        stored_state = request.session.get("oauth_state")
        if not stored_state or not self.jwt_manager.verify_state_token(stored_state, OAuthProvider.FACEBOOK):
            logger.warning("Invalid or missing OAuth state parameter")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid state parameter"
            )
        
        # Clear state from session
        request.session.pop("oauth_state", None)
        
        try:
            # Exchange authorization code for access token
            token_data = self.oauth_config.exchange_facebook_code_for_token(code, state)
            access_token = token_data.get("access_token")
            
            if not access_token:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to obtain access token"
                )
            
            # Get user information from Facebook
            user_info = self.oauth_config.get_facebook_user_info(access_token)
            
            # Create OAuth user object
            oauth_user = OAuthUser.from_facebook_data(user_info)
            
            # Create JWT tokens
            user_data = oauth_user.to_dict()
            jwt_access_token = self.jwt_manager.create_access_token(user_data)
            refresh_token = self.jwt_manager.create_refresh_token(oauth_user.id)
            
            logger.info(f"Facebook OAuth login successful for user: {oauth_user.email}")
            
            return TokenResponse(
                access_token=jwt_access_token,
                refresh_token=refresh_token,
                token_type="bearer"
            )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Facebook OAuth callback error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OAuth authentication failed"
            )
    
    def get_facebook_user_profile(self, access_token: str) -> Dict[str, Any]:
        """Get Facebook user profile using access token"""
        try:
            user_data = self.jwt_manager.get_user_from_token(access_token)
            
            # Verify this is a Facebook user
            if user_data.get("provider") != OAuthProvider.FACEBOOK:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Token is not for a Facebook user"
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
            logger.error(f"Failed to get Facebook user profile: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get user profile"
            )


# Global Facebook OAuth handler instance
facebook_oauth_handler = FacebookOAuthHandler()


def get_facebook_oauth_handler() -> FacebookOAuthHandler:
    """Get the global Facebook OAuth handler instance"""
    return facebook_oauth_handler


# FastAPI route handlers
async def facebook_login_route(request: Request) -> RedirectResponse:
    """FastAPI route for Facebook OAuth login"""
    handler = get_facebook_oauth_handler()
    return handler.initiate_facebook_login(request)


async def facebook_callback_route(request: Request, code: str, state: str) -> Dict[str, Any]:
    """FastAPI route for Facebook OAuth callback"""
    handler = get_facebook_oauth_handler()
    token_response = await handler.handle_facebook_callback(request, code, state)
    return token_response.to_dict()


async def facebook_profile_route(access_token: str) -> Dict[str, Any]:
    """FastAPI route for getting Facebook user profile"""
    handler = get_facebook_oauth_handler()
    return handler.get_facebook_user_profile(access_token)

