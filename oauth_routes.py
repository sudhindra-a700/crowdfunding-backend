"""
OAuth Routes for FastAPI Application
Integrates Google and Facebook OAuth authentication routes
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status, Request, Depends, Query
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from .google_oauth import get_google_oauth_handler
from .facebook_oauth import get_facebook_oauth_handler
from .jwt_utils import get_jwt_manager
from .oauth_config import get_oauth_config

logger = logging.getLogger(__name__)

# Create OAuth router
oauth_router = APIRouter(prefix="/auth", tags=["OAuth Authentication"])

# Security scheme for JWT tokens
security = HTTPBearer()

# Pydantic models for request/response
class TokenRefreshRequest(BaseModel):
    refresh_token: str

class UserProfileResponse(BaseModel):
    id: str
    email: str
    name: str
    picture: Optional[str] = None
    provider: str
    provider_id: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    refresh_token: Optional[str] = None

class OAuthStatusResponse(BaseModel):
    google_available: bool
    facebook_available: bool
    message: str

# Dependency to get current user from JWT token
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Get current user from JWT token"""
    jwt_manager = get_jwt_manager()
    try:
        user_data = jwt_manager.get_user_from_token(credentials.credentials)
        return user_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get current user: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )

# OAuth status endpoint
@oauth_router.get("/status", response_model=OAuthStatusResponse)
async def oauth_status():
    """Get OAuth providers availability status"""
    oauth_config = get_oauth_config()
    
    return OAuthStatusResponse(
        google_available=oauth_config.is_google_configured,
        facebook_available=oauth_config.is_facebook_configured,
        message="OAuth providers status"
    )

# Google OAuth routes
@oauth_router.get("/google")
async def google_login(request: Request):
    """Initiate Google OAuth login"""
    google_handler = get_google_oauth_handler()
    return google_handler.initiate_google_login(request)

@oauth_router.get("/google/callback", response_model=TokenResponse)
async def google_callback(
    request: Request,
    code: str = Query(..., description="Authorization code from Google"),
    state: str = Query(..., description="State parameter for security")
):
    """Handle Google OAuth callback"""
    from google_oauth import google_callback_route
    return await google_callback_route(request, code, state)

# Facebook OAuth routes
@oauth_router.get("/facebook")
async def facebook_login(request: Request):
    """Initiate Facebook OAuth login"""
    facebook_handler = get_facebook_oauth_handler()
    return facebook_handler.initiate_facebook_login(request)

@oauth_router.get("/facebook/callback", response_model=TokenResponse)
async def facebook_callback(
    request: Request,
    code: str = Query(..., description="Authorization code from Facebook"),
    state: str = Query(..., description="State parameter for security")
):
    """Handle Facebook OAuth callback"""
    from facebook_oauth import facebook_callback_route
    return await facebook_callback_route(request, code, state)

# User profile endpoint
@oauth_router.get("/profile", response_model=UserProfileResponse)
async def get_user_profile(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get current user profile"""
    return UserProfileResponse(
        id=current_user.get("id"),
        email=current_user.get("email"),
        name=current_user.get("name"),
        picture=current_user.get("picture"),
        provider=current_user.get("provider"),
        provider_id=current_user.get("provider_id")
    )

# Token refresh endpoint
@oauth_router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: TokenRefreshRequest):
    """Refresh access token using refresh token"""
    jwt_manager = get_jwt_manager()
    
    try:
        # Verify refresh token and get user ID
        user_id = jwt_manager.verify_refresh_token(request.refresh_token)
        
        # For now, we'll need to store user data to recreate the token
        # In a real application, you'd fetch user data from database
        # This is a simplified implementation
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Token refresh requires user data storage implementation"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )

# Logout endpoint
@oauth_router.post("/logout")
async def logout(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Logout user (invalidate token)"""
    # In a real application, you might want to blacklist the token
    # For now, we'll just return success since JWT tokens are stateless
    logger.info(f"User {current_user.get('id')} logged out")
    return {"message": "Successfully logged out"}

# Test protected endpoint
@oauth_router.get("/test")
async def test_protected_route(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Test protected route that requires authentication"""
    return {
        "message": "This is a protected route",
        "user": current_user
    }

# Error handling is done at the app level, not router level

def get_oauth_router() -> APIRouter:
    """Get the OAuth router for inclusion in main app"""
    return oauth_router

