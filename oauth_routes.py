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
from google_oauth import get_google_oauth_handler, google_callback_route, google_login_route
from facebook_oauth import get_facebook_oauth_handler, facebook_callback_route, facebook_login_route
from jwt_utils import get_jwt_manager
from oauth_config import get_oauth_config
from firebase_admin import auth, firestore  # Import Firebase Admin SDK

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
    firebase_custom_token: Optional[str] = None  # Added for Firebase integration

class OAuthStatusResponse(BaseModel):
    google_available: bool
    facebook_available: bool
    message: str

# Dependency to get current user from JWT token
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Get current user from JWT token"""
    jwt_manager = get_jwt_manager()
    user_info = jwt_manager.get_user_from_token(credentials.credentials)
    
    if not user_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user_info

# --- OAuth Routes ---

# Login endpoints
@oauth_router.get("/google/login")
async def google_login(request: Request):
    """Initiate Google OAuth login flow"""
    return await google_login_route(request)  # ✅ FIXED: Added await

@oauth_router.get("/facebook/login")
async def facebook_login(request: Request):
    """Initiate Facebook OAuth login flow"""
    return await facebook_login_route(request)  # ✅ FIXED: Added await

# Callback endpoints
@oauth_router.get("/google/callback")
async def google_callback(
    request: Request,
    code: str,
    state: str,
    firebase_auth_client: Any = Depends(lambda: auth.get_auth()),
    firestore_db: firestore.Client = Depends(lambda: firestore.client())
) -> TokenResponse:
    """Handle Google OAuth callback and return tokens"""
    return await google_callback_route(request, code, state, firebase_auth_client, firestore_db)

@oauth_router.get("/facebook/callback")
async def facebook_callback(
    request: Request,
    code: str,
    state: str,
    firebase_auth_client: Any = Depends(lambda: auth.get_auth()),
    firestore_db: firestore.Client = Depends(lambda: firestore.client())
) -> TokenResponse:
    """Handle Facebook OAuth callback and return tokens"""
    return await facebook_callback_route(request, code, state, firebase_auth_client, firestore_db)

@oauth_router.get("/status", response_model=OAuthStatusResponse)
async def get_oauth_status() -> OAuthStatusResponse:
    """Check if OAuth providers are configured"""
    config = get_oauth_config()  # ✅ This function is synchronous, no await needed
    return OAuthStatusResponse(
        google_available=config.is_google_configured,
        facebook_available=config.is_facebook_configured,
        message="OAuth providers status retrieved successfully"
    )

@oauth_router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: Request, refresh_request: TokenRefreshRequest) -> TokenResponse:
    """Refresh an access token using a refresh token"""
    try:
        # In a real application, you would use the refresh token to get a new access token
        # You'll need to store user data to recreate the token
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

