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
from firebase_admin import auth, firestore

logger = logging.getLogger(__name__)

oauth_router = APIRouter(prefix="/auth", tags=["OAuth Authentication"])
security = HTTPBearer()

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
    firebase_custom_token: Optional[str] = None

class OAuthStatusResponse(BaseModel):
    google_available: bool
    facebook_available: bool
    message: str

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    jwt_manager = get_jwt_manager()
    user_info = jwt_manager.get_user_from_token(credentials.credentials)
    if not user_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_info

@oauth_router.get("/google/login")
async def google_login(request: Request):
    return await google_login_route(request)

@oauth_router.get("/facebook/login")
async def facebook_login(request: Request):
    return await facebook_login_route(request)

@oauth_router.get("/google/callback")
async def google_callback(
    request: Request, code: str, state: str,
    firebase_auth_client: Any = Depends(lambda: auth),
    firestore_db: firestore.Client = Depends(lambda: firestore.client())
) -> TokenResponse:
    return await google_callback_route(request, code, state, firebase_auth_client, firestore_db)

@oauth_router.get("/facebook/callback")
async def facebook_callback(
    request: Request, code: str, state: str,
    firebase_auth_client: Any = Depends(lambda: auth),
    firestore_db: firestore.Client = Depends(lambda: firestore.client())
) -> TokenResponse:
    return await facebook_callback_route(request, code, state, firebase_auth_client, firestore_db)

@oauth_router.get("/status", response_model=OAuthStatusResponse)
async def get_oauth_status() -> OAuthStatusResponse:
    config = get_oauth_config()
    return OAuthStatusResponse(
        google_available=config.is_google_configured,
        facebook_available=config.is_facebook_configured,
        message="OAuth providers status retrieved successfully"
    )

@oauth_router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: Request, refresh_request: TokenRefreshRequest) -> TokenResponse:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Token refresh requires user data storage implementation"
    )

@oauth_router.post("/logout")
async def logout(current_user: Dict[str, Any] = Depends(get_current_user)):
    logger.info(f"User {current_user.get('id')} logged out")
    return {"message": "Successfully logged out"}

@oauth_router.get("/test")
async def test_protected_route(current_user: Dict[str, Any] = Depends(get_current_user)):
    return {"message": "This is a protected route", "user": current_user}

def get_oauth_router() -> APIRouter:
    return oauth_router
