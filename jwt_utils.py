"""
JWT Utilities for OAuth Authentication
Handles JWT token creation, validation, and user session management
"""

import os
import jwt
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from fastapi import HTTPException, status
import logging
from pydantic import BaseModel, Field # Import BaseModel and Field

logger = logging.getLogger(__name__)

class JWTManager:
    """JWT token management for OAuth authentication"""
    
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET_KEY", self._generate_secret_key())
        self.algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        self.expiration_hours = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
        
        if not os.getenv("JWT_SECRET_KEY"):
            logger.warning("JWT_SECRET_KEY not set. Using generated key. This will invalidate tokens on restart.")
    
    def _generate_secret_key(self) -> str:
        """Generate a random secret key"""
        return secrets.token_urlsafe(32)
    
    def create_access_token(self, user_data: Dict[str, Any]) -> str:
        """Create JWT access token for user"""
        now = datetime.now(timezone.utc)
        expire = now + timedelta(hours=self.expiration_hours)
        
        payload = {
            "sub": user_data.get("uid"),  # Subject (Firebase UID)
            "email": user_data.get("email"),
            "name": user_data.get("name"),
            "user_photo": user_data.get("user_photo"),
            "exp": expire,
            "iat": now,
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def get_user_from_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Decode and validate access token and return user data"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return {
                "uid": payload.get("sub"),
                "email": payload.get("email"),
                "name": payload.get("name"),
                "user_photo": payload.get("user_photo"),
            }
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
            return None
        
    def create_state_token(self, provider: str) -> str:
        """Create state token for OAuth flow"""
        now = datetime.now(timezone.utc)
        expire = now + timedelta(minutes=15) # State token expires quickly
        payload = {
            "type": "state_token",
            "provider": provider,
            "exp": expire,
            "iat": now,
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_state_token(self, token: str, provider: str) -> bool:
        """Verify state token for OAuth flow"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            return (
                payload.get("type") == "state_token" and
                payload.get("provider") == provider
            )
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
            return False


# Global JWT manager instance
jwt_manager = JWTManager()


def get_jwt_manager() -> JWTManager:
    """Get the global JWT manager instance"""
    return jwt_manager


class TokenResponse(BaseModel): # Inherit from BaseModel
    """Token response model"""
    access_token: str
    token_type: str = "bearer"
    refresh_token: Optional[str] = None
    expires_in: int = Field(default=3600, description="Expires in seconds") # 24 hours in seconds
    user_info: Optional[Dict[str, Any]] = None # Added user_info to the response model
    firebase_custom_token: Optional[str] = None # Added a field for the Firebase custom token

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "access_token": self.access_token,
            "token_type": self.token_type,
            "expires_in": self.expires_in
        }
        
        if self.refresh_token:
            result["refresh_token"] = self.refresh_token
        if self.user_info:
            result["user_info"] = self.user_info
        if self.firebase_custom_token:
            result["firebase_custom_token"] = self.firebase_custom_token
        return result
