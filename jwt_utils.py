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
            "user_type": user_data.get("user_type"), # Include user_type
            "picture": user_data.get("picture"),
            "provider": user_data.get("provider"), # For OAuth users
            "provider_id": user_data.get("provider_id"), # For OAuth users
            "iat": now,  # Issued at
            "exp": expire,  # Expiration
            "type": "access_token"
        }
        
        try:
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            logger.info(f"Created access token for user {user_data.get('uid')}")
            return token
        except Exception as e:
            logger.error(f"Failed to create access token: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create access token"
            )
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check if token is expired
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp, timezone.utc) < datetime.now(timezone.utc):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            
            # Check token type
            if payload.get("type") != "access_token":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            
            return payload
        
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"}
            )
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"}
            )
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token verification failed",
                headers={"WWW-Authenticate": "Bearer"}
            )
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create refresh token for long-term authentication"""
        now = datetime.now(timezone.utc)
        expire = now + timedelta(days=30)  # Refresh tokens last longer
        
        payload = {
            "sub": user_id, # Firebase UID
            "iat": now,
            "exp": expire,
            "type": "refresh_token"
        }
        
        try:
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            logger.info(f"Created refresh token for user {user_id}")
            return token
        except Exception as e:
            logger.error(f"Failed to create refresh token: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create refresh token"
            )
    
    def verify_refresh_token(self, token: str) -> str:
        """Verify refresh token and return user ID"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check token type
            if payload.get("type") != "refresh_token":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            
            return payload.get("sub") # Returns Firebase UID
        
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token has expired",
                headers={"WWW-Authenticate": "Bearer"}
            )
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid refresh token: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
                headers={"WWW-Authenticate": "Bearer"}
            )
    
    def get_user_from_token(self, token: str) -> Dict[str, Any]:
        """Extract user information from token"""
        payload = self.verify_token(token)
        # Map 'sub' to 'id' for consistency with frontend user_info
        user_data = {
            "id": payload.get("sub"), # This is the Firebase UID
            "uid": payload.get("sub"), # Keep uid for backend use
            "email": payload.get("email"),
            "name": payload.get("name"),
            "user_type": payload.get("user_type"),
            "picture": payload.get("picture"),
            "provider": payload.get("provider"),
            "provider_id": payload.get("provider_id")
        }
        return user_data
    
    def create_state_token(self, provider: str) -> str:
        """Create state token for OAuth flow security"""
        now = datetime.now(timezone.utc)
        expire = now + timedelta(minutes=10)  # Short-lived state token
        
        payload = {
            "provider": provider,
            "iat": now,
            "exp": expire,
            "type": "state_token",
            "nonce": secrets.token_urlsafe(16)
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
        
        return result

