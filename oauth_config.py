"""
OAuth 2.0 Configuration Module
Handles Google and Facebook OAuth configuration and client setup
"""

import os
from typing import Optional, Dict, Any
from authlib.integrations.requests_client import OAuth2Session
import secrets
import logging

logger = logging.getLogger(__name__)

class OAuthConfig:
    """OAuth configuration and client management"""
    
    def __init__(self):
        # Google OAuth configuration
        self.google_client_id = os.getenv("GOOGLE_CLIENT_ID")
        self.google_client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        self.facebook_client_id = os.getenv("FACEBOOK_CLIENT_ID")
        self.facebook_client_secret = os.getenv("FACEBOOK_CLIENT_SECRET")
        
        # Redirect URIs - Now fetched from environment variables
        self.google_redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")
        self.facebook_redirect_uri = os.getenv("FACEBOOK_REDIRECT_URI", "http://localhost:8000/auth/facebook/callback")
        
        # OAuth endpoints
        self.google_auth_url = "https://accounts.google.com/o/oauth2/auth"
        self.google_token_url = "https://oauth2.googleapis.com/token"
        self.google_userinfo_url = "https://www.googleapis.com/oauth2/v2/userinfo"
        
        self.facebook_auth_url = "https://www.facebook.com/v18.0/dialog/oauth"
        self.facebook_token_url = "https://graph.facebook.com/v18.0/oauth/access_token"
        self.facebook_userinfo_url = "https://graph.facebook.com/v18.0/me"
        
        # OAuth scopes
        self.google_scopes = ["openid", "email", "profile"]
        self.facebook_scopes = ["email", "public_profile"]
        
        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate OAuth configuration"""
        missing_vars = []
        
        if not self.google_client_id:
            missing_vars.append("GOOGLE_CLIENT_ID")
        if not self.google_client_secret:
            missing_vars.append("GOOGLE_CLIENT_SECRET")
        if not self.google_redirect_uri:  # Check redirect URI too
            missing_vars.append("GOOGLE_REDIRECT_URI")
            
        if not self.facebook_client_id:
            missing_vars.append("FACEBOOK_CLIENT_ID")
        if not self.facebook_client_secret:
            missing_vars.append("FACEBOOK_CLIENT_SECRET")
        if not self.facebook_redirect_uri:  # Check redirect URI too
            missing_vars.append("FACEBOOK_REDIRECT_URI")
            
        if missing_vars:
            logger.warning(f"Missing OAuth environment variables: {', '.join(missing_vars)}")
            logger.warning("OAuth functionality will be limited. Please set the missing variables.")

    def create_google_oauth_session(self, state: Optional[str] = None) -> OAuth2Session:
        """Create Google OAuth session"""
        if not state:
            state = secrets.token_urlsafe(32)
            
        return OAuth2Session(
            client_id=self.google_client_id,
            client_secret=self.google_client_secret,
            redirect_uri=self.google_redirect_uri,
            scope=self.google_scopes,
            state=state
        )

    def create_facebook_oauth_session(self, state: Optional[str] = None) -> OAuth2Session:
        """Create Facebook OAuth session"""
        if not state:
            state = secrets.token_urlsafe(32)
            
        return OAuth2Session(
            client_id=self.facebook_client_id,
            client_secret=self.facebook_client_secret,
            redirect_uri=self.facebook_redirect_uri,
            scope=self.facebook_scopes,
            state=state
        )

    def get_google_auth_url(self, state: str) -> str:
        """Generate Google OAuth authorization URL"""
        oauth_session = self.create_google_oauth_session(state)
        # ✅ FIXED: Use create_authorization_url instead of authorization_url
        auth_url = oauth_session.create_authorization_url(
            self.google_auth_url,
            access_type="offline",
            include_granted_scopes="true"
        )
        return auth_url

    def get_facebook_auth_url(self, state: str) -> str:
        """Generate Facebook OAuth authorization URL"""
        oauth_session = self.create_facebook_oauth_session(state)
        # ✅ FIXED: Use create_authorization_url instead of authorization_url
        auth_url = oauth_session.create_authorization_url(self.facebook_auth_url)
        return auth_url

    def exchange_google_code_for_token(self, code: str, state: str) -> Dict[str, Any]:
        """Exchange Google authorization code for access token"""
        oauth_session = self.create_google_oauth_session(state)
        token = oauth_session.fetch_token(
            self.google_token_url,
            code=code,
            client_secret=self.google_client_secret
        )
        return token

    def exchange_facebook_code_for_token(self, code: str, state: str) -> Dict[str, Any]:
        """Exchange Facebook authorization code for access token"""
        oauth_session = self.create_facebook_oauth_session(state)
        token = oauth_session.fetch_token(
            self.facebook_token_url,
            code=code,
            client_secret=self.facebook_client_secret
        )
        return token

    @property
    def is_google_configured(self) -> bool:
        """Check if Google OAuth is properly configured"""
        return bool(
            self.google_client_id and 
            self.google_client_secret and 
            self.google_redirect_uri
        )

    @property
    def is_facebook_configured(self) -> bool:
        """Check if Facebook OAuth is properly configured"""
        return bool(
            self.facebook_client_id and 
            self.facebook_client_secret and 
            self.facebook_redirect_uri
        )

# Global configuration instance
_oauth_config = None

def get_oauth_config() -> OAuthConfig:
    """Get the global OAuth configuration instance"""
    global _oauth_config
    if _oauth_config is None:
        _oauth_config = OAuthConfig()
    return _oauth_config

# OAuth Provider Enum
from enum import Enum

class OAuthProvider(Enum):
    GOOGLE = "google"
    FACEBOOK = "facebook"

# OAuth User Model
from pydantic import BaseModel

class OAuthUser(BaseModel):
    id: str
    email: str
    name: str
    picture: Optional[str] = None
    provider: str
    provider_id: str

