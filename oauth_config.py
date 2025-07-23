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
        self.google_client_id = os.getenv("GOOGLE_CLIENT_ID")
        self.google_client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        self.facebook_client_id = os.getenv("FACEBOOK_CLIENT_ID")
        self.facebook_client_secret = os.getenv("FACEBOOK_CLIENT_SECRET")
        
        # Redirect URIs
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
        if not self.facebook_client_id:
            missing_vars.append("FACEBOOK_CLIENT_ID")
        if not self.facebook_client_secret:
            missing_vars.append("FACEBOOK_CLIENT_SECRET")
        
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
        auth_url, _ = oauth_session.authorization_url(
            self.google_auth_url,
            access_type="offline",
            include_granted_scopes="true"
        )
        return auth_url
    
    def get_facebook_auth_url(self, state: str) -> str:
        """Generate Facebook OAuth authorization URL"""
        oauth_session = self.create_facebook_oauth_session(state)
        auth_url, _ = oauth_session.authorization_url(self.facebook_auth_url)
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
    
    def get_google_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get Google user information using access token"""
        oauth_session = OAuth2Session(token={"access_token": access_token})
        response = oauth_session.get(self.google_userinfo_url)
        response.raise_for_status()
        return response.json()
    
    def get_facebook_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get Facebook user information using access token"""
        oauth_session = OAuth2Session(token={"access_token": access_token})
        response = oauth_session.get(
            self.facebook_userinfo_url,
            params={"fields": "id,name,email,picture"}
        )
        response.raise_for_status()
        return response.json()
    
    @property
    def is_google_configured(self) -> bool:
        """Check if Google OAuth is properly configured"""
        return bool(self.google_client_id and self.google_client_secret)
    
    @property
    def is_facebook_configured(self) -> bool:
        """Check if Facebook OAuth is properly configured"""
        return bool(self.facebook_client_id and self.facebook_client_secret)
    
    @property
    def is_configured(self) -> bool:
        """Check if at least one OAuth provider is configured"""
        return self.is_google_configured or self.is_facebook_configured


# Global OAuth configuration instance
oauth_config = OAuthConfig()


class OAuthProvider:
    """OAuth provider enumeration"""
    GOOGLE = "google"
    FACEBOOK = "facebook"


class OAuthUser:
    """OAuth user data model"""
    
    def __init__(self, provider: str, provider_id: str, email: str, name: str, picture: str = None):
        self.provider = provider
        self.provider_id = provider_id
        self.email = email
        self.name = name
        self.picture = picture
        self.id = f"{provider}_{provider_id}"  # Unique user ID
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "provider": self.provider,
            "provider_id": self.provider_id,
            "email": self.email,
            "name": self.name,
            "picture": self.picture
        }
    
    @classmethod
    def from_google_data(cls, user_data: Dict[str, Any]) -> "OAuthUser":
        """Create OAuthUser from Google user data"""
        return cls(
            provider=OAuthProvider.GOOGLE,
            provider_id=user_data.get("id"),
            email=user_data.get("email"),
            name=user_data.get("name"),
            picture=user_data.get("picture")
        )
    
    @classmethod
    def from_facebook_data(cls, user_data: Dict[str, Any]) -> "OAuthUser":
        """Create OAuthUser from Facebook user data"""
        picture_url = None
        if "picture" in user_data and "data" in user_data["picture"]:
            picture_url = user_data["picture"]["data"].get("url")
        
        return cls(
            provider=OAuthProvider.FACEBOOK,
            provider_id=user_data.get("id"),
            email=user_data.get("email"),
            name=user_data.get("name"),
            picture=picture_url
        )


def get_oauth_config() -> OAuthConfig:
    """Get the global OAuth configuration instance"""
    return oauth_config

