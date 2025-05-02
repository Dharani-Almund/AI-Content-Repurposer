from fastapi import Depends, HTTPException, Header, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
import jwt
from datetime import datetime, timedelta
import logging
from backend.src.config import config  # Import the configuration


class AuthenticationError(HTTPException):
    """Custom exception for authentication-related errors."""

    def __init__(self, detail: str):
        super().__init__(status_code=401, detail=detail)


class AuthorizationError(HTTPException):
    """Custom exception for authorization-related errors."""

    def __init__(self, detail: str):
        super().__init__(status_code=403, detail=detail)


def create_access_token(
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a new JWT access token.

    Args:
        data (Dict[str, Any]): Payload data to encode
        expires_delta (Optional[timedelta]): Token expiration time

    Returns:
        str: Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        # Default expiration of 15 minutes
        expire = datetime.utcnow() + timedelta(minutes=15)

    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(
        to_encode,
        config.SECRET_KEY,
        algorithm=config.JWT_ALGORITHM
    )

    return encoded_jwt


def get_current_user(
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Dict[str, Any]:
    """
    Extract and validate user from JWT token.

    Args:
        request: FastAPI request object
        credentials: Authorization credentials

    Returns:
        Dict[str, Any]: Decoded user information
    """
    # Check for token in different locations
    token = None

    # Try getting token from Authorization header
    if credentials:
        token = credentials.credentials

    # Fallback to manual header check
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header:
            try:
                _, token = auth_header.split()
            except ValueError:
                raise AuthenticationError("Invalid Authorization header format")

    if not token:
        raise AuthenticationError("No authentication token provided")

    try:
        payload = jwt.decode(
            token,
            config.SECRET_KEY,
            algorithms=[config.JWT_ALGORITHM]
        )

        # Validate required claims
        required_claims = ["sub", "role"]
        if not all(claim in payload for claim in required_claims):
            raise AuthenticationError("Token is missing required claims")

        return {
            "user_id": payload.get("sub"),
            "role": payload.get("role"),
            "email": payload.get("email")
        }

    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired")
    except jwt.InvalidTokenError:
        raise AuthenticationError("Invalid token")
    except Exception as e:
        logging.error(f"Unexpected authentication error: {str(e)}")
        raise AuthenticationError("Authentication failed")


class RoleChecker:
    """
    Dependency to check user roles with granular access control.
    """

    def __init__(self, allowed_roles: list[str]):
        """
        Initialize RoleChecker with allowed roles.

        Args:
            allowed_roles (list[str]): List of roles with access
        """
        self.allowed_roles = allowed_roles

    def __call__(self, user: Dict[str, Any] = Depends(get_current_user)):
        """
        Check if user's role is in the allowed roles.

        Args:
            user (Dict[str, Any]): Authenticated user information

        Raises:
            AuthorizationError: If user role is not allowed
        """
        if user.get("role") not in self.allowed_roles:
            raise AuthorizationError(
                f"Access denied. Required roles: {', '.join(self.allowed_roles)}"
            )
        return user


# Example role-based access control
admin_required = RoleChecker(["admin"])
user_required = RoleChecker(["user", "admin"])