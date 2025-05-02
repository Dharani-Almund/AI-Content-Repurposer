import jwt
import os
import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get JWT secret from environment variables or use a default for development
JWT_SECRET = os.getenv("JWT_SECRET", "development_secret_key_change_this_in_production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
TOKEN_EXPIRY = int(os.getenv("TOKEN_EXPIRY", "3600"))  # 1 hour default

def generate_admin_token(username):
    """Generate a JWT token for admin users"""
    payload = {
        "sub": username,
        "role": "admin",
        "exp": datetime.datetime.utcnow() + datetime.timedelta(seconds=TOKEN_EXPIRY)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def generate_user_token(username):
    """Generate a JWT token for regular users"""
    payload = {
        "sub": username,
        "role": "user",
        "exp": datetime.datetime.utcnow() + datetime.timedelta(seconds=TOKEN_EXPIRY)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def validate_token(token):
    """Validate a JWT token and return the payload"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None  # Token has expired
    except jwt.InvalidTokenError:
        return None  # Invalid token

def is_token_valid(token):
    """Check if a token is valid"""
    return validate_token(token) is not None

def is_admin(token):
    """Check if a token belongs to an admin user"""
    payload = validate_token(token)
    if payload:
        return payload.get("role") == "admin"
    return False

# Generate test tokens for development - only runs when directly executed
if __name__ == "__main__":
    admin_token = generate_admin_token("admin")
    user_token = generate_user_token("user")

    print(f"Admin token: {admin_token}")
    print(f"User token: {user_token}")

