# Authentication Configuration
# Simple username/password authentication for isolated network

# Default credentials (change these for production)
DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "ecg2024"  # Change this to a strong password

# Security settings
SECRET_KEY = "your-secret-key-change-this-in-production"  # Change this to a random string
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Session settings
SESSION_COOKIE_NAME = "ecg_session"
SESSION_COOKIE_SECURE = False  # Set to True if using HTTPS
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "lax" 