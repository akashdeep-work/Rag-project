from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from db import get_db
from models.app_models import User

import hashlib
import hmac
import secrets
import json
import base64
import time

# ======================================================================
# CONFIG
# ======================================================================

SECRET_KEY = "CHANGE_THIS_TO_A_SUPER_SECRET_KEY"  # MUST be replaced in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

# ======================================================================
# PASSWORD HASHING (PBKDF2-HMAC-SHA256)
# ======================================================================

PBKDF2_ITERATIONS = 100_000
SALT_BYTES = 16  # 128-bit salt


def hash_password(password: str) -> str:
    """
    Hash password using PBKDF2-HMAC-SHA256.
    Format:
        pbkdf2_sha256$iterations$salthex$hashhex
    """
    salt = secrets.token_bytes(SALT_BYTES)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)

    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(plain_password: str, stored_hash: str) -> bool:
    """
    Verify a password using constant-time comparison.
    """
    try:
        scheme, iterations_s, salt_hex, dk_hex = stored_hash.split("$")
        assert scheme == "pbkdf2_sha256"
    except Exception:
        return False

    iterations = int(iterations_s)
    salt = bytes.fromhex(salt_hex)
    expected_dk = bytes.fromhex(dk_hex)

    new_dk = hashlib.pbkdf2_hmac("sha256", plain_password.encode(), salt, iterations)
    return hmac.compare_digest(new_dk, expected_dk)


# ======================================================================
# JWT IMPLEMENTATION (HS256) — NO THIRD-PARTY LIBRARIES
# ======================================================================

def _b64url_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _sign_hs256(message: bytes, secret: str) -> bytes:
    return hmac.new(secret.encode(), message, hashlib.sha256).digest()


def create_jwt(payload: Dict[str, Any], secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}

    header_b = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    payload_b = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()

    h_b64 = _b64url_encode(header_b)
    p_b64 = _b64url_encode(payload_b)

    signing_input = f"{h_b64}.{p_b64}".encode()
    sig = _sign_hs256(signing_input, secret)
    s_b64 = _b64url_encode(sig)

    return f"{h_b64}.{p_b64}.{s_b64}"


def decode_jwt(token: str, secret: str) -> Dict[str, Any]:
    try:
        h_b64, p_b64, s_b64 = token.split(".")
    except Exception:
        raise ValueError("Malformed token")

    signing_input = f"{h_b64}.{p_b64}".encode()
    signature = _b64url_decode(s_b64)
    expected = _sign_hs256(signing_input, secret)

    if not hmac.compare_digest(signature, expected):
        raise ValueError("Invalid signature")

    payload_json = _b64url_decode(p_b64)
    payload = json.loads(payload_json)

    # Validate exp
    if "exp" in payload and time.time() > payload["exp"]:
        raise ValueError("Token expired")

    return payload


# ======================================================================
# ACCESS TOKEN CREATION
# ======================================================================

def create_access_token(data: dict):
    to_encode = data.copy()
    expire_ts = int(time.time()) + (ACCESS_TOKEN_EXPIRE_MINUTES * 60)
    to_encode["exp"] = expire_ts
    return create_jwt(to_encode, SECRET_KEY)


# ======================================================================
# FASTAPI DEPENDENCY — GET CURRENT USER
# ======================================================================

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_jwt(token, SECRET_KEY)
    except Exception:
        raise credentials_exception

    username: str = payload.get("sub")
    if username is None:
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise credentials_exception

    return user
