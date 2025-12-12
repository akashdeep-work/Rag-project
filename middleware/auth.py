from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status, Header
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
import uuid

# ======================================================================
# CONFIG
# ======================================================================

SECRET_KEY = "CHANGE_THIS_TO_A_SUPER_SECRET_KEY"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Allow missing tokens for guest users
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token", auto_error=False)

# ======================================================================
# PBKDF2 PASSWORD HASHING
# ======================================================================

PBKDF2_ITERATIONS = 100_000
SALT_BYTES = 16

def hash_password(password: str) -> str:
    salt = secrets.token_bytes(SALT_BYTES)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${dk.hex()}"

def verify_password(plain_password: str, stored_hash: str) -> bool:
    try:
        scheme, it_s, salt_hex, dk_hex = stored_hash.split("$")
        assert scheme == "pbkdf2_sha256"
    except Exception:
        return False

    salt = bytes.fromhex(salt_hex)
    iterations = int(it_s)
    expected_dk = bytes.fromhex(dk_hex)
    new_dk = hashlib.pbkdf2_hmac("sha256", plain_password.encode(), salt, iterations)

    return hmac.compare_digest(new_dk, expected_dk)

# ======================================================================
# JWT (HS256) WITHOUT EXTERNAL LIBRARIES
# ======================================================================

def _b64url_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()

def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)

def _sign_hs256(message: bytes, secret: str) -> bytes:
    return hmac.new(secret.encode(), message, hashlib.sha256).digest()

def create_jwt(payload: Dict[str, Any], secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    h = _b64url_encode(json.dumps(header).encode())
    p = _b64url_encode(json.dumps(payload).encode())

    signing_input = f"{h}.{p}".encode()
    s = _b64url_encode(_sign_hs256(signing_input, secret))

    return f"{h}.{p}.{s}"

def decode_jwt(token: str, secret: str) -> Dict[str, Any]:
    try:
        h, p, s = token.split(".")
    except Exception:
        raise ValueError("Malformed token")

    signing_input = f"{h}.{p}".encode()

    if not hmac.compare_digest(_b64url_decode(s), _sign_hs256(signing_input, secret)):
        raise ValueError("Invalid signature")

    payload = json.loads(_b64url_decode(p))

    if "exp" in payload and time.time() > payload["exp"]:
        raise ValueError("Token expired")

    return payload

# ======================================================================
# ACCESS TOKEN CREATION
# ======================================================================

def create_access_token(data: dict):
    to_encode = data.copy()
    to_encode["exp"] = int(time.time()) + ACCESS_TOKEN_EXPIRE_MINUTES * 60
    return create_jwt(to_encode, SECRET_KEY)

# ======================================================================
# GUEST SESSION TOKEN
# ======================================================================

def create_guest_token(guest_id: str | None = None) -> str:
    """
    Create a JWT for a guest user.
    If guest_id is provided, reuse it (session restore).
    """
    if guest_id is None:
        guest_id = f"guest_{uuid.uuid4().hex}"

    payload = {
        "sub": guest_id,
        "guest": True,
        "exp": int(time.time()) + ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }

    return create_jwt(payload, SECRET_KEY)

# ======================================================================
# OPTIONAL USER FROM TOKEN
# ======================================================================

async def get_current_user_optional(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> Optional[User | dict]:

    if not token:
        return None  # no token → guest based on "required" method

    # Decode token
    try:
        payload = decode_jwt(token, SECRET_KEY)
    except Exception:
        return None

    user_id = payload.get("sub")
    if not user_id:
        return None

    # CASE 1 — NORMAL AUTHENTICATED USER
    if not payload.get("guest"):
        return db.query(User).filter(User.username == user_id).first()

    # CASE 2 — GUEST USER
    return {
        "id": user_id,
        "username": user_id,
        "is_guest": True,
        "token": token,
    }


# ======================================================================
# REQUIRED USER (Guest fallback)
# ======================================================================

class AuthUser:
    def __init__(self, id: str, username: str, is_guest: bool, token: str):
        self.id = id
        self.username = username
        self.is_guest = is_guest
        self.token = token

async def get_current_user(
    opt_user: Optional[User | dict] = Depends(get_current_user_optional)
) -> AuthUser:

    # CASE 1 — Normal authenticated user
    if opt_user and isinstance(opt_user, User):
        return AuthUser(
            id=str(opt_user.id),
            username=opt_user.username,
            is_guest=False,
            token=create_access_token({"sub": opt_user.username})
        )

    # CASE 2 — Existing guest user (token reused)
    if opt_user and isinstance(opt_user, dict) and opt_user.get("is_guest"):
        return AuthUser(
            id=opt_user["id"],
            username=opt_user["username"],
            is_guest=True,
            token=opt_user["token"]
        )

    # CASE 3 — No token → create new guest identity
    new_guest_id = f"guest_{uuid.uuid4().hex}"
    new_token = create_guest_token(new_guest_id)

    return AuthUser(
        id=new_guest_id,
        username=new_guest_id,
        is_guest=True,
        token=new_token
    )
