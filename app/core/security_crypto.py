"""Fernet encryption for fields persisted by SQLAlchemy."""

from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import settings


# Fernet instances are thread-safe; sharing one also avoids rebuilding it per row.
_fernet = Fernet(settings.PHI_ENCRYPTION_KEY.encode("ascii"))


def encrypt_field(plaintext: Optional[str]) -> Optional[str]:
    if plaintext is None:
        return None
    return _fernet.encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt_field(ciphertext: Optional[str]) -> Optional[str]:
    if ciphertext is None:
        return None
    if not ciphertext.startswith("gAAAA"):
        # Existing rows may still be plaintext until the backfill migration runs.
        return ciphertext
    try:
        return _fernet.decrypt(ciphertext.encode("ascii")).decode("utf-8")
    except InvalidToken:
        # A token that looks encrypted must never be returned as plaintext.
        raise
