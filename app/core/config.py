import base64
import hashlib
import logging
from typing import Optional

from cryptography.fernet import Fernet
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
  PROJECT_NAME: str = "MzaziCare API"
  VERSION: str = "1.0.0"
  API_V1_STR: str = "/api/v1"

  # Database & Security configurations
  DATABASE_URL: str = (
      "postgresql+asyncpg://postgres:postgres@localhost:5432/mzazicare_db"
  )
  SECRET_KEY: str = "mzazicare_development_secret_key_change_in_production_2026"
  PHI_ENCRYPTION_KEY: Optional[str] = None
  ALGORITHM: str = "HS256"
  ACCESS_TOKEN_EXPIRE_MINUTES: int = 120
  REFRESH_TOKEN_EXPIRE_DAYS: int = 30
  GOOGLE_CLIENT_IDS: str = ""
  GOOGLE_CLIENT_ID: str = ""
  ADMIN_EMAIL: Optional[str] = None

  @model_validator(mode="after")
  def configure_phi_encryption_key(self):
      if not self.PHI_ENCRYPTION_KEY:
          digest = hashlib.sha256(
              b"mzazicare-phi-encryption-v1\0" + self.SECRET_KEY.encode("utf-8")
          ).digest()
          self.PHI_ENCRYPTION_KEY = base64.urlsafe_b64encode(digest).decode("ascii")
          logging.getLogger(__name__).warning(
              "PHI_ENCRYPTION_KEY is unset; deriving it from SECRET_KEY. "
              "Set a dedicated persistent Fernet key before production use."
          )
      # Reject invalid explicit keys at startup rather than silently losing access to data.
      Fernet(self.PHI_ENCRYPTION_KEY.encode("ascii"))
      return self

  model_config = SettingsConfigDict(
      env_file=".env",
      env_file_encoding="utf-8",
      case_sensitive=False,
      extra="ignore",
  )


settings = Settings()
