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
  ALGORITHM: str = "HS256"
  ACCESS_TOKEN_EXPIRE_MINUTES: int = 120

  model_config = SettingsConfigDict(
      env_file=".env",
      env_file_encoding="utf-8",
      case_sensitive=False,
      extra="ignore",
  )


settings = Settings()