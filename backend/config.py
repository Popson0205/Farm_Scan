"""FarmScan's settings, from the environment (or backend/.env)."""
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://hub:hub@localhost:5432/hub"
    # FarmScan keeps its tables in this schema of the platform database (Geohub's); PostGIS stays where it is.
    db_schema: str = "farm"
    db_pool_max: int = 5
    app_url: str = "http://localhost:8000"     # where people open FarmScan

    # Signing in: GeoEstate ID, through Geohub (Geohub's ID_CLIENTS must list "farm" with the same secret)
    geohub_url: str = "http://localhost:5173"  # Geohub's address (the issuer)
    geohub_api: str = "http://localhost:8000"  # Geohub's API (…/api in production)
    id_client: str = "farm"
    id_secret: str = ""
    cookie_secure: bool | None = None          # default: secure when APP_URL is https

    # Files (the soil dataset): the platform's S3-compatible bucket, under S3_PREFIX.
    # Left empty (development, tests): files stay in backend/uploads.
    s3_endpoint: str = ""
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_bucket: str = ""
    s3_region: str = ""
    s3_prefix: str = "farm/"

    # Practice with synthetic sample data (sales demonstrations, tests). Off for customers: sample results must
    # never be mistaken for their farm.
    demo_tools: bool = False

    @model_validator(mode="after")
    def _checks(self):
        self.app_url = self.app_url.strip().rstrip("/")
        self.geohub_url = self.geohub_url.strip().rstrip("/")
        self.geohub_api = self.geohub_api.strip().rstrip("/")
        if not self.db_schema.isidentifier() or self.db_schema.lower() == "public":
            raise ValueError("DB_SCHEMA must be a plain name other than public, e.g. farm")
        if self.cookie_secure is None:
            self.cookie_secure = self.app_url.startswith("https://")
        self.s3_prefix = self.s3_prefix.strip("/") + "/" if self.s3_prefix.strip("/") else ""
        return self

    @property
    def s3_on(self) -> bool:
        return bool(self.s3_endpoint and self.s3_bucket and self.s3_access_key)


settings = Settings()
