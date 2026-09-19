from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MAT_", extra="ignore")

    model_backend: Literal["baseline", "transformer", "auto"] = "baseline"
    baseline_artifact_dir: Path = Path("artifacts/baseline")
    transformer_artifact_dir: Path = Path("artifacts/transformer/champion")
    normalization_catalog_path: Path = Path("configs/normalization.yaml")
    normalization_lock_path: Path = Path("configs/normalization.lock.json")
    max_request_bytes: int = Field(default=65_536, ge=1)
    # Comma-separated allow-list for the browser demo UI (frontend :3000);
    # origins outside the list get no CORS headers. Empty string disables CORS.
    cors_allow_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    uncertain_threshold: float = Field(default=0.60, gt=0.0, lt=1.0)
    allow_backend_override: bool = False
    service_name: str = "mat"
    environment: Literal["dev", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
