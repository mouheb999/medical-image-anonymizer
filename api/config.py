"""
config.py
---------
Environment-based configuration for the API.
All sensitive values loaded from environment variables.
"""

import os
from dataclasses import dataclass


@dataclass
class Settings:
    # MinIO
    minio_endpoint: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    minio_access_key: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    minio_secret_key: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    minio_bucket: str = os.getenv("MINIO_BUCKET", "anonymized-images")
    minio_secure: bool = os.getenv("MINIO_SECURE", "false").lower() == "true"

    # Pipeline
    output_dir: str = os.getenv("OUTPUT_DIR", "./output")
    temp_dir: str = os.getenv("TEMP_DIR", "./api/temp")


settings = Settings()
