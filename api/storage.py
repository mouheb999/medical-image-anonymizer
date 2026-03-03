"""
storage.py
----------
MinIO S3 storage client for saving anonymized medical images.
"""

import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class MinIOStorage:
    """MinIO S3 client wrapper for storing anonymized images.
    
    Parameters
    ----------
    endpoint:
        MinIO server endpoint (e.g. "localhost:9000")
    access_key:
        MinIO access key
    secret_key:
        MinIO secret key
    bucket_name:
        Target bucket name (default: "anonymized-images")
    secure:
        Use HTTPS (default: False for local MinIO)
    """

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket_name: str = "anonymized-images",
        secure: bool = False
    ) -> None:
        try:
            from minio import Minio
        except ImportError:
            raise ImportError(
                "minio is required. Run: pip install minio"
            )

        self.bucket_name = bucket_name
        self.endpoint = endpoint

        self._client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )

        # Create bucket if it doesn't exist
        self._ensure_bucket()
        logger.info(f"MinIO client initialized: {endpoint}/{bucket_name}")

    def _ensure_bucket(self) -> None:
        """Create bucket if it does not exist."""
        if not self._client.bucket_exists(self.bucket_name):
            self._client.make_bucket(self.bucket_name)
            logger.info(f"Created bucket: {self.bucket_name}")

    def upload_file(
        self,
        file_path: str,
        object_name: Optional[str] = None,
        category: Optional[str] = None
    ) -> str:
        """Upload a file to MinIO and return its URI.
        
        Parameters
        ----------
        file_path:
            Local path to the file to upload
        object_name:
            Name to store in MinIO. If None, uses:
            category/YYYY/MM/DD/filename format for organization
        category:
            Classification category (e.g., "chest-xray", "dental", "non-medical")
            Used to organize files by type
            
        Returns
        -------
        str
            Full MinIO URI in format:
            minio://{endpoint}/{bucket}/{object_name}
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if object_name is None:
            # Organize by category only: chest/anonymized_image.jpg
            
            # Sanitize category for use as directory name
            CATEGORY_MAP = {
                "chest": "chest",
                "dental": "dental", 
                "pelvic": "pelvic",
                "skull": "skull",
                "non_medical": "non_medical",
                "non-medical": "non_medical",
                "other_medical": "other_medical",
                "other": "other_medical",
            }
            
            if category:
                category_lower = category.lower()
                safe_category = "other_medical"  # default
                
                for key, folder in CATEGORY_MAP.items():
                    if key in category_lower:
                        safe_category = folder
                        break
            else:
                safe_category = "uncategorized"
            
            object_name = f"{safe_category}/{file_path.name}"

        self._client.fput_object(
            self.bucket_name,
            object_name,
            str(file_path)
        )

        uri = f"minio://{self.endpoint}/{self.bucket_name}/{object_name}"
        logger.info(f"Uploaded: {file_path.name} → {uri}")
        return uri

    def get_url(self, object_name: str, expires_hours: int = 24) -> str:
        """Generate a presigned URL for downloading an object.
        
        Parameters
        ----------
        object_name:
            Object path in the bucket
        expires_hours:
            URL expiry in hours (default: 24)
            
        Returns
        -------
        str
            Presigned HTTP URL valid for expires_hours
        """
        from datetime import timedelta
        url = self._client.presigned_get_object(
            self.bucket_name,
            object_name,
            expires=timedelta(hours=expires_hours)
        )
        return url
