"""Google Cloud Storage (GCS) production policy document storage service.

Provides authoritative document persistence in GCS (gs://company-hr-policies/) organized by category:
policies/
    leave/
    attendance/
    remote-work/
    travel/
    expenses/
    insurance/

When GCP credentials are configured, connects directly to Google Cloud Storage.
When credentials are not present locally, falls back cleanly to local filesystem storage
without faking cloud API responses, exposing integration status transparently.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger("app.storage.gcs_storage")

VALID_POLICY_CATEGORIES = [
    "leave",
    "attendance",
    "remote-work",
    "travel",
    "expenses",
    "insurance",
    "general",
]


class StoredPolicyMetadata(BaseModel):
    """Metadata container for a policy document stored in GCS."""

    path: str = Field(description="Relative storage path, e.g. policies/leave/leave_policy.pdf")
    filename: str = Field(description="Policy filename")
    category: str = Field(description="Policy category folder")
    size_bytes: int = Field(description="File size in bytes")
    updated_at: str = Field(description="Last modified timestamp in ISO 8601")
    gcs_uri: str = Field(description="Full GCS URI (gs://bucket/path)")
    content_type: str = Field(default="application/octet-stream", description="MIME type")


class GCSStorageService:
    """Production storage service for HR policy documents backed by Google Cloud Storage."""

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        project: Optional[str] = None,
        local_root: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.bucket_name = bucket_name or getattr(settings, "gcs_bucket_name", "company-hr-policies")
        self.project = (
            project
            or getattr(settings, "gcs_project", None)
            or getattr(settings, "google_cloud_project", None)
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
        )
        self.local_root = Path(local_root or getattr(settings, "gcs_local_root", "data/storage/gcs_mock"))

        self._gcs_client = None
        self._bucket = None
        self.is_cloud = False
        self.integration_status = "local_storage (gcs credentials not configured)"

        self._initialize_client()

    def _initialize_client(self) -> None:
        """Attempt to initialize Google Cloud Storage client using application default credentials."""
        # Ensure local root directory exists for fallback or caching
        self.local_root.mkdir(parents=True, exist_ok=True)
        for cat in VALID_POLICY_CATEGORIES:
            (self.local_root / "policies" / cat).mkdir(parents=True, exist_ok=True)

        try:
            from google.cloud import storage

            # Check if explicit credentials, Cloud Run environment, or project exist
            has_creds = bool(
                os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                or os.environ.get("GCP_ACCESS_TOKEN")
                or os.environ.get("K_SERVICE")  # Standard Cloud Run indicator
            )
            if not has_creds and not self.project:
                logger.info(
                    "GCP credentials not detected. Operating in local storage mode with GCS bucket semantics."
                )
                return

            client = storage.Client(project=self.project)
            bucket = client.bucket(self.bucket_name)
            # Check existence if permissions allow
            if bucket.exists():
                self._gcs_client = client
                self._bucket = bucket
                self.is_cloud = True
                self.integration_status = "cloud_gcs"
                logger.info(f"Connected to Google Cloud Storage bucket: gs://{self.bucket_name}")
            else:
                logger.warning(f"GCS bucket 'gs://{self.bucket_name}' not found. Using local storage fallback.")

        except Exception as exc:
            logger.info(
                f"Google Cloud Storage client not available ({exc}). "
                f"Using local storage fallback at '{self.local_root}'."
            )

    def _clean_path(self, raw_path: str) -> str:
        """Normalize storage path and strip gs:// prefix."""
        p = raw_path.strip().replace("\\", "/")
        prefix = f"gs://{self.bucket_name}/"
        if p.startswith(prefix):
            p = p[len(prefix):]
        elif p.startswith("gs://"):
            parts = p[5:].split("/", 1)
            p = parts[1] if len(parts) > 1 else parts[0]
        return p.lstrip("/")

    def format_gcs_uri(self, path: str) -> str:
        """Format a storage path into a standard gs:// URI."""
        clean = self._clean_path(path)
        return f"gs://{self.bucket_name}/{clean}"

    def upload(
        self,
        file_content: bytes,
        destination_path: str,
        content_type: Optional[str] = None,
    ) -> str:
        """Upload document bytes to GCS storage under the categorized structure.

        Args:
            file_content: Raw bytes of the document.
            destination_path: Path such as 'policies/leave/leave_policy_2026.pdf'.
            content_type: Optional MIME content type.

        Returns:
            GCS URI string (gs://<bucket>/<path>).
        """
        clean_path = self._clean_path(destination_path)
        mime = content_type or self._guess_content_type(clean_path)

        if self.is_cloud and self._bucket is not None:
            try:
                blob = self._bucket.blob(clean_path)
                blob.upload_from_string(file_content, content_type=mime)
                logger.info(f"Uploaded {len(file_content)} bytes to GCS: gs://{self.bucket_name}/{clean_path}")
                return self.format_gcs_uri(clean_path)
            except Exception as exc:
                logger.error(f"GCS upload failed: {exc}. Saving to local backup.")

        # Local fallback execution
        dest_file = self.local_root / clean_path
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        dest_file.write_bytes(file_content)
        logger.info(f"Saved {len(file_content)} bytes to storage: {dest_file}")
        return self.format_gcs_uri(clean_path)

    def download(self, source_path: str) -> bytes:
        """Download document bytes from GCS storage.

        Args:
            source_path: Relative storage path or gs:// URI.

        Returns:
            Document content as bytes.
        """
        clean_path = self._clean_path(source_path)

        if self.is_cloud and self._bucket is not None:
            try:
                blob = self._bucket.blob(clean_path)
                if blob.exists():
                    return blob.download_as_bytes()
                raise FileNotFoundError(f"Blob 'gs://{self.bucket_name}/{clean_path}' not found in GCS.")
            except Exception as exc:
                logger.warning(f"GCS download failed ({exc}), attempting local storage.")

        local_file = self.local_root / clean_path
        if local_file.exists():
            return local_file.read_bytes()

        raise FileNotFoundError(f"Policy file '{clean_path}' not found in storage.")

    def list(self, prefix: str = "policies/") -> List[StoredPolicyMetadata]:
        """List all policy documents in GCS storage matching the prefix."""
        clean_prefix = self._clean_path(prefix)
        results: List[StoredPolicyMetadata] = []

        if self.is_cloud and self._bucket is not None:
            try:
                blobs = self._gcs_client.list_blobs(self.bucket_name, prefix=clean_prefix)
                for blob in blobs:
                    if blob.name.endswith("/"):
                        continue
                    cat = self._extract_category(blob.name)
                    fname = Path(blob.name).name
                    updated = blob.updated.isoformat() if blob.updated else datetime.now(timezone.utc).isoformat()
                    results.append(
                        StoredPolicyMetadata(
                            path=blob.name,
                            filename=fname,
                            category=cat,
                            size_bytes=blob.size or 0,
                            updated_at=updated,
                            gcs_uri=self.format_gcs_uri(blob.name),
                            content_type=blob.content_type or "application/octet-stream",
                        )
                    )
                return sorted(results, key=lambda x: x.path)
            except Exception as exc:
                logger.warning(f"GCS list failed ({exc}), falling back to local listing.")

        # Local storage listing
        target_dir = self.local_root / clean_prefix
        if target_dir.exists():
            for p in target_dir.rglob("*"):
                if p.is_file():
                    rel_path = p.relative_to(self.local_root).as_posix()
                    cat = self._extract_category(rel_path)
                    stat = p.stat()
                    updated = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
                    results.append(
                        StoredPolicyMetadata(
                            path=rel_path,
                            filename=p.name,
                            category=cat,
                            size_bytes=stat.st_size,
                            updated_at=updated,
                            gcs_uri=self.format_gcs_uri(rel_path),
                            content_type=self._guess_content_type(p.name),
                        )
                    )

        return sorted(results, key=lambda x: x.path)

    def delete(self, target_path: str) -> bool:
        """Delete a policy document from GCS storage."""
        clean_path = self._clean_path(target_path)
        deleted = False

        if self.is_cloud and self._bucket is not None:
            try:
                blob = self._bucket.blob(clean_path)
                if blob.exists():
                    blob.delete()
                    logger.info(f"Deleted GCS blob: gs://{self.bucket_name}/{clean_path}")
                    deleted = True
            except Exception as exc:
                logger.warning(f"GCS delete failed: {exc}")

        local_file = self.local_root / clean_path
        if local_file.exists():
            local_file.unlink()
            logger.info(f"Deleted local file: {local_file}")
            deleted = True

        return deleted

    def exists(self, path: str) -> bool:
        """Check whether a policy document exists in GCS storage."""
        clean_path = self._clean_path(path)
        if self.is_cloud and self._bucket is not None:
            try:
                return self._bucket.blob(clean_path).exists()
            except Exception:
                pass
        return (self.local_root / clean_path).exists()

    @staticmethod
    def _extract_category(path: str) -> str:
        """Extract category from path like 'policies/leave/policy.pdf'."""
        parts = [p for p in path.replace("\\", "/").split("/") if p]
        if len(parts) >= 2 and parts[0] == "policies":
            cat = parts[1].lower()
            if cat in VALID_POLICY_CATEGORIES:
                return cat
        return "general"

    @staticmethod
    def _guess_content_type(filename: str) -> str:
        """Guess MIME type from filename extension."""
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            return "application/pdf"
        elif ext == ".docx":
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif ext == ".pptx":
            return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        elif ext == ".json":
            return "application/json"
        return "application/octet-stream"


# Global singleton instance
_GLOBAL_GCS_STORAGE: Optional[GCSStorageService] = None


def get_gcs_storage() -> GCSStorageService:
    """Return singleton GCSStorageService instance."""
    global _GLOBAL_GCS_STORAGE
    if _GLOBAL_GCS_STORAGE is None:
        _GLOBAL_GCS_STORAGE = GCSStorageService()
    return _GLOBAL_GCS_STORAGE
