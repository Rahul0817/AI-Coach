"""Media storage: Cloudinary in production, local disk otherwise.

Uploads are health data — lab reports especially — so two rules apply
regardless of backend:

* **Never trust the client's filename.** It is attacker-controlled and is the
  classic path-traversal vector. Stored names are generated, and the original
  is kept only as a display label.
* **Never trust the client's content type.** The declared MIME type is checked
  against the file's actual magic bytes, so a ``.exe`` renamed to ``.png`` with
  ``Content-Type: image/png`` is rejected.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings
from app.core.exceptions import UnsupportedMediaError
from app.core.logging import get_logger

logger = get_logger(__name__)

#: Leading bytes that identify each accepted format.
MAGIC_SIGNATURES: dict[str, list[bytes]] = {
    "application/pdf": [b"%PDF-"],
    "image/jpeg": [b"\xff\xd8\xff"],
    "image/png": [b"\x89PNG\r\n\x1a\n"],
    "image/webp": [b"RIFF"],           # plus a "WEBP" tag at offset 8
    "image/tiff": [b"II*\x00", b"MM\x00*"],
}

EXTENSIONS = {
    "application/pdf": ".pdf",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/tiff": ".tiff",
}


@dataclass(slots=True)
class StoredFile:
    url: str
    storage: str
    public_id: str
    size_bytes: int


def verify_magic_bytes(data: bytes, content_type: str) -> bool:
    """Confirm the payload really is the format it claims to be."""
    signatures = MAGIC_SIGNATURES.get(content_type)
    if not signatures:
        return False
    if not any(data.startswith(sig) for sig in signatures):
        return False
    if content_type == "image/webp":
        # RIFF is a container; the WEBP tag at offset 8 is what makes it a WebP.
        return len(data) > 12 and data[8:12] == b"WEBP"
    return True


class StorageService:
    """Uploads media, choosing the backend from configuration."""

    def __init__(self) -> None:
        self.use_cloudinary = settings.cloudinary_enabled
        self.local_root = Path(settings.local_upload_dir)

    def validate(self, data: bytes, content_type: str, allowed: set[str]) -> None:
        """Reject anything oversized, disallowed, or lying about its type."""
        max_bytes = settings.max_upload_mb * 1024 * 1024
        if len(data) > max_bytes:
            raise UnsupportedMediaError(
                f"File exceeds the {settings.max_upload_mb}MB limit."
            )
        if not data:
            raise UnsupportedMediaError("The uploaded file is empty.")
        if content_type not in allowed:
            raise UnsupportedMediaError(
                f"'{content_type}' is not accepted here. Allowed: "
                f"{', '.join(sorted(allowed))}."
            )
        if not verify_magic_bytes(data, content_type):
            raise UnsupportedMediaError(
                "The file content does not match its declared type. It may be "
                "corrupted, or renamed from another format."
            )

    async def upload(
        self, data: bytes, content_type: str, folder: str, user_id: uuid.UUID
    ) -> StoredFile:
        """Store a file and return its retrievable location."""
        # Generated, non-guessable name — never derived from user input.
        public_id = f"{user_id.hex[:8]}-{uuid.uuid4().hex}"
        extension = EXTENSIONS.get(content_type, ".bin")

        if self.use_cloudinary:
            return await self._upload_cloudinary(data, folder, public_id, content_type)
        return self._upload_local(data, folder, public_id + extension)

    async def _upload_cloudinary(
        self, data: bytes, folder: str, public_id: str, content_type: str
    ) -> StoredFile:
        import asyncio

        import cloudinary
        import cloudinary.uploader

        cloudinary.config(
            cloud_name=settings.cloudinary_cloud_name,
            api_key=settings.cloudinary_api_key,
            api_secret=settings.cloudinary_api_secret,
            secure=True,
        )

        def _do_upload() -> dict:
            return cloudinary.uploader.upload(
                data,
                folder=f"oviora/{folder}",
                public_id=public_id,
                resource_type="raw" if content_type == "application/pdf" else "image",
                overwrite=False,
                # Lab reports must never be publicly enumerable, so uploads are
                # private and served through signed, expiring URLs.
                type="private",
            )

        # The Cloudinary SDK is synchronous; running it in a thread keeps it
        # from blocking the event loop and stalling every other request.
        result = await asyncio.to_thread(_do_upload)
        return StoredFile(
            url=result.get("secure_url", ""),
            storage="cloudinary",
            public_id=result.get("public_id", public_id),
            size_bytes=len(data),
        )

    def _upload_local(self, data: bytes, folder: str, filename: str) -> StoredFile:
        target_dir = self.local_root / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / filename
        path.write_bytes(data)

        logger.info(
            "file stored locally",
            extra={"path": str(path), "bytes": len(data)},
        )
        return StoredFile(
            url=f"/uploads/{folder}/{filename}",
            storage="local",
            public_id=filename,
            size_bytes=len(data),
        )

    def backend_name(self) -> str:
        return "cloudinary" if self.use_cloudinary else "local_disk"


_storage: StorageService | None = None


def get_storage() -> StorageService:
    global _storage
    if _storage is None:
        _storage = StorageService()
    return _storage
