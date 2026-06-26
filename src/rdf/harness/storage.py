"""Abstract ObjectStore with Local and S3 implementations.

Backend selected by RDF_STORAGE env var: local | s3.
"""

from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path


class ObjectStore(ABC):
    @abstractmethod
    def put_bytes(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def get_bytes(self, key: str) -> bytes: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    @abstractmethod
    def list_keys(self, prefix: str) -> list[str]: ...


class LocalObjectStore(ObjectStore):
    def __init__(self, root: str | None = None):
        self.root = Path(root or os.environ.get("RDF_LOCAL_STORAGE_PATH", "/tmp/rdf/storage"))
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key

    def put_bytes(self, key: str, data: bytes) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete(self, key: str) -> None:
        p = self._path(key)
        if p.exists():
            p.unlink()

    def list_keys(self, prefix: str) -> list[str]:
        root_prefix = self.root / prefix
        if not root_prefix.exists():
            return []
        if root_prefix.is_file():
            return [prefix]
        return [
            str(p.relative_to(self.root))
            for p in root_prefix.rglob("*")
            if p.is_file()
        ]


class S3ObjectStore(ObjectStore):
    def __init__(self, bucket: str | None = None):
        import boto3
        self.bucket = bucket or os.environ.get("RDF_S3_RAW_BUCKET", "rdf-raw")
        self.client = boto3.client("s3")

    def put_bytes(self, key: str, data: bytes) -> None:
        import io
        self.client.upload_fileobj(io.BytesIO(data), self.bucket, key)

    def get_bytes(self, key: str) -> bytes:
        import io
        buf = io.BytesIO()
        self.client.download_fileobj(self.bucket, key, buf)
        return buf.getvalue()

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def list_keys(self, prefix: str) -> list[str]:
        paginator = self.client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys


def get_object_store(backend: str | None = None, **kwargs) -> ObjectStore:
    backend = backend or os.environ.get("RDF_STORAGE", "local")
    if backend == "local":
        return LocalObjectStore(**kwargs)
    if backend == "s3":
        return S3ObjectStore(**kwargs)
    raise ValueError(f"Unknown RDF_STORAGE backend: {backend!r}")
