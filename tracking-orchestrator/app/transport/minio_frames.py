"""Async MinIO/S3 frame image fetcher for the tracking pipeline."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from importlib import import_module
from io import BytesIO
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class MinioFrameConfig:
    """Connection settings for S3-compatible frame storage."""

    endpoint_url: str
    bucket: str
    access_key_id: str
    secret_access_key: str
    region_name: str = "us-east-1"
    secure: bool = False


class MinioFrameFetcher:
    """Fetch JPEG frames from MinIO and decode them into RGB numpy arrays."""

    def __init__(self, config: MinioFrameConfig) -> None:
        self._config = config
        self._client_cm: Any = None
        self._client: Any = None

    async def connect(self) -> None:
        if self._client is not None:
            return
        from aiobotocore.session import get_session  # type: ignore[import-untyped]

        session = get_session()
        self._client_cm = session.create_client(
            "s3",
            endpoint_url=self._config.endpoint_url,
            aws_access_key_id=self._config.access_key_id,
            aws_secret_access_key=self._config.secret_access_key,
            region_name=self._config.region_name,
            verify=self._config.secure,
            use_ssl=self._config.secure,
        )
        self._client = await self._client_cm.__aenter__()

    async def disconnect(self) -> None:
        if self._client_cm is not None:
            await self._client_cm.__aexit__(None, None, None)
            self._client_cm = None
            self._client = None

    async def fetch_rgb(self, minio_key: str) -> npt.NDArray[np.uint8]:
        if self._client is None:
            raise RuntimeError("MinioFrameFetcher is not connected")

        response = await self._client.get_object(
            Bucket=self._config.bucket,
            Key=minio_key,
        )
        body = response["Body"]
        try:
            data = await body.read()
        finally:
            with suppress(Exception):
                body.close()

        try:
            image_module: Any = import_module("PIL.Image")
        except ImportError as exc:
            raise ImportError(
                "Pillow is required to decode JPEG frames fetched from MinIO."
            ) from exc

        with image_module.open(BytesIO(data)) as image:
            rgb = image.convert("RGB")
            return np.asarray(rgb, dtype=np.uint8)
