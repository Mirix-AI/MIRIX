#!/usr/bin/env python3
"""
Evaluate Mirix token usage by uploading the sample images from set1.

This mirrors the add-image logic from run_client.py but focuses solely on
feeding the .local/images/set1 assets into memory so we can inspect usage
statistics returned by the API.
"""

import base64
import logging
import mimetypes
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image, ImageOps

from mirix import MirixClient

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
IMAGE_SET_PATH = Path(".local/images/set1")
MAX_IMAGE_DIMENSION = 512


def _save_resized_image(image_path: Path) -> Tuple[bytes, str]:
    """Return the image bytes (resized if needed) and its mime type."""
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        if max(img.size) > MAX_IMAGE_DIMENSION:
            img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)

        buffer = BytesIO()
        save_format = (img.format or mime_type.split("/")[-1]).upper()
        if save_format == "JPG":
            save_format = "JPEG"
        try:
            img.save(buffer, format=save_format)
        except ValueError:
            # Some formats do not support saving; fall back to PNG.
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            mime_type = "image/png"

    return buffer.getvalue(), mime_type


def encode_image_to_data_url(image_path: Path) -> str:
    """Return a data URL string for an image on disk."""
    image_bytes, mime_type = _save_resized_image(image_path)
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def get_images_from_set(set_path: Path) -> List[Path]:
    """Collect all supported images from the provided directory."""
    if not set_path.exists():
        raise FileNotFoundError(f"Image directory does not exist: {set_path}")

    image_paths = sorted(
        path for path in set_path.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(f"No supported images found in {set_path}")

    logger.info("Found %d images in %s", len(image_paths), set_path)
    return image_paths


def add_images_to_memory(client: MirixClient, user_id: str, image_paths: Iterable[Path]) -> None:
    """Send each image to the memory-add endpoint."""
    for image_path in image_paths:
        data_url = encode_image_to_data_url(image_path)
        logger.info("Adding %s ...", image_path.name)

        response = client.add(
            user_id=user_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": ("Please store this reference image for later: " f"{image_path.name}"),
                        },
                        {
                            "type": "image_data",
                            "image_data": {
                                "data": data_url,
                                "detail": "high",
                            },
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"I've received {image_path.name}.",
                        }
                    ],
                },
            ],
            chaining=False,
        )

        success = response.get("success")
        usage = response.get("usage")
        logger.info("Response success=%s usage=%s", success, usage)


def main() -> None:
    user_id = "token-usage-user"
    org_id = "token-usage-org"

    client = MirixClient(
        org_id=org_id,
        api_key=None,
        debug=True,
    )

    client.initialize_meta_agent(
        user_id=user_id,
        config_path="mirix/configs/examples/mirix_gemini.yaml",
        update_agents=False,
    )

    images = get_images_from_set(IMAGE_SET_PATH)
    add_images_to_memory(client, user_id=user_id, image_paths=images)


if __name__ == "__main__":
    main()
