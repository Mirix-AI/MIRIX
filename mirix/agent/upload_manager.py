import asyncio
import logging
import os
import shutil
import sys
import uuid

logger = logging.getLogger(__name__)


class UploadManager:
    """
    Async upload manager that handles each image upload independently.
    Each upload gets a timeout and either succeeds or fails.
    Uses asyncio for concurrency instead of ThreadPoolExecutor.
    """

    def __init__(self, google_client, client, existing_files, uri_to_create_time):
        self.google_client = google_client
        self.client = client
        self.existing_files = existing_files
        self.uri_to_create_time = uri_to_create_time

        self.logger = logging.getLogger("Mirix.UploadManager")
        self.logger.setLevel(logging.INFO)

        self._upload_status = {}
        self._upload_lock = asyncio.Lock()

    async def _compress_image(self, image_path, quality=85, max_size=(1920, 1080)):
        """Compress image via async subprocess (zero GIL impact).

        Primary: vipsthumbnail (libvips CLI, very fast).
        Fallback: Pillow in a child Python process.
        """
        base_path = os.path.splitext(image_path)[0]
        compressed_path = f"{base_path}_compressed.jpg"

        if shutil.which("vipsthumbnail"):
            try:
                process = await asyncio.create_subprocess_exec(
                    "vipsthumbnail", image_path,
                    "--size", f"{max_size[0]}x{max_size[1]}",
                    "-o", f"{compressed_path}[Q={quality},optimize-coding,strip]",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr_bytes = await process.communicate()

                if process.returncode == 0 and os.path.exists(compressed_path):
                    return compressed_path

                logger.warning(
                    "vipsthumbnail failed for %s (rc=%d): %s",
                    image_path, process.returncode,
                    stderr_bytes.decode() if stderr_bytes else "",
                )
            except Exception as e:
                logger.warning(
                    "vipsthumbnail error for %s: %s; falling back to Pillow",
                    image_path, e,
                )

        # Fallback: Pillow in a child process (still async, separate process)
        try:
            script = (
                "from PIL import Image; "
                f"img = Image.open({image_path!r}); "
                "img = img.convert('RGB') "
                "if img.mode in ('RGBA','LA','P') else img; "
                f"img.thumbnail(({max_size[0]},{max_size[1]}),"
                " Image.Resampling.LANCZOS); "
                f"img.save({compressed_path!r},'JPEG',"
                f"quality={quality},optimize=True)"
            )
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-c", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr_bytes = await process.communicate()

            if process.returncode == 0 and os.path.exists(compressed_path):
                return compressed_path

            logger.error(
                "Pillow subprocess failed for %s (rc=%d): %s",
                image_path, process.returncode,
                stderr_bytes.decode() if stderr_bytes else "",
            )
            return None
        except Exception as e:
            logger.error("Image compression failed for %s: %s", image_path, e)
            return None

    async def _upload_single_file(self, upload_uuid, filename, timestamp, compressed_file):
        """Upload a single file asynchronously."""
        try:
            check_result = await self.client.server.cloud_file_mapping_manager.check_if_existing(
                local_file_id=filename,
            )
            if check_result:
                cloud_file_name = await self.client.server.cloud_file_mapping_manager.get_cloud_file(
                    local_file_id=filename,
                )
                file_ref = [x for x in self.existing_files if x.name == cloud_file_name][0]

                async with self._upload_lock:
                    self._upload_status[upload_uuid] = {
                        "status": "completed",
                        "result": file_ref,
                    }
                return

            upload_file = compressed_file if compressed_file and os.path.exists(compressed_file) else filename

            import time
            upload_start_time = time.time()
            file_ref = await self.google_client.aio.files.upload(file=upload_file)
            upload_duration = time.time() - upload_start_time

            self.logger.info(f"Upload completed in {upload_duration:.2f} seconds for file {upload_file}")

            self.uri_to_create_time[file_ref.uri] = {
                "create_time": file_ref.create_time,
                "filename": file_ref.name,
            }
            await self.client.server.cloud_file_mapping_manager.add_mapping(
                local_file_id=filename,
                cloud_file_id=file_ref.uri,
                timestamp=timestamp,
                force_add=True,
            )

            if compressed_file and compressed_file != filename and upload_file == compressed_file:
                try:
                    os.remove(compressed_file)
                    logger.info("Removed compressed file: %s", compressed_file)
                except Exception:
                    pass

            async with self._upload_lock:
                self._upload_status[upload_uuid] = {
                    "status": "completed",
                    "result": file_ref,
                }

        except Exception as e:
            logger.error("Upload failed for %s: %s", filename, e)
            async with self._upload_lock:
                self._upload_status[upload_uuid] = {"status": "failed", "result": None}

            if compressed_file and compressed_file != filename and os.path.exists(compressed_file):
                try:
                    os.remove(compressed_file)
                except Exception:
                    pass

    async def upload_file_async(self, filename, timestamp, compress=True):
        """Start an async upload and return immediately with a placeholder."""
        upload_uuid = str(uuid.uuid4())

        compressed_file = None
        if compress and filename.lower().endswith((".png", ".jpg", ".jpeg")):
            compressed_file = await self._compress_image(filename)

        async with self._upload_lock:
            self._upload_status[upload_uuid] = {"status": "pending", "result": None}

        async def _upload_with_timeout():
            try:
                await asyncio.wait_for(
                    self._upload_single_file(upload_uuid, filename, timestamp, compressed_file),
                    timeout=10.0,
                )
            except asyncio.TimeoutError:
                self.logger.info(f"Upload timeout for {filename}, marking as failed")
                async with self._upload_lock:
                    if self._upload_status.get(upload_uuid, {}).get("status") == "pending":
                        self._upload_status[upload_uuid] = {
                            "status": "failed",
                            "result": None,
                        }

        asyncio.create_task(_upload_with_timeout())

        return {"upload_uuid": upload_uuid, "filename": filename, "pending": True}

    async def get_upload_status(self, placeholder):
        """Get upload status and result."""
        if not isinstance(placeholder, dict) or not placeholder.get("pending"):
            return {"status": "completed", "result": placeholder}

        upload_uuid = placeholder["upload_uuid"]

        async with self._upload_lock:
            if upload_uuid not in self._upload_status:
                return {"status": "unknown", "result": None}

            status_info = self._upload_status.get(upload_uuid, {})
            status = status_info.get("status", "pending")
            result = status_info.get("result")
            return {"status": status, "result": result}

    async def try_resolve_upload(self, placeholder):
        """Try to resolve an upload placeholder."""
        status_info = await self.get_upload_status(placeholder)
        if status_info["status"] == "completed":
            return status_info["result"]
        return None

    async def wait_for_upload(self, placeholder, timeout=30):
        """Wait for upload to complete."""
        if not isinstance(placeholder, dict) or not placeholder.get("pending"):
            return placeholder

        import time
        start_time = time.time()
        while time.time() - start_time < timeout:
            upload_status = await self.get_upload_status(placeholder)

            if upload_status["status"] == "completed":
                return upload_status["result"]
            elif upload_status["status"] == "failed":
                raise Exception(f"Upload failed for {placeholder['filename']}")

            await asyncio.sleep(0.1)

        raise TimeoutError(f"Upload timeout after {timeout}s for {placeholder['filename']}")

    async def upload_file(self, filename, timestamp):
        """Upload a file and wait for completion."""
        placeholder = await self.upload_file_async(filename, timestamp)
        return await self.wait_for_upload(placeholder, timeout=10)

    async def cleanup_resolved_upload(self, placeholder):
        """Clean up resolved upload from tracking."""
        if not isinstance(placeholder, dict) or not placeholder.get("pending"):
            return

        upload_uuid = placeholder["upload_uuid"]
        async with self._upload_lock:
            self._upload_status.pop(upload_uuid, None)

    async def get_upload_status_summary(self):
        """Get a summary of current upload statuses."""
        async with self._upload_lock:
            summary = {}
            for uid, info in self._upload_status.items():
                status = info.get("status", "unknown")
                summary[status] = summary.get(status, 0) + 1
            return summary
