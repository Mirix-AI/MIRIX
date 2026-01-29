import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from PIL import Image


class UploadManager:
    """
    Simplified upload manager that handles each image upload independently.
    Each upload gets a 10-second timeout and either succeeds or fails immediately.
    """

    def __init__(self, google_client, client, existing_files, uri_to_create_time):
        self.google_client = google_client
        self.client = client
        self.existing_files = existing_files
        self.uri_to_create_time = uri_to_create_time

        # Initialize logger
        self.logger = logging.getLogger("Mirix.UploadManager")
        self.logger.setLevel(logging.INFO)

        # Simple tracking: upload_uuid -> {'status': 'pending'/'completed'/'failed', 'result': file_ref or None}
        self._upload_status = {}
        self._upload_lock = threading.Lock()

        # Thread pool for concurrent uploads (max 4 simultaneous uploads)
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="upload_worker")

    def _compress_image(self, image_path, quality=85, max_size=(1920, 1080)):
        """Compress image to reduce upload time while maintaining reasonable quality"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ("RGBA", "LA", "P"):
                    img = img.convert("RGB")

                # Resize if too large
                img.thumbnail(max_size, Image.Resampling.LANCZOS)

                # Create compressed version
                base_path = os.path.splitext(image_path)[0]
                compressed_path = f"{base_path}_compressed.jpg"
                img.save(compressed_path, "JPEG", quality=quality, optimize=True)

                return compressed_path if os.path.exists(compressed_path) else None

        except Exception as e:
            logger.error("Image compression failed for %s: %s", image_path, e)
            return None

    def _upload_single_file(self, upload_uuid, filename, timestamp, compressed_file):
        """Upload a single file with 5-second timeout"""
        try:
            # Check if file already exists in cloud
            if self.client.server.cloud_file_mapping_manager.check_if_existing(local_file_id=filename):
                cloud_file_name = self.client.server.cloud_file_mapping_manager.get_cloud_file(local_file_id=filename)
                file_ref = [x for x in self.existing_files if x.name == cloud_file_name][0]

                with self._upload_lock:
                    self._upload_status[upload_uuid] = {
                        "status": "completed",
                        "result": file_ref,
                    }
                return

            # Choose file to upload (compressed if available, otherwise original)
            upload_file = compressed_file if compressed_file and os.path.exists(compressed_file) else filename

            # Upload with 5-second timeout
            upload_start_time = time.time()
            file_ref = self.google_client.files.upload(file=upload_file)
            upload_duration = time.time() - upload_start_time

            self.logger.info(f"Upload completed in {upload_duration:.2f} seconds for file {upload_file}")

            # Update tracking and database
            self.uri_to_create_time[file_ref.uri] = {
                "create_time": file_ref.create_time,
                "filename": file_ref.name,
            }
            self.client.server.cloud_file_mapping_manager.add_mapping(
                local_file_id=filename,
                cloud_file_id=file_ref.uri,
                timestamp=timestamp,
                force_add=True,
            )

            # Clean up compressed file if it was created and used
            if compressed_file and compressed_file != filename and upload_file == compressed_file:
                try:
                    os.remove(compressed_file)
                    logger.info("Removed compressed file: %s", compressed_file)
                except Exception:
                    pass  # Ignore cleanup errors

            # Mark as completed
            with self._upload_lock:
                self._upload_status[upload_uuid] = {
                    "status": "completed",
                    "result": file_ref,
                }

        except Exception as e:
            logger.error("Upload failed for %s: %s", filename, e)
            # Mark as failed
            with self._upload_lock:
                self._upload_status[upload_uuid] = {"status": "failed", "result": None}

            # Clean up compressed file on failure too
            if compressed_file and compressed_file != filename and os.path.exists(compressed_file):
                try:
                    os.remove(compressed_file)
                except Exception:
                    pass

    def upload_file_async(self, filename, timestamp, compress=True):
        """Start an async upload and return immediately with a placeholder"""
        upload_uuid = str(uuid.uuid4())

        # Compress image if requested
        compressed_file = None
        if compress and filename.lower().endswith((".png", ".jpg", ".jpeg")):
            compressed_file = self._compress_image(filename)

        # Initialize status
        with self._upload_lock:
            self._upload_status[upload_uuid] = {"status": "pending", "result": None}

        # Submit upload task with 5-second timeout
        future = self._executor.submit(self._upload_single_file, upload_uuid, filename, timestamp, compressed_file)

        # Set up automatic timeout handling
        def timeout_handler():
            time.sleep(10.0)  # Wait 10 seconds
            with self._upload_lock:
                if self._upload_status.get(upload_uuid, {}).get("status") == "pending":
                    self.logger.info(f"Upload timeout (5s) for {filename}, marking as failed")
                    self._upload_status[upload_uuid] = {
                        "status": "failed",
                        "result": None,
                    }
                    future.cancel()  # Try to cancel the upload

        # Start timeout handler in separate thread
        timeout_thread = threading.Thread(target=timeout_handler, daemon=True)
        timeout_thread.start()

        # Return placeholder
        return {"upload_uuid": upload_uuid, "filename": filename, "pending": True}

    def get_upload_status(self, placeholder):
        """Get upload status and result in one call"""
        if not isinstance(placeholder, dict) or not placeholder.get("pending"):
            return {"status": "completed", "result": placeholder}  # Already resolved

        upload_uuid = placeholder["upload_uuid"]

        with self._upload_lock:
            if upload_uuid not in self._upload_status:
                # Upload was either never started or already cleaned up
                # For cleaned up uploads, we can't tell if they succeeded or failed
                return {"status": "unknown", "result": None}

            status_info = self._upload_status.get(upload_uuid, {})
            status = status_info.get("status", "pending")
            result = status_info.get("result")

            # Don't clean up here - let cleanup_resolved_upload handle it
            return {"status": status, "result": result}

    def try_resolve_upload(self, placeholder):
        """Legacy method for backward compatibility"""
        status_info = self.get_upload_status(placeholder)
        if status_info["status"] == "completed":
            return status_info["result"]
        else:
            return None

    def wait_for_upload(self, placeholder, timeout=30):
        """Wait for upload to complete (legacy method, now just polls get_upload_status)"""
        if not isinstance(placeholder, dict) or not placeholder.get("pending"):
            return placeholder

        start_time = time.time()
        while time.time() - start_time < timeout:
            upload_status = self.get_upload_status(placeholder)

            if upload_status["status"] == "completed":
                return upload_status["result"]
            elif upload_status["status"] == "failed":
                raise Exception(f"Upload failed for {placeholder['filename']}")

            time.sleep(0.1)

        raise TimeoutError(f"Upload timeout after {timeout}s for {placeholder['filename']}")

    def upload_file(self, filename, timestamp):
        """Legacy synchronous upload method"""
        placeholder = self.upload_file_async(filename, timestamp)
        return self.wait_for_upload(placeholder, timeout=10)  # Reduced timeout since individual uploads timeout at 5s

    def cleanup_resolved_upload(self, placeholder):
        """Clean up resolved upload from tracking"""
        if not isinstance(placeholder, dict) or not placeholder.get("pending"):
            return  # Not a pending placeholder

        upload_uuid = placeholder["upload_uuid"]
        with self._upload_lock:
            self._upload_status.pop(upload_uuid, None)

    def cleanup_upload_workers(self):
        """Gracefully shut down the thread pool"""
        try:
            self._executor.shutdown(wait=True, timeout=10)
        except Exception:
            pass  # Ignore shutdown errors

    def get_upload_status_summary(self):
        """Get a summary of current upload statuses (for debugging)"""
        with self._upload_lock:
            summary = {}
            for uuid, info in self._upload_status.items():
                status = info.get("status", "unknown")
                summary[status] = summary.get(status, 0) + 1
            return summary
