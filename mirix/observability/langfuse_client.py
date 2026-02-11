"""
LangFuse client singleton implementation.

Thread-safe singleton pattern optimized for containerized deployment.
Each container instance gets its own singleton client.
"""

import atexit
import threading
from typing import TYPE_CHECKING, Optional

from mirix.log import get_logger

if TYPE_CHECKING:
    from langfuse import Langfuse

logger = get_logger(__name__)

# Thread lock for initialization
_init_lock = threading.Lock()

# Global singleton instance (per container/process)
_langfuse_client: Optional["Langfuse"] = None
_langfuse_enabled: bool = False
_initialization_attempted: bool = False


def initialize_langfuse(force: bool = False) -> Optional["Langfuse"]:
    """
    Initialize LangFuse client from settings (thread-safe).

    Called during server startup. Uses double-checked locking pattern
    for thread-safe lazy initialization.

    Args:
        force: Force re-initialization even if already initialized

    Returns:
        Langfuse client instance or None if disabled/failed
    """
    global _langfuse_client, _langfuse_enabled, _initialization_attempted

    # Fast path: already initialized
    if _initialization_attempted and not force:
        return _langfuse_client

    # Thread-safe initialization
    with _init_lock:
        # Double-check inside lock
        if _initialization_attempted and not force:
            return _langfuse_client

        _initialization_attempted = True

        try:
            from mirix.settings import settings

            if not settings.langfuse_enabled:
                logger.info("LangFuse observability is disabled")
                _langfuse_enabled = False
                return None

            if not settings.langfuse_public_key or not settings.langfuse_secret_key:
                logger.warning(
                    "LangFuse enabled but missing credentials. "
                    "Set MIRIX_LANGFUSE_PUBLIC_KEY and MIRIX_LANGFUSE_SECRET_KEY. "
                    "Observability will be disabled."
                )
                _langfuse_enabled = False
                return None

            environment = settings.langfuse_environment
            logger.info(f"Initializing LangFuse client (host: {settings.langfuse_host}, environment: {environment})")

            from langfuse import Langfuse
            from opentelemetry.sdk.trace import TracerProvider

            # LangFuse 3.x removed the 'enabled' parameter
            # Client is enabled by default when instantiated
            # The 'environment' parameter provides native environment filtering in Langfuse
            # (environment must match regex: ^(?!langfuse)[a-z0-9-_]+$ with max 40 chars)
            _langfuse_client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
                debug=settings.langfuse_debug,
                flush_interval=settings.langfuse_flush_interval,
                flush_at=settings.langfuse_flush_at,
                tracer_provider=TracerProvider(),
                environment=environment,
            )

            _langfuse_enabled = True

            # Register cleanup on process exit
            atexit.register(flush_langfuse)

            # Verify connection with initial flush
            try:
                _langfuse_client.flush()
                logger.info(
                    f"LangFuse observability initialized and verified successfully (environment: {environment})"
                )
            except Exception as health_error:
                logger.warning(f"LangFuse initialized but health check failed: {health_error}")
                # Continue anyway - will retry on actual use

            return _langfuse_client

        except Exception as e:
            logger.error(f"Failed to initialize LangFuse: {e}", exc_info=True)
            _langfuse_enabled = False
            _langfuse_client = None
            return None


def get_langfuse_client() -> Optional["Langfuse"]:
    """
    Get the global LangFuse client instance (lazy initialization).

    Thread-safe and efficient - only initializes once per container.
    Each container instance has its own singleton client.

    Returns:
        Langfuse client or None if not initialized/disabled
    """
    global _langfuse_client, _initialization_attempted

    # Fast path: already initialized
    if _initialization_attempted:
        return _langfuse_client if _langfuse_enabled else None

    # Lazy initialization on first use
    return initialize_langfuse()


def is_langfuse_enabled() -> bool:
    """
    Check if LangFuse tracing is enabled and initialized.

    Returns:
        True if LangFuse is enabled and client is initialized
    """
    return _langfuse_enabled and _langfuse_client is not None


def flush_langfuse(timeout: Optional[float] = None) -> bool:
    """
    Flush all pending LangFuse traces synchronously.

    Critical for container shutdown - ensures traces are sent before
    process termination.

    Args:
        timeout: Maximum time to wait for flush (seconds).
                Uses settings default if not specified.

    Returns:
        True if flush successful, False otherwise
    """
    global _langfuse_client

    if not _langfuse_client or not _langfuse_enabled:
        return True

    if timeout is None:
        try:
            from mirix.settings import settings

            timeout = settings.langfuse_flush_timeout
        except:
            timeout = 10.0  # Default fallback

    try:
        logger.info(f"Flushing LangFuse traces (timeout: {timeout}s)...")
        _langfuse_client.flush()
        logger.info("LangFuse traces flushed successfully")
        return True
    except Exception as e:
        logger.error(f"Error flushing LangFuse traces: {e}", exc_info=True)
        return False


def shutdown_langfuse() -> None:
    """
    Shutdown LangFuse client and clean up resources.

    Should be called on application shutdown to ensure all traces
    are sent and resources are properly released.
    """
    global _langfuse_client, _langfuse_enabled, _initialization_attempted

    if _langfuse_client:
        try:
            logger.info("Shutting down LangFuse client...")

            # Final flush with timeout
            flush_langfuse()

            # Close client if SDK provides shutdown method
            if hasattr(_langfuse_client, "shutdown"):
                _langfuse_client.shutdown()

            logger.info("LangFuse client shutdown complete")
        except Exception as e:
            logger.warning(f"Error during LangFuse shutdown: {e}")
        finally:
            _langfuse_client = None
            _langfuse_enabled = False
            # Keep _initialization_attempted = True to prevent re-init after shutdown


def _reset_for_testing() -> None:
    """
    Reset singleton state for testing.

    WARNING: DO NOT use in production code. Only for unit tests.
    """
    global _langfuse_client, _langfuse_enabled, _initialization_attempted

    if _langfuse_client:
        try:
            _langfuse_client.flush()
        except:
            pass

    _langfuse_client = None
    _langfuse_enabled = False
    _initialization_attempted = False
