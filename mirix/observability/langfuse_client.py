"""
LangFuse client singleton implementation.

Async-native singleton pattern optimized for containerized deployment.
Each container instance gets its own singleton client.
"""

import asyncio
import atexit
from typing import TYPE_CHECKING, Optional

from mirix.log import get_logger

if TYPE_CHECKING:
    from langfuse import Langfuse

logger = get_logger(__name__)

_init_lock = asyncio.Lock()

_langfuse_client: Optional["Langfuse"] = None
_langfuse_enabled: bool = False
_initialization_attempted: bool = False


async def initialize_langfuse(force: bool = False) -> Optional["Langfuse"]:
    """
    Initialize LangFuse client from settings (async, coroutine-safe).

    Called during server startup via the FastAPI lifespan hook.
    Uses double-checked locking pattern with asyncio.Lock.

    Args:
        force: Force re-initialization even if already initialized

    Returns:
        Langfuse client instance or None if disabled/failed
    """
    global _langfuse_client, _langfuse_enabled, _initialization_attempted

    if _initialization_attempted and not force:
        return _langfuse_client

    async with _init_lock:
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

            _langfuse_client = await asyncio.to_thread(
                Langfuse,
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

            atexit.register(_sync_flush_for_atexit)

            try:
                await asyncio.to_thread(_langfuse_client.flush)
                logger.info(
                    f"LangFuse observability initialized and verified successfully (environment: {environment})"
                )
            except Exception as health_error:
                logger.warning(f"LangFuse initialized but health check failed: {health_error}")

            return _langfuse_client

        except Exception as e:
            logger.error(f"Failed to initialize LangFuse: {e}", exc_info=True)
            _langfuse_enabled = False
            _langfuse_client = None
            return None


def get_langfuse_client() -> Optional["Langfuse"]:
    """
    Get the global LangFuse client instance.

    Returns the cached singleton. Initialization must have been called
    via ``await initialize_langfuse()`` during server startup; if not
    yet initialized this returns None.

    Returns:
        Langfuse client or None if not initialized/disabled
    """
    if _langfuse_enabled and _langfuse_client is not None:
        return _langfuse_client
    return None


def is_langfuse_enabled() -> bool:
    """
    Check if LangFuse tracing is enabled and initialized.

    Returns:
        True if LangFuse is enabled and client is initialized
    """
    return _langfuse_enabled and _langfuse_client is not None


def _sync_flush_for_atexit() -> None:
    """Synchronous flush for the atexit handler (runs outside the event loop)."""
    if _langfuse_client and _langfuse_enabled:
        try:
            _langfuse_client.flush()
        except Exception:
            pass


async def flush_langfuse(timeout: Optional[float] = None) -> bool:
    """
    Flush all pending LangFuse traces asynchronously.

    Args:
        timeout: Maximum time to wait for flush (seconds).
                Uses settings default if not specified.

    Returns:
        True if flush successful, False otherwise
    """
    if not _langfuse_client or not _langfuse_enabled:
        return True

    if timeout is None:
        try:
            from mirix.settings import settings

            timeout = settings.langfuse_flush_timeout
        except Exception:
            timeout = 10.0

    try:
        logger.info(f"Flushing LangFuse traces (timeout: {timeout}s)...")
        await asyncio.to_thread(_langfuse_client.flush)
        logger.info("LangFuse traces flushed successfully")
        return True
    except Exception as e:
        logger.error(f"Error flushing LangFuse traces: {e}", exc_info=True)
        return False


async def shutdown_langfuse() -> None:
    """
    Shutdown LangFuse client and clean up resources.

    Should be called on application shutdown to ensure all traces
    are sent and resources are properly released.
    """
    global _langfuse_client, _langfuse_enabled

    if _langfuse_client:
        try:
            logger.info("Shutting down LangFuse client...")

            await flush_langfuse()

            if hasattr(_langfuse_client, "shutdown"):
                await asyncio.to_thread(_langfuse_client.shutdown)

            logger.info("LangFuse client shutdown complete")
        except Exception as e:
            logger.warning(f"Error during LangFuse shutdown: {e}")
        finally:
            _langfuse_client = None
            _langfuse_enabled = False


async def _reset_for_testing() -> None:
    """
    Reset singleton state for testing.

    WARNING: DO NOT use in production code. Only for unit tests.
    """
    global _langfuse_client, _langfuse_enabled, _initialization_attempted

    if _langfuse_client:
        try:
            await asyncio.to_thread(_langfuse_client.flush)
        except Exception:
            pass

    _langfuse_client = None
    _langfuse_enabled = False
    _initialization_attempted = False
