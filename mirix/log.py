import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from mirix.settings import settings


def get_log_level() -> int:
    """Get the configured log level."""
    if settings.debug:
        return logging.DEBUG

    # Map string level to logging constant
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    return level_map.get(settings.log_level.upper(), logging.INFO)


selected_log_level = get_log_level()


class _TidLogFilter(logging.Filter):
    """Inject the current request's transaction id (TID) onto each record.

    Reads the ``current_tid`` contextvar (set on the request path and
    re-established in the queue worker via trace propagation) so that every log
    line can be correlated by TID — including worker/agent logs that run outside
    any HTTP request context. When no TID is set, a stable ``-`` placeholder is
    used so the format string never errors.

    Imported lazily to avoid an import cycle at module load
    (``observability`` imports ``mirix.log``).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        tid = None
        try:
            from mirix.observability.context import current_tid

            tid = current_tid.get()
        except Exception:
            tid = None
        record.tid = tid or "-"
        return True


_tid_log_filter = _TidLogFilter()


def validate_log_file_path(log_file_path: Path) -> Path:
    """
    Validate that the log file path is writable.

    Checks:
    - Path is not a directory
    - Parent directory exists or can be created
    - We have write permissions to the directory

    Args:
        log_file_path: Path to the log file

    Returns:
        Path: Validated absolute path

    Raises:
        ValueError: If the path is invalid or not writable
    """
    # Convert to absolute path
    log_file_path = log_file_path.expanduser().resolve()

    # Check if path exists and is a directory (not allowed)
    if log_file_path.exists() and log_file_path.is_dir():
        raise ValueError(
            f"Invalid log file path: '{log_file_path}' is a directory. "
            f"MIRIX_LOG_FILE must be a file path, not a directory."
        )

    # Get parent directory
    parent_dir = log_file_path.parent

    # Try to create parent directory if it doesn't exist
    try:
        parent_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        raise ValueError(f"Invalid log file path: Cannot create directory '{parent_dir}'. Error: {e}") from e

    # Check if parent directory is writable
    if not os.access(parent_dir, os.W_OK):
        raise ValueError(
            f"Invalid log file path: Directory '{parent_dir}' is not writable. Check permissions for MIRIX_LOG_FILE."
        )

    # If file exists, check if it's writable
    if log_file_path.exists() and not os.access(log_file_path, os.W_OK):
        raise ValueError(
            f"Invalid log file path: File '{log_file_path}' exists but is not writable. "
            f"Check file permissions for MIRIX_LOG_FILE."
        )

    return log_file_path


def get_logger(name: Optional[str] = None) -> "logging.Logger":
    """
    Get the Mirix logger with configured handlers.

    Log Level Configuration:
        - Single log level (MIRIX_LOG_LEVEL) applies to ALL handlers
        - Controlled by: MIRIX_LOG_LEVEL or MIRIX_DEBUG environment variables
        - Same level used for both console and file output

    Handler Configuration (Default Behavior):
        - Console: ALWAYS enabled UNLESS explicitly disabled (MIRIX_LOG_TO_CONSOLE=false)
        - File: Automatically enabled if MIRIX_LOG_FILE is set with a valid path
        - Handlers determine WHERE logs go, NOT what level they use

    Returns:
        logging.Logger: Configured logger instance

    Raises:
        ValueError: If MIRIX_LOG_FILE is set but the path is invalid or not writable
    """
    logger = logging.getLogger("Mirix")

    # Set the log level ONCE for the entire logger
    # This single level applies to all handlers (console and file)
    logger.setLevel(selected_log_level)

    # Add handlers if not already configured
    # Handlers control WHERE logs go (console/file), not WHAT level they use
    if not logger.handlers:
        # Create a single formatter for consistency across all handlers.
        # ``tid=`` carries the transaction id for cross-signal correlation
        # (populated by _TidLogFilter; "-" when no request context).
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s [tid=%(tid)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        handlers_added = []

        # Console handler - ALWAYS enabled unless explicitly disabled
        # Console logging is the default behavior
        if settings.log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.addFilter(_tid_log_filter)
            logger.addHandler(console_handler)
            handlers_added.append("console")

        # File handler - ONLY enabled if MIRIX_LOG_FILE is configured
        # Automatically enabled when MIRIX_LOG_FILE is set
        if settings.log_file is not None:
            # Validate and get absolute path
            # This will raise ValueError if path is invalid
            log_file = validate_log_file_path(Path(settings.log_file))

            # Create rotating file handler
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=settings.log_max_bytes,
                backupCount=settings.log_backup_count,
            )
            file_handler.setFormatter(formatter)
            file_handler.addFilter(_tid_log_filter)
            logger.addHandler(file_handler)
            handlers_added.append(f"file ({log_file})")

        # Log where logs are being written (if any handlers were added)
        if handlers_added:
            destinations = " and ".join(handlers_added)
            log_level_name = logging.getLevelName(selected_log_level)
            logger.info("Logging to: %s (level: %s)", destinations, log_level_name)
        else:
            # No handlers configured - add NullHandler to prevent warnings
            # This only happens if console is explicitly disabled AND file is not configured
            logger.addHandler(logging.NullHandler())

        # Prevent propagation to root logger to avoid duplicate messages
        logger.propagate = False

    return logger


def safe_traceback(exc: BaseException) -> str:
    """Render exc's frame stack with exception-message text stripped.

    The frame stack itself (file:line:function) carries no PII; the leak
    surface in a normal traceback is the final line ``ExceptionType:
    <msg>``, which echoes whatever ``__str__`` returns. We render only
    the frames plus the chain of exception class names.

    Output is shaped to match stdlib ``traceback.format_exception`` as
    closely as possible — same ``Traceback (most recent call last):``
    header, same frame format. Splunk alerts and dashboards that key on
    the literal "Traceback (most recent call last)" string continue to
    match. The final line is ``ExceptionType`` (no ``: <message>``)
    instead of ``ExceptionType: <message>``; alerts grepping for the
    type name still match, alerts grepping for specific exception-
    message text don't.
    """
    import traceback

    frames = "".join(traceback.format_tb(exc.__traceback__))
    chain = []
    seen: set = set()
    cur: Optional[BaseException] = exc
    # Walk __cause__ / __context__ chain with cycle detection. Pathological
    # exception chains (manually constructed ``raise … from prior`` loops)
    # would otherwise spin forever; stdlib's traceback.format_exception
    # uses the same id-tracking pattern.
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        chain.append(type(cur).__name__)
        cur = cur.__cause__ or cur.__context__
    return f"Traceback (most recent call last):\n{frames}{' -> '.join(chain)}"
