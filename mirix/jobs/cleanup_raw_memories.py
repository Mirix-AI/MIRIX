"""
Cleanup job for raw memories with 14-day TTL.

This job should be run nightly via cron or Celery beat to delete
raw memories older than 14 days (based on updated_at timestamp).
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Dict

from mirix.schemas.client import Client as PydanticClient
from mirix.services.raw_memory_manager import RawMemoryManager

logger = logging.getLogger(__name__)


def delete_stale_raw_memories(days_threshold: int = 14) -> Dict:
    """
    Hard delete raw memories older than the specified threshold (based on updated_at).

    This job should be run nightly via cron or Celery beat.

    Args:
        days_threshold: Number of days after which memories are considered stale (default: 14)

    Returns:
        Dict with deletion statistics
    """
    cutoff = datetime.now(UTC) - timedelta(days=days_threshold)

    logger.info(
        "Starting cleanup of raw memories older than %s (cutoff: %s)",
        f"{days_threshold} days",
        cutoff.isoformat(),
    )

    manager = RawMemoryManager()
    deleted_count = 0
    error_count = 0

    # Query memories older than cutoff and delete them
    with manager.session_maker() as session:
        from sqlalchemy import select

        from mirix.orm.raw_memory import RawMemory

        # Query stale memories
        stmt = select(RawMemory).where(RawMemory.updated_at < cutoff)
        result = session.execute(stmt)
        stale_memories = result.scalars().all()

        logger.info(f"Found {len(stale_memories)} stale raw memories to delete")

        # Create system actor for deletion
        # Note: This bypasses organization-level access control for cleanup
        system_actor = PydanticClient(
            id="system-cleanup-job",
            organization_id="system",
            name="Cleanup Job",
        )

        for memory in stale_memories:
            try:
                manager.delete_raw_memory(memory.id, system_actor)
                deleted_count += 1
            except Exception as e:
                logger.error(f"Failed to delete raw memory {memory.id}: {e}")
                error_count += 1

    result = {
        "success": True,
        "deleted_count": deleted_count,
        "error_count": error_count,
        "cutoff_date": cutoff.isoformat(),
        "days_threshold": days_threshold,
    }

    logger.info(
        "Cleanup completed: deleted %d memories, %d errors",
        deleted_count,
        error_count,
    )

    return result


if __name__ == "__main__":
    # Allow running directly from command line
    import sys

    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Get threshold from command line args if provided
    threshold = int(sys.argv[1]) if len(sys.argv) > 1 else 14

    print(f"Running raw memory cleanup with {threshold}-day threshold...")
    result = delete_stale_raw_memories(threshold)
    print(f"Cleanup result: {result}")
