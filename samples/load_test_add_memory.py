#!/usr/bin/env python3
"""
Mirix Load Test - Add Messages

This script performs a load test by adding messages to the Mirix system,
covering all memory types: core, episodic, semantic, procedural, resource, and knowledge vault.

Prerequisites:
- Start the server first: python scripts/start_server.py --reload
- Ensure test message file exists: samples/load_test_messages.txt

Usage:
    python load_test_add_memory.py [num_messages]

    num_messages: Number of messages to add (1-4159, default: 100)

Examples:
    python load_test_add_memory.py           # Add 100 messages
    python load_test_add_memory.py 500       # Add 500 messages
    python load_test_add_memory.py 4159      # Add all available messages
"""

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

from mirix.client import MirixClient

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LoadTestConfig:
    """Configuration for load test."""

    MAX_AVAILABLE_MESSAGES = 4159  # Maximum messages available in file
    DEFAULT_MESSAGE_COUNT = 100  # Default number of messages to add
    BATCH_SIZE = 50  # Process messages in batches
    DELAY_BETWEEN_BATCHES = 1  # seconds

    # Account IDs (Column B)
    ACCOUNT_IDS = [
        "ACC-001",
        "ACC-002",
        "ACC-003",
        "ACC-004",
        "ACC-005",
        "ACC-101",
        "ACC-102",
        "ACC-103",
        "ACC-104",
        "ACC-105",
        "ACC-201",
        "ACC-202",
        "ACC-203",
        "ACC-204",
        "ACC-205",
        "ACC-301",
        "ACC-302",
        "ACC-303",
        "ACC-304",
        "ACC-305",
        "ACC-401",
        "ACC-402",
        "ACC-403",
        "ACC-404",
        "ACC-405",
        "ACC-501",
        "ACC-502",
        "ACC-503",
        "ACC-504",
        "ACC-505",
    ]

    # Regions (Column C)
    REGIONS = [
        "West",
        "East",
        "Central",
        "South",
        "Northeast",
        "Northwest",
        "Southeast",
        "Southwest",
        "Midwest",
        "International",
    ]

    # Product Lines (Column D)
    PRODUCT_LINES = [
        "Enterprise",
        "SMB",
        "Analytics",
        "Financial",
        "Cloud",
        "Security",
        "Marketing",
        "Operations",
        "Integration",
        "Platform",
    ]


def load_messages_from_file(file_path: Path) -> List[Tuple[str, str, str, Dict[str, Any]]]:
    """
    Load messages from file.

    Returns:
        List of tuples: (user_id, memory_type, message_text, filter_tags)
    """
    messages = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue

                # Parse format: user_id|memory_type|message_text|filter_tags_json
                parts = line.split("|", 3)
                if len(parts) != 4:
                    logger.warning("Skipping invalid line %d: %s", line_num, line[:50])
                    continue

                user_id, memory_type, message_text, filter_tags_json = parts

                try:
                    filter_tags = json.loads(filter_tags_json)
                    messages.append((user_id.strip(), memory_type.strip(), message_text.strip(), filter_tags))
                except json.JSONDecodeError as e:
                    logger.warning("Invalid JSON in line %d: %s - Error: %s", line_num, line[:50], e)
                    continue

        logger.info("Loaded %d messages from %s", len(messages), file_path)
        return messages

    except FileNotFoundError:
        logger.error("Message file not found: %s", file_path)
        raise


def generate_occurred_at() -> str:
    """
    Generate a random timestamp within the last 30 days.

    Returns:
        ISO 8601 formatted timestamp string
    """
    now = datetime.now()
    days_ago = random.randint(0, 30)
    hours = random.randint(0, 23)
    minutes = random.randint(0, 59)

    timestamp = now - timedelta(days=days_ago, hours=hours, minutes=minutes)
    return timestamp.strftime("%Y-%m-%dT%H:%M:%S")


def prepare_messages_with_timestamps(
    messages: List[Tuple[str, str, str, Dict[str, Any]]],
) -> List[Tuple[str, str, str, Dict[str, Any], str]]:
    """
    Add timestamps to loaded messages.

    Args:
        messages: List of messages (user_id, memory_type, message_text, filter_tags)

    Returns:
        List of tuples: (user_id, memory_type, message_text, filter_tags, occurred_at)
    """
    messages_with_timestamps = []

    for user_id, memory_type, message_text, filter_tags in messages:
        # Generate timestamp
        occurred_at = generate_occurred_at()
        messages_with_timestamps.append((user_id, memory_type, message_text, filter_tags, occurred_at))

    return messages_with_timestamps


def add_message_to_mirix(
    client: MirixClient, user_id: str, message_text: str, filter_tags: Dict[str, Any], occurred_at: str
) -> bool:
    """
    Add a single message to Mirix system.

    Args:
        client: MirixClient instance
        user_id: User ID
        message_text: Message content
        filter_tags: Filter tags for categorization
        occurred_at: Timestamp when event occurred

    Returns:
        True if successful, False otherwise
    """
    try:
        result = client.add(
            user_id=user_id,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": message_text}]},
                {"role": "assistant", "content": [{"type": "text", "text": "I've saved this information to memory."}]},
            ],
            chaining=True,
            filter_tags=filter_tags,
            occurred_at=occurred_at,
        )
        return result.get("success", False)
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Failed to add message: %s", e)
        return False


def run_load_test(client: MirixClient, messages: List[Tuple[str, str, str, Dict[str, Any], str]], org_id: str):
    """
    Execute load test by adding all messages.

    Args:
        client: MirixClient instance
        messages: List of messages to add (user_id, memory_type, message_text, filter_tags, occurred_at)
        org_id: Organization ID
    """
    total = len(messages)
    success_count = 0
    failure_count = 0
    start_time = time.time()

    # Track unique users
    unique_users = set()
    for user_id, _, _, _, _ in messages:
        unique_users.add(user_id)

    # Create/get all users upfront
    logger.info("Creating/verifying %d users...", len(unique_users))
    for user_id in unique_users:
        try:
            client.create_or_get_user(user_id=user_id, user_name=f"Load Test User {user_id}", org_id=org_id)
            logger.info("  ✓ User ready: %s", user_id)
        except Exception as e:  # pylint: disable=broad-except
            logger.error("  ✗ Failed to create user %s: %s", user_id, e)

    logger.info("=" * 80)
    logger.info("STARTING LOAD TEST")
    logger.info("=" * 80)
    logger.info("Target messages: %d", total)
    logger.info("Users: %d", len(unique_users))
    logger.info("Batch size: %d", LoadTestConfig.BATCH_SIZE)
    logger.info("=" * 80)

    # Process messages in batches
    for batch_num in range(0, total, LoadTestConfig.BATCH_SIZE):
        batch = messages[batch_num : batch_num + LoadTestConfig.BATCH_SIZE]
        batch_start = time.time()
        batch_success = 0

        for user_id, _, message_text, filter_tags, occurred_at in batch:
            if add_message_to_mirix(client, user_id, message_text, filter_tags, occurred_at):
                success_count += 1
                batch_success += 1
            else:
                failure_count += 1

        batch_elapsed = time.time() - batch_start
        progress = (batch_num + len(batch)) / total * 100

        logger.info(
            "Batch %d: %d/%d succeeded in %.2fs | Progress: %.1f%% (%d/%d)",
            batch_num // LoadTestConfig.BATCH_SIZE + 1,
            batch_success,
            len(batch),
            batch_elapsed,
            progress,
            success_count + failure_count,
            total,
        )

        # Delay between batches to avoid overwhelming the server
        if batch_num + LoadTestConfig.BATCH_SIZE < total:
            time.sleep(LoadTestConfig.DELAY_BETWEEN_BATCHES)

    elapsed_time = time.time() - start_time

    # Print summary
    logger.info("=" * 80)
    logger.info("LOAD TEST COMPLETED")
    logger.info("=" * 80)
    logger.info("Total messages processed: %d", total)
    logger.info("Users: %d", len(unique_users))
    logger.info("Successful: %d (%.1f%%)", success_count, success_count / total * 100)
    logger.info("Failed: %d (%.1f%%)", failure_count, failure_count / total * 100)
    logger.info("Total time: %.2f seconds", elapsed_time)
    logger.info("Average rate: %.2f messages/second", total / elapsed_time)
    logger.info("=" * 80)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Mirix Load Test - Add messages to the Mirix system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                 # Add 100 messages (default)
  %(prog)s 500             # Add 500 messages
  %(prog)s 4159            # Add all available messages
  %(prog)s --count 1000    # Add 1000 messages (using flag)
        """,
    )

    parser.add_argument(
        "count",
        nargs="?",
        type=int,
        default=LoadTestConfig.DEFAULT_MESSAGE_COUNT,
        help="Number of messages to add (1-{}, default: {})".format(
            LoadTestConfig.MAX_AVAILABLE_MESSAGES, LoadTestConfig.DEFAULT_MESSAGE_COUNT
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=LoadTestConfig.BATCH_SIZE,
        help=f"Number of messages per batch (default: {LoadTestConfig.BATCH_SIZE})",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=LoadTestConfig.DELAY_BETWEEN_BATCHES,
        help=f"Delay between batches in seconds (default: {LoadTestConfig.DELAY_BETWEEN_BATCHES})",
    )

    args = parser.parse_args()

    # Validate count
    if args.count < 1:
        parser.error("count must be at least 1")
    if args.count > LoadTestConfig.MAX_AVAILABLE_MESSAGES:
        parser.error(
            "count cannot exceed {} (maximum available messages)".format(LoadTestConfig.MAX_AVAILABLE_MESSAGES)
        )

    # Validate batch size
    if args.batch_size < 1:
        parser.error("batch-size must be at least 1")

    # Validate delay
    if args.delay < 0:
        parser.error("delay cannot be negative")

    return args


def main():
    """Main execution function."""
    # Parse command-line arguments
    args = parse_arguments()

    # Override config with command-line arguments
    LoadTestConfig.BATCH_SIZE = args.batch_size
    LoadTestConfig.DELAY_BETWEEN_BATCHES = args.delay

    logger.info("=" * 80)
    logger.info("MIRIX LOAD TEST - ADD MEMORY")
    logger.info("=" * 80)
    logger.info("Messages to add: %d", args.count)
    logger.info("Batch size: %d", LoadTestConfig.BATCH_SIZE)
    logger.info("Delay between batches: %.1fs", LoadTestConfig.DELAY_BETWEEN_BATCHES)
    logger.info("=" * 80)

    # Setup paths
    project_root = Path(__file__).parent.parent
    config_path = project_root / "mirix" / "configs" / "examples" / "mirix_gemini.yaml"
    messages_file = Path(__file__).parent / "load_test_messages.txt"

    # Load messages from file
    logger.info("Loading messages from %s", messages_file)
    try:
        loaded_messages = load_messages_from_file(messages_file)
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Failed to load messages: %s", e)
        sys.exit(1)

    if not loaded_messages:
        logger.error("No valid messages found in file")
        sys.exit(1)

    logger.info("Loaded %d messages", len(loaded_messages))

    # Select subset of messages based on user input
    if args.count < len(loaded_messages):
        logger.info("Selecting %d messages from %d available...", args.count, len(loaded_messages))
        # Randomly select messages to ensure variety across users and memory types
        selected_messages = random.sample(loaded_messages, args.count)
    else:
        logger.info("Using all %d available messages", len(loaded_messages))
        selected_messages = loaded_messages

    # Add timestamps to messages
    logger.info("Adding timestamps to messages...")
    messages = prepare_messages_with_timestamps(selected_messages)
    logger.info("Prepared %d messages with timestamps", len(messages))

    # Count by user and memory type
    user_counts = {}
    memory_type_counts = {}
    for user_id, memory_type, _, _, _ in messages:
        user_counts[user_id] = user_counts.get(user_id, 0) + 1
        memory_type_counts[memory_type] = memory_type_counts.get(memory_type, 0) + 1  # pylint: disable=unused-variable

    logger.info("User distribution:")
    for user, count in sorted(user_counts.items()):
        logger.info("  - %s: %d (%.1f%%)", user, count, count / len(messages) * 100)

    logger.info("Memory type distribution:")
    for mem_type, count in sorted(memory_type_counts.items()):
        logger.info("  - %s: %d (%.1f%%)", mem_type, count, count / len(messages) * 100)

    # Initialize Mirix client
    client_id = "load-test-client"
    org_id = "demo-org"

    logger.info("\nInitializing MirixClient...")
    client = MirixClient(
        api_key=None,
        client_id=client_id,
        client_name="Load Test Client",
        client_scope="Sales",
        org_id=org_id,
        debug=False,
    )
    logger.info("✓ Client initialized")

    # Initialize meta agent
    logger.info("Initializing meta agent from %s", config_path)
    try:
        client.initialize_meta_agent(config_path=str(config_path), update_agents=False)
        logger.info("✓ Meta agent initialized")
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Failed to initialize meta agent: %s", e)
        sys.exit(1)

    # Confirm before starting
    logger.info("\n" + "=" * 80)
    logger.info("READY TO START LOAD TEST")
    num_messages = len(messages)
    num_users = len(user_counts)
    estimated_batches = num_messages / LoadTestConfig.BATCH_SIZE
    estimated_time = estimated_batches * LoadTestConfig.DELAY_BETWEEN_BATCHES / 60
    logger.info("Messages to add: %d", num_messages)
    logger.info("Users involved: %d", num_users)
    logger.info("Estimated time: ~%.1f minutes", estimated_time)
    logger.info("=" * 80)

    response = input("\nProceed with load test? (yes/no): ")
    if response.lower() != "yes":
        logger.info("Load test cancelled by user")
        sys.exit(0)

    # Run load test
    try:
        run_load_test(client, messages, org_id)
    except KeyboardInterrupt:
        logger.info("\n\nLoad test interrupted by user")
        sys.exit(1)
    except Exception as e:  # pylint: disable=broad-except
        logger.error("\n\nLoad test failed with error: %s", e)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
