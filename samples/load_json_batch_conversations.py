#!/usr/bin/env python3
"""
Load Sales Samples into Mirix

This script reads conversation data from a JSON file and loads it into Mirix.
The JSON file should contain an array of conversation records, each with:
- user_id: The user identifier
- messages: Array of message objects with role and content
- filter_tags: Dictionary of tags for filtering (seller_id, conversation_id, account_id, etc.)
- occurred_at: ISO 8601 timestamp when the conversation occurred

Prerequisites:
- Start the server first: python scripts/start_server.py --reload

Usage:
    python load_sales_samples.py <json_file> [options]

Examples:
    python load_sales_samples.py /path/to/sample_input.json
    python load_sales_samples.py data.json --batch-size 20 --delay 0.5
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from mirix.client import MirixClient

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LoadConfig:
    """Configuration for loading sales samples."""

    DEFAULT_BATCH_SIZE = 10
    DEFAULT_DELAY = 0.5  # seconds between batches
    CLIENT_ID = "sales-loader-client"
    ORG_ID = "demo-org"


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Load sales conversation samples into Mirix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/sample_input.json
  %(prog)s data.json --batch-size 20
  %(prog)s data.json --batch-size 5 --delay 1.0
        """,
    )

    parser.add_argument("json_file", type=str, help="Path to JSON file containing conversation data")

    parser.add_argument(
        "--batch-size",
        type=int,
        default=LoadConfig.DEFAULT_BATCH_SIZE,
        help="Number of conversations to process per batch (default: {})".format(LoadConfig.DEFAULT_BATCH_SIZE),
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=LoadConfig.DEFAULT_DELAY,
        help="Delay between batches in seconds (default: {})".format(LoadConfig.DEFAULT_DELAY),
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Parse and validate JSON without actually adding to Mirix"
    )

    args = parser.parse_args()

    # Validate file exists
    if not Path(args.json_file).exists():
        parser.error("JSON file not found: {}".format(args.json_file))

    # Validate batch size
    if args.batch_size < 1:
        parser.error("batch-size must be at least 1")

    # Validate delay
    if args.delay < 0:
        parser.error("delay cannot be negative")

    return args


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load and validate JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        List of conversation records

    Raises:
        ValueError: If JSON is invalid or missing required fields
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError("Invalid JSON file: {}".format(e)) from e
    except Exception as e:
        raise ValueError("Failed to read file: {}".format(e)) from e

    if not isinstance(data, list):
        raise ValueError("JSON file must contain an array of conversation records")

    # Validate each record
    for i, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError("Record {} is not a valid object".format(i))

        required_fields = ["user_id", "messages", "filter_tags", "occurred_at"]
        for field in required_fields:
            if field not in record:
                raise ValueError("Record {} missing required field: {}".format(i, field))

        if not isinstance(record["messages"], list):
            raise ValueError("Record {}: 'messages' must be an array".format(i))

        if not isinstance(record["filter_tags"], dict):
            raise ValueError("Record {}: 'filter_tags' must be an object".format(i))

    return data


def transform_message_format(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transform message format from JSON to Mirix format.

    JSON format:
        {"role": "assistant", "content": "text here"}
        {"role": "human", "content": "text here"}

    Mirix format:
        {"role": "assistant", "content": [{"type": "text", "text": "text here"}]}
        {"role": "user", "content": [{"type": "text", "text": "text here"}]}

    Args:
        messages: List of messages in JSON format

    Returns:
        List of messages in Mirix format
    """
    transformed = []

    for msg in messages:
        # Map "human" role to "user" role for Mirix
        role = msg.get("role", "")
        if role == "human":
            role = "user"

        # Get content
        content = msg.get("content", "")

        # Check if content is already in Mirix format (a list)
        if isinstance(content, list):
            # Already in Mirix format, use as-is
            transformed_msg = {"role": role, "content": content}
        else:
            # Content is a string, transform to Mirix format
            transformed_msg = {"role": role, "content": [{"type": "text", "text": content}]}
        transformed.append(transformed_msg)

    return transformed


def add_conversation_to_mirix(
    client: MirixClient, user_id: str, messages: List[Dict[str, Any]], filter_tags: Dict[str, Any], occurred_at: str
) -> bool:
    """
    Add a single conversation to Mirix.

    Args:
        client: MirixClient instance
        user_id: User ID
        messages: List of messages (already transformed to Mirix format)
        filter_tags: Filter tags for categorization
        occurred_at: ISO 8601 timestamp

    Returns:
        True if successful, False otherwise
    """
    try:
        result = client.add(
            user_id=user_id, messages=messages, chaining=True, filter_tags=filter_tags, occurred_at=occurred_at
        )
        return result.get("success", False)
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Failed to add conversation for user %s: %s", user_id, e)
        return False


def process_conversations(
    client: MirixClient, conversations: List[Dict[str, Any]], batch_size: int, delay: float, dry_run: bool = False
):
    """
    Process and add conversations to Mirix.

    Args:
        client: MirixClient instance
        conversations: List of conversation records
        batch_size: Number of conversations per batch
        delay: Delay between batches in seconds
        dry_run: If True, only validate without adding
    """
    total = len(conversations)
    success_count = 0
    failure_count = 0
    start_time = time.time()

    # Collect unique users
    unique_users = set()
    for conv in conversations:
        unique_users.add(conv["user_id"])

    logger.info("=" * 80)
    logger.info("PROCESSING SALES CONVERSATIONS")
    logger.info("=" * 80)
    logger.info("Total conversations: %d", total)
    logger.info("Unique users: %d", len(unique_users))
    logger.info("Batch size: %d", batch_size)
    logger.info("Dry run: %s", dry_run)
    logger.info("=" * 80)

    if dry_run:
        logger.info("DRY RUN - Validating data only, not adding to Mirix")
        logger.info("=" * 80)

    # Create/verify users upfront (skip in dry run)
    if not dry_run:
        logger.info("Creating/verifying %d users...", len(unique_users))
        for user_id in unique_users:
            try:
                client.create_or_get_user(
                    user_id=user_id, user_name="Sales User {}".format(user_id), org_id=LoadConfig.ORG_ID
                )
                logger.info("  ✓ User ready: %s", user_id)
            except Exception as e:  # pylint: disable=broad-except
                logger.error("  ✗ Failed to create user %s: %s", user_id, e)

    # Process conversations in batches
    for batch_num in range(0, total, batch_size):
        batch = conversations[batch_num : batch_num + batch_size]
        batch_start = time.time()
        batch_success = 0

        for conv in batch:
            # Transform messages to Mirix format
            transformed_messages = transform_message_format(conv["messages"])

            if dry_run:
                # Just validate the transformation
                logger.debug(
                    "Would add conversation for user %s with %d messages", conv["user_id"], len(transformed_messages)
                )
                success_count += 1
                batch_success += 1
            else:
                # Actually add to Mirix
                if add_conversation_to_mirix(
                    client, conv["user_id"], transformed_messages, conv["filter_tags"], conv["occurred_at"]
                ):
                    success_count += 1
                    batch_success += 1
                    logger.info("Waiting 20 seconds for memory processing...")
                    time.sleep(20)
                else:
                    failure_count += 1

        batch_elapsed = time.time() - batch_start
        progress = (batch_num + len(batch)) / total * 100

        logger.info(
            "Batch %d: %d/%d succeeded in %.2fs | Progress: %.1f%% (%d/%d)",
            batch_num // batch_size + 1,
            batch_success,
            len(batch),
            batch_elapsed,
            progress,
            success_count + failure_count,
            total,
        )

        # Delay between batches
        if batch_num + batch_size < total and not dry_run:
            time.sleep(delay)

    elapsed_time = time.time() - start_time

    # Print summary
    logger.info("=" * 80)
    logger.info("PROCESSING COMPLETED")
    logger.info("=" * 80)
    logger.info("Total conversations: %d", total)
    logger.info("Successful: %d (%.1f%%)", success_count, success_count / total * 100)
    if not dry_run:
        logger.info("Failed: %d (%.1f%%)", failure_count, failure_count / total * 100)
    logger.info("Total time: %.2f seconds", elapsed_time)
    logger.info("Average rate: %.2f conversations/second", total / elapsed_time)
    logger.info("=" * 80)


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    logger.info("=" * 80)
    logger.info("MIRIX SALES SAMPLES LOADER")
    logger.info("=" * 80)
    logger.info("JSON file: %s", args.json_file)
    logger.info("=" * 80)

    # Load and validate JSON
    logger.info("Loading JSON file...")
    try:
        conversations = load_json_file(args.json_file)
        logger.info("✓ Loaded %d conversation records", len(conversations))
    except ValueError as e:
        logger.error("✗ %s", e)
        sys.exit(1)
    except Exception as e:  # pylint: disable=broad-except
        logger.error("✗ Unexpected error: %s", e)
        sys.exit(1)

    if not conversations:
        logger.warning("No conversations found in JSON file")
        sys.exit(0)

    # Show sample data
    logger.info("\nSample conversation:")
    sample = conversations[0]
    logger.info("  User ID: %s", sample["user_id"])
    logger.info("  Messages: %d", len(sample["messages"]))
    logger.info("  Filter tags: %s", sample["filter_tags"])
    logger.info("  Occurred at: %s", sample["occurred_at"])

    # Initialize client (skip if dry run)
    client = None
    if not args.dry_run:
        logger.info("\nInitializing MirixClient...")
        try:
            client = MirixClient(
                api_key=None,
                client_id=LoadConfig.CLIENT_ID,
                client_name="Sales Samples Loader",
                client_scope="Sales",
                org_id=LoadConfig.ORG_ID,
                debug=False,
            )
            logger.info("✓ Client initialized")

            # Initialize meta agent
            logger.info("Initializing meta agent...")
            project_root = Path(__file__).parent.parent
            config_path = project_root / "mirix" / "configs" / "examples" / "mirix_gemini.yaml"

            client.initialize_meta_agent(config_path=str(config_path), update_agents=False)
            logger.info("✓ Meta agent initialized")
        except Exception as e:  # pylint: disable=broad-except
            logger.error("✗ Failed to initialize: %s", e)
            sys.exit(1)

    # Confirm before processing
    logger.info("\n" + "=" * 80)
    logger.info("READY TO PROCESS")
    logger.info("Conversations to add: %d", len(conversations))
    if args.dry_run:
        logger.info("Mode: DRY RUN (validation only)")
    else:
        logger.info("Mode: LIVE (will add to Mirix)")
    logger.info("=" * 80)

    if not args.dry_run:
        response = input("\nProceed with loading? (yes/no): ")
        if response.lower() != "yes":
            logger.info("Loading cancelled by user")
            sys.exit(0)

    # Process conversations
    try:
        process_conversations(client, conversations, args.batch_size, args.delay, args.dry_run)
    except KeyboardInterrupt:
        logger.info("\n\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:  # pylint: disable=broad-except
        logger.error("\n\nProcessing failed with error: %s", e)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
