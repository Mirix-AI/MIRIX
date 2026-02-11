#!/usr/bin/env python3
"""
Mirix Memory Viewer - Demonstrates how to retrieve and visualize memories
for a specific user using the MirixClient API.

Prerequisites:
- Start the server first: python scripts/start_server.py --reload
- Or set MIRIX_API_URL environment variable to your server URL
"""

import logging
import os
from pathlib import Path

# Load .env file (optional - Mirix now loads .env automatically in mirix/settings.py)
from dotenv import load_dotenv  # noqa: E402

current_dir = os.path.dirname(os.path.abspath(__file__))
mirix_root = os.path.dirname(current_dir)
load_dotenv(os.path.join(mirix_root, ".env"))

# Import MirixClient
from mirix import MirixClient  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Main function to demonstrate memory retrieval and visualization."""

    # 1. Create MirixClient (represents the client application)
    client_id = "demo-client-app"
    org_id = "demo-org"
    user_id = "demo-user"  # End-user ID

    logger.info("Initializing MirixClient...")
    client = MirixClient(
        api_key=None,  # TODO: add authentication later
        client_id=client_id,
        org_id=org_id,
        debug=True,
    )

    # 2. Initialize meta agent (no user_id parameter)
    logger.info("Initializing meta agent...")
    config_path = Path(mirix_root) / "mirix" / "configs" / "examples" / "mirix_gemini.yaml"

    client.initialize_meta_agent(
        config_path=str(config_path),
        update_agents=False,
    )
    logger.info("✅ Meta agent initialized")

    # 3. Ensure user exists
    logger.info(f"Creating/getting user: {user_id}")
    user_id = client.create_or_get_user(user_id=user_id, user_name="Alice")
    logger.info(f"✅ User ready: {user_id}")

    # 4. Retrieve memories using conversation context
    logger.info("\nRetrieving memories for user...")
    print("=" * 80)

    try:
        # Retrieve all memories by providing a general query
        memories = client.retrieve_with_conversation(
            user_id=user_id,
            messages=[{"role": "user", "content": [{"type": "text", "text": "Show me all my memories"}]}],
            limit=20,  # Retrieve up to 20 items per memory type
        )

        if not memories.get("success"):
            logger.error("❌ Failed to retrieve memories")
            return

        logger.info("✅ Retrieved memories successfully\n")

        # 5. Display summary
        memory_data = memories.get("memories", {})
        print("\n" + "=" * 80)
        print("MEMORY SUMMARY")
        print("=" * 80)

        for memory_type, data in memory_data.items():
            if data:
                total_count = data.get("total_count", 0)
                print(f"  {memory_type.upper():<20}: {total_count} items")

        # 6. Display detailed memories by type
        print("\n" + "=" * 80)
        print("DETAILED MEMORIES")
        print("=" * 80)

        for memory_type, data in memory_data.items():
            if not data or data.get("total_count", 0) == 0:
                continue

            print(f"\n{'─' * 80}")
            print(f"📋 {memory_type.upper()} MEMORY (Total: {data['total_count']})")
            print(f"{'─' * 80}")

            # Get items (handle different structures)
            items = data.get("items", [])

            # Episodic memory has 'recent' and 'relevant' instead of 'items'
            if not items and memory_type == "episodic":
                recent = data.get("recent", [])
                relevant = data.get("relevant", [])
                # Merge and deduplicate by ID
                seen_ids = set()
                items = []
                for item in recent + relevant:
                    if item.get("id") not in seen_ids:
                        items.append(item)
                        seen_ids.add(item.get("id"))

            # Display each memory item
            for i, item in enumerate(items, 1):
                print(f"\n  [{i}] ID: {item.get('id', 'N/A')}")

                # Display key fields based on memory type
                if memory_type == "episodic":
                    print(f"      Content: {item.get('content', 'N/A')[:100]}...")
                    print(f"      Created: {item.get('created_at', 'N/A')}")

                elif memory_type == "semantic":
                    print(f"      Name: {item.get('name', 'N/A')}")
                    print(f"      Content: {item.get('content', 'N/A')[:100]}...")

                elif memory_type == "procedural":
                    print(f"      Name: {item.get('name', 'N/A')}")
                    print(f"      Procedure: {item.get('procedure_text', 'N/A')[:100]}...")

                elif memory_type == "resources":
                    print(f"      Title: {item.get('title', 'N/A')}")
                    print(f"      Type: {item.get('resource_type', 'N/A')}")
                    print(f"      Summary: {item.get('summary', 'N/A')[:100]}...")

                elif memory_type == "knowledge":
                    print(f"      Name: {item.get('name', 'N/A')}")
                    print(f"      Value: {str(item.get('value', 'N/A'))[:100]}...")

                # Show filter tags if present
                if item.get("filter_tags"):
                    print(f"      Tags: {item.get('filter_tags')}")

        print("\n" + "=" * 80)
        logger.info("✅ Memory visualization complete")

    except Exception as e:
        logger.error(f"❌ Error retrieving memories: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
