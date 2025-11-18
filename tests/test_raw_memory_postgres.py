#!/usr/bin/env python3
"""
Test script for raw_memory with PostgreSQL database.

This script tests with the PostgreSQL database instead of SQLite.
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Force PostgreSQL mode and disable embeddings for faster testing
os.environ.setdefault("MIRIX_PG_URI", "postgresql+pg8000://power@localhost:5432/mirix")
os.environ["BUILD_EMBEDDINGS_FOR_MEMORY"] = "false"

from mirix.helpers.ocr_url_extractor import OCRUrlExtractor
from mirix.services.raw_memory_manager import RawMemoryManager
from mirix.schemas.user import User as PydanticUser


def test_raw_memory_storage():
    """Test writing and reading from raw_memory table in PostgreSQL."""
    print("\n" + "=" * 80)
    print("TEST: Raw Memory PostgreSQL Database Storage")
    print("=" * 80)

    try:
        # Import database dependencies
        from mirix.server.server import db_context
        from mirix.orm import Organization
        from mirix.orm.user import User

        # Create manager
        raw_memory_manager = RawMemoryManager()

        print("\n✅ RawMemoryManager initialized")

        # Get or create test user
        with db_context() as session:
            # Try to get first user
            user = session.query(User).first()

            if not user:
                # Create test organization
                org = Organization(
                    id="test-org-postgres-123",
                    name="Test Organization PostgreSQL"
                )
                session.add(org)

                # Create test user
                user = User(
                    id="test-user-postgres-123",
                    name="Test User PostgreSQL",
                    timezone="UTC",
                    organization_id=org.id
                )
                session.add(user)
                session.commit()
                print(f"✅ Created test user: {user.id}")
            else:
                print(f"✅ Using existing user: {user.id}")

            # Convert to Pydantic model
            pydantic_user = PydanticUser(
                id=user.id,
                name=user.name,
                timezone=user.timezone,
                organization_id=user.organization_id
            )

        # Test data with real screenshot paths
        test_screenshots = [
            {
                "screenshot_path": "/Users/power/.mirix/tmp/images/screenshot-2025-08-22T08-33-13-376Z.png",
                "source_app": "Chrome",
                "ocr_text": "deepwiki.com wujiabao77/shipany-template-one | DeepWiki new.web.cafe Web.Cafe",
                "urls": ["https://deepwiki.com", "https://new.web.cafe", "https://Web.Cafe"]
            },
            {
                "screenshot_path": "/Users/power/.mirix/tmp/images/screenshot-2025-09-05T06-30-37-992Z.png",
                "source_app": "Safari",
                "ocr_text": "youtube.com/watch?v=VDREHIOd80k usemotion.com",
                "urls": ["https://youtube.com/watch?v=VDREHIOd80k", "https://usemotion.com"]
            },
            {
                "screenshot_path": "/Users/power/.mirix/tmp/images/test-imagilabs.jpg",
                "source_app": "Firefox",
                "ocr_text": "imagilabs.com SHOP NOW Bring code to life with imagiCharms",
                "urls": ["https://imagilabs.com"]
            }
        ]

        print(f"\n📝 Inserting {len(test_screenshots)} test records into PostgreSQL...")
        inserted_ids = []

        for i, test_data in enumerate(test_screenshots, 1):
            print(f"\n  Record {i}:")
            print(f"    App: {test_data['source_app']}")
            print(f"    Path: {test_data['screenshot_path']}")
            print(f"    Text: {test_data['ocr_text'][:60]}...")
            print(f"    URLs: {test_data['urls']}")

            raw_memory = raw_memory_manager.insert_raw_memory(
                actor=pydantic_user,
                screenshot_path=test_data["screenshot_path"],
                source_app=test_data["source_app"],
                captured_at=datetime.now(timezone.utc),
                ocr_text=test_data["ocr_text"],
                source_url=test_data["urls"][0] if test_data["urls"] else None,
                metadata={
                    "test_urls": test_data["urls"],
                    "test_index": i,
                    "test_run": "postgres_migration_validation"
                }
            )

            inserted_ids.append(raw_memory.id)
            print(f"    ✅ Inserted with ID: {raw_memory.id}")

        print(f"\n✅ Successfully inserted {len(inserted_ids)} records into PostgreSQL")

        # Test retrieval
        print(f"\n📖 Testing retrieval from PostgreSQL...")

        for raw_memory_id in inserted_ids:
            retrieved = raw_memory_manager.get_raw_memory_by_id(
                raw_memory_id=raw_memory_id,
                user_id=pydantic_user.id
            )

            if retrieved:
                print(f"  ✅ Retrieved {raw_memory_id}:")
                print(f"     App: {retrieved.source_app}")
                print(f"     URL: {retrieved.source_url}")
                print(f"     OCR: {retrieved.ocr_text[:60] if retrieved.ocr_text else 'None'}...")
                print(f"     Has Embedding: {'Yes' if retrieved.ocr_text_embedding else 'No'}")
            else:
                print(f"  ❌ Failed to retrieve {raw_memory_id}")
                return False

        # Test bulk retrieval
        print(f"\n📦 Testing bulk retrieval from PostgreSQL...")
        bulk_retrieved = raw_memory_manager.get_raw_memories_by_ids(
            raw_memory_ids=inserted_ids,
            user_id=pydantic_user.id
        )

        print(f"  ✅ Retrieved {len(bulk_retrieved)} records in bulk")

        # Test filtering by source_app
        print(f"\n🔍 Testing filter by source_app...")
        chrome_memories = raw_memory_manager.get_raw_memories_by_source_app(
            user_id=pydantic_user.id,
            organization_id=pydantic_user.organization_id,
            source_app="Chrome"
        )
        print(f"  ✅ Found {len(chrome_memories)} Chrome screenshots")

        # Clean up test data
        print(f"\n🧹 Cleaning up test data...")
        for raw_memory_id in inserted_ids:
            raw_memory_manager.delete_raw_memory(raw_memory_id)
        print(f"  ✅ Deleted {len(inserted_ids)} test records")

        print(f"\n{'=' * 80}")
        print("✅ All PostgreSQL database storage tests PASSED")
        print(f"{'=' * 80}")

        return True

    except Exception as e:
        print(f"\n❌ PostgreSQL database storage test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run PostgreSQL tests."""
    print("\n" + "=" * 80)
    print("🧪 MIRIX Raw Memory PostgreSQL Test")
    print("=" * 80)

    success = test_raw_memory_storage()

    print("\n" + "=" * 80)
    if success:
        print("🎉 POSTGRESQL TEST PASSED!")
    else:
        print("⚠️  POSTGRESQL TEST FAILED")
    print("=" * 80 + "\n")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
