"""
Create mock raw_memory data and linked episodic memories for testing.
Run this script to populate your database with test data.

Usage:
    python scripts/create_raw_memory_mock_data.py
"""

import uuid
from datetime import datetime, timezone, timedelta

from mirix.orm.episodic_memory import EpisodicEvent
from mirix.orm.semantic_memory import SemanticMemoryItem
from mirix.schemas.user import User as PydanticUser
from mirix.server.server import db_context
from mirix.services.raw_memory_manager import RawMemoryManager


def create_mock_data():
    """Create comprehensive mock data for testing raw_memory references."""

    print("🚀 Creating mock raw_memory test data...\n")

    # Get or create test user
    with db_context() as session:
        from mirix.orm.user import User as OrmUser

        test_user_orm = session.query(OrmUser).filter(
            OrmUser.name == "Test User"
        ).first()

        if not test_user_orm:
            test_user_orm = OrmUser(
                id=f"user-{uuid.uuid4()}",
                name="Test User",
                organization_id=f"org-{uuid.uuid4()}",
                timezone="UTC",
                status="active"
            )
            session.add(test_user_orm)
            session.commit()
            session.refresh(test_user_orm)
            print(f"✅ Created test user: {test_user_orm.id}")
        else:
            print(f"✅ Using existing test user: {test_user_orm.id}")

    # Convert to Pydantic
    test_user = PydanticUser(
        id=test_user_orm.id,
        name=test_user_orm.name,
        organization_id=test_user_orm.organization_id,
        timezone=test_user_orm.timezone,
        status=test_user_orm.status
    )

    # Create raw memory manager
    raw_memory_manager = RawMemoryManager()

    # Scenario 1: Chrome - GitHub coding session
    print("\n📸 Scenario 1: GitHub Coding Session")
    raw_mem_github = raw_memory_manager.insert_raw_memory(
        actor=test_user,
        screenshot_path="/fake/screenshots/github_mirix_20251218_103000.png",
        source_app="Chrome",
        captured_at=datetime(2025, 12, 18, 10, 30, 0, tzinfo=timezone.utc),
        ocr_text="MIRIX Repository - Phase 1 Implementation\n\nFiles changed:\n- mirix/agent/agent.py: Added raw_memory_references collection\n- mirix/server/fastapi_server.py: Updated streaming endpoint\n- frontend/src/components/ChatBubble.js: Added memory badges\n\nCommit: Implement raw_memory reference pipeline",
        source_url="https://github.com/user/mirix/commit/abc123",
    )
    print(f"  ✅ Created: {raw_mem_github.id}")
    print(f"  📱 App: {raw_mem_github.source_app}")
    print(f"  🔗 URL: {raw_mem_github.source_url}")
    print(f"  📅 Captured: {raw_mem_github.captured_at}")

    # Scenario 2: Safari - Python documentation
    print("\n📸 Scenario 2: Python Documentation Research")
    raw_mem_python = raw_memory_manager.insert_raw_memory(
        actor=test_user,
        screenshot_path="/fake/screenshots/python_docs_20251217_154500.png",
        source_app="Safari",
        captured_at=datetime(2025, 12, 17, 15, 45, 0, tzinfo=timezone.utc),
        ocr_text="asyncio — Asynchronous I/O\n\nCoroutines and Tasks\n\nCoroutines declared with async/await syntax are the preferred way of writing asyncio applications.\n\nExample:\nimport asyncio\n\nasync def main():\n    print('Hello')\n    await asyncio.sleep(1)\n    print('World')\n\nasyncio.run(main())",
        source_url="https://docs.python.org/3/library/asyncio-task.html",
    )
    print(f"  ✅ Created: {raw_mem_python.id}")
    print(f"  📱 App: {raw_mem_python.source_app}")
    print(f"  🔗 URL: {raw_mem_python.source_url}")

    # Scenario 3: Notion - Project planning
    print("\n📸 Scenario 3: Project Planning in Notion")
    raw_mem_notion = raw_memory_manager.insert_raw_memory(
        actor=test_user,
        screenshot_path="/fake/screenshots/notion_planning_20251216_090000.png",
        source_app="Notion",
        captured_at=datetime(2025, 12, 16, 9, 0, 0, tzinfo=timezone.utc),
        ocr_text="MIRIX Memory Architecture - Phase 1 Tasks\n\n✅ Task 7: Modify system prompts to include source information\n✅ Task 8: Create GET /memory/raw/{id} endpoint\n🔄 Task 9: Frontend display memory references\n⏳ Task 15: Test integration\n\nNotes:\n- Raw memory stores original screenshots\n- Higher-level memories reference raw_memory via JSON array\n- Frontend displays purple badges with app icons",
        source_url="notion.so/workspace/mirix-project-planning",
    )
    print(f"  ✅ Created: {raw_mem_notion.id}")
    print(f"  📱 App: {raw_mem_notion.source_app}")
    print(f"  🔗 URL: {raw_mem_notion.source_url}")

    # Scenario 4: Firefox - Stack Overflow debugging
    print("\n📸 Scenario 4: Stack Overflow Debugging")
    raw_mem_stackoverflow = raw_memory_manager.insert_raw_memory(
        actor=test_user,
        screenshot_path="/fake/screenshots/stackoverflow_20251218_140000.png",
        source_app="Firefox",
        captured_at=datetime(2025, 12, 18, 14, 0, 0, tzinfo=timezone.utc),
        ocr_text="How to properly handle SQLAlchemy sessions in FastAPI?\n\nQuestion: I'm getting 'Object is already attached to session' errors...\n\nAnswer (Accepted): You should use session.refresh() after commit or create a new session. Here's the pattern:\n\nwith db_context() as session:\n    obj = session.add(new_obj)\n    session.commit()\n    session.refresh(obj)  # Important!\n    return obj",
        source_url="https://stackoverflow.com/questions/12345/sqlalchemy-session-fastapi",
    )
    print(f"  ✅ Created: {raw_mem_stackoverflow.id}")
    print(f"  📱 App: {raw_mem_stackoverflow.source_app}")
    print(f"  🔗 URL: {raw_mem_stackoverflow.source_url}")

    # Create episodic memories referencing these screenshots
    print("\n💾 Creating Episodic Memories with References...")

    with db_context() as session:
        # Episodic 1: Coding session (references GitHub + Stack Overflow)
        episodic_1 = EpisodicEvent(
            id=f"episodic-{uuid.uuid4()}",
            summary="Worked on MIRIX Phase 1 implementation",
            details="Implemented raw_memory reference pipeline. Fixed SQLAlchemy session issues by using refresh after commit. Updated frontend ChatBubble to display memory badges with purple gradient styling.",
            event_type="work",
            occurred_at=datetime(2025, 12, 18, 14, 30, 0, tzinfo=timezone.utc),
            actor="user",
            user_id=test_user.id,
            organization_id=test_user.organization_id,
            raw_memory_references=[raw_mem_github.id, raw_mem_stackoverflow.id],
            tree_path=["work", "coding", "mirix", "phase1"]
        )
        session.add(episodic_1)
        print(f"  ✅ Episodic Memory 1: {episodic_1.id}")
        print(f"     References: {len(episodic_1.raw_memory_references)} screenshots")

        # Episodic 2: Research session (references Python docs + Notion)
        episodic_2 = EpisodicEvent(
            id=f"episodic-{uuid.uuid4()}",
            summary="Researched Python async patterns for MIRIX",
            details="Studied asyncio documentation to understand async/await patterns. Documented findings in Notion for future reference. Planning to implement async processing for memory agents.",
            event_type="research",
            occurred_at=datetime(2025, 12, 17, 16, 0, 0, tzinfo=timezone.utc),
            actor="user",
            user_id=test_user.id,
            organization_id=test_user.organization_id,
            raw_memory_references=[raw_mem_python.id, raw_mem_notion.id],
            tree_path=["learning", "python", "async"]
        )
        session.add(episodic_2)
        print(f"  ✅ Episodic Memory 2: {episodic_2.id}")
        print(f"     References: {len(episodic_2.raw_memory_references)} screenshots")

        # Episodic 3: Planning (references only Notion)
        episodic_3 = EpisodicEvent(
            id=f"episodic-{uuid.uuid4()}",
            summary="Planned MIRIX Phase 1 tasks in Notion",
            details="Created comprehensive task breakdown for Phase 1 implementation. Prioritized memory reference features and frontend integration.",
            event_type="planning",
            occurred_at=datetime(2025, 12, 16, 9, 30, 0, tzinfo=timezone.utc),
            actor="user",
            user_id=test_user.id,
            organization_id=test_user.organization_id,
            raw_memory_references=[raw_mem_notion.id],
            tree_path=["work", "planning", "mirix"]
        )
        session.add(episodic_3)
        print(f"  ✅ Episodic Memory 3: {episodic_3.id}")
        print(f"     References: {len(episodic_3.raw_memory_references)} screenshot")

        session.commit()

    # Create semantic memory for bonus testing
    print("\n💾 Creating Semantic Memory with References...")

    with db_context() as session:
        semantic_1 = SemanticMemoryItem(
            id=f"semantic-{uuid.uuid4()}",
            name="Python Async/Await Patterns",
            summary="Comprehensive understanding of Python asyncio programming",
            details="Async/await syntax is the preferred way of writing asyncio applications. Key concepts: coroutines, tasks, event loops, concurrent execution.",
            source="Python Official Documentation",
            user_id=test_user.id,
            organization_id=test_user.organization_id,
            raw_memory_references=[raw_mem_python.id],
            tree_path=["knowledge", "python", "async"]
        )
        session.add(semantic_1)
        session.commit()
        print(f"  ✅ Semantic Memory: {semantic_1.id}")
        print(f"     References: {len(semantic_1.raw_memory_references)} screenshot")

    print("\n" + "="*60)
    print("✅ Mock data creation complete!")
    print("="*60)

    print("\n📊 Summary:")
    print(f"  • Created 4 raw_memory items")
    print(f"  • Created 3 episodic memories with references")
    print(f"  • Created 1 semantic memory with reference")
    print(f"  • User ID: {test_user.id}")

    print("\n🔍 Raw Memory IDs for testing:")
    print(f"  • GitHub (Chrome):      {raw_mem_github.id}")
    print(f"  • Python (Safari):      {raw_mem_python.id}")
    print(f"  • Notion:               {raw_mem_notion.id}")
    print(f"  • Stack Overflow (Firefox): {raw_mem_stackoverflow.id}")

    return {
        "user": test_user,
        "raw_memories": {
            "github": raw_mem_github,
            "python": raw_mem_python,
            "notion": raw_mem_notion,
            "stackoverflow": raw_mem_stackoverflow,
        },
        "episodic": [episodic_1, episodic_2, episodic_3],
        "semantic": [semantic_1]
    }


if __name__ == "__main__":
    try:
        data = create_mock_data()
        print("\n✨ Ready for testing! See RAW_MEMORY_TESTING_GUIDE.md for next steps.")
    except Exception as e:
        print(f"\n❌ Error creating mock data: {e}")
        import traceback
        traceback.print_exc()
