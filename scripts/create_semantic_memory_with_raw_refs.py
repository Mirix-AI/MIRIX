"""
Create a semantic memory that references existing raw_memory items.
This allows testing the frontend display of semantic memories with raw_memory references.

Usage:
    BUILD_EMBEDDINGS_FOR_MEMORY=false PYTHONPATH=. python scripts/create_semantic_memory_with_raw_refs.py
"""

import os
import uuid
from pathlib import Path

# Load .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"📝 Loaded environment from {env_path}")
except ImportError:
    print("⚠️  python-dotenv not installed")

from mirix.orm.raw_memory import RawMemoryItem
from mirix.orm.semantic_memory import SemanticMemoryItem
from mirix.server.server import db_context


def create_semantic_memory_with_refs():
    """Create a semantic memory referencing the most recent raw_memory items."""

    print("🚀 Creating semantic memory with raw_memory references...\n")

    with db_context() as session:
        # Get the most recent raw_memory items
        raw_memories = session.query(RawMemoryItem).order_by(
            RawMemoryItem.created_at.desc()
        ).limit(4).all()

        if not raw_memories:
            print("❌ No raw_memory items found in database!")
            print("   Please run: python scripts/create_raw_memory_mock_data_simple.py")
            return None

        print(f"✅ Found {len(raw_memories)} recent raw_memory items:")
        for rm in raw_memories:
            print(f"   • {rm.source_app}: {rm.source_url}")

        # Get the user from the first raw_memory
        user_id = raw_memories[0].user_id
        org_id = raw_memories[0].organization_id

        print(f"\n👤 User: {user_id}")

        # Collect raw_memory IDs
        raw_memory_ids = [rm.id for rm in raw_memories]

        # Create semantic memory
        semantic_memory = SemanticMemoryItem(
            id=f"semantic-{uuid.uuid4()}",
            name="MIRIX Phase 1 Development Knowledge",
            summary="Comprehensive understanding of MIRIX Phase 1 implementation including raw_memory pipeline, async patterns, and frontend integration",
            details="""Key learnings from MIRIX Phase 1 development:

1. **Raw Memory Architecture**: Implemented a foundational memory layer that stores screenshots with OCR-extracted text. Higher-level memories (episodic, semantic) reference raw_memory via JSON arrays.

2. **Python Async Patterns**: Studied asyncio documentation for async/await patterns. Key concepts include coroutines, tasks, event loops, and concurrent execution. This knowledge is essential for implementing async memory processing agents.

3. **Frontend Integration**: Created ChatBubble component with purple gradient memory badges displaying app icons (Chrome, Safari, Firefox, Notion), URLs, capture dates, and OCR text previews.

4. **Database Design**: Used PostgreSQL for production with proper foreign key constraints. Raw memory references are stored as JSON arrays in higher-level memory tables.

5. **Agent System Prompt**: Modified agent to collect raw_memory_references during retrieval and format source information in system prompts as [Sources: Chrome: github.com, ...].

This knowledge combines insights from GitHub code reviews, Python documentation research, project planning in Notion, and Stack Overflow debugging sessions.""",
            source="Multiple sources: GitHub, Python Docs, Notion, Stack Overflow",
            user_id=user_id,
            organization_id=org_id,
            raw_memory_references=raw_memory_ids,
            tree_path=["knowledge", "mirix", "phase1", "implementation"]
        )

        session.add(semantic_memory)
        session.commit()
        session.refresh(semantic_memory)

        print(f"\n✅ Created semantic memory:")
        print(f"   ID: {semantic_memory.id}")
        print(f"   Name: {semantic_memory.name}")
        print(f"   References: {len(semantic_memory.raw_memory_references)} raw_memory items")
        print(f"   User: {semantic_memory.user_id}")

        print("\n" + "="*70)
        print("✅ Semantic memory created successfully!")
        print("="*70)

        print("\n📋 Referenced Raw Memory Items:")
        for rm in raw_memories:
            print(f"   • {rm.source_app}: {rm.source_url}")

        print("\n🔍 How to View in Frontend:")
        print("\n1. Start the backend server:")
        print("   python -m mirix.server.fastapi_server")

        print("\n2. Start the frontend:")
        print("   cd frontend && npm start")

        print("\n3. Open http://localhost:3000")

        print("\n4. Ask a question that would retrieve this semantic memory:")
        print("   - 'Tell me about MIRIX Phase 1 implementation'")
        print("   - 'What do you know about async patterns?'")
        print("   - 'How was the raw memory pipeline implemented?'")

        print("\n5. Expected Result:")
        print("   You should see purple memory badges showing:")
        for rm in raw_memories:
            print(f"   • {rm.source_app} ({rm.source_url})")

        print("\n📊 To verify in database:")
        print(f"""
python -c "
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path('.env'))
from mirix.server.server import db_context
from mirix.orm.semantic_memory import SemanticMemoryItem

with db_context() as session:
    sem = session.get(SemanticMemoryItem, '{semantic_memory.id}')
    if sem:
        print(f'Name: {{sem.name}}')
        print(f'References: {{len(sem.raw_memory_references)}} items')
        print(f'IDs: {{sem.raw_memory_references}}')
"
""")

        return semantic_memory


if __name__ == "__main__":
    try:
        result = create_semantic_memory_with_refs()
        if result:
            print("\n✨ Ready for frontend testing!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
