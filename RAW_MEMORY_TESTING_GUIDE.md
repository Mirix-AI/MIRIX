# Raw Memory Reference Testing Guide

## Table of Contents
1. [Memory Agent Output Differences](#memory-agent-output-differences)
2. [Data Flow Overview](#data-flow-overview)
3. [Creating Mock Data](#creating-mock-data)
4. [Verification Steps](#verification-steps)
5. [Testing Instructions](#testing-instructions)

---

## Memory Agent Output Differences

### How Different Memory Types Show in UX

**All memory types show the SAME memory reference badges in the UX**, but they differ in:
- **When** they appear (based on what memories are retrieved)
- **Context** (the main message content differs based on memory type)
- **System Prompt** (backend shows different source info formatting)

#### Memory Type Comparison

| Memory Type | System Prompt Display | UX Badge Display | When It Appears |
|------------|----------------------|------------------|-----------------|
| **Episodic** | `[Sources: Chrome: github.com]` after each event summary | Same purple badges | When recalling past events/activities |
| **Semantic** | Currently NO source info (not implemented yet) | Same purple badges | When retrieving knowledge/concepts |
| **Procedural** | Currently NO source info (not implemented yet) | Same purple badges | When recalling how-to procedures |
| **Resource** | Currently NO source info (not implemented yet) | Same purple badges | When accessing saved resources |
| **Knowledge Vault** | Currently NO source info (not implemented yet) | Same purple badges | When retrieving sensitive knowledge |

**Important Notes:**
- ✅ **System Prompt Sources**: Only implemented for **Episodic Memory** (AI sees where info came from)
- ✅ **UX Memory Badges**: Implemented for **ALL memory types** (users see purple badges)
- ⚠️ **Current Limitation**: Other memory types collect references but don't show sources in system prompts yet

### Example UX Output

Regardless of memory type, the frontend displays:

```
┌────────────────────────────────────────────────┐
│ 🤖 Assistant                      3:45 PM      │
├────────────────────────────────────────────────┤
│ ┌─ Memory Sources ────────────────── 3 ─┐     │
│ │ ┌─────────────────────────────────────┐│     │
│ │ │ 🌐 Chrome                           ││     │
│ │ │ github.com/mirix                    ││     │
│ │ │ Dec 18, 2025                        ││     │
│ │ │ "GitHub repository showing MIRIX..." ││     │
│ │ └─────────────────────────────────────┘│     │
│ │ ┌─────────────────────────────────────┐│     │
│ │ │ 🧭 Safari                           ││     │
│ │ │ docs.python.org                     ││     │
│ │ │ Dec 17, 2025                        ││     │
│ │ │ "Python asyncio documentation..."    ││     │
│ │ └─────────────────────────────────────┘│     │
│ │ ┌─────────────────────────────────────┐│     │
│ │ │ 📝 Notion                           ││     │
│ │ │ notion.so/workspace                 ││     │
│ │ │ Dec 16, 2025                        ││     │
│ │ │ "Project planning notes for MIRIX..."││     │
│ │ └─────────────────────────────────────┘│     │
│ └─────────────────────────────────────────┘     │
│                                                 │
│ Based on your GitHub activity and Python       │
│ research (from the screenshots above), here's   │
│ what I found about async programming...         │
└─────────────────────────────────────────────────┘
```

---

## Data Flow Overview

### Complete Pipeline: Screenshot → Database → Frontend

```
┌─────────────────┐
│ 1. Screenshot   │
│    Captured     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. raw_memory   │ ← Store in database
│    Table        │   - screenshot_path
│                 │   - source_app (Chrome/Safari/etc)
│                 │   - source_url
│                 │   - ocr_text
│                 │   - captured_at
│                 │   ID: rawmem-{uuid}
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. Memory Agent │ ← Process and create memory
│    Processing   │   (Episodic/Semantic/etc)
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. Memory Table │ ← Store with references
│    (Episodic/   │   - memory fields
│     Semantic)   │   - raw_memory_references: [rawmem-uuid1, rawmem-uuid2]
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 5. User Query   │
│    to AI        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 6. Agent        │ ← Retrieve memories
│    Retrieves    │   Collect all raw_memory_references
│    Memories     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 7. Backend      │ ← Fetch raw_memory details
│    Response     │   Return with memoryReferences
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 8. Frontend     │ ← Display memory badges
│    ChatBubble   │
└─────────────────┘
```

---

## Creating Mock Data

### Script to Create Test Data

Create and run this script to populate your database with test data:

```python
# File: scripts/create_raw_memory_mock_data.py

"""
Create mock raw_memory data and linked episodic memories for testing.
Run this script to populate your database with test data.

Usage:
PYTHONPATH=. python scripts/create_raw_memory_mock_data_simple.py
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
```

---

## Verification Steps

### Step 1: Verify Raw Memory in Database

```bash
# Open PostgreSQL/SQLite console
# For PostgreSQL:
psql -d mirix_db

# Query raw_memory table
SELECT
    id,
    source_app,
    source_url,
    captured_at,
    LEFT(ocr_text, 50) as ocr_preview,
    processed,
    user_id
FROM raw_memory
ORDER BY captured_at DESC
LIMIT 10;
```

**Expected Output:**
```
id                          | source_app | source_url                | captured_at          | ocr_preview
----------------------------|------------|---------------------------|----------------------|--------------------------------------------------
rawmem-abc-123              | Chrome     | github.com/user/mirix     | 2025-12-18 10:30:00  | MIRIX Repository - Phase 1 Implementation...
rawmem-def-456              | Safari     | docs.python.org/3/library | 2025-12-17 15:45:00  | asyncio — Asynchronous I/O...
rawmem-ghi-789              | Notion     | notion.so/workspace       | 2025-12-16 09:00:00  | MIRIX Memory Architecture - Phase 1 Tasks...
rawmem-jkl-012              | Firefox    | stackoverflow.com/questions | 2025-12-18 14:00:00 | How to properly handle SQLAlchemy sessions...
```

### Step 2: Verify Episodic Memory References

```sql
SELECT
    id,
    summary,
    event_type,
    raw_memory_references,
    occurred_at
FROM episodic_memory
ORDER BY occurred_at DESC
LIMIT 5;
```

**Expected Output:**
```
id             | summary                           | event_type | raw_memory_references                        | occurred_at
---------------|-----------------------------------|------------|----------------------------------------------|------------------
episodic-...   | Worked on MIRIX Phase 1          | work       | ["rawmem-abc-123", "rawmem-jkl-012"]         | 2025-12-18 14:30
episodic-...   | Researched Python async patterns | research   | ["rawmem-def-456", "rawmem-ghi-789"]         | 2025-12-17 16:00
episodic-...   | Planned MIRIX Phase 1 tasks      | planning   | ["rawmem-ghi-789"]                           | 2025-12-16 09:30
```

### Step 3: Test API Endpoint

```bash
# Test the GET /memory/raw/{id} endpoint
curl http://localhost:47284/memory/raw/rawmem-abc-123 | jq

# Expected response:
{
  "id": "rawmem-abc-123",
  "screenshot_path": "/fake/screenshots/github_mirix_20251218_103000.png",
  "source_app": "Chrome",
  "source_url": "https://github.com/user/mirix/commit/abc123",
  "captured_at": "2025-12-18T10:30:00+00:00",
  "ocr_text": "MIRIX Repository - Phase 1 Implementation...",
  "google_cloud_url": null,
  "metadata_": {},
  "processed": false,
  "processing_count": 0,
  "user_id": "user-...",
  "organization_id": "org-...",
  "created_at": "2025-12-18T10:30:00+00:00",
  "has_embedding": true
}
```

### Step 4: Test Memory Retrieval

```bash
# Test the /retrieve_from_memory endpoint
curl -X POST http://localhost:47284/retrieve_from_memory \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_history": [
      {"role": "user", "content": "What did I work on with MIRIX?"}
    ],
    "memory_types": ["episodic"],
    "limit": 5
  }' | jq
```

**Expected Response:**
```json
{
  "memory": {
    "episodic": [
      {
        "timestamp": "2025-12-18T14:30:00+00:00",
        "summary": "Worked on MIRIX Phase 1 implementation",
        "details": "Implemented raw_memory reference pipeline...",
        "event_type": "work",
        "tree_path": ["work", "coding", "mirix", "phase1"]
      }
    ]
  },
  "topic": "What did I work on with MIRIX?"
}
```

---

## Testing Instructions

### Prerequisites

1. **Start the backend servers:**
```bash
# Terminal 1: Start main FastAPI server
python -m mirix.server.fastapi_server

# Terminal 2: Start memory server
python -m mirix.server.memory_server
```

2. **Start the frontend:**
```bash
cd frontend
npm start
```

### Test Procedure

#### Test 1: Create Mock Data

```bash
# Create the scripts directory if it doesn't exist
mkdir -p scripts

# Copy the create_raw_memory_mock_data.py script above to:
# scripts/create_raw_memory_mock_data.py

# Run it
python scripts/create_raw_memory_mock_data.py
```

**Expected Output:**
```
🚀 Creating mock raw_memory test data...

✅ Using existing test user: user-abc-123

📸 Scenario 1: GitHub Coding Session
  ✅ Created: rawmem-abc-123
  📱 App: Chrome
  🔗 URL: https://github.com/user/mirix/commit/abc123
  📅 Captured: 2025-12-18 10:30:00+00:00

...

✅ Mock data creation complete!
```

#### Test 2: Verify Database

```bash
# Check raw_memory table
python -c "
from mirix.services.raw_memory_manager import RawMemoryManager
from mirix.schemas.user import User
from mirix.server.server import db_context
from mirix.orm.user import User as OrmUser

with db_context() as session:
    user = session.query(OrmUser).filter(OrmUser.name == 'Test User').first()

rm_manager = RawMemoryManager()

# Get all raw memories for test user
from mirix.orm.raw_memory import RawMemoryItem
with db_context() as session:
    raw_mems = session.query(RawMemoryItem).filter(
        RawMemoryItem.user_id == user.id
    ).all()

    print(f'Found {len(raw_mems)} raw_memory items:')
    for rm in raw_mems:
        print(f'  - {rm.id}: {rm.source_app} - {rm.source_url}')
"
```

#### Test 3: Test Agent Memory Retrieval

```python
# File: test_agent_retrieval.py

from mirix.agent.agent import Agent
from mirix.schemas.agent import AgentState
from mirix.schemas.user import User
from mirix.server.server import db_context
from mirix.orm.user import User as OrmUser

# Get test user
with db_context() as session:
    user_orm = session.query(OrmUser).filter(OrmUser.name == "Test User").first()

user = User(
    id=user_orm.id,
    name=user_orm.name,
    organization_id=user_orm.organization_id,
    timezone=user_orm.timezone,
    status=user_orm.status
)

# Create agent
agent_state = AgentState(
    id="test-agent",
    name="chat_agent",
    topic="MIRIX development",
    system_prompt="You are a helpful assistant",
    tools=[],
    user_id=user.id,
    organization_id=user.organization_id
)

agent = Agent(agent_state=agent_state, user=user)

# Trigger memory retrieval
system_prompt, retrieved_memories = agent.build_system_prompt_with_memories(
    raw_system="Test",
    topics="MIRIX Python coding work"
)

# Check collected references
print(f"\n✅ Collected {len(agent.current_raw_memory_refs)} raw_memory_references:")
for ref_id in agent.current_raw_memory_refs:
    print(f"  - {ref_id}")

# Check if source info is in system prompt
if "[Sources:" in system_prompt:
    print("\n✅ System prompt contains source information!")
    # Extract and print source sections
    for line in system_prompt.split('\n'):
        if '[Sources:' in line:
            print(f"  {line}")
else:
    print("\n⚠️ No source information in system prompt (might be normal if no episodic memories)")
```

Run it:
```bash
python test_agent_retrieval.py
```

**Expected Output:**
```
✅ Collected 4 raw_memory_references:
  - rawmem-abc-123
  - rawmem-def-456
  - rawmem-ghi-789
  - rawmem-jkl-012

✅ System prompt contains source information!
  [0] Timestamp: 2025-12-18 14:30:00 - Worked on MIRIX Phase 1 implementation - Path: work > coding > mirix > phase1 [Sources: Chrome: github.com, Firefox: stackoverflow.com]
  [1] Timestamp: 2025-12-17 16:00:00 - Researched Python async patterns - Path: learning > python > async [Sources: Safari: docs.python.org, Notion: notion.so]
```

#### Test 4: Test Frontend Display

1. **Open the frontend:** http://localhost:3000

2. **Send a test message:**
```
What did I work on recently with MIRIX?
```

3. **Look for memory badges** in the AI response:

Expected to see:
- Purple gradient "Memory Sources" section at top of response
- Individual badges showing:
  - 🌐 Chrome icon with "github.com"
  - 🦊 Firefox icon with "stackoverflow.com"
  - 🧭 Safari icon with "docs.python.org"
  - 📝 Notion icon with "notion.so"
- OCR text previews
- Capture dates

#### Test 5: Inspect Network Request

Open browser DevTools → Network tab:

1. Send message in chat
2. Look for `/send_streaming_message` request
3. Check the SSE response

Expected SSE data format:
```json
{
  "type": "final",
  "response": "Based on your recent work...",
  "memoryReferences": [
    {
      "id": "rawmem-abc-123",
      "source_app": "Chrome",
      "source_url": "https://github.com/user/mirix/commit/abc123",
      "captured_at": "2025-12-18T10:30:00+00:00",
      "ocr_text": "MIRIX Repository - Phase 1 Implementation..."
    },
    {
      "id": "rawmem-jkl-012",
      "source_app": "Firefox",
      "source_url": "https://stackoverflow.com/questions/12345/...",
      "captured_at": "2025-12-18T14:00:00+00:00",
      "ocr_text": "How to properly handle SQLAlchemy sessions..."
    }
  ]
}
```

---

## Troubleshooting

### Issue: No memory badges appear

**Check:**
1. ✅ Raw memories exist in database
2. ✅ Episodic/semantic memories have `raw_memory_references` populated
3. ✅ Agent retrieves these memories (check system prompt)
4. ✅ Backend sends `memoryReferences` in response
5. ✅ Frontend receives the data

**Debug:**
```javascript
// In browser console
console.log(lastMessage); // Should show memoryReferences array
```

### Issue: "Cannot find raw_memory"

**Cause:** Raw memory was deleted or wrong user_id

**Fix:**
```bash
# Re-run mock data creation
python scripts/create_raw_memory_mock_data.py
```

### Issue: Empty raw_memory_references array

**Cause:** Episodic memory doesn't have references

**Fix:**
```python
# Check episodic memory
from mirix.server.server import db_context
from mirix.orm.episodic_memory import EpisodicEvent

with db_context() as session:
    events = session.query(EpisodicEvent).all()
    for event in events:
        print(f"{event.summary}: {event.raw_memory_references}")
```

---

## Next Steps

After successful testing:

1. ✅ Verify all 4 raw_memory items are in database
2. ✅ Verify episodic memories reference them
3. ✅ Verify agent collects references during retrieval
4. ✅ Verify frontend displays purple memory badges
5. ✅ Test with real screenshots from monitor

**For Production:**
- Enable screen monitoring to auto-create raw_memory
- Configure memory agents to link screenshots
- Monitor memory reference creation in logs
