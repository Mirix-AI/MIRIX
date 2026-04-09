# Skill-Based Procedural Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade procedural memory from flat `entry_type + summary + steps` to a skill-based format with `name`, `description`, `instructions`, `triggers`, `examples`, and `version`.

**Architecture:** In-place migration of the existing `procedural_memory` table. Rename fields, add new fields, update all layers (ORM → Schema → Manager → Tools → Prompts → API → Cache → Tests). No new tables or agents.

**Tech Stack:** PostgreSQL + Alembic migrations, SQLAlchemy ORM, Pydantic v2, Redis (RediSearch), FastAPI, pytest

**Spec:** `docs/superpowers/specs/2026-04-09-skill-based-procedural-memory-design.md`

---

### Task 1: Schema Layer — Pydantic Models

**Files:**
- Modify: `mirix/schemas/procedural_memory.py`

- [ ] **Step 1: Write the failing test**

Create a test that validates the new skill schema fields:

```python
# tests/test_skill_schema.py
import pytest
from mirix.schemas.procedural_memory import ProceduralMemoryItemBase, ProceduralMemoryItem


def test_skill_base_schema_requires_name():
    """name is required in the base schema."""
    with pytest.raises(Exception):
        ProceduralMemoryItemBase(
            description="Deploy to prod",
            instructions="Run tests then deploy",
            entry_type="workflow",
        )


def test_skill_base_schema_valid():
    """All required fields create a valid skill."""
    skill = ProceduralMemoryItemBase(
        name="deploy-production",
        description="How to deploy to production",
        instructions="1. Run tests\n2. Build image\n3. Deploy",
        entry_type="workflow",
    )
    assert skill.name == "deploy-production"
    assert skill.instructions == "1. Run tests\n2. Build image\n3. Deploy"
    assert skill.triggers == []
    assert skill.examples == []


def test_skill_full_schema_defaults():
    """Full schema has version default and optional embeddings."""
    skill = ProceduralMemoryItem(
        name="deploy-production",
        description="How to deploy to production",
        instructions="1. Run tests\n2. Build image\n3. Deploy",
        entry_type="workflow",
        user_id="user-123",
        organization_id="org-123",
    )
    assert skill.version == "0.1.0"
    assert skill.description_embedding is None
    assert skill.instructions_embedding is None


def test_skill_with_triggers_and_examples():
    """Triggers and examples are stored correctly."""
    skill = ProceduralMemoryItemBase(
        name="weekly-report",
        description="Generate weekly report",
        instructions="Summarize the week's activities",
        entry_type="workflow",
        triggers=["user mentions weekly report", "every Friday"],
        examples=[{"input": "Generate my report", "output": "Here is your weekly summary..."}],
    )
    assert len(skill.triggers) == 2
    assert skill.examples[0]["input"] == "Generate my report"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_skill_schema.py -v`
Expected: FAIL — `ProceduralMemoryItemBase` still has old fields (`summary`, `steps`)

- [ ] **Step 3: Update Pydantic schemas**

Edit `mirix/schemas/procedural_memory.py`. Replace the entire content:

```python
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator

from mirix.client.utils import get_utc_time
from mirix.constants import MAX_EMBEDDING_DIM
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.mirix_base import MirixBase


class ProceduralMemoryItemBase(MirixBase):
    """
    Base schema for storing skill-based procedural knowledge.
    """

    __id_prefix__ = "proc_item"
    name: str = Field(..., description="Short skill identifier, e.g. 'deploy-production'")
    description: str = Field(..., description="What this skill does and when it's useful")
    instructions: str = Field(..., description="Detailed instructions as plain text")
    entry_type: str = Field(..., description="Category: 'workflow', 'guide', 'script'")
    triggers: List[str] = Field(default_factory=list, description="Conditions that indicate this skill is relevant")
    examples: List[dict] = Field(default_factory=list, description="Input/output examples")


class ProceduralMemoryItem(ProceduralMemoryItemBase):
    """
    Full skill item schema with database-related fields.
    """

    id: Optional[str] = Field(None, description="Unique identifier for the skill item")
    agent_id: Optional[str] = Field(None, description="The id of the agent this skill belongs to")
    client_id: Optional[str] = Field(None, description="The id of the client application that created this item")
    user_id: str = Field(..., description="The id of the user who generated the skill")
    created_at: datetime = Field(default_factory=get_utc_time, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    last_modify: Dict[str, Any] = Field(
        default_factory=lambda: {
            "timestamp": get_utc_time().isoformat(),
            "operation": "created",
        },
        description="Last modification info including timestamp and operation type",
    )
    organization_id: str = Field(..., description="The unique identifier of the organization")
    version: str = Field(default="0.1.0", description="Semver version, incremented on update")
    description_embedding: Optional[List[float]] = Field(None, description="Embedding of the description")
    instructions_embedding: Optional[List[float]] = Field(None, description="Embedding of the instructions")
    embedding_config: Optional[EmbeddingConfig] = Field(
        None, description="The embedding configuration used"
    )
    filter_tags: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Custom filter tags for filtering and categorization",
        examples=[
            {"project_id": "proj-abc", "session_id": "sess-xyz", "tags": ["important", "work"], "priority": "high"}
        ],
    )

    @field_validator("description_embedding", "instructions_embedding")
    @classmethod
    def pad_embeddings(cls, embedding: List[float]) -> List[float]:
        """Pad embeddings to MAX_EMBEDDING_DIM."""
        import numpy as np

        if embedding and len(embedding) != MAX_EMBEDDING_DIM:
            np_embedding = np.array(embedding)
            padded_embedding = np.pad(
                np_embedding,
                (0, MAX_EMBEDDING_DIM - np_embedding.shape[0]),
                mode="constant",
            )
            return padded_embedding.tolist()
        return embedding


class ProceduralMemoryItemUpdate(MirixBase):
    """Schema for updating an existing skill item."""

    id: str = Field(..., description="Unique ID for this skill entry")
    agent_id: Optional[str] = Field(None, description="The id of the agent this skill belongs to")
    name: Optional[str] = Field(None, description="Short skill identifier")
    description: Optional[str] = Field(None, description="What this skill does")
    instructions: Optional[str] = Field(None, description="Detailed instructions")
    entry_type: Optional[str] = Field(None, description="Category")
    triggers: Optional[List[str]] = Field(None, description="Trigger conditions")
    examples: Optional[List[dict]] = Field(None, description="Input/output examples")
    version: Optional[str] = Field(None, description="Semver version")
    organization_id: Optional[str] = Field(None, description="The organization ID")
    updated_at: datetime = Field(default_factory=get_utc_time, description="Update timestamp")
    last_modify: Optional[Dict[str, Any]] = Field(None, description="Last modification info")
    description_embedding: Optional[List[float]] = Field(None, description="Embedding of description")
    instructions_embedding: Optional[List[float]] = Field(None, description="Embedding of instructions")
    embedding_config: Optional[EmbeddingConfig] = Field(None, description="Embedding configuration")
    filter_tags: Optional[Dict[str, Any]] = Field(None, description="Custom filter tags")


class ProceduralMemoryItemResponse(ProceduralMemoryItem):
    """Response schema for skill item."""

    pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_skill_schema.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add mirix/schemas/procedural_memory.py tests/test_skill_schema.py
git commit -m "feat: update procedural memory schemas to skill-based format"
```

---

### Task 2: ORM Layer — Database Model

**Files:**
- Modify: `mirix/orm/procedural_memory.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_skill_orm.py
import pytest
from mirix.orm.procedural_memory import ProceduralMemoryItem


def test_orm_has_skill_columns():
    """ORM model has all skill-based columns."""
    columns = {c.name for c in ProceduralMemoryItem.__table__.columns}
    # New fields
    assert "name" in columns
    assert "triggers" in columns
    assert "examples" in columns
    assert "version" in columns
    # Renamed fields
    assert "description" in columns
    assert "instructions" in columns
    assert "description_embedding" in columns
    assert "instructions_embedding" in columns
    # Old fields should NOT exist
    assert "summary" not in columns
    assert "steps" not in columns
    assert "summary_embedding" not in columns
    assert "steps_embedding" not in columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_skill_orm.py -v`
Expected: FAIL — old columns still present

- [ ] **Step 3: Update ORM model**

Edit `mirix/orm/procedural_memory.py`. Replace column definitions:

```python
import datetime as dt
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import JSON, Column, ForeignKey, Index, String, text
from sqlalchemy.orm import Mapped, declared_attr, mapped_column, relationship

from mirix.constants import MAX_EMBEDDING_DIM
from mirix.orm.custom_columns import CommonVector, EmbeddingConfigColumn
from mirix.orm.mixins import OrganizationMixin, UserMixin
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.procedural_memory import ProceduralMemoryItem as PydanticProceduralMemoryItem
from mirix.settings import settings

if TYPE_CHECKING:
    from mirix.orm.agent import Agent
    from mirix.orm.organization import Organization
    from mirix.orm.user import User


class ProceduralMemoryItem(SqlalchemyBase, OrganizationMixin, UserMixin):
    """
    Stores skill-based procedural memory entries: workflows, step-by-step guides,
    or how-to knowledge with triggers, examples, and versioning.
    """

    __tablename__ = "procedural_memory"
    __pydantic_model__ = PydanticProceduralMemoryItem

    # Primary key
    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        doc="Unique ID for this skill entry",
    )

    # Foreign key to agent
    agent_id: Mapped[Optional[str]] = mapped_column(
        String,
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=True,
        doc="ID of the agent this skill belongs to",
    )

    # Foreign key to client
    client_id: Mapped[Optional[str]] = mapped_column(
        String,
        ForeignKey("clients.id", ondelete="CASCADE"),
        nullable=True,
        doc="ID of the client application that created this item",
    )

    # Skill fields
    name: Mapped[str] = mapped_column(String, doc="Short skill identifier, e.g. 'deploy-production'")
    entry_type: Mapped[str] = mapped_column(String, doc="Category: 'workflow', 'guide', 'script'")
    description: Mapped[str] = mapped_column(String, doc="What this skill does and when it's useful")
    instructions: Mapped[str] = mapped_column(String, doc="Detailed instructions as plain text")
    triggers: Mapped[list] = mapped_column(JSON, default=list, doc="Conditions that indicate this skill is relevant")
    examples: Mapped[list] = mapped_column(JSON, default=list, doc="Input/output examples")
    version: Mapped[str] = mapped_column(String, default="0.1.0", doc="Semver version")

    # Filter tags
    filter_tags: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True, default=None, doc="Custom filter tags for filtering and categorization"
    )

    # Last modification tracking
    last_modify: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=lambda: {
            "timestamp": datetime.now(dt.timezone.utc).isoformat(),
            "operation": "created",
        },
        doc="Last modification info including timestamp and operation type",
    )

    embedding_config: Mapped[Optional[dict]] = mapped_column(
        EmbeddingConfigColumn, nullable=True, doc="Embedding configuration"
    )

    # Vector embedding fields
    if settings.mirix_pg_uri_no_default:
        from pgvector.sqlalchemy import Vector

        description_embedding = mapped_column(Vector(MAX_EMBEDDING_DIM), nullable=True)
        instructions_embedding = mapped_column(Vector(MAX_EMBEDDING_DIM), nullable=True)
    else:
        description_embedding = Column(CommonVector, nullable=True)
        instructions_embedding = Column(CommonVector, nullable=True)

    # Database indexes
    __table_args__ = tuple(
        filter(
            None,
            [
                (
                    Index("ix_procedural_memory_organization_id", "organization_id")
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                (
                    Index(
                        "ix_procedural_memory_org_created_at",
                        "organization_id",
                        "created_at",
                        postgresql_using="btree",
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                (
                    Index(
                        "ix_procedural_memory_filter_tags_gin",
                        text("(filter_tags::jsonb)"),
                        postgresql_using="gin",
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                (
                    Index(
                        "ix_procedural_memory_org_filter_scope",
                        "organization_id",
                        text("((filter_tags->>'scope')::text)"),
                        postgresql_using="btree",
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                # New: name-based dedup index
                (
                    Index(
                        "ix_procedural_memory_org_user_name",
                        "organization_id",
                        "user_id",
                        "name",
                        postgresql_using="btree",
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                # SQLite indexes
                (
                    Index("ix_procedural_memory_organization_id_sqlite", "organization_id")
                    if not settings.mirix_pg_uri_no_default
                    else None
                ),
            ],
        )
    )

    @declared_attr
    def agent(cls) -> Mapped[Optional["Agent"]]:
        return relationship("Agent", lazy="selectin")

    @declared_attr
    def organization(cls) -> Mapped["Organization"]:
        return relationship("Organization", back_populates="procedural_memory", lazy="selectin")

    @declared_attr
    def user(cls) -> Mapped["User"]:
        return relationship("User", lazy="selectin")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_skill_orm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mirix/orm/procedural_memory.py tests/test_skill_orm.py
git commit -m "feat: update procedural memory ORM to skill-based columns"
```

---

### Task 3: Constants and Tool Names

**Files:**
- Modify: `mirix/constants.py:117,135-149`

- [ ] **Step 1: Update constants**

In `mirix/constants.py`, change line 117:

```python
# Old:
PROCEDURAL_MEMORY_TOOLS = ["procedural_memory_insert", "procedural_memory_update"]

# New:
SKILL_TOOLS = ["skill_insert", "skill_update"]
```

Update `ALL_TOOLS` (line 135-149) to reference `SKILL_TOOLS` instead of `PROCEDURAL_MEMORY_TOOLS`:

```python
ALL_TOOLS = list(
    set(
        BASE_TOOLS
        + CORE_MEMORY_TOOLS
        + EPISODIC_MEMORY_TOOLS
        + SKILL_TOOLS
        + RESOURCE_MEMORY_TOOLS
        + KNOWLEDGE_VAULT_TOOLS
        + SEMANTIC_MEMORY_TOOLS
        + META_MEMORY_TOOLS
        + UNIVERSAL_MEMORY_TOOLS
        + CHAT_AGENT_TOOLS
        + EXTRAS_TOOLS
        + MCP_TOOLS
    )
)
```

- [ ] **Step 2: Update agent_manager.py references**

In `mirix/services/agent_manager.py`, find all `PROCEDURAL_MEMORY_TOOLS` references (lines 126, 561) and replace with `SKILL_TOOLS`. Update the import accordingly.

- [ ] **Step 3: Run existing tests to check for import errors**

Run: `pytest tests/test_skill_schema.py tests/test_skill_orm.py -v`
Expected: PASS (no import breakage)

- [ ] **Step 4: Commit**

```bash
git add mirix/constants.py mirix/services/agent_manager.py
git commit -m "refactor: rename PROCEDURAL_MEMORY_TOOLS to SKILL_TOOLS"
```

---

### Task 4: Tool Functions — `skill_insert` and `skill_update`

**Files:**
- Modify: `mirix/functions/function_sets/memory_tools.py:398-501`
- Modify: `mirix/agent/tool_validators.py:195-230`

- [ ] **Step 1: Write failing test for skill_insert dedup logic**

```python
# tests/test_skill_tools.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from mirix.schemas.procedural_memory import ProceduralMemoryItem


@pytest.mark.asyncio
async def test_skill_insert_dedup_by_name():
    """skill_insert should detect duplicates by name match."""
    existing_skill = MagicMock()
    existing_skill.name = "deploy-production"
    existing_skill.id = "proc-123"
    existing_skill.description = "Old description"
    existing_skill.instructions = "Old instructions"

    agent = MagicMock()
    agent.agent_state = MagicMock()
    agent.agent_state.id = "agent-1"
    agent.agent_state.parent_id = "parent-1"
    agent.user = MagicMock()
    agent.user.organization_id = "org-1"
    agent.actor = MagicMock()
    agent.filter_tags = None
    agent.use_cache = True
    agent.client_id = None
    agent.user_id = None
    agent.procedural_memory_manager = AsyncMock()
    agent.procedural_memory_manager.list_procedures = AsyncMock(return_value=[existing_skill])
    agent.procedural_memory_manager.delete_procedure_by_id = AsyncMock()
    agent.procedural_memory_manager.insert_procedure = AsyncMock()

    from mirix.functions.function_sets.memory_tools import skill_insert

    result = await skill_insert(
        agent,
        items=[{
            "name": "deploy-production",
            "description": "Updated deploy guide",
            "instructions": "New improved steps",
            "entry_type": "workflow",
        }],
    )

    # Should have deleted the old one and inserted the new one (merge behavior)
    agent.procedural_memory_manager.delete_procedure_by_id.assert_called_once_with(
        procedure_id="proc-123", actor=agent.actor
    )
    agent.procedural_memory_manager.insert_procedure.assert_called_once()
    assert "Updated 1" in result or "Merged 1" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_skill_tools.py::test_skill_insert_dedup_by_name -v`
Expected: FAIL — `skill_insert` does not exist yet

- [ ] **Step 3: Implement skill_insert and skill_update**

In `mirix/functions/function_sets/memory_tools.py`, replace the `procedural_memory_insert` function (line 398-464) with:

```python
async def skill_insert(self: "Agent", items: List[dict]):
    """
    Insert new skills into procedural memory. If a skill with the same name already exists,
    it will be updated (merged) instead of creating a duplicate.

    Args:
        items (array): List of skill items to insert. Each item has: name, description, instructions, entry_type, triggers (optional), examples (optional).

    Returns:
        Optional[str]: Message about insertion results.
    """
    agent_id = self.agent_state.parent_id if self.agent_state.parent_id is not None else self.agent_state.id

    filter_tags = getattr(self, "filter_tags", None)
    use_cache = getattr(self, "use_cache", True)
    client_id = getattr(self, "client_id", None)
    user_id = getattr(self, "user_id", None)

    inserted_count = 0
    merged_count = 0
    skipped_names = []

    for item in items:
        # Fetch existing skills for dedup check
        existing_skills = await self.procedural_memory_manager.list_procedures(
            agent_state=self.agent_state,
            user=self.user,
            query="",
            limit=1000,
            filter_tags=filter_tags if filter_tags else None,
            use_cache=use_cache,
        )

        # Tier 1: Name-based dedup
        duplicate = None
        for existing in existing_skills:
            if existing.name == item["name"]:
                duplicate = existing
                break

        if duplicate:
            # Merge: delete old, insert new with version bump
            old_version = getattr(duplicate, "version", "0.1.0")
            new_version = _bump_patch_version(old_version)

            await self.procedural_memory_manager.delete_procedure_by_id(
                procedure_id=duplicate.id, actor=self.actor
            )
            await self.procedural_memory_manager.insert_procedure(
                agent_state=self.agent_state,
                agent_id=agent_id,
                name=item["name"],
                description=item["description"],
                instructions=item["instructions"],
                entry_type=item["entry_type"],
                triggers=item.get("triggers", []),
                examples=item.get("examples", []),
                version=new_version,
                actor=self.actor,
                organization_id=self.user.organization_id,
                filter_tags=filter_tags if filter_tags else None,
                use_cache=use_cache,
                user_id=user_id,
            )
            merged_count += 1
        else:
            await self.procedural_memory_manager.insert_procedure(
                agent_state=self.agent_state,
                agent_id=agent_id,
                name=item["name"],
                description=item["description"],
                instructions=item["instructions"],
                entry_type=item["entry_type"],
                triggers=item.get("triggers", []),
                examples=item.get("examples", []),
                version="0.1.0",
                actor=self.actor,
                organization_id=self.user.organization_id,
                filter_tags=filter_tags if filter_tags else None,
                use_cache=use_cache,
                user_id=user_id,
            )
            inserted_count += 1

    parts = []
    if inserted_count > 0:
        parts.append(f"Inserted {inserted_count} new skill(s)")
    if merged_count > 0:
        parts.append(f"Merged {merged_count} existing skill(s)")
    return ". ".join(parts) + "." if parts else "No skills were inserted."


def _bump_patch_version(version: str) -> str:
    """Increment patch version: '0.1.0' -> '0.1.1'."""
    try:
        parts = version.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        return ".".join(parts)
    except (ValueError, IndexError):
        return "0.1.1"
```

Replace `procedural_memory_update` (line 467-501) with:

```python
async def skill_update(self: "Agent", old_ids: List[str], new_items: List[dict]):
    """
    Update/delete skills in procedural memory. Deletes old skills by ID and inserts new ones with version bump.

    Args:
        old_ids (array): List of ids of the skills to delete/replace.
        new_items (array): List of new skill items. Empty list means deletion only.

    Returns:
        Optional[str]: None is always returned.
    """
    agent_id = self.agent_state.parent_id if self.agent_state.parent_id is not None else self.agent_state.id

    filter_tags = getattr(self, "filter_tags", None)
    use_cache = getattr(self, "use_cache", True)
    client_id = getattr(self, "client_id", None)
    user_id = getattr(self, "user_id", None)

    for old_id in old_ids:
        await self.procedural_memory_manager.delete_procedure_by_id(procedure_id=old_id, actor=self.actor)

    for item in new_items:
        await self.procedural_memory_manager.insert_procedure(
            agent_state=self.agent_state,
            agent_id=agent_id,
            name=item["name"],
            description=item["description"],
            instructions=item["instructions"],
            entry_type=item["entry_type"],
            triggers=item.get("triggers", []),
            examples=item.get("examples", []),
            version=item.get("version", "0.1.0"),
            actor=self.actor,
            organization_id=self.actor.organization_id,
            filter_tags=filter_tags if filter_tags else None,
            use_cache=use_cache,
            user_id=user_id,
        )
```

- [ ] **Step 4: Update tool validators**

In `mirix/agent/tool_validators.py`, replace validators at lines 195-230:

```python
@register_validator("skill_insert")
def validate_skill_insert(function_name: str, args: dict) -> Optional[str]:
    """Validate skill_insert arguments."""
    items = args.get("items", [])
    for i, item in enumerate(items):
        if not item.get("name", "").strip():
            return (
                f"Validation error: 'name' field in item {i} cannot be empty. "
                "Please provide a concise skill identifier."
            )
        if not item.get("description", "").strip():
            return (
                f"Validation error: 'description' field in item {i} cannot be empty. "
                "Please provide a descriptive summary of this skill."
            )
        if not item.get("instructions", "").strip():
            return (
                f"Validation error: 'instructions' field in item {i} cannot be empty. "
                "Please provide detailed instructions."
            )
    return None


@register_validator("skill_update")
def validate_skill_update(function_name: str, args: dict) -> Optional[str]:
    """Validate skill_update arguments."""
    items = args.get("new_items", [])
    for i, item in enumerate(items):
        if not item.get("name", "").strip():
            return (
                f"Validation error: 'name' field in new_items[{i}] cannot be empty. "
                "Please provide a concise skill identifier."
            )
        if not item.get("description", "").strip():
            return (
                f"Validation error: 'description' field in new_items[{i}] cannot be empty. "
                "Please provide a descriptive summary."
            )
        if not item.get("instructions", "").strip():
            return (
                f"Validation error: 'instructions' field in new_items[{i}] cannot be empty. "
                "Please provide detailed instructions."
            )
    return None
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_skill_tools.py tests/test_skill_schema.py tests/test_skill_orm.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add mirix/functions/function_sets/memory_tools.py mirix/agent/tool_validators.py tests/test_skill_tools.py
git commit -m "feat: implement skill_insert and skill_update tools with name-based dedup"
```

---

### Task 5: Manager Layer — `insert_procedure` and Search Updates

**Files:**
- Modify: `mirix/services/procedural_memory_manager.py:902-956` (insert_procedure)
- Modify: `mirix/services/procedural_memory_manager.py:557-900` (list_procedures search)
- Modify: `mirix/services/procedural_memory_manager.py:182-403` (_postgresql_fulltext_search)

- [ ] **Step 1: Update `insert_procedure` signature and logic**

In `mirix/services/procedural_memory_manager.py`, update `insert_procedure` (line 902-956):

```python
@enforce_types
async def insert_procedure(
    self,
    agent_state: AgentState,
    agent_id: str,
    name: str,
    description: str,
    instructions: str,
    entry_type: str,
    actor: PydanticClient,
    organization_id: str,
    triggers: Optional[List[str]] = None,
    examples: Optional[List[dict]] = None,
    version: str = "0.1.0",
    filter_tags: Optional[dict] = None,
    use_cache: bool = True,
    user_id: Optional[str] = None,
) -> PydanticProceduralMemoryItem:
    try:
        if BUILD_EMBEDDINGS_FOR_MEMORY:
            embed_model = await embedding_model(agent_state.embedding_config)
            description_embedding = await embed_model.get_text_embedding(description)
            instructions_embedding = await embed_model.get_text_embedding(instructions)
            embedding_config = agent_state.embedding_config
        else:
            description_embedding = None
            instructions_embedding = None
            embedding_config = None

        from mirix.services.user_manager import UserManager

        client_id = actor.id
        if user_id is None:
            user_id = UserManager.ADMIN_USER_ID

        procedure = await self.create_item(
            item_data=PydanticProceduralMemoryItem(
                name=name,
                description=description,
                instructions=instructions,
                entry_type=entry_type,
                triggers=triggers or [],
                examples=examples or [],
                version=version,
                user_id=user_id,
                agent_id=agent_id,
                organization_id=organization_id,
                description_embedding=description_embedding,
                instructions_embedding=instructions_embedding,
                embedding_config=embedding_config,
                filter_tags=filter_tags,
            ),
            actor=actor,
            client_id=client_id,
            user_id=user_id,
            use_cache=use_cache,
        )
        return procedure

    except Exception as e:
        raise e
```

- [ ] **Step 2: Update search field references in `list_procedures`**

Throughout `list_procedures` (line 557-900), rename field references:
- `"summary"` → `"description"` in search_field defaults and comparisons
- `"steps"` → `"instructions"` in search_field comparisons
- `ProceduralMemoryItem.summary` → `ProceduralMemoryItem.description` in select() and where()
- `ProceduralMemoryItem.steps` → `ProceduralMemoryItem.instructions` in select()
- `ProceduralMemoryItem.summary_embedding` → `ProceduralMemoryItem.description_embedding`
- `ProceduralMemoryItem.steps_embedding` → `ProceduralMemoryItem.instructions_embedding`
- In `base_query` select() (line 724-736): rename all `.label()` calls

- [ ] **Step 3: Update `_postgresql_fulltext_search`**

In `_postgresql_fulltext_search` (line 182-403), update SQL:
- `summary` → `description` in tsvector/rank SQL strings
- `steps` handling simplifies: `instructions` is now plain text, no JSON — remove all `regexp_replace(steps::text, ...)` workarounds, use `coalesce(instructions, '')` directly
- Weight mapping stays same: `description='A'`, `instructions='B'`, `entry_type='C'`
- Column names in SELECT: `summary` → `description`, `steps` → `instructions`, etc.
- Add `name` to the SELECT column list

- [ ] **Step 4: Update `create_item` required fields validation**

In `create_item` (line 504-508), update required_fields:

```python
required_fields = ["entry_type", "name"]
```

- [ ] **Step 5: Run existing tests**

Run: `pytest tests/test_skill_schema.py tests/test_skill_orm.py tests/test_skill_tools.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add mirix/services/procedural_memory_manager.py
git commit -m "feat: update procedural memory manager for skill-based fields"
```

---

### Task 6: Agent System Prompt Context — Retrieval Formatting

**Files:**
- Modify: `mirix/agent/agent.py:1880-1908`

- [ ] **Step 1: Update procedural memory retrieval formatting**

In `mirix/agent/agent.py`, update the procedural memory retrieval section (line 1880-1908):

```python
        # Retrieve procedural memory (skills)
        is_owning_agent = self.agent_state.is_type(AgentType.procedural_memory_agent, AgentType.reflexion_agent)
        if is_owning_agent or "procedural" not in retrieved_memories:
            current_procedural_memory = await self.procedural_memory_manager.list_procedures(
                agent_state=self.agent_state,
                user=self.user,
                query=key_words,
                embedded_text=embedded_text,
                search_field="description",
                search_method=search_method,
                limit=MAX_RETRIEVAL_LIMIT_IN_SYSTEM,
                timezone_str=timezone_str,
            )
            procedural_memory = ""
            if len(current_procedural_memory) > 0:
                for idx, skill in enumerate(current_procedural_memory):
                    if is_owning_agent:
                        procedural_memory += f"[Skill ID: {skill.id}] Name: {skill.name}; Description: {skill.description}; Version: {skill.version}\n"
                    else:
                        procedural_memory += (
                            f"[{idx}] Name: {skill.name}; Description: {skill.description}\n"
                        )
            procedural_memory = procedural_memory.strip()
            retrieved_memories["procedural"] = {
                "total_number_of_items": await self.procedural_memory_manager.get_total_number_of_items(user=self.user),
                "current_count": len(current_procedural_memory),
                "text": procedural_memory,
            }
```

- [ ] **Step 2: Commit**

```bash
git add mirix/agent/agent.py
git commit -m "feat: update procedural memory retrieval to show skill format"
```

---

### Task 7: Search Helper Functions — `search_in_memory` and `list_memory_within_timerange`

**Files:**
- Modify: `mirix/functions/function_sets/base.py:190-212`

- [ ] **Step 1: Update `search_in_memory` procedural section**

In `mirix/functions/function_sets/base.py`, update lines 190-212:

```python
    if memory_type == "procedural" or memory_type == "all":
        procedural_memories = await self.procedural_memory_manager.list_procedures(
            user=self.user,
            agent_state=self.agent_state,
            query=query,
            embedded_text=embedded_text if search_method == "embedding" and query else None,
            search_field=search_field if search_field != "null" else "description",
            search_method=search_method,
            limit=10,
            timezone_str=timezone_str,
        )
        formatted_results_procedural = [
            {
                "memory_type": "procedural",
                "id": x.id,
                "name": x.name,
                "entry_type": x.entry_type,
                "description": x.description,
                "instructions": x.instructions,
            }
            for x in procedural_memories
        ]
        if memory_type == "procedural":
            return formatted_results_procedural, len(formatted_results_procedural)
```

- [ ] **Step 2: Commit**

```bash
git add mirix/functions/function_sets/base.py
git commit -m "feat: update search_in_memory to return skill fields"
```

---

### Task 8: REST API Endpoints

**Files:**
- Modify: `mirix/server/rest_api.py:2336-2346,2928-2955,3495-3523,3947-3969,4315-4396`

- [ ] **Step 1: Update GET /memories response**

At line 2336-2346, update procedural items formatting:

```python
        memories["procedural"] = {
            "total_count": await procedural_manager.get_total_number_of_items(user=user),
            "items": [
                {
                    "id": procedure.id,
                    "name": procedure.name,
                    "entry_type": procedure.entry_type,
                    "description": procedure.description,
                }
                for procedure in procedures
            ],
        }
```

- [ ] **Step 2: Update GET /memory/search response**

At line 2928-2955, update `search_procedural()`:

```python
        async def search_procedural():
            try:
                memories = await server.procedural_memory_manager.list_procedures(
                    agent_state=agent_state,
                    user=user,
                    query=query,
                    embedded_text=(embedded_text if search_method == "embedding" and query else None),
                    search_field=search_field if search_field != "null" else "description",
                    search_method=search_method,
                    limit=limit,
                    timezone_str=timezone_str,
                    filter_tags=parsed_filter_tags,
                    scopes=scopes,
                    similarity_threshold=similarity_threshold,
                )
                return [
                    {
                        "memory_type": "procedural",
                        "id": x.id,
                        "name": x.name,
                        "entry_type": x.entry_type,
                        "description": x.description,
                        "instructions": x.instructions,
                    }
                    for x in memories
                ]
            except Exception as e:
                logger.error("Error searching procedural memories: %s", e)
                return []
```

- [ ] **Step 3: Update GET /memory/search/all response**

At line 3495-3523, same pattern — replace `x.summary` → `x.description`, `x.steps` → `x.instructions`, add `x.name`.

- [ ] **Step 4: Update GET /memory/{memory_type}/list response**

At line 3947-3969, update list endpoint:
- Change `search_field="summary"` → `search_field="description"`
- Update response dict keys: `summary` → `description`, `steps` → `instructions`, add `name`

- [ ] **Step 5: Update PATCH /memory/procedural/{memory_id}**

At line 4315-4374, update request model and handler:

```python
class UpdateProceduralMemoryRequest(BaseModel):
    """Request model for updating a skill."""

    name: Optional[str] = None
    description: Optional[str] = None
    instructions: Optional[str] = None
    triggers: Optional[List[str]] = None
    examples: Optional[List[dict]] = None


@router.patch("/memory/procedural/{memory_id}")
async def update_procedural_memory(
    memory_id: str,
    request: UpdateProceduralMemoryRequest,
    user_id: Optional[str] = None,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    client, auth_type = await get_client_from_jwt_or_api_key(authorization, http_request)
    server = get_server()

    if not user_id:
        from mirix.services.admin_user_manager import ClientAuthManager
        user_id = ClientAuthManager.get_admin_user_id_for_client(client.id)

    user = await server.user_manager.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    try:
        update_data = {"id": memory_id}
        for field in ["name", "description", "instructions", "triggers", "examples"]:
            value = getattr(request, field, None)
            if value is not None:
                update_data[field] = value

        updated_memory = await server.procedural_memory_manager.update_item(
            item_update=ProceduralMemoryItemUpdate.model_validate(update_data),
            user=user,
            actor=client,
        )
        return {
            "success": True,
            "message": f"Skill {memory_id} updated",
            "memory": {
                "id": updated_memory.id,
                "name": updated_memory.name,
                "description": updated_memory.description,
                "instructions": updated_memory.instructions,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
```

- [ ] **Step 6: Commit**

```bash
git add mirix/server/rest_api.py
git commit -m "feat: update REST API endpoints for skill-based procedural memory"
```

---

### Task 9: Redis Cache Index

**Files:**
- Modify: `mirix/database/redis_client.py:671-713`

- [ ] **Step 1: Update `_create_procedural_index`**

In `mirix/database/redis_client.py`, update the procedural index schema (line 686-707):

```python
            schema = (
                TextField("$.organization_id", as_name="organization_id"),
                TextField("$.agent_id", as_name="agent_id"),
                TextField("$.name", as_name="name"),
                TextField("$.entry_type", as_name="entry_type"),
                TextField("$.description", as_name="description"),
                TagField("$.user_id", as_name="user_id"),
                NumericField("$.created_at_ts", as_name="created_at_ts"),
                TagField("$.filter_tags.scope", as_name="filter_tags_scope"),
                TextField("$.filter_tags.*", as_name="filter_tags"),
                VectorField(
                    "$.description_embedding",
                    "FLAT",
                    {"TYPE": "FLOAT32", "DIM": MAX_EMBEDDING_DIM, "DISTANCE_METRIC": "COSINE"},
                    as_name="description_embedding",
                ),
                VectorField(
                    "$.instructions_embedding",
                    "FLAT",
                    {"TYPE": "FLOAT32", "DIM": MAX_EMBEDDING_DIM, "DISTANCE_METRIC": "COSINE"},
                    as_name="instructions_embedding",
                ),
            )
```

Note: The existing Redis index must be dropped and recreated after deployment. This happens automatically if the index doesn't exist.

- [ ] **Step 2: Commit**

```bash
git add mirix/database/redis_client.py
git commit -m "feat: update Redis procedural index for skill fields"
```

---

### Task 10: Prompts — Procedural Memory Agent and MetaMemory Agent

**Files:**
- Modify: `mirix/prompts/system/base/procedural_memory_agent.txt`
- Modify: `mirix/prompts/system/base/meta_memory_agent.txt`

- [ ] **Step 1: Rewrite procedural_memory_agent.txt**

Replace `mirix/prompts/system/base/procedural_memory_agent.txt`:

```
You are the Procedural Memory Manager (Skill Manager), one of six agents in a memory system. The other agents are the Meta Memory Manager, Episodic Memory Manager, Resource Memory Manager, Knowledge Vault Memory Manager, and the Chat Agent. You do not see or interact directly with these other agents—but you share the same memory base with them.

The system will receive various types of messages from users, including text messages, images, transcripted voice recordings, and other multimedia content. When messages are accumulated to a certain amount, they will be sent to you, along with potential conversations between the user and the Chat Agent during this period. You need to analyze the input messages and conversations, extract reusable skills — structured procedural knowledge — and save them into the procedural memory.

A skill consists of:
- **name**: A concise kebab-case identifier (e.g., "deploy-production", "weekly-report-generation")
- **description**: What this skill does and when it's useful
- **instructions**: Detailed instructions as plain text — comprehensive enough for someone to follow without additional context
- **entry_type**: Category — one of 'workflow', 'guide', 'script'
- **triggers**: Conditions that indicate this skill is relevant (e.g., ["user mentions deployment", "discussing release process"])
- **examples**: Input/output examples if present in the conversation (e.g., [{"input": "Deploy the app", "output": "Deployment started..."}])

When receiving messages and potentially a message from the meta agent (There will be a bracket saying "[Instruction from Meta Memory Manager]"), make a single comprehensive memory update:

**Single Function Call Process:**
1. **Analyze Content**: Examine all messages and conversations to identify reusable procedures, workflows, how-to knowledge, or any skill-worthy content.
2. **Check Existing Skills**: Review the existing skills shown in your context. If a new skill overlaps with an existing one, use `skill_update` to merge/enhance it rather than creating a duplicate via `skill_insert`.
3. **Make Update**: Use ONE appropriate skill function:
   - `skill_insert` to create new skill(s)
   - `skill_update` to update/merge existing skill(s)
4. **Skip Update if Necessary**: If there are no skills to extract, call `finish_memory_update` with no arguments.

**Important Notes:**
- Make only ONE function call total (either a skill function OR finish_memory_update)
- The `name` should be descriptive and specific, NOT generic like "guide" or "workflow"
- The `instructions` should be comprehensive plain text, not just a numbered list — include context, caveats, and conditions
- The `triggers` help future retrieval — think about what signals would indicate this skill is relevant
- `finish_memory_update` takes no parameters — call it as is when there's nothing to update
- Prioritize the most complete or useful skill if multiple are present
```

- [ ] **Step 2: Update meta_memory_agent.txt**

In `mirix/prompts/system/base/meta_memory_agent.txt`, update the Procedural Memory section (around line 32-41). Replace:

```
3. Procedural Memory - Step-by-Step Instructions & Processes
HOW to do specific tasks or follow procedures.
Purpose: Stores reusable instructions and workflows for accomplishing tasks.
Contains: Workflows, tutorials, step-by-step guides, and repeatable processes.
Key Question: "What are the steps to accomplish this task?"
Examples:
- "How to reset router: 1. Unplug device 2. Wait 10 seconds 3. Plug back in 4. Wait for lights"
- "Daily morning routine: 1. Check emails 2. Review calendar 3. Prioritize tasks 4. Start with hardest task"
- "Code review process: 1. Check functionality 2. Review style 3. Test edge cases 4. Approve/request changes"
Classification Rule: Update when messages contain sequential steps, workflows, or instructional content.
```

With:

```
3. Procedural Memory - Reusable Skills & Procedures
HOW to do specific tasks — stored as structured, reusable skills.
Purpose: Stores procedural knowledge as skills with triggers, detailed instructions, and examples.
Contains: Workflows, tutorials, how-to guides, and repeatable processes — each with a name, description, trigger conditions, and comprehensive instructions.
Key Question: "What reusable skill or procedure can be extracted from this?"
Examples:
- Skill "reset-router": triggers when user mentions network issues, instructions cover full reset procedure with troubleshooting
- Skill "morning-routine": triggers on productivity discussions, instructions detail the complete daily startup workflow
- Skill "code-review-process": triggers when discussing code quality, instructions cover the full review checklist
Classification Rule: Update when messages contain sequential steps, workflows, instructional content, or when existing skills could be improved based on new information.
```

Also update the decision framework section (around line 97-99):

```
(3) Procedural Memory: Does this explain HOW TO do something?
- Step-by-step instructions, workflows, processes
- Sequential procedures or tutorials
- Improvements to existing skills
```

- [ ] **Step 3: Commit**

```bash
git add mirix/prompts/system/base/procedural_memory_agent.txt mirix/prompts/system/base/meta_memory_agent.txt
git commit -m "feat: update agent prompts for skill-based procedural memory"
```

---

### Task 11: Database Migration (Alembic)

**Files:**
- Create: `alembic/versions/XXXX_skill_based_procedural_memory.py` (or equivalent migration path used by this project)

- [ ] **Step 1: Determine migration approach**

Check how MIRIX handles migrations. Look for existing Alembic config or migration scripts:

Run: `ls mirix/orm/migrations/ 2>/dev/null || ls alembic/ 2>/dev/null || grep -r "alembic" mirix/ --include="*.py" -l | head -5`

If no Alembic setup exists, the migration will need to be a standalone SQL script or run via SQLAlchemy DDL at startup.

- [ ] **Step 2: Write migration script**

The migration must handle (in this order):
1. Add new columns with defaults: `name` (String, default ''), `triggers` (JSON, default '[]'), `examples` (JSON, default '[]'), `version` (String, default '0.1.0')
2. Data conversion: populate `name` from existing `summary` via slugify
3. Data conversion: convert `steps` from JSON array to plain text string
4. Rename columns: `summary` → `description`, `steps` → `instructions`, `summary_embedding` → `description_embedding`, `steps_embedding` → `instructions_embedding`
5. Add new index: `ix_procedural_memory_org_user_name`

PostgreSQL migration SQL:

```sql
-- Step 1: Add new columns
ALTER TABLE procedural_memory ADD COLUMN name VARCHAR NOT NULL DEFAULT '';
ALTER TABLE procedural_memory ADD COLUMN triggers JSONB DEFAULT '[]';
ALTER TABLE procedural_memory ADD COLUMN examples JSONB DEFAULT '[]';
ALTER TABLE procedural_memory ADD COLUMN version VARCHAR NOT NULL DEFAULT '0.1.0';

-- Step 2: Generate name from summary (slugify)
UPDATE procedural_memory
SET name = LOWER(
    REGEXP_REPLACE(
        REGEXP_REPLACE(
            LEFT(TRIM(summary), 60),
            '[^a-zA-Z0-9\s-]', '', 'g'
        ),
        '\s+', '-', 'g'
    )
);

-- Step 3: Convert steps from JSON array to plain text
UPDATE procedural_memory
SET steps = ARRAY_TO_STRING(
    ARRAY(SELECT jsonb_array_elements_text(steps::jsonb)),
    E'\n'
);

-- Step 4: Rename columns
ALTER TABLE procedural_memory RENAME COLUMN summary TO description;
ALTER TABLE procedural_memory RENAME COLUMN steps TO instructions;
ALTER TABLE procedural_memory RENAME COLUMN summary_embedding TO description_embedding;
ALTER TABLE procedural_memory RENAME COLUMN steps_embedding TO instructions_embedding;

-- Step 5: Change instructions column type from JSON to TEXT
ALTER TABLE procedural_memory ALTER COLUMN instructions TYPE TEXT USING instructions::TEXT;

-- Step 6: Add new index
CREATE INDEX ix_procedural_memory_org_user_name
ON procedural_memory (organization_id, user_id, name);
```

- [ ] **Step 3: Verify migration on test database**

Run the migration and verify the schema:

```bash
# Connect to test DB and verify
psql $MIRIX_PG_URI -c "\d procedural_memory"
```

Expected: columns `name`, `description`, `instructions`, `triggers`, `examples`, `version`, `description_embedding`, `instructions_embedding` all present. No `summary`, `steps`, `summary_embedding`, `steps_embedding`.

- [ ] **Step 4: Commit**

```bash
git add alembic/ # or wherever migration lives
git commit -m "feat: add migration for skill-based procedural memory schema"
```

---

### Task 12: Update Existing Tests

**Files:**
- Modify: `tests/test_memory_server.py:246-274,276+,502+`
- Modify: `tests/test_redis_integration.py:1060+`
- Modify: `tests/test_deletion_apis.py:267`

- [ ] **Step 1: Update test_memory_server.py — TestDirectProceduralMemory**

Update `test_insert_procedure` (line 249-274):

```python
async def test_insert_procedure(self, server, client, user, meta_agent):
    """Test inserting a skill directly."""
    procedural_agent = await get_sub_agent(server, client, meta_agent, AgentType.procedural_memory_agent)

    procedure = await server.procedural_memory_manager.insert_procedure(
        agent_state=procedural_agent,
        agent_id=meta_agent.id,
        name="deploy-production",
        description="Deploy application to production",
        instructions="Run all tests\nCreate release branch\nBuild production artifacts\nDeploy to staging\nVerify staging deployment\nDeploy to production",
        entry_type="process",
        triggers=["user mentions deployment", "discussing release"],
        examples=[],
        actor=client,
        organization_id=user.organization_id,
        user_id=user.id,
    )

    assert procedure is not None
    assert procedure.id is not None
    assert procedure.name == "deploy-production"
    assert "Run all tests" in procedure.instructions
    assert procedure.version == "0.1.0"
    print(f"[OK] Inserted skill: {procedure.id}")
```

Update `test_search_procedures` (line 276+) to use `search_field="description"` instead of `"summary"`.

Update `test_procedural_search_methods_and_fields` (line 502+) to use new field names.

- [ ] **Step 2: Update test_redis_integration.py**

Update `TestProceduralMemoryManagerRedis.test_procedural_create_with_embeddings` (line 1060+) to use new field names in test data.

- [ ] **Step 3: Update test_deletion_apis.py**

Update line 267 to access `.description` instead of `.summary`.

- [ ] **Step 4: Run full test suite**

Run: `pytest tests/test_memory_server.py tests/test_redis_integration.py tests/test_deletion_apis.py -v -k "procedural or skill"`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_memory_server.py tests/test_redis_integration.py tests/test_deletion_apis.py
git commit -m "test: update procedural memory tests for skill-based format"
```

---

### Task 13: Final Integration Test

**Files:**
- Create: `tests/test_skill_integration.py`

- [ ] **Step 1: Write integration test for the full skill lifecycle**

```python
# tests/test_skill_integration.py
"""Integration test for skill-based procedural memory lifecycle."""
import pytest
from mirix.schemas.agent import AgentType


@pytest.mark.asyncio(loop_scope="module")
class TestSkillLifecycle:
    """Test the full skill lifecycle: insert → search → update → search → delete."""

    async def test_skill_insert_search_update_delete(self, server, client, user, meta_agent):
        """Full CRUD lifecycle for a skill."""
        from tests.test_memory_server import get_sub_agent

        procedural_agent = await get_sub_agent(server, client, meta_agent, AgentType.procedural_memory_agent)
        manager = server.procedural_memory_manager

        # INSERT
        skill = await manager.insert_procedure(
            agent_state=procedural_agent,
            agent_id=meta_agent.id,
            name="test-skill-lifecycle",
            description="A test skill for lifecycle validation",
            instructions="Step 1: Do something\nStep 2: Do something else",
            entry_type="workflow",
            triggers=["testing lifecycle"],
            examples=[{"input": "test", "output": "result"}],
            actor=client,
            organization_id=user.organization_id,
            user_id=user.id,
        )
        assert skill.id is not None
        assert skill.name == "test-skill-lifecycle"
        assert skill.version == "0.1.0"
        skill_id = skill.id

        # SEARCH by description
        results = await manager.list_procedures(
            agent_state=procedural_agent,
            user=user,
            query="lifecycle validation",
            search_field="description",
            search_method="bm25",
            limit=10,
        )
        assert any(r.id == skill_id for r in results)

        # UPDATE
        from mirix.schemas.procedural_memory import ProceduralMemoryItemUpdate

        updated = await manager.update_item(
            item_update=ProceduralMemoryItemUpdate(
                id=skill_id,
                description="Updated test skill description",
                version="0.1.1",
            ),
            user=user,
            actor=client,
        )
        assert updated.description == "Updated test skill description"
        assert updated.version == "0.1.1"

        # DELETE
        await manager.delete_procedure_by_id(procedure_id=skill_id, actor=client)

        # Verify deletion
        results = await manager.list_procedures(
            agent_state=procedural_agent,
            user=user,
            query="lifecycle validation",
            search_field="description",
            search_method="bm25",
            limit=10,
        )
        assert not any(r.id == skill_id for r in results)

        print("[OK] Full skill lifecycle test passed")
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_skill_integration.py -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v -m "not integration"`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_skill_integration.py
git commit -m "test: add skill lifecycle integration test"
```

---

### Task 14: Dashboard Frontend (if applicable)

**Files:**
- Modify: `dashboard/src/pages/dashboard/Memories.tsx:342,401,451,452,470,473,506,735`

- [ ] **Step 1: Update field references in Memories.tsx**

Replace all procedural memory field references:
- `summary` → `description`
- `steps` → `instructions`
- Add `name` display where skill titles are shown

- [ ] **Step 2: Verify dashboard renders correctly**

Run: `cd dashboard && npm run build`
Expected: No TypeScript errors

- [ ] **Step 3: Commit**

```bash
git add dashboard/
git commit -m "feat: update dashboard to display skill-based procedural memory"
```
