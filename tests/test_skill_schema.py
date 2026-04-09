"""Tests for the skill-based procedural memory schema."""

import pytest

from mirix.schemas.procedural_memory import (
    ProceduralMemoryItem,
    ProceduralMemoryItemBase,
    ProceduralMemoryItemUpdate,
    ProceduralMemoryItemResponse,
)


class TestProceduralMemoryItemBase:
    """Tests for ProceduralMemoryItemBase skill schema."""

    def test_name_is_required(self):
        """name field must be provided; omitting it raises a ValidationError."""
        with pytest.raises(Exception):
            ProceduralMemoryItemBase(
                entry_type="workflow",
                description="Deploy the app",
                instructions="Step 1: build\nStep 2: deploy",
            )

    def test_valid_skill_creation(self):
        """All required fields produce a valid skill object."""
        item = ProceduralMemoryItemBase(
            name="deploy-production",
            entry_type="workflow",
            description="Deploy the production application",
            instructions="Step 1: build the image\nStep 2: push to registry\nStep 3: deploy",
        )
        assert item.name == "deploy-production"
        assert item.entry_type == "workflow"
        assert item.description == "Deploy the production application"
        assert isinstance(item.instructions, str)

    def test_triggers_defaults_to_empty_list(self):
        """triggers should default to an empty list."""
        item = ProceduralMemoryItemBase(
            name="deploy-production",
            entry_type="workflow",
            description="Deploy the production application",
            instructions="Step 1: build",
        )
        assert item.triggers == []

    def test_examples_defaults_to_empty_list(self):
        """examples should default to an empty list."""
        item = ProceduralMemoryItemBase(
            name="deploy-production",
            entry_type="workflow",
            description="Deploy the production application",
            instructions="Step 1: build",
        )
        assert item.examples == []

    def test_triggers_and_examples_can_be_set(self):
        """triggers and examples can be explicitly provided."""
        item = ProceduralMemoryItemBase(
            name="deploy-production",
            entry_type="workflow",
            description="Deploy the production application",
            instructions="Step 1: build",
            triggers=["user says deploy", "CI pipeline triggers"],
            examples=[{"input": "deploy now", "output": "deployed v1.2.3"}],
        )
        assert len(item.triggers) == 2
        assert len(item.examples) == 1


class TestProceduralMemoryItem:
    """Tests for the full ProceduralMemoryItem with DB fields."""

    def test_version_defaults_to_0_1_0(self):
        """version should default to '0.1.0'."""
        item = ProceduralMemoryItem(
            name="deploy-production",
            entry_type="workflow",
            description="Deploy the app",
            instructions="Step 1: build",
            user_id="user-123",
            organization_id="org-456",
        )
        assert item.version == "0.1.0"

    def test_description_embedding_defaults_to_none(self):
        """description_embedding should be Optional and default to None."""
        item = ProceduralMemoryItem(
            name="deploy-production",
            entry_type="workflow",
            description="Deploy the app",
            instructions="Step 1: build",
            user_id="user-123",
            organization_id="org-456",
        )
        assert item.description_embedding is None

    def test_instructions_embedding_defaults_to_none(self):
        """instructions_embedding should be Optional and default to None."""
        item = ProceduralMemoryItem(
            name="deploy-production",
            entry_type="workflow",
            description="Deploy the app",
            instructions="Step 1: build",
            user_id="user-123",
            organization_id="org-456",
        )
        assert item.instructions_embedding is None

    def test_version_can_be_set(self):
        """version can be explicitly provided."""
        item = ProceduralMemoryItem(
            name="deploy-production",
            entry_type="workflow",
            description="Deploy the app",
            instructions="Step 1: build",
            user_id="user-123",
            organization_id="org-456",
            version="1.2.3",
        )
        assert item.version == "1.2.3"


class TestProceduralMemoryItemUpdate:
    """Tests for the update schema."""

    def test_update_has_renamed_fields(self):
        """Update schema uses description/instructions instead of summary/steps."""
        update = ProceduralMemoryItemUpdate(
            id="proc_item-abc12345",
            description="Updated description",
            instructions="Updated instructions",
        )
        assert update.description == "Updated description"
        assert update.instructions == "Updated instructions"

    def test_update_has_name_field(self):
        """Update schema includes optional name field."""
        update = ProceduralMemoryItemUpdate(
            id="proc_item-abc12345",
            name="new-skill-name",
        )
        assert update.name == "new-skill-name"


class TestProceduralMemoryItemResponse:
    """Tests for the response schema."""

    def test_response_inherits_from_item(self):
        """Response schema should be a subclass of ProceduralMemoryItem."""
        assert issubclass(ProceduralMemoryItemResponse, ProceduralMemoryItem)
