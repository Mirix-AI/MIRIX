"""Tests for the skill-based ProceduralMemoryItem ORM model."""

import pytest
from sqlalchemy import inspect as sa_inspect

from mirix.orm.procedural_memory import ProceduralMemoryItem


class TestProceduralMemoryORM:
    """Verify the ORM model has all new skill-based columns and no old columns."""

    def _column_names(self):
        mapper = sa_inspect(ProceduralMemoryItem)
        return {col.key for col in mapper.columns}

    def test_has_new_columns(self):
        cols = self._column_names()
        for expected in [
            "name",
            "triggers",
            "examples",
            "version",
            "description",
            "instructions",
            "description_embedding",
            "instructions_embedding",
        ]:
            assert expected in cols, f"Missing expected column: {expected}"

    def test_no_old_columns(self):
        cols = self._column_names()
        for removed in ["summary", "steps", "summary_embedding", "steps_embedding"]:
            assert removed not in cols, f"Old column still present: {removed}"
