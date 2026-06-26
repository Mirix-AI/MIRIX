"""Memory id generation contract.

Memory rows must use the prefixed-uuid4 id scheme (``<prefix>-<uuid4>``) that
the IPS-R field mapper consumes as a real UUID. The legacy short-id scheme
(``ep_A7K9``) had a ~1.2M keyspace per prefix and collided under load, landing
as ``*_pkey`` ProviderConflictErrors after the deterministic uuid5 mapping.
"""

import inspect
import re
import uuid

from mirix.schemas.episodic_memory import EpisodicEvent as PydanticEpisodicEvent
from mirix.schemas.knowledge_vault import KnowledgeVaultItem as PydanticKnowledgeVaultItem
from mirix.schemas.procedural_memory import ProceduralMemoryItem as PydanticProceduralMemoryItem
from mirix.schemas.raw_memory import RawMemoryItem as PydanticRawMemoryItem
from mirix.schemas.resource_memory import ResourceMemoryItem as PydanticResourceMemoryItem
from mirix.schemas.semantic_memory import SemanticMemoryItem as PydanticSemanticMemoryItem

MEMORY_SCHEMAS = [
    (PydanticEpisodicEvent, "ep_mem"),
    (PydanticSemanticMemoryItem, "sem_item"),
    (PydanticProceduralMemoryItem, "proc_item"),
    (PydanticResourceMemoryItem, "res_item"),
    (PydanticKnowledgeVaultItem, "kv_item"),
    (PydanticRawMemoryItem, "raw_mem"),
]


def _assert_prefixed_uuid4(value: str, prefix: str) -> None:
    assert value.startswith(f"{prefix}-"), f"{value!r} missing {prefix!r} prefix"
    suffix = value[len(prefix) + 1 :]
    # Must be a real UUID (the IPS-R field mapper extracts this as BaseEntity.id).
    parsed = uuid.UUID(suffix)
    assert parsed.version == 4, f"{value!r} suffix is not a uuid4"


class TestMemoryIdContract:
    def test_each_memory_schema_generates_prefixed_uuid4(self):
        for schema, prefix in MEMORY_SCHEMAS:
            generated = schema._generate_id()
            _assert_prefixed_uuid4(generated, prefix)

    def test_ids_are_unique_at_scale(self):
        # The legacy short-id keyspace (26*36**3 ~= 1.2M) hit a birthday-paradox
        # collision well under 2k rows. uuid4 must not.
        for schema, _prefix in MEMORY_SCHEMAS:
            ids = {schema._generate_id() for _ in range(20_000)}
            assert len(ids) == 20_000, f"{schema.__name__} produced a collision"

    def test_no_short_underscore_random_ids(self):
        # Guard against regressing to the `ep_A7K9` form (prefix + underscore +
        # 4 random chars, no uuid).
        legacy = re.compile(r"^[a-z_]+_[A-Z][A-Z0-9]{3}$")
        for schema, _prefix in MEMORY_SCHEMAS:
            generated = schema._generate_id()
            assert not legacy.match(generated), f"{generated!r} looks like a legacy short id"


class TestManagersDoNotUseShortIdGenerator:
    """The 6 memory managers must furnish ids via the prefixed-uuid4 scheme,
    not the legacy short-id keyspace generator that collided under load.
    """

    MANAGERS = [
        "mirix.services.episodic_memory_manager",
        "mirix.services.semantic_memory_manager",
        "mirix.services.procedural_memory_manager",
        "mirix.services.resource_memory_manager",
        "mirix.services.knowledge_vault_manager",
        "mirix.services.raw_memory_manager",
    ]

    def test_no_manager_references_short_id_generator(self):
        import importlib

        offenders = []
        for mod_name in self.MANAGERS:
            mod = importlib.import_module(mod_name)
            src = inspect.getsource(mod)
            if "generate_unique_short_id" in src or "generate_short_id" in src:
                offenders.append(mod_name)
        assert not offenders, f"these managers still use the short-id generator: {offenders}"

    def test_short_id_generators_removed_from_utils(self):
        import mirix.utils as utils

        for name in ("generate_short_id", "generate_unique_short_id", "generate_unique_short_id_async"):
            assert not hasattr(utils, name), f"mirix.utils.{name} should be deleted"
