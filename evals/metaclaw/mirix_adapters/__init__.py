"""MIRIX-backed adapters for MetaClaw's SkillManager / SkillEvolver duck-types.

Current production-aligned surface:

* ``MirixSkillsAdapter`` retrieves procedural memory through ``/memory/search``.
* ``MirixGenericMemoryAdapter`` ingests visible turns through ``/memory/add_sync``.

Direct skill lifecycle/evolution endpoints are intentionally not modeled here.
"""
