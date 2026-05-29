"""MIRIX-backed adapters for MetaClaw's SkillManager / SkillEvolver duck-types.

Populated incrementally across the multi-slice plan:

* slice #2 — :mod:`._stub` no-op adapters (this slice; lets the D6 dispatch
  wiring be exercised without requiring a live MIRIX deployment).
* slice #3 — ``MirixSkillsAdapter`` (real ``retrieve``/``add_skill`` against
  MIRIX procedural memory).
* slice #4 — ``MirixEvolverAdapter`` (real ``evolve`` writes new procedures).
"""
