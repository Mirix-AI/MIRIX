# Work-item specs (the-loop 3-phase model)

> Per-work-item Kiro-style specs live here, one directory per work item:
> `docs/specs/<id>/` where `<id>` is the ticket id (e.g. `ECMS-42`).
> This repo is one of several story-target repos in the ECMS workspace; the SDLC
> configuration (`.sdlc/config.yaml`, spec templates, workspace registry) lives in the
> **`ecms-parent` workspace hub** (github.intuit.com/expertise-help/ecms-parent), which
> is cloned as the parent directory of this repo in a set-up workspace — hub paths below
> are workspace-local, not paths in this repo.

Each work item is specified in three human-reviewed phases, then executed:

```
docs/specs/<id>/
├── requirements.md   # or bugfix.md for the minimal bug path (phase: requirements-definition)
├── design.md         # phase: design
├── tasks.md          # DAG of TDD tasks (phase: tasks-breakdown)
└── execution-log.md  # self-checking progress ledger (implementation → complete)
```

Templates are instantiated per work item from the hub's `.sdlc/templates/`
(`requirements.md`, `bugfix.md`, `design.md`, `tasks.md`, `execution-log.md`).

## Rules

- **Every work item has a ticket** (Jira, per the hub's `ticketing.system` config).
- **Human review per phase** — each phase is pushed to the work item's evolving PR and
  reviewed there; the phase docs' front-matter `status:`/`approvedBy:` record that review.
- **Single source of truth** — the ticket *references* these files by path; it never
  duplicates them. Subsequent changes are edits to the docs, not new ticket comments.
- **Phase tags** — the work item's phase is mirrored as a ticket label
  `<phaseLabelPrefix><phase>` (e.g. `loop:design`).
- **Paper trail** — every decision/opinion taken from a human lands as a comment on
  the ticket or PR; messaging channels only *signal* that attention is needed.
- **Which repo gets the spec** — a story's spec lives in its Target Repo per the hub's
  `.sdlc/workspace.md`; cross-cutting stories default to `context-and-memory-service`.
  The hub repo itself tracks no specs.
