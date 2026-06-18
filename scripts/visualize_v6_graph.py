#!/usr/bin/env python3
"""Generate Graphviz visualizations for the v6 Neo4j graph.

The output is intentionally sampled. Rendering all nodes in the LongMem-S
graph is unreadable, so this script emits:

- an overview with the highest-degree entities and a few memory refs each
- topic-focused subgraphs with more memory refs per seed entity
"""

from __future__ import annotations

import argparse
import html
import os
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from neo4j import GraphDatabase


ENTITY_COLORS = {
    "Person": ("#d8e9ff", "#3a6ea5"),
    "Location": ("#dff4df", "#4a8f4a"),
    "Organization": ("#ffe7bf", "#bf7a12"),
    "Concept": ("#eee4ff", "#7c5bc7"),
    "Content": ("#ffdcdc", "#bf4f4f"),
    "Event": ("#e4f0ff", "#4b6fb5"),
    "Method": ("#e0f7f4", "#368a7f"),
    "Other": ("#eeeeee", "#777777"),
}


@dataclass
class Entity:
    id: str
    name: str
    entity_type: str
    degree: int


@dataclass
class MemoryRef:
    id: str
    memory_type: str
    summary: str
    title: str | None
    occurred_at: str | None


@dataclass
class Edge:
    entity_id: str
    memory_id: str
    rel_type: str


def dot_escape(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("\\", "\\\\").replace('"', '\\"')


def dot_id(prefix: str, value: str) -> str:
    clean = "".join(ch if ch.isalnum() else "_" for ch in value)
    return f"{prefix}_{clean}"


def wrap(value: str, width: int = 34, max_chars: int = 150) -> str:
    text = " ".join((value or "").split())
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "..."
    return "\\n".join(textwrap.wrap(text, width=width)) or "(empty)"


def graph_driver():
    uri = os.environ.get("MIRIX_NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("MIRIX_NEO4J_USER", "neo4j")
    password = os.environ.get("MIRIX_NEO4J_PASSWORD", "mirix_neo4j_dev")
    return GraphDatabase.driver(uri, auth=(user, password))


def fetch_top_entities(driver, user_id: str, limit: int) -> list[Entity]:
    query = """
    MATCH (e:V6Entity {user_id: $user_id})-[r]-(m:V6MemoryRef)
    WITH e, count(r) AS degree
    RETURN e.id AS id, e.name AS name, e.entity_type AS entity_type, degree
    ORDER BY degree DESC, name ASC
    LIMIT $limit
    """
    with driver.session(database=os.environ.get("MIRIX_NEO4J_DATABASE", "neo4j")) as session:
        return [
            Entity(
                id=rec["id"],
                name=rec["name"] or "",
                entity_type=rec["entity_type"] or "Other",
                degree=int(rec["degree"] or 0),
            )
            for rec in session.run(query, user_id=user_id, limit=limit)
        ]


def fetch_topic_entities(driver, user_id: str, terms: list[str], limit: int) -> list[Entity]:
    query = """
    MATCH (e:V6Entity {user_id: $user_id})-[r]-(m:V6MemoryRef)
    WITH e, count(r) AS degree
    WHERE any(term IN $terms WHERE toLower(e.name) CONTAINS term)
    RETURN e.id AS id, e.name AS name, e.entity_type AS entity_type, degree
    ORDER BY degree DESC, name ASC
    LIMIT $limit
    """
    with driver.session(database=os.environ.get("MIRIX_NEO4J_DATABASE", "neo4j")) as session:
        return [
            Entity(
                id=rec["id"],
                name=rec["name"] or "",
                entity_type=rec["entity_type"] or "Other",
                degree=int(rec["degree"] or 0),
            )
            for rec in session.run(query, user_id=user_id, terms=[t.lower() for t in terms], limit=limit)
        ]


def fetch_edges_for_entities(
    driver, entity_ids: list[str], refs_per_entity: int
) -> tuple[dict[str, MemoryRef], list[Edge]]:
    if not entity_ids:
        return {}, []
    query = """
    UNWIND $entity_ids AS entity_id
    MATCH (e:V6Entity {id: entity_id})-[r]-(m:V6MemoryRef)
    WITH e, r, m
    ORDER BY e.id, type(r), coalesce(m.occurred_at, m.semantic_created_at, m.created_at) DESC
    WITH e, collect({
      rel_type: type(r),
      memory_id: m.id,
      memory_type: coalesce(m.memory_type, CASE WHEN m:V6EpisodeRef THEN 'episodic' ELSE 'semantic' END),
      summary: coalesce(m.summary, ''),
      title: coalesce(m.title, ''),
      occurred_at: coalesce(toString(m.occurred_at), toString(m.semantic_created_at), toString(m.created_at))
    })[..$refs_per_entity] AS refs
    RETURN e.id AS entity_id, refs
    """
    memories: dict[str, MemoryRef] = {}
    edges: list[Edge] = []
    with driver.session(database=os.environ.get("MIRIX_NEO4J_DATABASE", "neo4j")) as session:
        for rec in session.run(query, entity_ids=entity_ids, refs_per_entity=refs_per_entity):
            entity_id = rec["entity_id"]
            for ref in rec["refs"]:
                memory_id = ref["memory_id"]
                memories[memory_id] = MemoryRef(
                    id=memory_id,
                    memory_type=ref["memory_type"] or "",
                    summary=ref["summary"] or "",
                    title=ref["title"] or None,
                    occurred_at=ref["occurred_at"] or None,
                )
                edges.append(Edge(entity_id=entity_id, memory_id=memory_id, rel_type=ref["rel_type"]))
    return memories, edges


def make_dot(
    *,
    title: str,
    entities: list[Entity],
    memories: dict[str, MemoryRef],
    edges: list[Edge],
) -> str:
    lines = [
        "digraph G {",
        '  graph [rankdir=LR, bgcolor="white", splines=true, overlap=false, nodesep=0.35, ranksep=1.0];',
        '  node [fontname="Helvetica", fontsize=10, style="filled,rounded", penwidth=1.2];',
        '  edge [fontname="Helvetica", fontsize=8, color="#9aa4b2", arrowsize=0.55];',
        f'  label="{dot_escape(title)}";',
        '  labelloc="t";',
        '  fontsize=20;',
        "",
        "  subgraph cluster_entities {",
        '    label="V6Entity";',
        '    color="#d8dee9";',
        '    style="rounded";',
    ]

    for entity in entities:
        fill, border = ENTITY_COLORS.get(entity.entity_type, ENTITY_COLORS["Other"])
        label = f"{wrap(entity.name, 24, 80)}\\n({entity.entity_type}, d={entity.degree})"
        lines.append(
            f'    {dot_id("e", entity.id)} [shape=ellipse, fillcolor="{fill}", color="{border}", label="{dot_escape(label)}"];'
        )

    lines += [
        "  }",
        "",
        "  subgraph cluster_memories {",
        '    label="V6MemoryRef -> PG flat memory";',
        '    color="#d8dee9";',
        '    style="rounded";',
    ]

    for memory in memories.values():
        if memory.memory_type == "episodic":
            fill, border, shape = "#fff2bf", "#b38a00", "box"
            prefix = "E"
        else:
            fill, border, shape = "#dff7ff", "#2f7f9f", "component"
            prefix = "S"
        stamp = f"{memory.occurred_at[:10]}\\n" if memory.occurred_at else ""
        summary = memory.title or memory.summary or memory.id
        label = f"{prefix}: {stamp}{wrap(summary, 34, 135)}"
        lines.append(
            f'    {dot_id("m", memory.id)} [shape={shape}, fillcolor="{fill}", color="{border}", label="{dot_escape(label)}"];'
        )

    lines += ["  }", ""]

    for edge in edges:
        color = "#5f8dd3" if edge.rel_type == "APPEARS_IN" else "#9b6bd3"
        lines.append(
            f'  {dot_id("e", edge.entity_id)} -> {dot_id("m", edge.memory_id)} '
            f'[label="{dot_escape(edge.rel_type)}", color="{color}"];'
        )

    lines.append("}")
    return "\n".join(lines)


def render_dot(dot: str, out_dot: Path, out_svg: Path) -> None:
    out_dot.write_text(dot, encoding="utf-8")
    subprocess.run(["dot", "-Tsvg", str(out_dot), "-o", str(out_svg)], check=True)
    subprocess.run(["dot", "-Tpng", str(out_dot), "-o", str(out_svg.with_suffix(".png"))], check=True)


def write_index(out_dir: Path, figures: list[tuple[str, str, int, int]]) -> None:
    sections = []
    for title, filename, node_n, edge_n in figures:
        sections.append(
            f"""
            <section>
              <h2>{html.escape(title)}</h2>
              <p>{node_n} nodes, {edge_n} sampled edges</p>
              <object data="{html.escape(filename)}" type="image/svg+xml"></object>
            </section>
            """
        )
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>v6 LongMem-S Graph Visualizations</title>
  <style>
    body {{ font: 15px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2937; }}
    h1 {{ margin-bottom: 4px; }}
    .note {{ max-width: 920px; color: #4b5563; }}
    section {{ margin-top: 36px; }}
    object {{ width: 100%; min-height: 760px; border: 1px solid #d1d5db; border-radius: 8px; background: white; }}
  </style>
</head>
<body>
  <h1>v6 LongMem-S Graph Visualizations</h1>
  <p class="note">
    Sampled from <code>user_id=longmem_s_0</code>. The full graph has
    7,281 nodes and 14,125 edges; these views show readable slices.
    Entity nodes link to V6MemoryRef nodes, which point back to PostgreSQL
    episodic and semantic flat memory.
  </p>
  {''.join(sections)}
</body>
</html>
"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def build_view(
    *,
    driver,
    user_id: str,
    out_dir: Path,
    slug: str,
    title: str,
    entities: list[Entity],
    refs_per_entity: int,
) -> tuple[str, str, int, int]:
    memories, edges = fetch_edges_for_entities(driver, [e.id for e in entities], refs_per_entity)
    dot = make_dot(title=title, entities=entities, memories=memories, edges=edges)
    dot_path = out_dir / f"{slug}.dot"
    svg_path = out_dir / f"{slug}.svg"
    render_dot(dot, dot_path, svg_path)
    return title, svg_path.name, len(entities) + len(memories), len(edges)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", default="longmem_s_0")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/graph_memory_v6/visualizations"),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures: list[tuple[str, str, int, int]] = []

    topics = [
        ("aquarium", "Aquarium / Fish Detail", ["aquarium", "fish", "betta", "gourami", "tetra", "pleco"], 14, 8),
        ("career", "Career / Campaign Work Detail", ["campaign", "work hours", "resume", "linkedin", "marketing"], 14, 8),
        ("fitness", "Fitness / Health Devices Detail", ["fitness", "bodypump", "zumba", "yoga", "fitbit", "health"], 14, 8),
        ("travel", "Travel / Places Detail", ["vatican", "rome", "speyer", "moncayo", "hawaii", "yosemite"], 14, 8),
    ]

    with graph_driver() as driver:
        overview_entities = fetch_top_entities(driver, args.user_id, 28)
        figures.append(
            build_view(
                driver=driver,
                user_id=args.user_id,
                out_dir=args.out_dir,
                slug="overview_top_entities",
                title="Overview: Top-Degree Entities",
                entities=overview_entities,
                refs_per_entity=4,
            )
        )

        for slug, title, terms, entity_limit, refs_per_entity in topics:
            entities = fetch_topic_entities(driver, args.user_id, terms, entity_limit)
            figures.append(
                build_view(
                    driver=driver,
                    user_id=args.user_id,
                    out_dir=args.out_dir,
                    slug=f"topic_{slug}",
                    title=title,
                    entities=entities,
                    refs_per_entity=refs_per_entity,
                )
            )

    write_index(args.out_dir, figures)
    print(f"Wrote {len(figures)} graph views to {args.out_dir}")
    for title, filename, node_n, edge_n in figures:
        print(f"- {filename}: {title} ({node_n} nodes, {edge_n} edges)")


if __name__ == "__main__":
    main()
