"""Render a v7.10 HYPERGRAPH neighborhood: V7Fact nodes (reified triples) as
hyperedges linking a SUBJECT anchor + OBJECT anchor + the source MEMORY, stamped
with role. Shows the distinctive shape — facts are first-class nodes, not edges.
"""
import os
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from neo4j import GraphDatabase

U = "longmem_s_0"
d = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "mirix_neo4j_dev"))

# Two thematically-clean memories (a travel/flight cluster) so the picture reads.
SEED_MEMS = ["ep_DU49", "ep_EJ78"]
rows = []
with d.session() as s:
    # Pick facts seeded from these memories, then show ALL their source memories so the
    # post-refinement multi-citation structure (one fact, N citations) is visible.
    q = """
    MATCH (f:V7Fact {user_id:$u})-[:V7_FACT_FROM]->(seed:V7MemoryRef)
    WHERE seed.memory_id IN $mids
    MATCH (subj:V7Anchor)<-[:V7_FACT_SUBJECT]-(f)-[:V7_FACT_OBJECT]->(obj:V7Anchor)
    MATCH (f)-[:V7_FACT_FROM]->(m:V7MemoryRef)
    WHERE subj.name IS NOT NULL AND obj.name IS NOT NULL
    RETURN subj.name AS s, f.predicate AS p, obj.name AS o, f.role AS role,
           collect(DISTINCT m.memory_id) AS mids
    """
    for r in s.run(q, u=U, mids=SEED_MEMS):
        for mid in r["mids"]:
            rows.append((r["s"], r["p"], r["o"], r["role"] or "shared", mid))
d.close()

G = nx.Graph()
kind = {}
def short(t, w=16):
    return "\n".join(textwrap.wrap(t, w))[:60]

for s_, p, o, role, mid in rows:
    # key the fact by its triple so a fact cited by N memories stays ONE node
    fid = f"F::{s_}|{p}|{o}"
    sN, oN, mN = f"A::{s_}", f"A::{o}", f"M::{mid}"
    for n, k, lbl in [(sN, "anchor", short(s_)), (oN, "anchor", short(o)),
                      (mN, "memory", mid), (fid, "fact", p)]:
        if n not in G:
            G.add_node(n, label=lbl); kind[n] = k
    G.add_edge(fid, sN, etype="subj")
    G.add_edge(fid, oN, etype="obj")
    G.add_edge(fid, mN, etype="from")

pos = nx.spring_layout(G, k=0.9, iterations=200, seed=7)
plt.figure(figsize=(17, 12))

col = {"anchor": "#4C78A8", "memory": "#E45756", "fact": "#F2C744"}
size = {"anchor": 1500, "memory": 2600, "fact": 700}
for k in col:
    ns = [n for n in G if kind[n] == k]
    nx.draw_networkx_nodes(G, pos, nodelist=ns, node_color=col[k], node_size=size[k],
                           node_shape=("s" if k == "memory" else ("D" if k == "fact" else "o")),
                           edgecolors="white", linewidths=1.5)
ecol = {"subj": "#59A14F", "obj": "#B279A2", "from": "#BAB0AC"}
for et, c in ecol.items():
    es = [(a, b) for a, b, dd in G.edges(data=True) if dd["etype"] == et]
    nx.draw_networkx_edges(G, pos, edgelist=es, edge_color=c, width=1.6, alpha=0.8)

anchor_lbls = {n: G.nodes[n]["label"] for n in G if kind[n] == "anchor"}
mem_lbls = {n: G.nodes[n]["label"] for n in G if kind[n] == "memory"}
nx.draw_networkx_labels(G, pos, labels=anchor_lbls, font_size=7.5, font_color="white")
nx.draw_networkx_labels(G, pos, labels=mem_lbls, font_size=9, font_color="white", font_weight="bold")
fact_lbls = {n: G.nodes[n]["label"] for n in G if kind[n] == "fact"}
nx.draw_networkx_labels(G, pos, labels=fact_lbls, font_size=6, font_color="#5a4a00")

legend = [mpatches.Patch(color=col["anchor"], label="V7Anchor (entity/concept)"),
          mpatches.Patch(color=col["fact"], label="V7Fact  (reified triple = HYPEREDGE)"),
          mpatches.Patch(color=col["memory"], label="V7MemoryRef (source memory)"),
          mpatches.Patch(color=ecol["subj"], label="FACT_SUBJECT"),
          mpatches.Patch(color=ecol["obj"], label="FACT_OBJECT"),
          mpatches.Patch(color=ecol["from"], label="FACT_FROM")]
plt.legend(handles=legend, loc="upper left", fontsize=9, framealpha=0.9)
plt.title(f"v7.10 hypergraph neighborhood — {len(rows)} V7Fact hyperedges from 2 memories "
          f"(longmem_s_0)\nEach yellow diamond is a fact linking a subject + object anchor to its source memory",
          fontsize=12)
plt.axis("off"); plt.tight_layout()
OUT = os.path.expanduser("~/MIRIX_eval/hypergraph_neighborhood.png")
plt.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"wrote {OUT}  ({len(rows)} facts, {G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
