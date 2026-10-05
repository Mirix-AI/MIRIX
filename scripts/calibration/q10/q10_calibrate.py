"""Calibrate the skill retrieval gate on gemini-embedding-001.

The retrieval gate (``MIRIX_SKILL_RETRIEVAL_MIN_SIMILARITY`` /
``MIRIX_SKILL_RETRIEVAL_RELATIVE_RATIO``) keeps a skill only when its cosine
similarity to the query reaches an absolute floor AND a ratio of the best
candidate's cosine. This script measures the cosine distributions those two
numbers are chosen from, on ``q10_calibration_data.json``: 30 skills from
neutral domains (none from ALFWorld, OfficeQA or DocVQA), 60 queries with one
relevant skill each (a natural-language form and a topic-keyword form, the
shape MIRIX topic extraction produces), and 16 queries no skill matches.

Each skill is embedded as its retrieval text (name, triggers one per line,
description: ``mirix.services.skill_history.skill_retrieval_text``), exactly
what MIRIX stores in ``retrieval_embedding``. It writes ``q10_results.json``:

* ``summary`` -- top-1 accuracy; percentiles of the relevant and non-relevant
  cosines, of each one's ratio to the query's best cosine, and of the best
  cosine an unmatched query reaches; token usage and cost;
* ``per_query`` -- each query's relevant, best and best non-relevant cosine;
* ``grid`` -- floor 0.50..0.80 x ratio 0.80..0.95: recall of the relevant
  skill, non-relevant skills passed per query, and skills passed for the
  unmatched queries.

Re-running it (it embeds the set through OpenRouter; nothing else calls a
provider)::

    export OPENROUTER_API_KEY=...        # read from the environment only
    .venv/bin/python scripts/calibration/q10/q10_calibrate.py \\
        --embeddings-cache /tmp/q10_embeddings.json

``--embeddings-cache`` keeps the vectors (about 7 MB, deliberately not
committed) so the analysis can be redone with ``--offline`` and no key.
``--out`` writes the results elsewhere instead of over the committed
``q10_results.json``; ``--endpoint`` points at another OpenAI-compatible
``/embeddings`` URL (the tests use the offline fake server). The key is
never printed or written anywhere, and it is only ever sent where it belongs:

* the endpoint must be ``https://``, or ``http://`` on a loopback host (the
  offline fake server); anything else is refused before a request is made,
  and an endpoint other than OpenRouter's gets a warning on stderr;
* a redirect is never followed (``urllib`` would turn the POST into a GET and
  forward the ``Authorization`` header, to another host too);
* only a transient failure is retried (429, 5xx, a network error or a
  timeout); a 400, 401, 403 ... stops at once, so a bad or revoked key is
  sent once, not four times.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "q10_calibration_data.json"
RESULTS_PATH = HERE / "q10_results.json"
MODEL = "google/gemini-embedding-001"
ENDPOINT = "https://openrouter.ai/api/v1/embeddings"
KEY_ENV = "OPENROUTER_API_KEY"
BATCH = 32
FLOORS = [x / 100 for x in range(50, 81, 2)]
RATIOS = [0.80, 0.85, 0.88, 0.90, 0.92, 0.95]

Vectors = Dict[str, List[float]]


def retrieval_text(skill: Dict[str, Any]) -> str:
    """A skill's retrieval text, as ``skill_retrieval_text`` builds it."""
    parts = [
        skill.get("name"),
        *(skill.get("triggers") or []),
        skill.get("description"),
    ]
    return "\n".join(str(p) for p in parts if p)


def texts_to_embed(data: Dict[str, Any]) -> Dict[str, str]:
    """``{id: text}`` for every skill, query and unmatched query."""
    texts = {s["id"]: retrieval_text(s) for s in data["skills"]}
    texts.update({q["id"]: q["text"] for q in data["queries"]})
    texts.update({u["id"]: u["text"] for u in data["unmatched"]})
    return texts


#: HTTP statuses worth another attempt: rate limiting and server-side failures.
RETRY_STATUSES = frozenset({408, 429}) | frozenset(range(500, 600))
ATTEMPTS = 4


class EndpointRefused(ValueError):
    """An ``--endpoint`` the key must not be sent to."""


def check_endpoint(endpoint: str) -> Optional[str]:
    """Refuse an endpoint the key would leak to; return a warning for a non-default one.

    ``https://`` anywhere is allowed; ``http://`` only on a loopback host (the
    offline fake server), since plain HTTP would carry the key in clear text.
    """
    parts = urllib.parse.urlsplit(endpoint)
    host = parts.hostname or ""
    if parts.scheme == "https" and host:
        pass
    elif parts.scheme == "http" and _is_loopback(host):
        pass
    else:
        raise EndpointRefused(
            f"refusing to send the key to {endpoint!r}: use https://, or http:// "
            "on a loopback host"
        )
    if endpoint != ENDPOINT:
        return f"warning: sending the key to {endpoint}, not to {ENDPOINT}"
    return None


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect: urllib would re-send the Authorization header."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _post(request: urllib.request.Request, timeout: float) -> Any:
    with _OPENER.open(request, timeout=timeout) as response:
        return json.load(response)


def _transient(exc: BaseException) -> bool:
    """A failure worth another attempt: 408/429/5xx, a network error, a timeout."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in RETRY_STATUSES
    return isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError))


def fetch_embeddings(
    texts: Dict[str, str],
    todo: Sequence[str],
    *,
    api_key: str,
    endpoint: str = ENDPOINT,
    model: str = MODEL,
    sleep: Callable[[float], Any] = time.sleep,
) -> tuple[Vectors, Dict[str, Any]]:
    """Embed ``todo`` in batches of 32. A batch is tried up to 4 times on a
    transient failure only (:func:`_transient`); any other failure, and a
    redirect, stops the run at once (nothing is written)."""
    vectors: Vectors = {}
    usage: Dict[str, Any] = {"prompt_tokens": 0, "cost": 0.0}
    for start in range(0, len(todo), BATCH):
        ids = list(todo[start : start + BATCH])
        body = json.dumps({"model": model, "input": [texts[k] for k in ids]}).encode()
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        for attempt in range(ATTEMPTS):
            try:
                answer = _post(request, timeout=120)
                break
            except Exception as exc:  # noqa: BLE001 - classified below
                status = getattr(exc, "code", None)
                label = f"{type(exc).__name__}" + (f" {status}" if status else "")
                if not _transient(exc):
                    raise SystemExit(
                        f"embedding failed with {label} (not retried); nothing written"
                    ) from None
                print(f"retry {attempt}: {label}", file=sys.stderr)
                if attempt + 1 < ATTEMPTS:
                    sleep(2**attempt)
        else:
            raise SystemExit(f"embedding failed {ATTEMPTS} times; nothing written")
        rows = sorted(answer["data"], key=lambda row: row["index"])
        if len(rows) != len(ids):
            raise SystemExit(f"asked for {len(ids)} embeddings, got {len(rows)}")
        for key, row in zip(ids, rows):
            vectors[key] = row["embedding"]
        batch_usage = answer.get("usage") or {}
        usage["prompt_tokens"] += batch_usage.get("prompt_tokens", 0) or 0
        usage["cost"] += float(batch_usage.get("cost", 0) or 0)
    return vectors, usage


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    )


def percentile(values: Iterable[float], p: float) -> float:
    """Linear interpolation between closest ranks."""
    xs = sorted(values)
    k = (len(xs) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


Scores = Dict[str, Dict[str, float]]


def score_table(data: Dict[str, Any], vectors: Vectors) -> Scores:
    """``{query id: {skill id: cosine}}`` for every query and unmatched query,
    computed once (the grid reuses it)."""
    skills = [s["id"] for s in data["skills"]]
    return {
        q["id"]: {s: cosine(vectors[q["id"]], vectors[s]) for s in skills}
        for q in [*data["queries"], *data["unmatched"]]
    }


def gate_eval(
    data: Dict[str, Any], scores: Scores, floor: float, ratio: float
) -> Dict[str, Any]:
    """What the gate (cos >= floor AND cos >= best x ratio) lets through."""
    hits = nonrelevant_passed = 0
    for query in data["queries"]:
        row = scores[query["id"]]
        best = max(row.values())
        passed = {s for s, v in row.items() if v >= floor and v >= best * ratio}
        hits += int(query["rel"] in passed)
        nonrelevant_passed += len(passed - {query["rel"]})
    unmatched_passed = 0
    for query in data["unmatched"]:
        row = scores[query["id"]]
        best = max(row.values())
        unmatched_passed += sum(
            1 for v in row.values() if v >= floor and v >= best * ratio
        )
    return {
        "floor": floor,
        "ratio": ratio,
        "recall": hits / len(data["queries"]),
        "nonrel_passed_per_query": nonrelevant_passed / len(data["queries"]),
        "unmatched_queries_items_passed": unmatched_passed,
        "unmatched_queries": len(data["unmatched"]),
    }


def analyse(
    data: Dict[str, Any], vectors: Vectors, usage: Dict[str, Any], model: str = MODEL
) -> Dict[str, Any]:
    """The ``{summary, per_query, grid}`` document of ``q10_results.json``."""
    skills = [s["id"] for s in data["skills"]]
    table = score_table(data, vectors)
    relevant, relevant_ratio, nonrelevant, nonrelevant_ratio = [], [], [], []
    unmatched_best: List[float] = []
    top1 = 0
    per_query = []
    for query in data["queries"]:
        scores = table[query["id"]]
        best = max(scores.values())
        rel = scores[query["rel"]]
        relevant.append(rel)
        relevant_ratio.append(rel / best)
        top1 += int(rel == best)
        for skill, value in scores.items():
            if skill != query["rel"]:
                nonrelevant.append(value)
                nonrelevant_ratio.append(value / best)
        per_query.append(
            {
                "id": query["id"],
                "rel_cos": round(rel, 4),
                "max_cos": round(best, 4),
                "top": max(scores, key=lambda s: scores[s]),
                "second_best_nonrel": round(
                    max(v for s, v in scores.items() if s != query["rel"]), 4
                ),
            }
        )
    for query in data["unmatched"]:
        unmatched_best.append(max(table[query["id"]].values()))

    def pcts(values: List[float], ps: Sequence[float]) -> Dict[str, float]:
        return {str(p): round(percentile(values, p), 4) for p in ps}

    grid = [
        gate_eval(data, table, round(f, 2), round(r, 2)) for f in FLOORS for r in RATIOS
    ]
    summary = {
        "model": model,
        "n_skills": len(skills),
        "n_queries": len(data["queries"]),
        "n_unmatched": len(data["unmatched"]),
        "usage": usage,
        "top1_accuracy": top1 / len(data["queries"]),
        "relevant_cos": pcts(relevant, (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)),
        "nonrelevant_cos": pcts(nonrelevant, (0.5, 0.9, 0.95, 0.99, 1.0)),
        "relevant_ratio_to_max": pcts(relevant_ratio, (0.0, 0.05, 0.1, 0.5)),
        "nonrelevant_ratio_to_max": pcts(nonrelevant_ratio, (0.5, 0.9, 0.95, 0.99)),
        "unmatched_max_cos": pcts(unmatched_best, (0.0, 0.5, 0.9, 1.0)),
    }
    return {"summary": summary, "per_query": per_query, "grid": grid}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    parser.add_argument(
        "--embeddings-cache",
        type=Path,
        default=None,
        help="JSON file of vectors to reuse and extend (not committed: ~7 MB)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use only --embeddings-cache; fail if a vector is missing",
    )
    parser.add_argument("--endpoint", default=ENDPOINT)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args(argv)

    data = json.loads(args.data.read_text())
    texts = texts_to_embed(data)
    cache: Vectors = {}
    if args.embeddings_cache and args.embeddings_cache.exists():
        cache = json.loads(args.embeddings_cache.read_text())
    todo = [k for k in texts if k not in cache]
    usage: Dict[str, Any] = {"prompt_tokens": 0, "cost": 0.0}
    if todo:
        if args.offline:
            print(
                f"--offline: {len(todo)} vectors missing from the cache",
                file=sys.stderr,
            )
            return 2
        api_key = os.environ.get(KEY_ENV)
        if not api_key:
            print(f"set {KEY_ENV} (read from the environment only)", file=sys.stderr)
            return 2
        try:
            warning = check_endpoint(args.endpoint)
        except EndpointRefused as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if warning:
            print(warning, file=sys.stderr)
        fetched, usage = fetch_embeddings(
            texts, todo, api_key=api_key, endpoint=args.endpoint, model=args.model
        )
        cache.update(fetched)
        if args.embeddings_cache:
            args.embeddings_cache.write_text(json.dumps(cache))
    results = analyse(data, cache, usage, model=args.model)
    args.out.write_text(json.dumps(results, indent=1))
    print(json.dumps(results["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
