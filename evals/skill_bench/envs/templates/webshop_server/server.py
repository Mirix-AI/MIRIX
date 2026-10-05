"""WebShop text-environment server for skill_bench.

Runs inside the image built from this directory (``Dockerfile``: python 3.9, openjdk 11,
upstream princeton-nlp/WebShop at a pinned commit, the full product set and its Lucene
index). One process holds one upstream ``SimServer``, built once, and every episode of
every cell is a ``WebAgentTextEnv`` sharing it, as AgentBench runs WebShop. The harness
side is ``evals/skill_bench/envs/webshop.py``; it talks to this server over loopback HTTP.

What this wrapper adds to upstream (whose code it runs unchanged):

* **Determinism.** ``random`` is seeded with the protocol seed immediately before
  ``SimServer()`` is built, because upstream draws product prices (``engine.py:180-192``,
  inside ``load_products``) and each goal's price limit (``goal.py:39``) from the global
  random state. In ``serve`` mode the server then compares every goal of the frozen
  manifest (the split files mounted at ``--manifest-dir``), the sha256 of the whole
  price table and the image's upstream commit, data hashes and index record hash (its
  ``build.json``) with the frozen values and refuses every episode on any difference
  (``/ready`` answers ``failed`` with the differences).
* **One upstream thread, one global lock.** Upstream is imported (which starts the JVM
  pyserini searches with) and called on one long-lived worker thread only, under one
  lock: upstream has no locks, the thread safety of pyjnius/Lucene is not established,
  and no HTTP connection thread ever attaches to the JVM.
* **Actions the shop cannot process.** An action on which upstream's own logic raises
  (a Python error of its page code, or Lucene's query-size limit) is treated as an
  action the shop does not accept: the session is restored to what it was before the
  action, the page stays, and the reply names the error (``shop_error``; counted on
  ``/ready``). Any other error (memory, I/O, the JVM) is the server's: a 500.
* **Session isolation.** Each episode has its own session prefix (a uuid the harness
  sends); upstream keys sessions by ``prefix + goal index``, so two cells playing one goal
  never share state. Deleting an episode removes every upstream session of its prefix,
  the one ``WebAgentTextEnv()``'s constructor opens with a random goal included, and
  sessions idle longer than ``--session-idle-s`` are dropped.
* **Refused special searches.** A ``search[...]`` whose first word is
  ``<x>`` (upstream's ``<r>``, ``<a>``, ``<c>``, ``<q>``: an unseeded random sample, a
  crash, scans by the goal's hidden query or category) never reaches upstream: the page
  stays as it is, like any action upstream does not accept.
* **No done page.** After Buy Now the reply carries the reward and no observation:
  the rendered done page keeps hidden goal fields (upstream's text mode keeps
  ``display:none`` text), so it never leaves this process.
* **Readiness.** ``GET /ready`` answers 503 ``loading`` until the product set, the index
  and the manifest check are done (several minutes), then 200 ``ready`` or 503
  ``failed``. In ``--prepare`` mode it answers 503 ``prepare``: the server then serves
  ``/goals`` and ``/diagnostics`` for ``tools/prepare_webshop.py`` and refuses episodes.

HTTP API (JSON bodies)::

    GET    /ready                       -> {"state", "info", "problems", "episodes",
                                        "action_errors", "recent_action_errors"}
    POST   /episodes                    {"goal_index", "session_prefix", "owner"}
                                        -> {"episode", "observation", "available_actions"}
    POST   /episodes/<id>/step          {"action"} -> {"observation" (null when done),
                                        "reward", "done", "accepted", "available_actions",
                                        "shop_error" (null unless the shop raised)}
    DELETE /episodes/<id>               -> {"deleted_sessions"}
    GET    /goals                       (prepare mode) -> {"goals": [...]}
    POST   /diagnostics                 (prepare mode) {"indices": [...]} -> {"items": {...}}

Subcommands: ``serve`` (the default command of the image), ``check-install`` and
``write-build-record`` (image build). The product files and the Lucene index are the
image's earlier layers, built by ``data_index.py`` next to this file (``fetch-data``,
``build-index``) before this file is copied in, so a change here never fetches or
indexes again; this file reads their paths and hashes from it. Standard library only,
apart from upstream's own imports; Python 3.9 compatible.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import traceback
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

# The image's data and index steps sit next to this file (``python server.py`` puts its
# directory first on ``sys.path``; a test that imports this file by its path does not).
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import data_index  # noqa: E402

#: Bumped when the HTTP API changes; the harness refuses another value.
API_VERSION = 1
WEBSHOP_DIR = data_index.WEBSHOP_DIR
BUILD_RECORD = Path(
    os.environ.get("WEBSHOP_BUILD_RECORD", "/opt/skill-bench/build.json")
)
DATA_PINS = data_index.DATA_PINS
file_sha256 = data_index.file_sha256
#: The goal fields the frozen manifest pins (upstream ``goal.py:56-66``).
MANIFEST_FIELDS = (
    "asin",
    "instruction_text",
    "attributes",
    "goal_options",
    "price_upper",
    "name",
    "query",
    "category",
    "product_category",
)
SPLIT_FILES = ("train.json", "val.json", "test.json", "pilot.json")
#: upstream ``engine.parse_action`` (``engine.py:117-128``).
ACTION_PATTERN = re.compile(r"(.+)\[(.+)\]")
#: A search whose first word matches this is refused.
SPECIAL_SEARCH = re.compile(r"<[a-z]>")
SESSION_PREFIX = re.compile(r"[0-9a-f]{32}-")
MAX_BODY_BYTES = 1 << 20
DIAGNOSTICS_MAX_INDICES = 50
#: The image's build record values a final manifest freezes (``source.json``) and the
#: server must have been built with (``build.json``): any other index, data or upstream
#: commit can rank and price products differently.
BUILD_PINS = ("upstream_commit", "data", "index_sha256")
#: The image's build context digest (the tag ``envs/webshop.py`` builds it under), passed
#: as a build argument and kept in ``build.json`` (``context``).
CONTEXT_ENV = "SKILL_BENCH_CONTEXT"
#: Python errors of upstream's own page logic: raised on an action, they are the action's
#: (the shop cannot process it), never the server's.
ACTION_ERRORS = (
    LookupError,
    ValueError,
    TypeError,
    AttributeError,
    ArithmeticError,
    AssertionError,
)
#: Lucene's query-size limit (``BooleanQuery.TooManyClauses``, "maxClauseCount is set to
#: 1024"): a search with too many distinct words, raised through pyjnius.
LUCENE_ACTION_ERRORS = ("TooManyClauses", "maxClauseCount")
ACTION_ERRORS_KEPT = 20
#: What :class:`UpstreamBackend` imports from upstream (``check-install`` imports them at
#: build time).
UPSTREAM_MODULES = (
    "web_agent_site.engine.engine",
    "web_agent_site.engine.goal",
    "web_agent_site.envs.web_agent_text_env",
)


# ---------------------------------------------------------------------------
# Pure helpers (shared with the harness tests)
# ---------------------------------------------------------------------------


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def price_table_sha256(prices: Mapping[str, Any]) -> str:
    """sha256 of the whole product price table (ASIN -> price) as canonical JSON."""
    return sha256_bytes(canonical_json({str(k): v for k, v in prices.items()}))


def parse_action(action: str) -> Tuple[str, Optional[str]]:
    """upstream ``engine.parse_action``: ``name[arg]`` (greedy), else ``(action, None)``."""
    match = re.match(ACTION_PATTERN, action)
    if match is None:
        return action, None
    name, arg = match.groups()
    return name, arg


def refused_search(action: str) -> bool:
    """Whether ``action`` is a search whose first word is a special engine command."""
    name, arg = parse_action(action)
    if name != "search" or arg is None:
        return False
    words = arg.lower().split()
    return bool(words) and SPECIAL_SEARCH.fullmatch(words[0]) is not None


def accepted_action(action: str, clickables: Iterable[str]) -> bool:
    """Whether upstream's ``WebAgentTextEnv.step`` acts on ``action`` (its lines 100-110)
    rather than leaving the page as it is."""
    name, arg = parse_action(action)
    if arg is not None:
        arg = arg.lower()
    if name == "search":
        return arg is not None and arg != ""
    return name == "click" and arg in set(clickables) and arg != "search"


def caused_by_the_action(exc: BaseException) -> bool:
    """Whether an error upstream raised while acting on an action is the action's (the
    shop cannot process it: :data:`ACTION_ERRORS`, or Lucene's query-size limit through
    pyjnius) rather than the server's (memory, I/O, any other JVM error)."""
    if type(exc).__name__ == "JavaException":
        text = f"{getattr(exc, 'classname', '')} {exc}"
        return any(marker in text for marker in LUCENE_ACTION_ERRORS)
    return isinstance(exc, ACTION_ERRORS)


def manifest_rows(
    manifest_dir: Path,
) -> Tuple[Dict[str, List[Mapping[str, Any]]], Dict[str, str]]:
    """The split files of the frozen manifest and the sha256 of every manifest file."""
    rows: Dict[str, List[Mapping[str, Any]]] = {}
    hashes: Dict[str, str] = {}
    for name in sorted(os.listdir(manifest_dir)):
        path = manifest_dir / name
        if path.is_file() and name.endswith(".json"):
            hashes[name] = file_sha256(path)
    for name in SPLIT_FILES:
        path = manifest_dir / name
        if path.is_file():
            rows[name[: -len(".json")]] = json.loads(path.read_text(encoding="utf-8"))
    return rows, hashes


def verify_manifest(
    goals: List[Mapping[str, Any]],
    prices_sha256: str,
    manifest_dir: Path,
    seed: int,
    build: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """``(info, problems)``: the frozen manifest at ``manifest_dir`` against the goals and
    price table this process built and the image's build record ``build``.
    ``problems`` empty: every pinned field of every item, the goal count, the seed, the
    price table and the image's upstream commit, data and index hashes are the frozen
    ones."""
    problems: List[str] = []
    try:
        rows, hashes = manifest_rows(manifest_dir)
        split_manifest = json.loads((manifest_dir / "split_manifest.json").read_text())
        source = json.loads((manifest_dir / "source.json").read_text())
    except (OSError, ValueError) as exc:
        return {}, [f"the manifest at {manifest_dir} is unreadable: {exc}"]
    info = {
        "status": split_manifest.get("status"),
        "files": hashes,
        "items": {split: len(items) for split, items in sorted(rows.items())},
    }
    if split_manifest.get("status") != "final":
        problems.append(
            f"the manifest is {split_manifest.get('status')!r}, not 'final': run "
            "tools/prepare_webshop.py freeze against this image first"
        )
        return info, problems
    if source.get("seed") != seed:
        problems.append(
            f"the manifest was frozen with seed {source.get('seed')}, this server uses {seed}"
        )
    if source.get("goal_count") != len(goals):
        problems.append(
            f"{len(goals)} goals, the manifest froze {source.get('goal_count')}"
        )
    if source.get("price_table_sha256") != prices_sha256:
        problems.append(
            f"the price table's sha256 is {prices_sha256}, the manifest froze "
            f"{source.get('price_table_sha256')}"
        )
    build = build or {}
    for key in BUILD_PINS:
        if source.get(key) is None:
            problems.append(f"the manifest froze no {key}")
        elif canonical_json(source.get(key)) != canonical_json(build.get(key)):
            problems.append(
                f"the image's {key} is {build.get(key)!r}, the manifest froze "
                f"{source.get(key)!r}"
            )
    for split, items in sorted(rows.items()):
        for row in items:
            index = row.get("goal_index")
            if not isinstance(index, int) or not 0 <= index < len(goals):
                problems.append(
                    f"{split} item {row.get('id')}: goal index {index!r} out of range"
                )
                continue
            goal = goals[index]
            for name in MANIFEST_FIELDS:
                if canonical_json(goal.get(name)) != canonical_json(row.get(name)):
                    problems.append(
                        f"{split} item {row.get('id')}: {name} is {goal.get(name)!r}, "
                        f"the manifest froze {row.get(name)!r}"
                    )
    return info, problems


def goal_record(index: int, goal: Mapping[str, Any]) -> Dict[str, Any]:
    """The manifest fields of one goal, JSON-ready."""
    return {"index": index, **{name: goal.get(name) for name in MANIFEST_FIELDS}}


# ---------------------------------------------------------------------------
# Backends: upstream (in the image) or a test double with the same surface
# ---------------------------------------------------------------------------


class UpstreamBackend:
    """Upstream's ``SimServer`` with the full product set and human goals, built after
    ``random.seed(seed)``. Built, and every method called, on the service's upstream
    thread under its lock (:class:`UpstreamWorker`)."""

    def __init__(self, data_dir: Path, seed: int) -> None:
        sys.path.insert(0, str(WEBSHOP_DIR))
        from web_agent_site.engine import engine  # noqa: E402 - upstream, in the image

        # The full product set (upstream's utils.py defaults to the 1k files; AgentBench
        # patches it, this wrapper sets the module names load_products reads instead).
        engine.DEFAULT_ATTR_PATH = str(data_dir / "items_ins_v2.json")
        engine.HUMAN_ATTR_PATH = str(data_dir / "items_human_ins.json")
        from web_agent_site.engine import goal as goal_module
        from web_agent_site.envs.web_agent_text_env import SimServer, WebAgentTextEnv

        self._engine = engine
        self._goal = goal_module
        self._env_class = WebAgentTextEnv
        random.seed(seed)  # immediately before SimServer() draws prices and goals
        self.sim = SimServer(
            "http://127.0.0.1:3000",
            str(data_dir / "items_shuffle.json"),
            filter_goals=None,
            limit_goals=-1,
            num_products=None,
            human_goals=1,
            show_attrs=False,
        )

    @property
    def goals(self) -> List[Mapping[str, Any]]:
        return self.sim.goals

    @property
    def product_prices(self) -> Mapping[str, Any]:
        return self.sim.product_prices

    def make_env(self, session_prefix: str) -> Any:
        return self._env_class(
            observation_mode="text", server=self.sim, session_prefix=session_prefix
        )

    def session_ids(self) -> List[str]:
        return list(self.sim.user_sessions)

    def drop_session(self, session_id: str) -> None:
        self.sim.user_sessions.pop(session_id, None)

    def save_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """What an action may change of an upstream session (``SimServer.receive``: the
        keywords, page, product, options, action counts, done and reward), copied; the
        goal is shared and never written."""
        session = self.sim.user_sessions.get(session_id)
        if session is None:
            return None
        return {
            key: value
            if key == "goal" or not isinstance(value, (dict, list, set))
            else copy.copy(value)
            for key, value in session.items()
        }

    def restore_session(
        self, session_id: str, saved: Optional[Mapping[str, Any]]
    ) -> None:
        """Put a session back as :meth:`save_session` found it (the same dict object,
        which upstream's environment holds by its key)."""
        session = self.sim.user_sessions.get(session_id)
        if saved is None:
            self.sim.user_sessions.pop(session_id, None)
        elif session is None:
            self.sim.user_sessions[session_id] = dict(saved)
        else:
            session.clear()
            session.update(saved)

    def best_reward(self, goal: Mapping[str, Any]) -> Dict[str, Any]:
        """The reward of buying the gold product with the goal's options."""
        product = self.sim.product_item_dict[goal["asin"]]
        wanted = goal.get("goal_options") or []
        # Upstream's get_reward takes a goal's options as a list or a dict (goal.py:241).
        values = list(wanted.values()) if isinstance(wanted, dict) else list(wanted)
        options = {str(i): str(o) for i, o in enumerate(values)}
        reward, details = self._goal.get_reward(
            product,
            goal,
            price=self.sim.product_prices.get(goal["asin"]),
            options=options,
            verbose=True,
        )
        values = {v for vs in (product.get("options") or {}).values() for v in vs}
        return {
            "best_reward": reward,
            "reward_info": {k: details.get(k) for k in sorted(details)},
            "options_available": all(
                str(o).lower() in values for o in options.values()
            ),
        }

    def search_rank(self, query: str, asin: str) -> Optional[int]:
        """The gold product's 0-based rank among the top 50 for ``query`` (None: absent)."""
        found = self._engine.get_top_n_product_from_keywords(
            query.lower().split(" "),
            self.sim.search_engine,
            self.sim.all_products,
            self.sim.product_item_dict,
        )
        asins = [p["asin"] for p in found]
        return asins.index(asin) if asin in asins else None

    def info(self) -> Dict[str, Any]:
        from thefuzz import fuzz

        meta = getattr(self._goal.nlp, "meta", {}) or {}
        return {
            "spacy_model": f"{meta.get('lang')}_{meta.get('name')}=={meta.get('version')}",
            "fuzzy_backend": getattr(fuzz.SequenceMatcher, "__module__", None),
            "products": len(self.sim.all_products),
        }


# ---------------------------------------------------------------------------
# The service: state, lock, episodes
# ---------------------------------------------------------------------------


class UpstreamWorker:
    """The one thread every call into upstream runs on (its import and ``SimServer()``
    included, so the JVM pyserini starts lives on it). pyjnius attaches each thread that
    calls into the JVM and cannot detach a thread that ends by itself; with one
    long-lived thread no HTTP connection thread ever attaches. The thread ends with the
    worker (``close``, or when the worker is collected)."""

    def __init__(self) -> None:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="upstream")
        self._close = weakref.finalize(self, executor.shutdown, wait=False)
        self._executor = executor

    def run(self, fn: Callable[[], Any]) -> Any:
        """``fn()`` on the upstream thread; its result, or its exception re-raised."""
        return self._executor.submit(fn).result()

    def close(self) -> None:
        self._close()


class ServiceError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class Episode:
    env: Any
    prefix: str
    goal_index: int
    owner: str
    created: float
    last_used: float
    done: bool = False


class Service:
    """The server's state machine (``loading`` -> ``ready`` | ``failed``; ``prepare``)."""

    def __init__(
        self,
        *,
        seed: int,
        manifest_dir: Optional[Path],
        prepare: bool = False,
        session_idle_s: float = 3600.0,
        build_record: Optional[Mapping[str, Any]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not prepare and manifest_dir is None:
            raise ValueError("serve mode needs the frozen manifest (--manifest-dir)")
        self.seed = seed
        self.manifest_dir = manifest_dir
        self.prepare = prepare
        self.session_idle_s = session_idle_s
        self.build_record = dict(build_record or {})
        self.clock = clock
        self.worker = UpstreamWorker()  # the one thread that calls into upstream
        self.lock = threading.Lock()  # held by every call into upstream
        self._state_lock = threading.Lock()
        self.state = "loading"
        self.problems: List[str] = []
        self.info: Dict[str, Any] = {
            "api_version": API_VERSION,
            "seed": seed,
            "mode": "prepare" if prepare else "serve",
            "build": self.build_record,
            "python": sys.version.split()[0],
        }
        self.backend: Any = None
        self.episodes: Dict[str, Episode] = {}
        self.allowed: Optional[set] = None
        #: Actions the shop could not process (``shop_error``): how many, and the last
        #: :data:`ACTION_ERRORS_KEPT` with their episode, action and error.
        self.action_errors = 0
        self.recent_action_errors: List[Dict[str, Any]] = []

    def _upstream(self, fn: Callable[[], Any]) -> Any:
        """``fn()`` on the upstream thread, under the global lock."""

        def locked() -> Any:
            with self.lock:
                return fn()

        return self.worker.run(locked)

    # -- start-up -------------------------------------------------------------------------

    def load(self, factory: Callable[[], Any]) -> None:
        """Build the backend (``factory`` imports, seeds and builds upstream, on the
        upstream thread) and verify the manifest; sets the final state. Never raises."""
        started = self.clock()

        def build() -> Tuple[Any, Dict[str, Any], List[str]]:
            backend = factory()
            prices = price_table_sha256(backend.product_prices)
            info: Dict[str, Any] = {
                "goal_count": len(backend.goals),
                "price_table_sha256": prices,
                **backend.info(),
            }
            problems: List[str] = []
            if not self.prepare:
                assert self.manifest_dir is not None
                manifest, problems = verify_manifest(
                    backend.goals,
                    prices,
                    self.manifest_dir,
                    self.seed,
                    self.build_record,
                )
                info["manifest"] = manifest
                if not problems:
                    rows, _ = manifest_rows(self.manifest_dir)
                    self.allowed = {
                        row["goal_index"] for items in rows.values() for row in items
                    }
            return backend, info, problems

        try:
            backend, info, problems = self._upstream(build)
        except Exception as exc:  # noqa: BLE001 - reported by /ready
            with self._state_lock:
                self.state = "failed"
                self.problems = [f"{type(exc).__name__}: {exc}"]
                self.info["traceback"] = traceback.format_exc()[-4000:]
            return
        with self._state_lock:
            self.backend = backend
            self.info.update(info)
            self.info["load_s"] = round(self.clock() - started, 1)
            self.problems = problems
            self.state = (
                "failed" if problems else ("prepare" if self.prepare else "ready")
            )

    def readiness(self) -> Tuple[int, Dict[str, Any]]:
        with self._state_lock:
            payload = {
                "state": self.state,
                "problems": list(self.problems),
                "info": dict(self.info),
                "episodes": len(self.episodes),
                "action_errors": self.action_errors,
                "recent_action_errors": list(self.recent_action_errors),
            }
        return (
            HTTPStatus.OK if self.state == "ready" else HTTPStatus.SERVICE_UNAVAILABLE
        ), payload

    def _require(self, *states: str) -> None:
        if self.state not in states:
            raise ServiceError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                f"the server is {self.state}"
                + (f": {'; '.join(self.problems[:5])}" if self.problems else ""),
            )

    # -- episodes -------------------------------------------------------------------------

    def _reap(self) -> None:
        """Drop episodes idle longer than ``session_idle_s`` (a harness that died)."""
        now = self.clock()
        for episode_id, episode in list(self.episodes.items()):
            if now - episode.last_used > self.session_idle_s:
                self._delete(episode_id)

    def _delete(self, episode_id: str) -> int:
        episode = self.episodes.pop(episode_id, None)
        prefix = episode.prefix if episode is not None else episode_id
        dropped = 0
        for session_id in self.backend.session_ids():
            if session_id.startswith(prefix):
                self.backend.drop_session(session_id)
                dropped += 1
        return dropped

    def create(self, body: Mapping[str, Any]) -> Dict[str, Any]:
        self._require("ready")
        index, prefix = body.get("goal_index"), body.get("session_prefix")
        owner = str(body.get("owner") or "")[:200]
        if not isinstance(index, int) or isinstance(index, bool):
            raise ServiceError(HTTPStatus.BAD_REQUEST, "goal_index must be an integer")
        if not isinstance(prefix, str) or SESSION_PREFIX.fullmatch(prefix) is None:
            raise ServiceError(
                HTTPStatus.BAD_REQUEST, "session_prefix must be 32 hex digits and '-'"
            )

        def run() -> Dict[str, Any]:
            self._reap()
            if not 0 <= index < len(self.backend.goals):
                raise ServiceError(HTTPStatus.BAD_REQUEST, f"no goal {index}")
            if self.allowed is not None and index not in self.allowed:
                raise ServiceError(
                    HTTPStatus.FORBIDDEN, f"goal {index} is not in the frozen manifest"
                )
            if prefix in self.episodes or any(
                s.startswith(prefix) for s in self.backend.session_ids()
            ):
                raise ServiceError(HTTPStatus.CONFLICT, f"episode {prefix} exists")
            try:
                env = self.backend.make_env(prefix)  # its constructor opens a session
                observation, _ = env.reset(session=index)
                available = env.get_available_actions()
            except BaseException:
                self._delete(prefix)  # the constructor's session must not stay behind
                raise
            now = self.clock()
            self.episodes[prefix] = Episode(env, prefix, index, owner, now, now)
            return {
                "episode": prefix,
                "observation": observation,
                "available_actions": available,
            }

        return self._upstream(run)

    def step(self, episode_id: str, body: Mapping[str, Any]) -> Dict[str, Any]:
        self._require("ready")
        action = body.get("action")
        if not isinstance(action, str):
            raise ServiceError(HTTPStatus.BAD_REQUEST, "action must be a string")

        def unchanged(env: Any, shop_error: Optional[str]) -> Dict[str, Any]:
            return {
                "observation": env.observation,
                "reward": 0.0,
                "done": False,
                "accepted": False,
                "available_actions": env.get_available_actions(),
                "shop_error": shop_error,
            }

        def run() -> Dict[str, Any]:
            episode = self.episodes.get(episode_id)
            if episode is None:
                raise ServiceError(HTTPStatus.NOT_FOUND, f"no episode {episode_id}")
            if episode.done:
                raise ServiceError(HTTPStatus.CONFLICT, "the episode is over")
            episode.last_used = self.clock()
            env = episode.env
            if refused_search(action):  # never reaches upstream
                return unchanged(env, None)
            clickables = env.get_available_actions()["clickables"]
            accepted = accepted_action(action, clickables)
            saved = self.backend.save_session(env.session)
            try:
                observation, reward, done, _ = env.step(action)
            except Exception as exc:
                if not caused_by_the_action(exc):
                    raise  # the server's: a 500, the harness re-runs the item
                # The shop cannot process this action: as if it did not accept it.
                self.backend.restore_session(env.session, saved)
                error = f"{type(exc).__name__}: {exc}"[:500]
                self._record_action_error(episode, action, error)
                return unchanged(env, error)
            if done:
                # The done page keeps hidden goal fields; only the reward leaves.
                episode.done = True
                return {
                    "observation": None,
                    "reward": float(reward),
                    "done": True,
                    "accepted": True,
                    "available_actions": None,
                    "shop_error": None,
                }
            return {
                "observation": observation,
                "reward": float(reward or 0.0),
                "done": False,
                "accepted": accepted,
                "available_actions": env.get_available_actions(),
                "shop_error": None,
            }

        return self._upstream(run)

    def _record_action_error(self, episode: Episode, action: str, error: str) -> None:
        with self._state_lock:
            self.action_errors += 1
            self.recent_action_errors.append(
                {
                    "episode": episode.prefix,
                    "goal_index": episode.goal_index,
                    "action": action[:200],
                    "error": error,
                }
            )
            del self.recent_action_errors[:-ACTION_ERRORS_KEPT]

    def delete(self, episode_id: str) -> Dict[str, Any]:
        if self.backend is None:
            raise ServiceError(HTTPStatus.NOT_FOUND, f"no episode {episode_id}")

        def run() -> Dict[str, Any]:
            if episode_id not in self.episodes:
                raise ServiceError(HTTPStatus.NOT_FOUND, f"no episode {episode_id}")
            return {"deleted_sessions": self._delete(episode_id)}

        return self._upstream(run)

    # -- prepare mode -----------------------------------------------------------------------

    def goals(self) -> Dict[str, Any]:
        self._require("prepare")
        return self._upstream(
            lambda: {
                "goals": [goal_record(i, g) for i, g in enumerate(self.backend.goals)]
            }
        )

    def diagnostics(self, body: Mapping[str, Any]) -> Dict[str, Any]:
        self._require("prepare")
        indices = body.get("indices")
        if (
            not isinstance(indices, list)
            or len(indices) > DIAGNOSTICS_MAX_INDICES
            or not all(isinstance(i, int) and not isinstance(i, bool) for i in indices)
        ):
            raise ServiceError(
                HTTPStatus.BAD_REQUEST,
                f"indices must be a list of at most {DIAGNOSTICS_MAX_INDICES} integers",
            )

        def run() -> Dict[str, Any]:
            items: Dict[str, Any] = {}
            goals = self.backend.goals
            for index in indices:
                if not 0 <= index < len(goals):
                    raise ServiceError(HTTPStatus.BAD_REQUEST, f"no goal {index}")
                goal = goals[index]
                items[str(index)] = {
                    **self.backend.best_reward(goal),
                    "gold_search_rank": self.backend.search_rank(
                        str(goal["instruction_text"]), str(goal["asin"])
                    ),
                }
            return {"items": items}

        return self._upstream(run)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def route(
    service: Service,
    method: str,
    target: str,
    read_body: Callable[[], Dict[str, Any]],
) -> Tuple[int, Dict[str, Any]]:
    """One request: ``(status, JSON payload)``. Pure apart from ``service``; the HTTP
    handler and the harness tests' in-process transport both call it."""
    path = target.split("?", 1)[0].rstrip("/")
    parts = [p for p in path.split("/") if p]
    try:
        if method == "GET" and parts == ["ready"]:
            return service.readiness()
        if method == "GET" and parts == ["goals"]:
            result = service.goals()
        elif method == "POST" and parts == ["diagnostics"]:
            result = service.diagnostics(read_body())
        elif method == "POST" and parts == ["episodes"]:
            result = service.create(read_body())
        elif (
            method == "POST"
            and len(parts) == 3
            and parts[0] == "episodes"
            and parts[2] == "step"
        ):
            result = service.step(parts[1], read_body())
        elif method == "DELETE" and len(parts) == 2 and parts[0] == "episodes":
            result = service.delete(parts[1])
        else:
            raise ServiceError(HTTPStatus.NOT_FOUND, f"no route {method} {path}")
    except ServiceError as exc:
        return exc.status, {"error": exc.message}
    except Exception as exc:  # noqa: BLE001 - an upstream error of one request
        return HTTPStatus.INTERNAL_SERVER_ERROR, {
            "error": f"{type(exc).__name__}: {exc}"[:2000]
        }
    return HTTPStatus.OK, result


def parse_body(raw: bytes) -> Dict[str, Any]:
    """A request body: a JSON object (an empty body is ``{}``)."""
    if len(raw) > MAX_BODY_BYTES:
        raise ServiceError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "body too large")
    try:
        body = json.loads(raw.decode("utf-8")) if raw else {}
    except ValueError:
        raise ServiceError(HTTPStatus.BAD_REQUEST, "the body is not JSON") from None
    if not isinstance(body, dict):
        raise ServiceError(HTTPStatus.BAD_REQUEST, "the body is not an object")
    return body


def make_handler(service: Service) -> type:
    class Handler(BaseHTTPRequestHandler):
        server_version = "skill-bench-webshop/%d" % API_VERSION
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet
            return

        def _body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                self.close_connection = True  # the unread body stays on the socket
                raise ServiceError(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "body too large"
                )
            return parse_body(self.rfile.read(length) if length else b"")

        def _route(self, method: str) -> None:
            status, payload = route(service, method, self.path, self._body)
            data = canonical_json(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            self._route("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._route("POST")

        def do_DELETE(self) -> None:  # noqa: N802
            self._route("DELETE")

    return Handler


def start_http(service: Service, host: str, port: int) -> ThreadingHTTPServer:
    """The HTTP server, bound but not yet serving (``serve_forever`` in the caller)."""
    httpd = ThreadingHTTPServer((host, port), make_handler(service))
    httpd.daemon_threads = True
    return httpd


# ---------------------------------------------------------------------------
# Image build steps
# ---------------------------------------------------------------------------


def check_install(import_upstream: bool = True) -> Dict[str, Any]:
    """The spaCy model and the fuzzy backend upstream's reward runs with; and
    (``import_upstream``) every upstream module the server imports, so a package missing
    from the allowlist, or one the pinned setuptools no longer serves, fails the image
    build rather than the server's first start."""
    import importlib
    import importlib.util

    import spacy
    from thefuzz import fuzz

    problems = []
    if importlib.util.find_spec("Levenshtein") is not None:
        problems.append("python-Levenshtein is installed (difflib is pinned)")
    if getattr(fuzz.SequenceMatcher, "__module__", None) != "difflib":
        problems.append("thefuzz does not use difflib")
    model = spacy.load("en_core_web_sm")
    version = model.meta.get("version")
    if version != "3.3.0":
        problems.append(f"en_core_web_sm {version} is installed (3.3.0 is pinned)")
    if import_upstream:
        sys.path.insert(0, str(WEBSHOP_DIR))
        try:
            for name in UPSTREAM_MODULES:
                importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - reported as the build's failure
            problems.append(f"upstream does not import: {type(exc).__name__}: {exc}")
    if problems:
        raise SystemExit("; ".join(problems))
    return {"spacy_model": f"en_core_web_sm=={version}", "fuzzy_backend": "difflib"}


def environment_record() -> Dict[str, Any]:
    """What the image runs that its pins do not fix (``build.json``): the Java runtime
    (bullseye's openjdk-11 package is not pinned by version), setuptools, a digest of
    every installed distribution and the build context digest."""
    import subprocess
    from importlib import metadata

    try:
        done = subprocess.run(
            ["java", "-version"], capture_output=True, text=True, timeout=60
        )
        java = (done.stderr or done.stdout).strip().splitlines()[0]
    except (OSError, IndexError, subprocess.SubprocessError):
        java = None
    packages = sorted(
        f"{d.metadata['Name']}=={d.version}"
        for d in metadata.distributions()
        if d.metadata["Name"]
    )
    return {
        "java": java,
        "setuptools": next(
            (p.split("==", 1)[1] for p in packages if p.startswith("setuptools==")),
            None,
        ),
        "packages_sha256": sha256_bytes("\n".join(packages).encode("utf-8")),
        "context": os.environ.get(CONTEXT_ENV) or None,
    }


def build_record(
    data_dir: Path,
    *,
    pins_path: Path = DATA_PINS,
    index_record: Path = data_index.INDEX_RECORD,
) -> Dict[str, Any]:
    """``build.json``: the upstream commit, the sha256 of each pinned product
    file as the image holds it, the sha256 of the index record ``data_index.py
    build-index`` wrote (a deterministic description of the index, not of its files),
    the install check and what no pin fixes (:func:`environment_record`, the build
    context's digest included)."""
    import subprocess

    commit = subprocess.run(
        ["git", "-C", str(WEBSHOP_DIR), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    pins = json.loads(pins_path.read_text(encoding="utf-8"))
    return {
        "upstream_commit": commit,
        "data": {name: file_sha256(data_dir / name) for name in sorted(pins["files"])},
        "index_sha256": file_sha256(index_record),
        **check_install(),
        **environment_record(),
    }


def read_build_record(path: Path = BUILD_RECORD) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="serve episodes (or --prepare)")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--seed", type=int, required=True)
    serve.add_argument("--data-dir", type=Path, default=data_index.DATA_DIR)
    serve.add_argument("--manifest-dir", type=Path)
    serve.add_argument("--prepare", action="store_true")
    serve.add_argument("--session-idle-s", type=float, default=3600.0)
    sub.add_parser("check-install")
    record = sub.add_parser("write-build-record")
    record.add_argument("--data-dir", type=Path, default=data_index.DATA_DIR)
    args = parser.parse_args(argv)

    if args.command == "check-install":
        print(json.dumps(check_install()))
        return 0
    if args.command == "write-build-record":
        build = build_record(args.data_dir)
        BUILD_RECORD.write_text(json.dumps(build, indent=1, sort_keys=True))
        print(json.dumps(build))
        return 0

    service = Service(
        seed=args.seed,
        manifest_dir=args.manifest_dir,
        prepare=args.prepare,
        session_idle_s=args.session_idle_s,
        build_record=read_build_record(),
    )
    httpd = start_http(service, args.host, args.port)
    # Waits for the upstream thread, which imports and builds upstream (minutes).
    loader = threading.Thread(
        target=service.load,
        args=(lambda: UpstreamBackend(args.data_dir, args.seed),),
        daemon=True,
    )
    loader.start()
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
