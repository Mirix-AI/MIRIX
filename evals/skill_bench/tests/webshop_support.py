"""An offline WebShop for the tests: the real ``server.py`` service over a scripted shop.

The image's ``server.py`` (``envs/templates/webshop_server``) runs here unchanged, in
process: :func:`server_module` imports it from its frozen path, :class:`FakeBackend` has the
surface of its ``UpstreamBackend`` (goals, price table, ``WebAgentTextEnv``-like episodes,
sessions keyed ``prefix + session``, the diagnostics), and :func:`shop_client_factory`
gives :class:`~evals.skill_bench.envs.webshop.WebShopEnv` an ``httpx`` transport that
calls the server's own ``route`` (no socket, no Java, no product data). Tests that need
the HTTP layer itself start ``server.start_http`` on a loopback port.

The scripted shop (:class:`FakeTextEnv`) renders text pages the way upstream's text mode
does: every page starts with ``WebShop [SEP] Instruction: [SEP] <instruction>``; a search
lists the goal's product and a distractor; an item page offers the goal's first two
options; Buy Now ends the episode with reward 1.0 (the gold product with those options),
0.5 (the gold product without them) or 0.0, and a done page that carries the hidden-goal
text upstream's done page carries, which must never leave the server. Special searches
(``<r>``, ``<a>``, ``<c>``, ``<q>``) are counted when they reach the shop: they never should.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import random
import string
import sys
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx

from evals.skill_bench import protocol
from evals.skill_bench.envs import webshop as ws

SERVER_PY = protocol.SKILL_BENCH_DIR / ws.SERVER_DIR / "server.py"
SHOP_URL = "http://webshop.test"
#: Buttons of the scripted pages (never product links or options).
NAV = (
    "back to search",
    "next >",
    "< prev",
    "description",
    "features",
    "reviews",
    "buy now",
)
#: The done page of upstream's text mode (``templates/done_page.html``): its hidden blocks
#: keep the goal. The server must never return it.
DONE_PAGE = (
    "Thank you for shopping with us! [SEP] Your code: [SEP] None [SEP] (Paste it in "
    "your MTurk interface.) [SEP] Purchased [SEP] asin [SEP] {asin} [SEP] options "
    "[SEP] {options} [SEP] attrs [SEP] {attributes} [SEP] category [SEP] {category} "
    "[SEP] query [SEP] {query} [SEP] product category [SEP] {product_category} [SEP] "
    "Target [SEP] asin [SEP] {asin} [SEP] price upper [SEP] {price_upper} [SEP] "
    "instuction text [SEP] {instruction_text} [SEP] Reward Details [SEP] "
    "Your score (min 0.0, max 1.0) [SEP] {reward}"
)

_SERVER: ModuleType | None = None


def server_module() -> ModuleType:
    """The frozen ``server.py``, imported from its path (it is not a package module)."""
    global _SERVER
    if _SERVER is None:
        name = "skill_bench_webshop_server"
        spec = importlib.util.spec_from_file_location(name, SERVER_PY)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # dataclasses resolve the module by name
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True  # no __pycache__ in the image's Docker context
        try:
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = previous
        _SERVER = module
    return _SERVER


# ---------------------------------------------------------------------------
# The scripted shop
# ---------------------------------------------------------------------------


@dataclass
class FakeProduct:
    asin: str
    title: str
    price: float
    options: tuple[str, ...]


def goal_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """A goal with every manifest field, from a frozen split row (provisional rows lack
    the fields only the real data has: they get deterministic stand-ins)."""
    index = int(row["goal_index"])
    instruction = str(row.get("instruction_text") or row["instruction"])
    if "instruction_text" not in row:
        instruction += f", and price lower than {40 + index % 30:.2f} dollars"
    return {
        "asin": str(row["asin"]),
        "instruction_text": instruction,
        "attributes": list(row.get("attributes") or []),
        "goal_options": list(row.get("goal_options") or []),
        "price_upper": row.get("price_upper", float(40 + index % 30)),
        "name": row.get("name", f"product {index}"),
        "query": row.get("query", "fake query"),
        "category": row.get("category", "beauty"),
        "product_category": row.get("product_category", "Beauty › Hair"),
    }


def synthetic_goal(index: int) -> dict[str, Any]:
    return goal_from_row(
        {
            "goal_index": index,
            "asin": f"B{index:09d}",
            "instruction": f"i need the fake product number {index}",
            "attributes": ["fake attribute"],
            "goal_options": ["red", "large"],
        }
    )


class FakeBackend:
    """``server.UpstreamBackend``'s surface over scripted goals and products."""

    def __init__(
        self,
        goals: Sequence[Mapping[str, Any]],
        *,
        seed: int = 0,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        self.goals = [dict(g) for g in goals]
        self.products: dict[str, FakeProduct] = {}
        for index, goal in enumerate(self.goals):
            asin = str(goal["asin"])
            if asin not in self.products:
                options = tuple(str(o) for o in goal.get("goal_options") or [])[:2]
                self.products[asin] = FakeProduct(
                    asin, f"Gold item {index}", 10.0 + index % 20, options
                )
            distractor = f"BX{index:08d}"
            self.products[distractor] = FakeProduct(
                distractor, f"Other item {index}", 5.0, ("blue",)
            )
        self.product_prices = {a: p.price for a, p in self.products.items()}
        self.sessions: dict[str, dict[str, Any]] = {}
        self.rng = random.Random(seed)
        self.special_searches: list[str] = []
        self.actions: list[tuple[str, str]] = []  # (session, action) reaching the shop
        #: The next step fails: True raises the server's own error (an ``OSError``), an
        #: exception instance is raised as it is, after the action changed the session
        #: (as upstream's ``search_results`` does before it searches).
        self.fail_next_step: bool | BaseException = False
        #: The thread every call into this backend ran on (``server.UpstreamWorker``).
        self.threads: set[int] = set()
        self._info = dict(
            info
            or {
                "spacy_model": protocol.WEBSHOP_SERVER["spacy_model"],
                "fuzzy_backend": protocol.WEBSHOP_SERVER["fuzzy_backend"],
                "products": len(self.products),
            }
        )

    # --- server.UpstreamBackend ---------------------------------------------------------

    def make_env(self, session_prefix: str) -> FakeTextEnv:
        self._seen()
        return FakeTextEnv(self, session_prefix)

    def session_ids(self) -> list[str]:
        self._seen()
        return list(self.sessions)

    def drop_session(self, session_id: str) -> None:
        self._seen()
        self.sessions.pop(session_id, None)

    def save_session(self, session_id: str) -> dict[str, Any] | None:
        self._seen()
        session = self.sessions.get(session_id)
        if session is None:
            return None
        return {
            k: copy.copy(v) if isinstance(v, (dict, list, set)) else v
            for k, v in session.items()
        }

    def restore_session(self, session_id: str, saved: Mapping[str, Any] | None) -> None:
        self._seen()
        if saved is None:
            self.sessions.pop(session_id, None)
            return
        session = self.sessions.setdefault(session_id, {})
        session.clear()
        session.update(saved)

    def _seen(self) -> None:
        self.threads.add(threading.get_ident())

    def best_reward(self, goal: Mapping[str, Any]) -> dict[str, Any]:
        product = self.products[str(goal["asin"])]
        return {
            "best_reward": 1.0 if len(goal.get("goal_options") or []) <= 2 else 0.9,
            "reward_info": {"r_att": 1.0, "r_option": 1.0},
            "options_available": all(
                o in product.options for o in (goal.get("goal_options") or [])
            ),
        }

    def search_rank(self, query: str, asin: str) -> int | None:
        return 0 if asin in self.products else None

    def info(self) -> dict[str, Any]:
        return dict(self._info)

    # --- the shop ---------------------------------------------------------------------

    def results(self, goal_index: int) -> list[FakeProduct]:
        goal = self.goals[goal_index]
        return [self.products[str(goal["asin"])], self.products[f"BX{goal_index:08d}"]]

    def reward(self, goal_index: int, asin: str, chosen: Iterable[str]) -> float:
        goal = self.goals[goal_index]
        if asin != goal["asin"]:
            return 0.0
        wanted = set(self.products[asin].options)
        return 1.0 if wanted <= set(chosen) else 0.5


class FakeTextEnv:
    """``WebAgentTextEnv`` (``observation_mode="text"``) over :class:`FakeBackend`: its
    constructor opens a session with a random name under the prefix, as upstream's does."""

    def __init__(self, backend: FakeBackend, session_prefix: str) -> None:
        self.backend = backend
        self.session_prefix = session_prefix
        self.session = ""
        self.reset()

    @property
    def _state(self) -> dict[str, Any]:
        return self.backend.sessions[self.session]

    def reset(
        self, session: int | str | None = None, instruction_text: str | None = None
    ) -> tuple[str, None]:
        if session is not None:
            name = str(session)
            goal = session if isinstance(session, int) else 0
        else:
            name = "".join(self.backend.rng.choices(string.ascii_lowercase, k=10))
            goal = self.backend.rng.randrange(len(self.backend.goals))
        self.session = self.session_prefix + name
        self.backend.sessions[self.session] = {
            "goal": goal,
            "page": "search",
            "query": "",
            "asin": None,
            "chosen": [],
            "done": False,
        }
        return self.observation, None

    def _header(self) -> str:
        goal = self.backend.goals[self._state["goal"]]
        # Upstream's text mode strips every text of the page.
        return f"WebShop [SEP] Instruction: [SEP] {goal['instruction_text'].strip()}"

    @property
    def observation(self) -> str:
        state = self._state
        header = self._header()
        if state["page"] == "search":
            return f"{header} [SEP] Search"
        if state["page"] == "results":
            listed = self.backend.results(state["goal"])
            rows = " [SEP] ".join(
                f"{p.asin} [SEP] {p.title} [SEP] ${p.price:.2f}" for p in listed
            )
            return (
                f"{header} [SEP] Back to Search [SEP] Page 1 (Total results: "
                f"{len(listed)}) [SEP] Next > [SEP] {rows}"
            )
        if state["page"] == "item":
            product = self.backend.products[state["asin"]]
            options = " [SEP] ".join(product.options)
            return (
                f"{header} [SEP] Back to Search [SEP] < Prev [SEP] options [SEP] "
                f"{options} [SEP] {product.title} [SEP] Price: ${product.price:.2f} "
                "[SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews "
                "[SEP] Buy Now"
            )
        goal = self.backend.goals[state["goal"]]
        return DONE_PAGE.format(reward=state["reward"], options=state["chosen"], **goal)

    def get_available_actions(self) -> dict[str, Any]:
        state = self._state
        if state["page"] == "search":
            return {"has_search_bar": True, "clickables": ["search"]}
        if state["page"] == "results":
            products = [p.asin.lower() for p in self.backend.results(state["goal"])]
            return {
                "has_search_bar": False,
                "clickables": ["back to search", "next >", *products],
            }
        if state["page"] == "item":
            product = self.backend.products[state["asin"]]
            return {
                "has_search_bar": False,
                "clickables": [*NAV[:1], *NAV[2:], *product.options],
            }
        return {"has_search_bar": False, "clickables": []}

    def step(self, action: str) -> tuple[str, float, bool, None]:
        """upstream ``WebAgentTextEnv.step``: an action it does not accept changes
        nothing."""
        server = server_module()
        self.backend._seen()
        self.backend.actions.append((self.session, action))
        state = self._state
        failure, self.backend.fail_next_step = self.backend.fail_next_step, False
        if failure is True:
            raise OSError("an upstream failure")
        if isinstance(failure, BaseException):
            state.update(page="results", query="half done")
            state["chosen"].append("half done")
            raise failure
        clickables = self.get_available_actions()["clickables"]
        name, arg = server.parse_action(action)
        arg = arg.lower() if arg is not None else None
        if name == "search" and arg:
            if arg.split()[0] in ("<r>", "<a>", "<c>", "<q>"):
                self.backend.special_searches.append(action)
            state.update(page="results", query=arg)
        elif name == "click" and arg in clickables and arg != "search":
            if arg == "back to search":
                state.update(page="search")
            elif arg == "< prev":
                state.update(page="results")
            elif arg == "buy now":
                reward = self.backend.reward(
                    state["goal"], state["asin"], state["chosen"]
                )
                state.update(page="done", done=True, reward=reward)
                return self.observation, reward, True, None
            elif state["page"] == "results" and arg not in NAV:
                state.update(page="item", asin=arg.upper(), chosen=[])
            elif state["page"] == "item" and arg not in NAV:
                state["chosen"].append(arg)
        return self.observation, 0.0, False, None


# ---------------------------------------------------------------------------
# Goals, manifests and the service
# ---------------------------------------------------------------------------


def frozen_rows() -> dict[int, dict[str, Any]]:
    """Every row of the repo's frozen WebShop splits, by goal index."""
    rows: dict[int, dict[str, Any]] = {}
    for split in ws.SPLIT_NAMES:
        for row in ws.load_split(split):
            rows[int(row["goal_index"])] = row
    return rows


def shop_goals(
    rows: Mapping[int, Mapping[str, Any]],
    count: int = protocol.WEBSHOP_GOAL_COUNT,
) -> list[dict[str, Any]]:
    """``count`` goals: the frozen rows at their indices, synthetic goals elsewhere."""
    return [
        goal_from_row(rows[i]) if i in rows else synthetic_goal(i) for i in range(count)
    ]


def write_manifest(
    directory: Path,
    backend: FakeBackend,
    splits: Mapping[str, Sequence[int]],
    *,
    status: str = ws.MANIFEST_FINAL,
    seed: int = protocol.SEED,
    goal_count: int | None = None,
    prices_sha256: str | None = None,
    source_changes: Mapping[str, Any] | None = None,
) -> Path:
    """A synthetic final manifest directory, for the
    goals of ``backend`` (``splits``: split name -> goal indices), its ``source.json``
    pinning :func:`build_record`'s commit, data and index (``source_changes`` on top)."""
    server = server_module()
    directory.mkdir(parents=True, exist_ok=True)
    for split, indices in splits.items():
        rows = []
        for index in indices:
            goal = backend.goals[index]
            rows.append(
                {
                    "id": ws.item_id(index),
                    "goal_index": index,
                    "instruction": goal["instruction_text"],
                    **{name: goal[name] for name in ws.MANIFEST_FIELDS},
                }
            )
        (directory / f"{split}.json").write_text(json.dumps(rows, indent=1))
    (directory / "split_manifest.json").write_text(json.dumps({"status": status}))
    record = build_record()
    (directory / "source.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "goal_count": len(backend.goals) if goal_count is None else goal_count,
                "price_table_sha256": prices_sha256
                or server.price_table_sha256(backend.product_prices),
                **{key: record[key] for key in server.BUILD_PINS},
                **dict(source_changes or {}),
            }
        )
    )
    return directory


@dataclass
class Shop:
    """A loaded service, its backend and the manifest it verified."""

    service: Any
    backend: FakeBackend
    manifest_dir: Path | None
    #: (method, path, JSON body) of every request, and (method, path, status, payload)
    #: of every reply, through :meth:`client_factory`'s clients.
    requests: list[tuple[str, str, Any]] = field(default_factory=list)
    replies: list[tuple[str, str, int, Any]] = field(default_factory=list)

    def client_factory(self) -> ws.ClientFactory:
        return shop_client_factory(self.service, self.requests, self.replies)

    def env(self, **kwargs: Any) -> ws.WebShopEnv:
        return ws.WebShopEnv(
            server_url=SHOP_URL, client_factory=self.client_factory(), **kwargs
        )

    def live_sessions(self) -> list[str]:
        return self.backend.session_ids()


def make_shop(
    tmp_path: Path,
    *,
    rows: Mapping[int, Mapping[str, Any]] | None = None,
    goals: Sequence[Mapping[str, Any]] | None = None,
    splits: Mapping[str, Sequence[int]] | None = None,
    prepare: bool = False,
    goal_count: int = protocol.WEBSHOP_GOAL_COUNT,
    session_idle_s: float = 3600.0,
    clock: Any = None,
    manifest_dir: Path | None = None,
    manifest_changes: Mapping[str, Any] | None = None,
    record: Mapping[str, Any] | None = None,
) -> Shop:
    """A loaded service over ``goals``, or the frozen rows (default: every row of the
    repo's splits) at their indices, serving a final manifest of ``splits`` (default:
    those rows by their split) written under ``tmp_path``, or the one at
    ``manifest_dir``."""
    server = server_module()
    if goals is None:
        goals = shop_goals(frozen_rows() if rows is None else rows, goal_count)
    backend = FakeBackend(goals)
    if not prepare and manifest_dir is None:
        if splits is None:
            splits = {
                s: [int(r["goal_index"]) for r in ws.load_split(s)]
                for s in ws.SPLIT_NAMES
            }
        manifest_dir = write_manifest(
            tmp_path / "manifest", backend, splits, **dict(manifest_changes or {})
        )
    options: dict[str, Any] = {"clock": clock} if clock is not None else {}
    service = server.Service(
        seed=protocol.SEED,
        manifest_dir=None if prepare else manifest_dir,
        prepare=prepare,
        session_idle_s=session_idle_s,
        build_record=build_record() if record is None else record,
        **options,
    )
    service.load(lambda: backend)
    return Shop(service, backend, None if prepare else manifest_dir)


#: The index hash of the scripted image (any 64 hex digits: no index is built here).
INDEX_SHA256 = "1" * 64


def build_record() -> dict[str, Any]:
    """What the image's build.json holds when the build matches the protocol and this
    checkout's server directory."""
    return {
        "upstream_commit": protocol.WEBSHOP_SERVER["upstream_commit"],
        "data": {n: p["sha256"] for n, p in protocol.WEBSHOP_DATA["files"].items()},
        "index_sha256": INDEX_SHA256,
        "spacy_model": protocol.WEBSHOP_SERVER["spacy_model"],
        "fuzzy_backend": protocol.WEBSHOP_SERVER["fuzzy_backend"],
        "java": 'openjdk version "11.0.24" 2024-07-16',
        "setuptools": "82.0.1",
        "packages_sha256": "2" * 64,
        "context": ws.context_digest(),
    }


def shop_transport(
    service: Any,
    requests: list[tuple[str, str, Any]] | None = None,
    replies: list[tuple[str, str, int, Any]] | None = None,
) -> httpx.MockTransport:
    """An ``httpx`` transport that hands each request to ``server.route``."""
    server = server_module()

    def handle(request: httpx.Request) -> httpx.Response:
        body = request.content
        if requests is not None:
            requests.append(
                (request.method, request.url.path, json.loads(body) if body else None)
            )
        status, payload = server.route(
            service,
            request.method,
            request.url.raw_path.decode("ascii"),
            lambda: server.parse_body(body),
        )
        if replies is not None:
            replies.append(
                (
                    request.method,
                    request.url.path,
                    int(status),
                    json.loads(server.canonical_json(payload)),
                )
            )
        return httpx.Response(
            int(status),
            content=server.canonical_json(payload),
            headers={"Content-Type": "application/json"},
        )

    return httpx.MockTransport(handle)


def shop_client_factory(
    service: Any,
    requests: list[tuple[str, str, Any]] | None = None,
    replies: list[tuple[str, str, int, Any]] | None = None,
) -> ws.ClientFactory:
    def make(base_url: str, timeout_s: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout_s,
            transport=shop_transport(service, requests, replies),
        )

    return make


# ---------------------------------------------------------------------------
# A scripted shopper (the e2e fake model and the rollout tests)
# ---------------------------------------------------------------------------


def respond(action: str, think: str = "Shopping.") -> str:
    return f"<think>{think}</think><action>{action}</action>"


def shopper_action(messages: Sequence[Mapping[str, Any]], *, right: bool) -> str:
    """What a scripted shopper does on the page its WebShop request shows (the last
    message of the conversation): search for the task, open the first result (``right``)
    or the second, pick each offered option once (its own earlier replies, in the
    conversation, say which it picked), then buy."""
    prompt = str(messages[-1]["content"])
    replies = [str(m["content"]) for m in messages if m.get("role") == "assistant"]
    task = prompt.split("Your task is to: ", 1)[1].split(".\n", 1)[0]
    available = prompt.rsplit(
        "Your admissible actions of the current situation are:", 1
    )[1]
    actions = [
        line.strip()[1:-2]
        for line in available.splitlines()
        if line.strip().startswith("'") and line.strip().endswith("',")
    ]
    if "search[<your query>]" in actions:
        return f"search[{task.split(',')[0]}]"
    clicks = [a[len("click[") : -1] for a in actions if a.startswith("click[")]
    if "buy now" in clicks:
        for option in (c for c in clicks if c not in NAV):
            picked = f"<action>click[{option}]</action>"
            if not any(picked in reply for reply in replies):
                return f"click[{option}]"
        return "click[buy now]"
    products = [c for c in clicks if c not in NAV]
    if products:
        return f"click[{products[0] if right or len(products) < 2 else products[1]}]"
    return "click[back to search]"
