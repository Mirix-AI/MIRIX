"""The WebShop server (``envs/templates/webshop_server/server.py``), run here on the
scripted backend of ``tests/webshop_support.py``: the readiness
gate, the frozen-manifest check, session isolation, the global lock, the guards, prepare
mode and the HTTP layer (a real ``ThreadingHTTPServer`` on a loopback port)."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import io
import json
import os
import random
import re
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from evals.skill_bench import protocol
from evals.skill_bench.envs import webshop as ws
from evals.skill_bench.tests.webshop_support import (
    INDEX_SHA256,
    FakeBackend,
    Shop,
    build_record,
    make_shop,
    respond,
    server_module,
    shop_goals,
    shopper_action,
    synthetic_goal,
    write_manifest,
)

S = server_module()
PREFIX = "0" * 32 + "-"
OTHER = "1" * 32 + "-"


def call(shop: Shop, method: str, path: str, body: Any = None) -> tuple[int, Any]:
    status, payload = S.route(shop.service, method, path, lambda: body or {})
    return int(status), payload


def small_shop(tmp_path: Path, **kw: Any) -> Shop:
    """Ten synthetic goals; goals 1-4 in the manifest."""
    goals = [synthetic_goal(i) for i in range(10)]
    kw.setdefault("splits", {"train": [1], "val": [2], "test": [3]})
    return make_shop(tmp_path, goals=goals, goal_count=10, **kw)


def create(shop: Shop, index: int = 3, prefix: str = PREFIX) -> tuple[int, Any]:
    return call(
        shop,
        "POST",
        "/episodes",
        {"goal_index": index, "session_prefix": prefix, "owner": "test"},
    )


# ---------------------------------------------------------------------------
# Readiness and the frozen manifest
# ---------------------------------------------------------------------------


def test_ready_only_after_loading_a_matching_manifest(tmp_path):
    goals = [synthetic_goal(i) for i in range(10)]
    backend = FakeBackend(goals)
    manifest = write_manifest(tmp_path / "m", backend, {"test": [3]})
    service = S.Service(
        seed=protocol.SEED, manifest_dir=manifest, build_record=build_record()
    )
    status, payload = service.readiness()
    assert (status, payload["state"]) == (503, "loading")
    status, payload = S.route(
        service,
        "POST",
        "/episodes",
        lambda: {"goal_index": 3, "session_prefix": PREFIX},
    )
    assert status == 503 and "loading" in payload["error"]
    service.load(lambda: backend)
    status, payload = service.readiness()
    assert (status, payload["state"], payload["problems"]) == (200, "ready", [])
    info = payload["info"]
    assert info["goal_count"] == 10 and info["seed"] == protocol.SEED
    assert info["price_table_sha256"] == S.price_table_sha256(backend.product_prices)
    assert info["manifest"]["status"] == "final"
    assert set(info["manifest"]["files"]) == {
        "test.json",
        "split_manifest.json",
        "source.json",
    }
    assert info["api_version"] == S.API_VERSION and "load_s" in info
    with pytest.raises(ValueError, match="manifest"):
        S.Service(seed=protocol.SEED, manifest_dir=None)


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"seed": 7}, "seed 7"),
        ({"goal_count": 9}, "the manifest froze 9"),
        ({"prices_sha256": "0" * 64}, "price table"),
        ({"status": "provisional"}, "not 'final'"),
        # The image's index, data and commit are the ones the manifest froze.
        (
            {"source_changes": {"index_sha256": "f" * 64}},
            f"the image's index_sha256 is '{INDEX_SHA256}'",
        ),
        ({"source_changes": {"index_sha256": None}}, "froze no index_sha256"),
        ({"source_changes": {"upstream_commit": "abc"}}, "upstream_commit"),
        ({"source_changes": {"data": {}}}, "the image's data"),
    ],
)
def test_a_manifest_mismatch_refuses_every_episode(tmp_path, changes, problem):
    shop = small_shop(tmp_path, manifest_changes=changes)
    status, payload = shop.service.readiness()
    assert (status, payload["state"]) == (503, "failed")
    assert any(problem in p for p in payload["problems"]), payload["problems"]
    status, payload = create(shop)
    assert status == 503 and "failed" in payload["error"]


@pytest.mark.parametrize("field", list(ws.MANIFEST_FIELDS))
def test_every_pinned_goal_field_is_compared(tmp_path, field):
    goals = [synthetic_goal(i) for i in range(10)]
    backend = FakeBackend(goals)
    manifest = write_manifest(tmp_path / "m", backend, {"test": [3]})
    rows = json.loads((manifest / "test.json").read_text())
    rows[0][field] = ["changed"] if isinstance(rows[0][field], list) else "changed"
    (manifest / "test.json").write_text(json.dumps(rows))
    info, problems = S.verify_manifest(
        goals,
        S.price_table_sha256(backend.product_prices),
        manifest,
        protocol.SEED,
        build_record(),
    )
    assert len(problems) == 1 and f": {field} is " in problems[0]
    assert info["status"] == "final"


def test_manifest_rows_out_of_range_and_unreadable(tmp_path):
    goals = [synthetic_goal(i) for i in range(10)]
    backend = FakeBackend(goals)
    prices = S.price_table_sha256(backend.product_prices)
    manifest = write_manifest(tmp_path / "m", backend, {"test": [3]})
    (manifest / "val.json").write_text(
        json.dumps([{"id": "fixed_99", "goal_index": 99}])
    )
    _, problems = S.verify_manifest(
        goals, prices, manifest, protocol.SEED, build_record()
    )
    assert problems == ["val item fixed_99: goal index 99 out of range"]
    # Without the image's build record, its index, data and commit are not the frozen.
    _, problems = S.verify_manifest(goals, prices, manifest, protocol.SEED)
    assert len(problems) == 4 and sum("the image's" in p for p in problems) == 3
    info, problems = S.verify_manifest(
        goals, prices, tmp_path / "absent", protocol.SEED
    )
    assert info == {} and "unreadable" in problems[0]


def test_a_backend_that_fails_to_load_is_failed_with_its_traceback(tmp_path):
    shop = small_shop(tmp_path)

    def broken() -> Any:
        raise OSError("items_shuffle.json: no such file")

    service = S.Service(seed=protocol.SEED, manifest_dir=shop.manifest_dir)
    service.load(broken)
    status, payload = service.readiness()
    assert (status, payload["state"]) == (503, "failed")
    assert payload["problems"] == ["OSError: items_shuffle.json: no such file"]
    assert "Traceback" in payload["info"]["traceback"]


def test_the_price_table_hash_is_canonical():
    prices = {"B2": 10.5, "B1": 3}
    assert S.price_table_sha256(prices) == S.price_table_sha256({"B1": 3, "B2": 10.5})
    assert S.price_table_sha256(prices) != S.price_table_sha256({"B1": 3, "B2": 10.51})
    assert (
        S.price_table_sha256(prices)
        == hashlib.sha256(b'{"B1":3,"B2":10.5}').hexdigest()
    )


# ---------------------------------------------------------------------------
# Episodes: isolation, guards, deletion
# ---------------------------------------------------------------------------


def test_create_validates_and_isolates(tmp_path):
    shop = small_shop(tmp_path)
    assert create(shop, index=True)[0] == 400  # type: ignore[arg-type]
    assert create(shop, prefix="not-hex")[0] == 400
    assert create(shop, prefix="A" * 32 + "-")[0] == 400
    assert create(shop, index=99)[0] == 400
    status, payload = create(shop, index=5)  # a goal outside the frozen manifest
    assert status == 403 and "not in the frozen manifest" in payload["error"]
    status, payload = create(shop)
    assert status == 200 and payload["episode"] == PREFIX
    assert payload["observation"].startswith("WebShop [SEP] Instruction: [SEP] ")
    assert payload["available_actions"] == {
        "has_search_bar": True,
        "clickables": ["search"],
    }
    assert sorted(shop.live_sessions()) == sorted(
        [s for s in shop.live_sessions() if s.startswith(PREFIX)]
    )
    assert f"{PREFIX}3" in shop.live_sessions() and len(shop.live_sessions()) == 2
    assert create(shop)[0] == 409  # the prefix exists
    assert create(shop, prefix=OTHER)[0] == 200  # the same goal, another episode
    assert len(shop.live_sessions()) == 4


def test_a_failed_create_leaves_no_session_behind(tmp_path):
    shop = small_shop(tmp_path)
    backend = shop.backend
    original = backend.make_env

    def failing(prefix: str) -> Any:
        env = original(prefix)  # its constructor opened a session

        def reset(*a: Any, **k: Any) -> Any:
            raise RuntimeError("upstream reset failed")

        env.reset = reset  # type: ignore[method-assign]
        return env

    backend.make_env = failing  # type: ignore[method-assign]
    status, payload = create(shop)
    assert status == 500 and "upstream reset failed" in payload["error"]
    assert shop.live_sessions() == [] and shop.service.episodes == {}


def test_step_guards(tmp_path):
    shop = small_shop(tmp_path)
    create(shop)
    path = f"/episodes/{PREFIX}/step"
    assert call(shop, "POST", path, {"action": 3})[0] == 400
    assert call(shop, "POST", f"/episodes/{OTHER}/step", {"action": "x"})[0] == 404
    before = list(shop.backend.actions)
    for special in (
        "search[<r>]",
        "search[<a> red]",
        "search[<Q> x]",
        "search[ <c> y]",
    ):
        status, reply = call(shop, "POST", path, {"action": special})
        assert status == 200 and reply["accepted"] is False and reply["done"] is False
        assert reply["observation"].endswith("[SEP] Search")
    assert shop.backend.actions == before and shop.backend.special_searches == []
    status, reply = call(shop, "POST", path, {"action": "click[nothing]"})
    assert reply["accepted"] is False and len(shop.backend.actions) == len(before) + 1
    status, reply = call(shop, "POST", path, {"action": "search[gold]"})
    assert reply["accepted"] is True and "Back to Search" in reply["observation"]
    asin = shop.backend.goals[3]["asin"].lower()
    call(shop, "POST", path, {"action": f"click[{asin}]"})
    call(shop, "POST", path, {"action": "click[red]"})
    call(shop, "POST", path, {"action": "click[large]"})
    status, reply = call(shop, "POST", path, {"action": "click[buy now]"})
    assert reply == {
        "observation": None,
        "reward": 1.0,
        "done": True,
        "accepted": True,
        "available_actions": None,
        "shop_error": None,
    }
    status, reply = call(shop, "POST", path, {"action": "click[buy now]"})
    assert status == 409


def test_accepted_action_is_upstreams_rule():
    clickables = ["back to search", "b0", "search", "Red"]
    assert S.accepted_action("search[red shoes]", clickables)
    assert not S.accepted_action("search[]", clickables)
    assert S.accepted_action("click[Back to Search]", clickables)  # lower-cased
    assert not S.accepted_action("click[search]", clickables)
    assert not S.accepted_action("click[Red]", clickables)  # options keep their case
    assert not S.accepted_action("buy it", clickables)
    assert S.parse_action("click[a[b]]") == ("click[a", "b]")  # greedy, as upstream


def test_delete_drops_every_session_of_the_prefix_only(tmp_path):
    shop = small_shop(tmp_path)
    create(shop)
    create(shop, prefix=OTHER)
    status, reply = call(shop, "DELETE", f"/episodes/{PREFIX}")
    assert (status, reply) == (200, {"deleted_sessions": 2})
    assert all(s.startswith(OTHER) for s in shop.live_sessions())
    assert len(shop.live_sessions()) == 2
    assert call(shop, "DELETE", f"/episodes/{PREFIX}")[0] == 404
    assert call(shop, "POST", f"/episodes/{PREFIX}/step", {"action": "x"})[0] == 404


def test_idle_episodes_are_reaped(tmp_path):
    now = [1000.0]
    shop = small_shop(tmp_path, clock=lambda: now[0], session_idle_s=60)
    create(shop)
    now[0] += 30
    call(shop, "POST", f"/episodes/{PREFIX}/step", {"action": "search[x]"})
    now[0] += 59
    create(shop, prefix=OTHER)  # reaps before it creates: PREFIX was used 59 s ago
    assert PREFIX in shop.service.episodes
    now[0] += 61
    create(shop, prefix="2" * 32 + "-")
    assert PREFIX not in shop.service.episodes and OTHER not in shop.service.episodes
    assert all(s.startswith("2" * 32) for s in shop.live_sessions())


def test_every_call_into_upstream_holds_the_global_lock(tmp_path):
    shop = small_shop(tmp_path)
    lock = shop.service.lock
    seen: list[bool] = []
    backend = shop.backend
    for name in ("make_env", "session_ids", "drop_session"):
        original = getattr(backend, name)

        def wrapped(*a: Any, _original: Any = original, **k: Any) -> Any:
            seen.append(lock.locked())
            return _original(*a, **k)

        setattr(backend, name, wrapped)
    create(shop)
    env = shop.service.episodes[PREFIX].env
    step = env.step

    def locked_step(action: str) -> Any:
        seen.append(lock.locked())
        return step(action)

    env.step = locked_step  # type: ignore[method-assign]
    call(shop, "POST", f"/episodes/{PREFIX}/step", {"action": "search[x]"})
    call(shop, "DELETE", f"/episodes/{PREFIX}")
    assert seen and all(seen)


def test_an_upstream_error_is_a_500_for_that_request_only(tmp_path):
    shop = small_shop(tmp_path)
    create(shop)
    shop.backend.fail_next_step = True
    status, reply = call(
        shop, "POST", f"/episodes/{PREFIX}/step", {"action": "search[x]"}
    )
    assert status == 500 and reply == {"error": "OSError: an upstream failure"}
    status, _ = call(shop, "POST", f"/episodes/{PREFIX}/step", {"action": "search[x]"})
    assert status == 200
    assert call(shop, "GET", "/nothing")[0] == 404
    assert shop.service.readiness()[1]["action_errors"] == 0


class JavaException(Exception):
    """pyjnius's ``jnius.JavaException`` (by name, as the server tells it apart), with
    the Java class name it carries."""

    def __init__(self, message: str, classname: str) -> None:
        super().__init__(message)
        self.classname = classname


@pytest.mark.parametrize(
    "error",
    [
        KeyError("B0LOWERCASE"),
        IndexError("list index out of range"),
        ZeroDivisionError("division by zero"),
        JavaException(
            "JVM exception occurred: maxClauseCount is set to 1024",
            "org.apache.lucene.search.BooleanQuery$TooManyClauses",
        ),
    ],
)
def test_an_action_the_shop_cannot_process_leaves_the_page_and_session(tmp_path, error):
    """An error of upstream's own logic on an action is the action's: the session is
    restored (upstream changed it before raising), the page stays, the reply says why,
    and /ready counts it. Nothing is a 500, so the item is not re-run for free."""
    shop = small_shop(tmp_path)
    create(shop)
    path = f"/episodes/{PREFIX}/step"
    _, page = call(shop, "POST", path, {"action": "search[gold]"})
    session = f"{PREFIX}3"
    before = json.loads(json.dumps(shop.backend.sessions[session]))
    held = shop.backend.sessions[session]
    shop.backend.fail_next_step = error
    status, reply = call(shop, "POST", path, {"action": "search[" + "w " * 2000 + "]"})
    assert status == 200 and reply["accepted"] is False and reply["done"] is False
    assert reply["shop_error"] == f"{type(error).__name__}: {error}"
    assert reply["observation"] == page["observation"]
    assert reply["available_actions"] == page["available_actions"]
    assert shop.backend.sessions[session] == before
    assert shop.backend.sessions[session] is held  # the same dict upstream holds
    ready = shop.service.readiness()[1]
    assert ready["action_errors"] == 1
    (recorded,) = ready["recent_action_errors"]
    assert recorded["goal_index"] == 3 and recorded["episode"] == PREFIX
    assert recorded["error"] == reply["shop_error"] and len(recorded["action"]) == 200
    # The episode goes on from the page it was on.
    asin = shop.backend.goals[3]["asin"].lower()
    status, reply = call(shop, "POST", path, {"action": f"click[{asin}]"})
    assert status == 200 and reply["accepted"] is True and reply["shop_error"] is None


@pytest.mark.parametrize(
    "error",
    [
        MemoryError(),
        OSError("index file gone"),
        RuntimeError("JVM is not running"),
        JavaException("JVM exception occurred: read past EOF", "java.io.EOFException"),
    ],
)
def test_an_error_of_the_server_itself_is_a_500(tmp_path, error):
    shop = small_shop(tmp_path)
    create(shop)
    shop.backend.fail_next_step = error
    status, reply = call(shop, "POST", f"/episodes/{PREFIX}/step", {"action": "x[y]"})
    assert status == 500 and reply["error"].startswith(type(error).__name__)
    assert shop.service.readiness()[1]["action_errors"] == 0


def test_caused_by_the_action():
    assert S.caused_by_the_action(KeyError("x"))
    assert S.caused_by_the_action(ValueError("x"))
    assert S.caused_by_the_action(UnicodeEncodeError("utf-8", "\ud83d", 0, 1, "x"))
    assert S.caused_by_the_action(AttributeError("x"))
    assert not S.caused_by_the_action(MemoryError())
    assert not S.caused_by_the_action(OSError("x"))
    assert not S.caused_by_the_action(RuntimeError("x"))
    assert not S.caused_by_the_action(KeyboardInterrupt())
    assert S.caused_by_the_action(
        JavaException("x", "org.apache.lucene.search.BooleanQuery$TooManyClauses")
    )
    assert S.caused_by_the_action(JavaException("maxClauseCount is set to 1024", ""))
    assert not S.caused_by_the_action(JavaException("x", "java.io.IOException"))


def test_every_call_into_upstream_runs_on_one_thread(tmp_path):
    """pyjnius attaches every thread that calls into the JVM; upstream (its import and
    SimServer() included) runs on the service's one upstream thread only, never on the
    caller's (an HTTP connection thread)."""
    goals = [synthetic_goal(i) for i in range(10)]
    backend = FakeBackend(goals)
    manifest = write_manifest(
        tmp_path / "m", backend, {"train": [1], "val": [2], "test": [3]}
    )
    service = S.Service(
        seed=protocol.SEED, manifest_dir=manifest, build_record=build_record()
    )
    built_on: list[int] = []

    def factory() -> FakeBackend:
        built_on.append(threading.get_ident())
        return backend

    service.load(factory)
    shop = Shop(service, backend, manifest)
    callers: set[int] = set()

    def from_a_thread(method: str, path: str, body: Any = None) -> None:
        def run() -> None:
            callers.add(threading.get_ident())
            call(shop, method, path, body)

        thread = threading.Thread(target=run)
        thread.start()
        thread.join()

    from_a_thread(
        "POST", "/episodes", {"goal_index": 3, "session_prefix": PREFIX, "owner": "t"}
    )
    from_a_thread("POST", f"/episodes/{PREFIX}/step", {"action": "search[gold]"})
    from_a_thread("DELETE", f"/episodes/{PREFIX}")
    assert len(backend.threads) == 1 and backend.threads == set(built_on)
    assert not backend.threads & (callers | {threading.get_ident()})
    service.worker.close()
    with pytest.raises(RuntimeError):
        service.worker.run(lambda: None)


# ---------------------------------------------------------------------------
# Source inspection mode
# ---------------------------------------------------------------------------


def test_prepare_mode_serves_goals_and_diagnostics_only(tmp_path):
    shop = small_shop(tmp_path, prepare=True)
    status, payload = shop.service.readiness()
    assert (status, payload["state"]) == (503, "prepare")
    assert payload["info"]["mode"] == "prepare"
    status, payload = call(shop, "GET", "/goals")
    assert status == 200 and len(payload["goals"]) == 10
    assert payload["goals"][3] == {
        "index": 3,
        **{name: shop.backend.goals[3][name] for name in ws.MANIFEST_FIELDS},
    }
    status, payload = call(shop, "POST", "/diagnostics", {"indices": [3, 4]})
    assert status == 200 and set(payload["items"]) == {"3", "4"}
    assert payload["items"]["3"]["best_reward"] == 1.0
    assert payload["items"]["3"]["gold_search_rank"] == 0
    for bad in ({"indices": list(range(51))}, {"indices": ["3"]}, {"indices": 3}):
        assert call(shop, "POST", "/diagnostics", bad)[0] == 400
    assert call(shop, "POST", "/diagnostics", {"indices": [10]})[0] == 400
    assert create(shop)[0] == 503  # no episodes in prepare mode
    served = small_shop(tmp_path / "serve")
    assert call(served, "GET", "/goals")[0] == 503


# ---------------------------------------------------------------------------
# HTTP (a real server on a loopback port)
# ---------------------------------------------------------------------------


@pytest.fixture
def http_shop(tmp_path: Path) -> Iterator[tuple[Shop, str]]:
    shop = make_shop(tmp_path)
    httpd = S.start_http(shop.service, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield shop, f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.mark.asyncio
async def test_an_episode_over_real_http(http_shop):
    shop, url = http_shop
    env = ws.WebShopEnv(server_url=url)  # the default httpx client
    task = env.load_tasks("test")[1]
    episode = env.episode(task)
    async with episode:
        reset = await episode.reset()
        fields = reset.fields
        for _ in range(10):
            request = env.actor_messages(fields)
            step = await episode.step(respond(shopper_action(request, right=True)))
            fields = step.fields
            if step.done:
                break
    assert step.done and step.private["reward"] == 1.0
    assert shop.live_sessions() == []
    async with httpx.AsyncClient(base_url=url, trust_env=False) as client:
        ready = await client.get("/ready")
        assert ready.status_code == 200 and ready.json()["state"] == "ready"
        bad = await client.post("/episodes", content=b"{not json")
        assert bad.status_code == 400 and "not JSON" in bad.json()["error"]
        bad = await client.post("/episodes", content=b"[1]")
        assert bad.status_code == 400 and "not an object" in bad.json()["error"]
        missing = await client.get("/nowhere")
        assert missing.status_code == 404
    # A body over the cap is refused unread (the connection then closes).
    with pytest.raises(S.ServiceError) as refused:
        S.parse_body(b" " * (S.MAX_BODY_BYTES + 1))
    assert refused.value.status == 413
    assert S.parse_body(b"") == {}


@pytest.mark.asyncio
async def test_concurrent_episodes_over_http_stay_apart(http_shop):
    shop, url = http_shop
    env = ws.WebShopEnv(server_url=url)
    tasks = env.load_tasks("test")[:4]

    async def one(task: Any, right: bool) -> float:
        episode = env.episode(task)
        async with episode:
            reset = await episode.reset()
            fields = reset.fields
            for _ in range(10):
                request = env.actor_messages(fields)
                step = await episode.step(respond(shopper_action(request, right=right)))
                fields = step.fields
                if step.done:
                    return float(step.private["reward"])
        raise AssertionError("no purchase")

    rewards = await asyncio.gather(
        *(one(task, right=i % 2 == 0) for i, task in enumerate(tasks)),
        one(tasks[0], right=False),  # the same goal twice at once
    )
    assert rewards[0] == rewards[2] == 1.0 and rewards[4] == 0.0
    assert shop.live_sessions() == []


# ---------------------------------------------------------------------------
# UpstreamBackend: the glue to upstream, over stand-in modules
# ---------------------------------------------------------------------------

WEBSHOP_UPSTREAM_DIR = os.environ.get("WEBSHOP_UPSTREAM_DIR")


class StubSimServer:
    """``web_agent_site.envs.web_agent_text_env.SimServer``'s surface; records how it
    was built: its arguments, the product paths the engine held and the next draw of
    the global random state (equal to ``random.seed(seed); random.random()`` only when
    nothing drew from it between the seeding and the construction)."""

    built: list[dict[str, Any]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        engine = sys.modules["web_agent_site.engine.engine"]
        self.built.append(
            {
                "args": args,
                "kwargs": kwargs,
                "attr_path": engine.DEFAULT_ATTR_PATH,
                "human_attr_path": engine.HUMAN_ATTR_PATH,
                "next_random": random.random(),
            }
        )
        self.goals = [{"asin": "B1", "goal_options": ["red"], "price_upper": 50.0}]
        self.product_prices = {"B1": 12.5, "B2": 3.0}
        self.product_item_dict = {
            "B1": {"asin": "B1", "options": {"color": ["red", "blue"]}},
            "B2": {"asin": "B2", "options": {}},
        }
        self.all_products = list(self.product_item_dict.values())
        self.search_engine = object()
        self.user_sessions: dict[str, dict[str, Any]] = {}


class StubTextEnv:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _upstream_stubs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    import difflib
    import types

    calls: dict[str, Any] = {"rewards": [], "searches": []}
    modules = {
        name: types.ModuleType(name)
        for name in (
            "web_agent_site",
            "web_agent_site.engine",
            "web_agent_site.engine.engine",
            "web_agent_site.engine.goal",
            "web_agent_site.envs",
            "web_agent_site.envs.web_agent_text_env",
            "thefuzz",
            "thefuzz.fuzz",
        )
    }
    engine = modules["web_agent_site.engine.engine"]
    engine.DEFAULT_ATTR_PATH = "data/items_ins_v2_1000.json"  # upstream's 1k default
    engine.HUMAN_ATTR_PATH = "data/items_human_ins.json"

    def top_n(keywords, search_engine, all_products, product_item_dict):  # noqa: ANN001
        calls["searches"].append(keywords)
        return [all_products[1], all_products[0]]

    engine.get_top_n_product_from_keywords = top_n
    goal = modules["web_agent_site.engine.goal"]

    def get_reward(purchased_product, goal, price, options, **kwargs):  # noqa: ANN001
        calls["rewards"].append((purchased_product, goal, price, options, kwargs))
        return 0.75, {"r_option": 0.5, "r_att": 1.0}

    goal.get_reward = get_reward
    goal.nlp = types.SimpleNamespace(
        meta={"lang": "en", "name": "core_web_sm", "version": "3.3.0"}
    )
    text_env = modules["web_agent_site.envs.web_agent_text_env"]
    text_env.SimServer = StubSimServer
    text_env.WebAgentTextEnv = StubTextEnv
    modules["thefuzz.fuzz"].SequenceMatcher = difflib.SequenceMatcher
    modules["web_agent_site"].engine = modules["web_agent_site.engine"]
    modules["web_agent_site"].envs = modules["web_agent_site.envs"]
    modules["web_agent_site.engine"].engine = engine
    modules["web_agent_site.engine"].goal = goal
    modules["web_agent_site.envs"].web_agent_text_env = text_env
    modules["thefuzz"].fuzz = modules["thefuzz.fuzz"]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    StubSimServer.built = []
    return calls


def test_the_upstream_backend_glue(tmp_path, monkeypatch):
    calls = _upstream_stubs(monkeypatch)
    backend = S.UpstreamBackend(tmp_path, protocol.SEED)
    (built,) = StubSimServer.built
    # The full product set, not upstream's 1k default, before SimServer() loads it.
    assert built["attr_path"] == str(tmp_path / "items_ins_v2.json")
    assert built["human_attr_path"] == str(tmp_path / "items_human_ins.json")
    assert built["args"] == (
        "http://127.0.0.1:3000",
        str(tmp_path / "items_shuffle.json"),
    )
    assert built["kwargs"] == {
        "filter_goals": None,
        "limit_goals": -1,
        "num_products": None,
        "human_goals": 1,
        "show_attrs": False,
    }
    # Seeded immediately before SimServer() (nothing drew in between).
    random.seed(protocol.SEED)
    assert built["next_random"] == random.random()
    assert backend.goals is backend.sim.goals
    assert backend.product_prices == {"B1": 12.5, "B2": 3.0}
    env = backend.make_env(PREFIX)
    assert env.kwargs == {
        "observation_mode": "text",
        "server": backend.sim,
        "session_prefix": PREFIX,
    }
    assert backend.info() == {
        "spacy_model": protocol.WEBSHOP_SERVER["spacy_model"],
        "fuzzy_backend": protocol.WEBSHOP_SERVER["fuzzy_backend"],
        "products": 2,
    }
    # The gold product bought with the goal's options, a list or a dict of them.
    best = backend.best_reward({"asin": "B1", "goal_options": ["red"]})
    assert best == {
        "best_reward": 0.75,
        "reward_info": {"r_att": 1.0, "r_option": 0.5},
        "options_available": True,
    }
    product, _goal, price, options, kwargs = calls["rewards"][-1]
    assert product["asin"] == "B1" and price == 12.5
    assert options == {"0": "red"} and kwargs == {"verbose": True}
    best = backend.best_reward({"asin": "B1", "goal_options": {"color": "green"}})
    assert calls["rewards"][-1][3] == {"0": "green"}
    assert best["options_available"] is False
    assert backend.search_rank("Red Shoes", "B1") == 1
    assert calls["searches"][-1] == ["red", "shoes"]
    assert backend.search_rank("red", "B9") is None


def test_the_upstream_backend_restores_a_session_as_it_was(tmp_path, monkeypatch):
    """What an action may change of upstream's session (SimServer.receive): restored in
    place (upstream's environment holds the dict by its key), the goal untouched."""
    from collections import defaultdict

    _upstream_stubs(monkeypatch)
    backend = S.UpstreamBackend(tmp_path, protocol.SEED)
    goal = backend.goals[0]
    session = {
        "goal": goal,
        "done": False,
        "keywords": ["red", "shoes"],
        "page": 1,
        "asin": "B1",
        "asins": {"B1"},
        "options": {"color": "red"},
        "actions": defaultdict(int, {"search": 1, "asin": 1}),
    }
    backend.sim.user_sessions[PREFIX + "0"] = session
    saved = backend.save_session(PREFIX + "0")
    # What search_results / item_page / done change before they can raise.
    session.update(keywords=["x"], page=2, asin=None, options={})
    session["asins"].add("B2")
    session["actions"]["search"] += 1
    session["actions"]["purchase"] += 1
    session["verbose_info"] = {}
    backend.restore_session(PREFIX + "0", saved)
    assert backend.sim.user_sessions[PREFIX + "0"] is session
    assert session == {
        "goal": goal,
        "done": False,
        "keywords": ["red", "shoes"],
        "page": 1,
        "asin": "B1",
        "asins": {"B1"},
        "options": {"color": "red"},
        "actions": {"search": 1, "asin": 1},
    }
    assert session["goal"] is goal and isinstance(session["actions"], defaultdict)
    assert session["actions"]["never"] == 0
    assert backend.save_session("absent") is None
    backend.restore_session("absent", None)
    assert "absent" not in backend.sim.user_sessions


def _upstream_tree(rel: str) -> Any:
    import ast

    assert WEBSHOP_UPSTREAM_DIR is not None
    return ast.parse((Path(WEBSHOP_UPSTREAM_DIR) / rel).read_text())


def _function(tree: Any, name: str, cls: str | None = None) -> Any:
    import ast

    body = tree.body
    if cls is not None:
        body = next(
            n.body for n in body if isinstance(n, ast.ClassDef) and n.name == cls
        )
    return next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)


@pytest.mark.skipif(
    not WEBSHOP_UPSTREAM_DIR, reason="WEBSHOP_UPSTREAM_DIR (WebShop clone) not set"
)
def test_the_stand_ins_have_upstreams_surface():
    """The modules the glue test stands in for, as upstream 64fa2a5 defines them: the
    names and signatures UpstreamBackend calls, the product paths load_products reads
    from the engine module (so setting them there takes the full set), and the session
    keys an action changes (what save_session copies)."""
    import ast

    def args(fn: Any) -> list[str]:
        return [a.arg for a in fn.args.args]

    engine = _upstream_tree("web_agent_site/engine/engine.py")
    load = _function(engine, "load_products")
    read = {n.id for n in ast.walk(load) if isinstance(n, ast.Name)}
    assert {"DEFAULT_ATTR_PATH", "HUMAN_ATTR_PATH"} <= read
    imported = {
        alias.name
        for n in engine.body
        if isinstance(n, ast.ImportFrom) and n.module == "web_agent_site.utils"
        for alias in n.names
    }
    assert {"DEFAULT_ATTR_PATH", "HUMAN_ATTR_PATH"} <= imported
    assert args(_function(engine, "get_top_n_product_from_keywords"))[:4] == [
        "keywords",
        "search_engine",
        "all_products",
        "product_item_dict",
    ]
    goal = _function(_upstream_tree("web_agent_site/engine/goal.py"), "get_reward")
    assert args(goal) == ["purchased_product", "goal", "price", "options"]
    assert goal.args.kwarg is not None and goal.args.kwarg.arg == "kwargs"
    text_env = _upstream_tree("web_agent_site/envs/web_agent_text_env.py")
    init = _function(text_env, "__init__", "SimServer")
    assert args(init) == [
        "self",
        "base_url",
        "file_path",
        "filter_goals",
        "limit_goals",
        "num_products",
        "human_goals",
        "show_attrs",
    ]
    assigned = {
        n.attr
        for n in ast.walk(init)
        if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
    }
    assert {
        "goals",
        "product_prices",
        "product_item_dict",
        "all_products",
        "search_engine",
        "user_sessions",
    } <= assigned
    env_init = _function(text_env, "__init__", "WebAgentTextEnv")
    assert args(env_init)[:2] == ["self", "observation_mode"]
    assert "server" in args(env_init) and env_init.args.kwarg is not None
    source = ast.unparse(env_init)
    assert "self.kwargs.get('session_prefix')" in source
    assert "self.session" in ast.unparse(
        _function(text_env, "reset", "WebAgentTextEnv")
    )
    # The session keys receive / search_results / item_page / done write; the goal is
    # written only under ``assigned_instruction_text``, which nothing but SimServer's
    # constructor (to None) ever sets.
    keys = {
        n.slice.value
        for fn in ("receive", "search_results", "item_page", "item_sub_page", "done")
        for n in ast.walk(_function(text_env, fn, "SimServer"))
        if isinstance(n, ast.Subscript)
        and isinstance(n.ctx, ast.Store)
        and isinstance(n.slice, ast.Constant)
        and ast.unparse(n.value).startswith(
            ("session", "self.user_sessions[session_id]")
        )
        and "['goal']" not in ast.unparse(n.value)
    } | {"keywords", "page", "asin", "asins", "options", "actions"}
    setters = [
        n
        for n in ast.walk(text_env)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Attribute) and t.attr == "assigned_instruction_text"
            for t in n.targets
        )
    ]
    assert len(setters) == 1 and ast.unparse(setters[0].value) == "None"
    assert keys <= {
        "goal",
        "done",
        "keywords",
        "page",
        "asin",
        "asins",
        "options",
        "actions",
        "verbose_info",
        "reward",
    } | {"search", "purchase"}, keys


def test_check_install_imports_upstream(monkeypatch):
    """Every upstream module the server imports is imported at build
    time, so an allowlist gap (or a setuptools without pkg_resources) fails the build."""
    import types

    _upstream_stubs(monkeypatch)
    spacy = types.ModuleType("spacy")
    spacy.load = lambda name: types.SimpleNamespace(meta={"version": "3.3.0"})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "spacy", spacy)
    monkeypatch.setattr(S, "WEBSHOP_DIR", Path("/nonexistent-webshop"))
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None if name == "Levenshtein" else real_find_spec(name, *a),
    )
    assert S.check_install() == {
        "spacy_model": "en_core_web_sm==3.3.0",
        "fuzzy_backend": "difflib",
    }
    monkeypatch.delitem(sys.modules, "web_agent_site.envs.web_agent_text_env")
    monkeypatch.setattr(
        S, "UPSTREAM_MODULES", (*S.UPSTREAM_MODULES, "web_agent_site.nowhere")
    )
    with pytest.raises(SystemExit, match="upstream does not import"):
        S.check_install()
    assert S.check_install(import_upstream=False)["fuzzy_backend"] == "difflib"


def test_the_environment_record(monkeypatch):
    monkeypatch.setenv(S.CONTEXT_ENV, "c" * 64)
    record = S.environment_record()
    assert record["context"] == "c" * 64
    assert re.fullmatch(r"[0-9a-f]{64}", record["packages_sha256"])
    assert record["setuptools"] is None or re.fullmatch(r"[\w.]+", record["setuptools"])
    assert record["java"] is None or isinstance(record["java"], str)
    monkeypatch.delenv(S.CONTEXT_ENV)
    assert S.environment_record()["context"] is None


# ---------------------------------------------------------------------------
# Image build steps that run offline
# ---------------------------------------------------------------------------


#: ``data_index.py``, as ``server.py`` imports it (one module object: a monkeypatch of it
#: is the server's too).
DI = S.data_index
SERVER_DIR = protocol.SKILL_BENCH_DIR / ws.SERVER_DIR


def test_fetch_data_verifies_size_and_hash(tmp_path, monkeypatch):
    data = b"product data"
    pins = {
        "hf_repo": "x/y",
        "hf_revision": "r",
        "files": {
            "items.json": {
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        },
    }
    pins_path = tmp_path / "data.json"
    pins_path.write_text(json.dumps(pins))
    fetched: list[str] = []

    def urlopen(url: str, timeout: float) -> io.BytesIO:
        fetched.append(url)
        return io.BytesIO(data)

    monkeypatch.setattr(DI.urllib.request, "urlopen", urlopen)
    out = tmp_path / "data"
    assert DI.fetch_data(out, pins_path) == {
        "items.json": pins["files"]["items.json"]["sha256"]
    }
    assert fetched == ["https://huggingface.co/datasets/x/y/resolve/r/items.json"]
    assert DI.fetch_data(out, pins_path)  # already there: not fetched again
    assert len(fetched) == 1
    (out / "items.json").write_bytes(b"other")
    monkeypatch.setattr(
        DI.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(b"x")
    )
    with pytest.raises(SystemExit, match="pinned"):
        DI.fetch_data(out, pins_path)


def test_fetch_data_from_the_command_line_reads_the_pins_next_to_it(
    tmp_path, monkeypatch, capsys
):
    pins = json.loads((SERVER_DIR / "data.json").read_text())
    assert DI.DATA_PINS == SERVER_DIR / "data.json"
    seen: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        DI,
        "fetch_data",
        lambda out, pins_path=DI.DATA_PINS: seen.append((out, pins_path)) or {},
    )
    assert DI.main(["fetch-data", "--out", str(tmp_path / "d")]) == 0
    assert seen == [(tmp_path / "d", DI.DATA_PINS)]
    assert DI.main(["fetch-data"]) == 0 and seen[-1][0] == DI.DATA_DIR
    assert set(pins["files"]) == set(protocol.WEBSHOP_DATA["files"])
    capsys.readouterr()


def test_the_index_documents_are_upstreams_conversion():
    """Upstream ``convert_product_file_format.py``: title, description, the first bullet
    point and the options, lower-cased, keyed by ASIN, the product kept raw."""
    product = {
        "asin": "B01",
        "Title": "Red Shoe",
        "Description": "A Shoe",
        "BulletPoints": ["Comfy", "ignored"],
        "options": {"size": ["8", "9"], "color": ["Red"]},
    }
    (doc,) = DI.documents([product])
    assert doc == {
        "id": "B01",
        "contents": "red shoe a shoe comfy size: 8, 9, and color: red",
        "product": product,
    }
    (bare,) = DI.documents([{**product, "options": None}])
    assert bare["contents"] == "red shoe a shoe comfy "
    argv = DI.index_argv(Path("/r"), Path("/i"))
    assert argv[1:3] == ["-m", "pyserini.index.lucene"]
    assert argv[argv.index("--threads") + 1] == "1"
    assert argv[argv.index("--input") + 1] == "/r"
    assert argv[argv.index("--index") + 1] == "/i"
    assert {"--storePositions", "--storeDocvectors", "--storeRaw"} <= set(argv)


PRODUCTS = [
    {
        "asin": f"B0{i}",
        "Title": f"Shoe {i}",
        "Description": "A shoe",
        "BulletPoints": ["Comfy"],
        "options": {"size": [str(i)]},
    }
    for i in range(3)
]


class FakeIndexer:
    """``subprocess.run`` of the indexer: reads the documents it is given (their sha256 and
    number) and writes index files holding a fresh random id, as Lucene does."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, int]] = []

    def __call__(self, argv: list[str], check: bool) -> Any:
        import subprocess

        assert check and argv[1:3] == ["-m", DI.INDEX_MODULE]
        source = Path(argv[argv.index("--input") + 1]) / DI.DOCUMENTS_FILE
        data = source.read_bytes()
        self.seen.append((hashlib.sha256(data).hexdigest(), data.count(b"\n")))
        index = Path(argv[argv.index("--index") + 1])
        index.mkdir(parents=True, exist_ok=True)
        (index / "_0.si").write_bytes(os.urandom(16) + data[:8])
        (index / "segments_1").write_bytes(os.urandom(16))
        return subprocess.CompletedProcess(argv, 0)


@pytest.fixture
def fake_upstream(monkeypatch) -> FakeIndexer:
    """``build_index``'s upstream engine, indexer, ``IndexReader`` and pyserini version,
    as test doubles (the image holds the real ones)."""
    import types

    engine = types.ModuleType("web_agent_site.engine.engine")
    engine.load_products = lambda filepath: (list(PRODUCTS), {}, {}, {})
    package = types.ModuleType("web_agent_site.engine")
    package.engine = engine
    monkeypatch.setitem(
        sys.modules, "web_agent_site", types.ModuleType("web_agent_site")
    )
    monkeypatch.setitem(sys.modules, "web_agent_site.engine", package)
    monkeypatch.setitem(sys.modules, "web_agent_site.engine.engine", engine)
    monkeypatch.setattr(sys, "path", list(sys.path))
    indexer = FakeIndexer()
    monkeypatch.setattr(DI.subprocess, "run", indexer)

    def stats(index_dir: Path) -> dict[str, int]:
        assert (index_dir / "segments_1").is_file()
        return {"documents": indexer.seen[-1][1], "total_terms": 12}

    monkeypatch.setattr(DI, "index_stats", stats)
    monkeypatch.setattr(DI, "pyserini_version", lambda: "0.17.0")
    return indexer


def test_the_index_record_describes_the_index_not_its_files(tmp_path, fake_upstream):
    """Lucene writes a random id into every index file, so two builds of the same
    documents never have the same files. The record (whose sha256 the build record
    keeps) is what decides how the index searches: the documents fed to the indexer
    (sha256 and number), the indexer's module, arguments (no paths) and pyserini version,
    and the index's statistics; the same bytes from every build, wherever it runs."""
    first = DI.build_index(tmp_path / "data", tmp_path / "a" / "indexes")
    second = DI.build_index(tmp_path / "data", tmp_path / "b" / "deep" / "indexes")
    assert DI.record_text(first) == DI.record_text(second)
    assert DI.index_hashes(tmp_path / "a" / "indexes") != DI.index_hashes(
        tmp_path / "b" / "deep" / "indexes"
    )
    # The documents' sha256 and number are those of the file the indexer read.
    (seen_sha256, seen_count), again = fake_upstream.seen
    assert again == (seen_sha256, seen_count)
    expected = b"".join(
        (json.dumps(doc) + "\n").encode() for doc in DI.documents(PRODUCTS)
    )
    assert seen_sha256 == hashlib.sha256(expected).hexdigest()
    assert first == {
        "documents": {"sha256": seen_sha256, "count": len(PRODUCTS)},
        "indexer": {
            "module": "pyserini.index.lucene",
            "args": list(DI.INDEX_ARGS),
            "pyserini": "0.17.0",
        },
        "index": {"documents": len(PRODUCTS), "total_terms": 12},
    }
    assert seen_count == len(PRODUCTS)
    assert str(tmp_path) not in DI.record_text(first)
    # The documents are removed once indexed; the index stays where it was asked for.
    assert not list(tmp_path.rglob(DI.DOCUMENTS_FILE))
    assert (tmp_path / "a" / "resources").is_dir()
    # Other documents, another record.
    PRODUCTS.append({**PRODUCTS[0], "asin": "B09"})
    try:
        third = DI.build_index(tmp_path / "data", tmp_path / "c" / "indexes")
    finally:
        PRODUCTS.pop()
    assert third["documents"]["count"] == len(PRODUCTS) + 1
    assert third["documents"]["sha256"] != first["documents"]["sha256"]


def test_the_index_record_argv_fills_in_only_the_two_directories():
    argv = DI.index_argv(Path("/r"), Path("/i"))
    assert argv[0] == sys.executable
    assert [a for a in argv[3:] if a not in ("/r", "/i")] == [
        a for a in DI.INDEX_ARGS if a not in ("{input}", "{index}")
    ]
    assert DI.INDEX_ARGS.count("{input}") == 1 and DI.INDEX_ARGS.count("{index}") == 1


@pytest.mark.parametrize("indexed", [0, len(PRODUCTS) + 1])
def test_build_index_refuses_an_index_without_its_documents(
    tmp_path, fake_upstream, monkeypatch, indexed
):
    """The indexer may skip a document it cannot index (the index then holds fewer than
    were written), never index none or more than were written."""
    monkeypatch.setattr(DI, "index_stats", lambda index_dir: {"documents": indexed})
    with pytest.raises(SystemExit, match=f"holds {indexed} documents of the 3"):
        DI.build_index(tmp_path / "data", tmp_path / "indexes")


def test_build_index_writes_the_record_and_the_files_hashes_apart(
    tmp_path, fake_upstream, monkeypatch, capsys
):
    """``build-index`` writes the record where ``write-build-record`` hashes it (sorted
    keys, indent 1), and the index files' sha256 next to it for diagnosis; a rebuild to
    another directory (``--index-dir``, ``--record``) writes the same record bytes."""
    root = tmp_path / "search_engine"
    monkeypatch.setattr(DI, "INDEX_DIR", root / "indexes")
    monkeypatch.setattr(DI, "INDEX_RECORD", root / "index.sha256.json")
    assert DI.main(["build-index", "--data-dir", str(tmp_path)]) == 0
    record = (root / "index.sha256.json").read_text()
    assert json.loads(record)["documents"]["count"] == len(PRODUCTS)
    assert record == json.dumps(json.loads(record), indent=1, sort_keys=True)
    files = root / DI.INDEX_FILES_RECORD.name
    assert files.read_text() == json.dumps(
        DI.index_hashes(root / "indexes"), indent=1, sort_keys=True
    )
    assert set(json.loads(files.read_text())) == {"_0.si", "segments_1"}
    again = tmp_path / "again"
    argv = ["build-index", "--data-dir", str(tmp_path)]
    argv += ["--index-dir", str(again / "indexes")]
    argv += ["--record", str(again / "index.sha256.json")]
    assert DI.main(argv) == 0
    assert (again / "index.sha256.json").read_text() == record
    assert (again / DI.INDEX_FILES_RECORD.name).read_text() != files.read_text()
    capsys.readouterr()


def test_the_build_record_holds_the_data_and_index_hashes(
    tmp_path, monkeypatch, capsys
):
    """``write-build-record`` (the image's last layer): the upstream commit, the sha256
    of each pinned product file as the image holds it, the sha256 of the index record
    ``data_index.py build-index`` wrote, the install check and the environment record
    (the build context's digest included). ``/ready`` and the harness compare these."""
    import subprocess

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pins = {"hf_repo": "x/y", "hf_revision": "r", "files": {}}
    for name, body in (("b.json", b"bee"), ("a.json", b"ay")):
        (data_dir / name).write_bytes(body)
        pins["files"][name] = {"size": len(body), "sha256": "unused"}
    pins_path = tmp_path / "data.json"
    pins_path.write_text(json.dumps(pins))
    index_record = tmp_path / "index.sha256.json"
    index_record.write_text('{"x": "1"}')
    commit = protocol.WEBSHOP_SERVER["upstream_commit"]

    def run(argv: list[str], **kw: Any) -> Any:
        assert argv[:2] == ["git", "-C"] and argv[-2:] == ["rev-parse", "HEAD"]
        return subprocess.CompletedProcess(argv, 0, stdout=commit + "\n", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(
        S, "check_install", lambda: {"spacy_model": "m", "fuzzy_backend": "difflib"}
    )
    monkeypatch.setattr(S, "environment_record", lambda: {"context": "c" * 64})
    record = S.build_record(data_dir, pins_path=pins_path, index_record=index_record)
    assert record == {
        "upstream_commit": commit,
        "data": {
            "a.json": hashlib.sha256(b"ay").hexdigest(),
            "b.json": hashlib.sha256(b"bee").hexdigest(),
        },
        "index_sha256": hashlib.sha256(b'{"x": "1"}').hexdigest(),
        "spacy_model": "m",
        "fuzzy_backend": "difflib",
        "context": "c" * 64,
    }
    assert set(S.BUILD_PINS) <= set(record)
    # The command writes it where /ready reads it, from the shared pins and record.
    out = tmp_path / "build.json"
    monkeypatch.setattr(S, "BUILD_RECORD", out)
    monkeypatch.setattr(S, "DATA_PINS", pins_path)
    monkeypatch.setattr(
        S,
        "build_record",
        lambda data_dir: {"data_dir": str(data_dir)},
    )
    assert S.main(["write-build-record", "--data-dir", str(data_dir)]) == 0
    assert json.loads(out.read_text()) == {"data_dir": str(data_dir)}
    assert S.main(["write-build-record"]) == 0
    assert json.loads(out.read_text()) == {"data_dir": str(DI.DATA_DIR)}
    capsys.readouterr()


def test_the_server_reads_its_paths_and_hashes_from_data_index():
    """One source for what the data and index layers wrote: the server's pins, its file
    hash and its default data directory are ``data_index.py``'s, which the image fetches
    to outside upstream's checkout (``WEBSHOP_DATA_DIR``)."""
    assert S.DATA_PINS == DI.DATA_PINS and S.file_sha256 is DI.file_sha256
    assert S.WEBSHOP_DIR == DI.WEBSHOP_DIR
    assert DI.INDEX_RECORD == DI.WEBSHOP_DIR / "search_engine" / "index.sha256.json"
    assert DI.INDEX_DIR.parent == DI.INDEX_RECORD.parent
    assert DI.INDEX_FILES_RECORD.parent == DI.INDEX_RECORD.parent
    assert DI.INDEX_FILES_RECORD != DI.INDEX_RECORD
    assert str(DI.DATA_DIR) == "/opt/webshop-data"  # the image's ENV WEBSHOP_DATA_DIR
    assert not str(DI.DATA_DIR).startswith(str(DI.WEBSHOP_DIR) + "/")
    assert not hasattr(S, "fetch_data") and not hasattr(S, "build_index")
    with pytest.raises(SystemExit):
        S.main(["fetch-data"])  # the data and index steps are data_index.py's


def test_the_build_record_is_optional(tmp_path):
    assert S.read_build_record(tmp_path / "absent.json") == {}
    path = tmp_path / "build.json"
    path.write_text(json.dumps({"upstream_commit": "abc"}))
    assert S.read_build_record(path) == {"upstream_commit": "abc"}


def test_serve_requires_the_seed():
    with pytest.raises(SystemExit):
        S.main(["serve", "--prepare"])


def test_shop_goals_place_the_frozen_rows():
    rows = {
        5: {
            "goal_index": 5,
            "asin": "B5",
            "instruction": "i need x",
            "goal_options": [],
        }
    }
    goals = shop_goals(rows, 8)
    assert len(goals) == 8 and goals[5]["asin"] == "B5"
    assert goals[5]["instruction_text"].startswith("i need x, and price lower than")
    assert goals[4]["asin"] == "B000000004"
