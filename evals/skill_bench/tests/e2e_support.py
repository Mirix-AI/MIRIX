"""The offline end-to-end matrix and helpers to tamper with its records.

:func:`build_matrix` runs, through ``run.py``'s :class:`~evals.skill_bench.run.CellRunner`,
every benchmark x the no-memory method and the two baselines (ReasoningBank, AWM), 3
learning and 3 test items each, on one real loopback gateway over the fake upstream
(``tests/fakes.py``): nothing reaches a provider and nothing is paid.

* **ALFWorld**: the real :class:`AlfWorldEnv` (SkillOpt's templates, parsing, fallbacks)
  on scripted games (:class:`E2EGame`): the fake model cannot play real TextWorld games,
  which ``test_env_alfworld.py`` covers. A third of the games cannot be won, so those
  episodes run to the 50-step cap. Smoke mode: the head of the seed-42 train and val
  orders.
* **WebShop**: the real :class:`WebShopEnv` and the image's real ``server.py`` service over
  the scripted shop of ``tests/webshop_support.py`` (no Java, no product data), the frozen
  splits, smoke mode. The fake shopper searches, opens the gold product when it is
  "right" (else a distractor), picks the offered options and buys.

The fake model (:class:`E2EModel`) answers the actor from the scorer's gold answers so that
scores vary (it is right on about half of the items without memory and three quarters
with a learned memory), the judge J with a deterministic verdict, and each baseline's
learner in its own output format. "With memory" means the prompt carries what the fake
learners produced (``LEARNED_MARKS``: RB's item titles, AWM's workflow), not
merely a non-empty memory slot, so a method that learned nothing gets no boost; on
ALFWorld the actor wins a game it is "right" on and waits out the others.

MIRIX joins through :func:`build_matrix` with ``"mirix"`` in ``methods``
(``test_e2e.py::test_mirix_joins_the_matrix``, an integration test): run.py starts a real
MIRIX server per cell on the compose Postgres, and :class:`~evals.skill_bench.tests.
mirix_model.MirixModel` plays MIRIX's topic extraction, distiller and curator in front of
the scripted model, so MIRIX learns a skill whose text is one of the ``LEARNED_MARKS``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evals.skill_bench import protocol
from evals.skill_bench.cell_records import CellPaths, read_json, write_json_atomic
from evals.skill_bench.envs import alfworld as alf
from evals.skill_bench.envs.base import Env
from evals.skill_bench.methods import awm as A
from evals.skill_bench.methods import judge as J
from evals.skill_bench.methods import reasoningbank as R
from evals.skill_bench.run import CellOutcome, CellRunner, CellSpec, RunOptions
from evals.skill_bench.tests.fakes import FakeUpstream, Reply
from evals.skill_bench.tests.mirix_model import MIRIX_LEARNED_MARK, MirixModel
from evals.skill_bench.tests.rollout_support import (
    benchmark_of,
    system_text,
    text_reply,
)
from evals.skill_bench.tests.s1_support import first_text, items_reply, judged
from evals.skill_bench.tests.test_env_alfworld import ScriptedGame, act
from evals.skill_bench.tests.webshop_support import make_shop, respond, shopper_action

E2E_METHODS: tuple[str, ...] = ("none", "reasoningbank", "awm")
E2E_MODEL = "luna"
N_ITEMS = 3
#: The gateway's upstream in the tests (``fakes.serve_gateway``).
TEST_UPSTREAM = "http://upstream"


def _bucket(text: str, buckets: int) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % buckets


# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------


#: The objects of the scripted games' goals: the actor's prompt differs across games, so
#: the fake actor can be right on some games and not on others.
E2E_OBJECTS = ("cloth", "mug", "apple", "book", "pen", "plate", "cup")


class E2EGame(ScriptedGame):
    """A scripted game that ``win`` wins, except in a third of the games (by game file),
    with a goal object of its own in the initial observation."""

    def __init__(self, gamefile: Path, seed: int, train_eval: str, **kw: Any) -> None:
        super().__init__(gamefile, seed, train_eval, **kw)
        # The game's own directory names (json_2.1.1/<split>/<task>/<trial>), not where
        # the data root happens to be, so the same games are winnable in every run.
        name = "/".join(Path(gamefile).parts[-4:])
        self.winnable = _bucket(f"e2e-game:{name}", 3) != 0
        self.goal = E2E_OBJECTS[_bucket(f"e2e-goal:{name}", len(E2E_OBJECTS))]

    def reset(self) -> tuple[str, Mapping[str, Any]]:
        observation, info = super().reset()
        return observation.replace("a clean cloth", f"a clean {self.goal}"), info

    def step(self, command: str) -> tuple[str, bool, Mapping[str, Any]]:
        if command == "win" and not self.winnable:
            command = "wait"
        return super().step(command)


def alfworld_env(root: Path) -> alf.AlfWorldEnv:
    def make(path: Path, seed: int, train_eval: str) -> E2EGame:
        return E2EGame(path, seed, train_eval)

    return alf.AlfWorldEnv(data_root=root / "alfworld", backend_factory=make)


# ---------------------------------------------------------------------------
# The fake model
# ---------------------------------------------------------------------------

WORKFLOW = (
    "## Workflow: answer a question about a document\n"
    "1. Locate the part of the document the question names.\n"
    "2. Read the value and keep its unit.\n"
    "3. Answer in the requested format."
)
_ALFWORLD_GOAL = re.compile(r"Your task is to: ([^\n]*)")
#: Text only a learned memory carries: RB's extracted item titles, AWM's induced workflow,
#: MIRIX's curated skill. An empty or stale memory (a method's template around nothing
#: learned) carries none of them.
LEARNED_MARKS = (
    "check the source first",
    "Locate the part of the document the question names",
    MIRIX_LEARNED_MARK,
)


def request_text(body: Mapping[str, Any]) -> str:
    """Every text of a chat request's messages, newline-joined."""
    texts = []
    for message in body.get("messages") or ():
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts += [str(p.get("text") or "") for p in content if isinstance(p, dict)]
    return "\n".join(texts)


def learned_memory(body: Mapping[str, Any]) -> bool:
    """True when an actor request carries what a fake learner produced."""
    text = request_text(body)
    return any(mark in text for mark in LEARNED_MARKS)


#: What the fake J reads as a failed trajectory: a game that never ends ("Nothing
#: happens"), a WebShop purchase whose score is 0.
FAILURE_MARKS = ("Nothing happens", "max 1.0): 0.00")


@dataclass
class E2EModel:
    """FakeUpstream responder for the whole matrix (see the module docstring)."""

    kinds: dict[str, int] = field(default_factory=dict)

    def _count(self, kind: str) -> None:
        self.kinds[kind] = self.kinds.get(kind, 0) + 1

    def __call__(self, endpoint: str, body: dict[str, Any]) -> Reply | None:
        if endpoint != "chat/completions":
            self._count("embedding")
            return None
        system = system_text(body)
        text = first_text(body)
        if system == J.system_prompt():
            self._count("judge")
            # Mostly right (a lost game, a wrong answer), wrong on a fifth of the inputs.
            failed = any(mark in text for mark in FAILURE_MARKS)
            if _bucket(text, 5) == 0:
                failed = not failed
            return text_reply(
                judged("failure" if failed else "success"), model=body["model"]
            )
        if system in (R.system_prompt("success"), R.system_prompt("failure")):
            self._count("rb_learner")
            return text_reply(
                items_reply("check the source first", "keep the answer format"),
                model=body["model"],
            )
        if len(body["messages"]) == 1 and text.startswith(A.instruction_text()):
            self._count("awm_learner")
            return text_reply(WORKFLOW, model=body["model"])
        benchmark = benchmark_of(body)
        self._count(f"actor:{benchmark}")
        if benchmark == "alfworld":
            goal = _ALFWORLD_GOAL.search(request_text(body))
            key = goal.group(1) if goal else ""
            right = _bucket(key, 4) < (3 if learned_memory(body) else 2)
            return text_reply(act("win" if right else "wait"), model=body["model"])
        if benchmark == "webshop":
            return self._webshop(body)
        return None  # MIRIX's own calls and anything else: the fake's default reply

    def _webshop(self, body: Mapping[str, Any]) -> Reply:
        # The conversation's last message is the current page.
        prompt = str(body["messages"][-1]["content"])
        task = prompt.split("Your task is to: ", 1)[1].split(".\n", 1)[0]
        right = _bucket(task, 4) < (3 if learned_memory(body) else 2)
        return text_reply(
            respond(shopper_action(body["messages"], right=right)), model=body["model"]
        )


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------


@dataclass
class Matrix:
    """The runs directory of an offline matrix and what built it."""

    root: Path
    runs: Path
    envs: dict[str, Env]
    outcomes: dict[str, CellOutcome]
    model: E2EModel
    upstream: FakeUpstream
    #: MIRIX's model roles, when the matrix has MIRIX cells.
    mirix: MirixModel | None = None

    def env_factory(self, benchmark: str) -> Env:
        return self.envs[benchmark]

    def cell(self, cell_id: str) -> CellPaths:
        return CellPaths(self.runs / cell_id)


def cell_spec(
    benchmark: str,
    method: str,
    model: str = E2E_MODEL,
    *,
    n_train: int = N_ITEMS,
    n_test: int = N_ITEMS,
) -> CellSpec:
    return CellSpec(benchmark, model, method, "smoke", n_train, n_test)


def build_envs(root: Path) -> tuple[dict[str, Env], E2EModel]:
    envs: dict[str, Env] = {
        "alfworld": alfworld_env(root),
        "webshop": make_shop(root / "webshop").env(),
    }
    return envs, E2EModel()


async def run_cells(
    runs: Path,
    base_url: str,
    envs: Mapping[str, Env],
    specs: Iterable[CellSpec],
    *,
    method_factories: Mapping[str, Any] | None = None,
    mirix_server_factory: Any = None,
) -> dict[str, CellOutcome]:
    """Every cell through ``run.py``'s runner (one after another, as a scheduler of
    concurrency 1 would), attached to the gateway at ``base_url``."""
    outcomes = {}
    for spec in specs:
        env = envs[spec.benchmark]
        runner = CellRunner(
            spec,
            RunOptions(
                out_dir=runs / spec.cell_id,
                gateway_url=base_url,
                gateway_runs_dir=runs,
                preflight=False,  # scripted games and shop
                idle_timeout_s=60.0,
                test_concurrency=2,
            ),
            env_factory=lambda _benchmark, env=env: env,
            method_factory=(method_factories or {}).get(spec.method),
        )
        if mirix_server_factory is not None:
            runner.mirix_server_factory = mirix_server_factory
        outcomes[spec.cell_id] = await runner.run()
    return outcomes


def build_matrix(
    root: Path,
    methods: Sequence[str] = E2E_METHODS,
    *,
    benchmarks: Sequence[str] | None = None,
    n_train: int = N_ITEMS,
    n_test: int = N_ITEMS,
    method_factories: Mapping[str, Any] | None = None,
    mirix_server_factory: Any = None,
    mirix_model: MirixModel | None = None,
) -> Matrix:
    """Run the offline matrix into ``root/runs`` (blocking). With ``"mirix"`` in
    ``methods`` the upstream also plays MIRIX's model (``mirix_model``, default a plain
    :class:`MirixModel`; the matrix's model is its fallback) and run.py starts a MIRIX
    server per MIRIX cell (``mirix_server_factory``, default run.py's)."""
    from evals.skill_bench.tests.fakes import serve_gateway

    envs, model = build_envs(root)
    mirix = None
    if "mirix" in methods:
        mirix = mirix_model if mirix_model is not None else MirixModel()
        if mirix.fallback is None:
            mirix.fallback = model
    fake = FakeUpstream(responder=mirix or model)
    specs = [
        cell_spec(benchmark, method, n_train=n_train, n_test=n_test)
        for benchmark in protocol.BENCHMARKS
        if benchmark in envs and (benchmarks is None or benchmark in benchmarks)
        for method in methods
    ]
    runs = root / "runs"
    with serve_gateway(root, fake, backoff_base_s=protocol.GATEWAY_BACKOFF_BASE_S) as (
        _gw,
        url,
    ):
        outcomes = asyncio.run(
            run_cells(
                runs,
                url,
                envs,
                specs,
                method_factories=method_factories,
                mirix_server_factory=mirix_server_factory,
            )
        )
    return Matrix(root, runs, envs, outcomes, model, fake, mirix)


# ---------------------------------------------------------------------------
# Copies to tamper with
# ---------------------------------------------------------------------------


def copy_cells(matrix: Matrix, dest: Path, cell_ids: Iterable[str]) -> Path:
    """Copy cells, their gateway ledgers and the gateway start records into ``dest/runs``;
    each attempt's recorded ledger path is rewritten to the copy (blobs stay shared)."""
    runs = dest / "runs"
    runs.mkdir(parents=True)
    shutil.copytree(matrix.runs / "_gateway", runs / "_gateway")
    for cell_id in cell_ids:
        shutil.copytree(matrix.runs / cell_id, runs / cell_id)
        paths = CellPaths(runs / cell_id)
        for number in paths.attempt_numbers():
            state = read_json(paths.attempt(number).state)
            run_id = state["run_id"]
            shutil.copytree(matrix.runs / run_id, runs / run_id)
            state["ledger"] = str(runs / run_id / "gateway.jsonl")
            write_json_atomic(paths.attempt(number).state, state)
    return runs


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def trajectory_paths(cell: CellPaths, phase: str, attempt: int = 1) -> list[Path]:
    return sorted(cell.attempt(attempt).trajectories(phase).glob("*.json"))


def edit_json(path: Path, change: Any) -> Any:
    data = json.loads(path.read_text())
    result = change(data)
    path.write_text(json.dumps(data))
    return result
