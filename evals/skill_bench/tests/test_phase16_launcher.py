"""``tools/phase16.sh`` and its systemd unit template ``infra/skill-bench@.service.in``:
one command per step of the phase16 studies, with the same options in tmux, in the
systemd unit and in the recovery after a reboot, and one budget at a time.

The script runs under the system's bash (3.2 on macOS, 5 on Linux); ``tmux`` is a fake
on ``PATH`` that logs its arguments, and the lock check uses this interpreter."""

from __future__ import annotations

import configparser
import fcntl
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from evals.skill_bench import protocol
from evals.skill_bench.tools import supervise

SCRIPT = protocol.SKILL_BENCH_DIR / "tools" / "phase16.sh"
UNIT_TEMPLATE = protocol.SKILL_BENCH_DIR / "infra" / "skill-bench@.service.in"
GATES = [
    "--min-free-memory-gb",
    "8",
    "--min-free-disk-gb",
    "15",
    "--memory-emergency-gb",
    "3",
]
#: Synthetic USD budgets for r1-r3 and dev (the launcher has no default).
BUDGET = "100"
DEV_BUDGET = "50"
#: The system's bash, found before a test narrows ``PATH``.
BASH = shutil.which("bash") or "bash"


def run(
    tmp_path: Path, *args: str, script: Path = SCRIPT, **env: str
) -> subprocess.CompletedProcess[str]:
    base = {
        "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
        "HOME": str(tmp_path / "home"),
        "P16_CONFIG": str(tmp_path / "no-config.env"),
        "P16_PYTHON": sys.executable,
        "TMUX_LOG": str(tmp_path / "tmux.log"),
    }
    return subprocess.run(
        [BASH, str(script), *args],
        env={**base, **env},
        capture_output=True,
        text=True,
        timeout=60,
    )


def printed(tmp_path: Path, step: str, *extra: str, **env: str) -> list[str]:
    env.setdefault("SKILL_BENCH_ENV_FILE", "/keys/skill-bench.env")
    env.setdefault("P16_DEV_BUDGET", DEV_BUDGET)
    done = run(tmp_path, "print", step, *extra, **env)
    assert done.returncode == 0, done.stderr
    return shlex.split(done.stdout)


def expected(step: str, *, budget: str, port: str, concurrency: str = "4") -> list[str]:
    study = {"dev": "phase16-dev"}.get(step, "phase16")
    argv = [
        sys.executable,
        "-m",
        "evals.skill_bench.tools.supervise",
        "--runs-dir",
        f"evals/skill_bench/runs/phase16-{step}",
        "--study",
        study,
        "--concurrency",
        concurrency,
        "--test-concurrency",
        "4",
        "--budget",
        budget,
        "--balance-floor",
        "10",
        "--env-file",
        "/keys/skill-bench.env",
        "--gateway-port",
        port,
    ]
    if step in ("r2", "r3"):
        argv += ["--budget-start-from", "evals/skill_bench/runs/phase16-r1"]
    return argv + GATES


def test_each_step_gets_its_options(tmp_path):
    assert printed(tmp_path, "dev") == expected("dev", budget=DEV_BUDGET, port="18545")
    for n in (1, 2, 3):
        assert printed(tmp_path, f"r{n}", P16_BUDGET=BUDGET) == expected(
            f"r{n}", budget=BUDGET, port=f"1854{n}"
        )
    argv = printed(tmp_path, "dev", P16_DEV_BUDGET="90")
    assert argv[argv.index("--budget") + 1] == "90"
    # A lower concurrency for a loaded host.
    for step in ("dev", "r1", "r2", "r3"):
        argv = printed(tmp_path, step, P16_BUDGET=BUDGET, P16_CONCURRENCY="3")
        want = "3"
        assert argv[argv.index("--concurrency") + 1] == want, step


def test_the_port_base_moves_every_gateway_port(tmp_path):
    """P16_PORT_BASE (default 18540): rN gets base+N, dev base+5."""
    ports = {
        step: printed(tmp_path, step, P16_BUDGET=BUDGET, P16_PORT_BASE="20000")
        for step in ("dev", "r1", "r2", "r3")
    }
    assert {
        step: argv[argv.index("--gateway-port") + 1] for step, argv in ports.items()
    } == {
        "dev": "20005",
        "r1": "20001",
        "r2": "20002",
        "r3": "20003",
    }
    done = run(tmp_path, "print", "dev", SKILL_BENCH_ENV_FILE="/k", P16_PORT_BASE="x1")
    assert done.returncode == 2 and "P16_PORT_BASE must be a number" in done.stderr


def test_the_options_are_the_supervisors_own(tmp_path):
    """Every step's command line parses with supervise.py's own parser (and the
    scheduler's behind it), so a relaunch never stops on a usage error."""
    key = tmp_path / "key.env"
    key.write_text("OPENROUTER_API_KEY=not-a-key\n")
    runs = tmp_path / "runs"
    for step in ("dev", "r1", "r2", "r3"):
        argv = printed(
            tmp_path,
            step,
            SKILL_BENCH_ENV_FILE=str(key),
            P16_BUDGET=BUDGET,
            P16_RUNS=str(runs),
        )
        options, _args = supervise.parse_args(argv[3:])
        assert options.runs_dir == Path(runs / f"phase16-{step}")
        study = options.scheduler_args[options.scheduler_args.index("--study") + 1]
        assert study == {"dev": "phase16-dev"}.get(
            step, "phase16"
        )


def test_one_off_options_and_a_notify_command_pass_through(tmp_path):
    argv = printed(
        tmp_path,
        "dev",
        "--",
        "--new-attempt-cells",
        "alfworld-luna-mirix-smoke-l39-t18",
        "--new-attempt-note",
        "MIRIX's distiller: it's changed",
        P16_NOTIFY_CMD="notify-send 'skill bench'",
    )
    assert argv[-6:] == [
        "--notify-cmd",
        "notify-send 'skill bench'",
        "--new-attempt-cells",
        "alfworld-luna-mirix-smoke-l39-t18",
        "--new-attempt-note",
        "MIRIX's distiller: it's changed",
    ]


def test_settings_come_from_the_config_file(tmp_path):
    config = tmp_path / "launch.env"
    config.write_text(
        "SKILL_BENCH_ENV_FILE=/keys/skill-bench.env\nP16_BUDGET=80\n"
        "P16_GATES='--min-free-memory-gb 4'\n"
    )
    done = run(tmp_path, "print", "r3", P16_CONFIG=str(config))
    assert done.returncode == 0, done.stderr
    argv = shlex.split(done.stdout)
    assert argv[argv.index("--budget") + 1] == "80"
    assert argv[-2:] == ["--min-free-memory-gb", "4"]


def test_the_default_config_is_launch_env_under_the_config_home(tmp_path):
    """Without P16_CONFIG: ``$XDG_CONFIG_HOME/skill-bench/launch.env``, and
    ``~/.config/skill-bench/launch.env`` without XDG_CONFIG_HOME."""
    env = dict(os.environ)
    env.pop("XDG_CONFIG_HOME", None)
    for home_config, extra in (
        (tmp_path / "home" / ".config", {}),
        (tmp_path / "xdg", {"XDG_CONFIG_HOME": str(tmp_path / "xdg")}),
    ):
        config = home_config / "skill-bench" / "launch.env"
        config.parent.mkdir(parents=True)
        config.write_text("SKILL_BENCH_ENV_FILE=/keys/skill-bench.env\nP16_BUDGET=70\n")
        done = subprocess.run(
            ["bash", str(SCRIPT), "print", "r1"],
            env={
                **{k: v for k, v in env.items() if not k.startswith("P16_")},
                "HOME": str(tmp_path / "home"),
                "P16_PYTHON": sys.executable,
                **extra,
            },
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert done.returncode == 0, done.stderr
        argv = shlex.split(done.stdout)
        assert argv[argv.index("--budget") + 1] == "70"
        config.unlink()


@pytest.mark.parametrize(
    ("args", "env", "message"),
    [
        (("print", "r1"), {"SKILL_BENCH_ENV_FILE": "/k"}, "set P16_BUDGET"),
        (("print", "dev"), {"SKILL_BENCH_ENV_FILE": "/k"}, "set P16_DEV_BUDGET"),
        (("exec", "dev"), {"SKILL_BENCH_ENV_FILE": "/k"}, "set P16_DEV_BUDGET"),
        (("print", "dev"), {"P16_DEV_BUDGET": DEV_BUDGET}, "set SKILL_BENCH_ENV_FILE"),
        (("print", "r4"), {"SKILL_BENCH_ENV_FILE": "/k"}, "unknown step 'r4'"),
        (("print", "dev", "r1"), {"SKILL_BENCH_ENV_FILE": "/k"}, "takes one step"),
        (("up",), {"SKILL_BENCH_ENV_FILE": "/k"}, "name the steps"),
        (("install-unit", "r1"), {}, "install-unit takes no arguments"),
        (("bogus",), {}, "phase16.sh print STEP"),
    ],
)
def test_what_it_refuses(tmp_path, args, env, message):
    done = run(tmp_path, *args, **env)
    assert done.returncode == 2 and message in done.stderr


def test_the_usage_is_the_header_only(tmp_path):
    """An unknown command prints the header (every command and setting), not the code."""
    done = run(tmp_path, "bogus")
    assert done.returncode == 2
    lines = done.stderr.splitlines()
    assert lines[0].startswith("# The supervisors of the phase16 studies' steps")
    assert all(line.startswith("#") for line in lines)
    for word in (
        "install-unit",
        "P16_BENCHMARKS",
        "P16_PORT_BASE",
        "P16_DEV_BUDGET",
        "launch.env",
    ):
        assert word in done.stderr, word
    assert "set -u" not in done.stderr


def fake_tmux(tmp_path: Path, sessions: str = "") -> Path:
    """A ``tmux`` that logs each call; ``has-session`` finds the sessions named."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    tmux = bin_dir / "tmux"
    tmux.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$TMUX_LOG"\n'
        'if [ "$1" = has-session ]; then\n'
        f'  case " {sessions} " in *" ${{3#=}} "*) exit 0 ;; esac\n'
        "  exit 1\n"
        "fi\n"
        "exit 0\n"
    )
    tmux.chmod(0o755)
    return tmp_path / "tmux.log"


def test_up_starts_only_what_no_supervisor_holds(tmp_path):
    """``up``: a runs directory whose supervisor lock is held is left as it is (no
    second supervisor, no tmux call); another gets a fresh session, and a leftover
    session (its supervisor exited: ``exec bash`` keeps the window) is replaced
    instead of tmux refusing a duplicate session."""
    log = fake_tmux(tmp_path, sessions="p16-r3")
    runs = tmp_path / "runs"
    held = runs / "phase16-r1" / "_supervise" / "supervisor.lock"
    held.parent.mkdir(parents=True)
    with open(held, "w") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        done = run(
            tmp_path,
            "up",
            "r1",
            "r2",
            "r3",
            "--",
            "--reset-resume",
            "webshop-luna-awm",
            SKILL_BENCH_ENV_FILE="/k",
            P16_BUDGET=BUDGET,
            P16_RUNS=str(runs),
        )
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == [
        f"r1: a supervisor holds {runs}/phase16-r1; left as it is",
        "r2: started its supervisor in tmux session p16-r2",
        "r3: started its supervisor in tmux session p16-r3",
    ]
    calls = log.read_text().splitlines()
    assert calls[0] == "has-session -t =p16-r2"
    assert calls[1].startswith("new-session -d -s p16-r2 ")
    assert calls[2:4] == ["has-session -t =p16-r3", "kill-session -t =p16-r3"]
    assert calls[4].startswith("new-session -d -s p16-r3 ")
    command = calls[1].split(" ", 4)[4]
    assert command.startswith(f"{SCRIPT} exec r2 -- --reset-resume webshop-luna-awm; ")
    assert command.endswith('; echo "supervise exited with $?"; exec bash')


def read_unit(text: str) -> configparser.ConfigParser:
    unit = configparser.ConfigParser(interpolation=None, strict=False)
    unit.optionxform = str  # type: ignore[assignment,method-assign]
    unit.read_string(text)
    return unit


def test_the_unit_template_runs_the_launcher_and_restarts_on_failure():
    unit = read_unit(UNIT_TEMPLATE.read_text())
    service = unit["Service"]
    assert service["ExecStart"] == (
        "@CHECKOUT@/evals/skill_bench/tools/phase16.sh exec %i"
    )
    assert service["WorkingDirectory"] == "@CHECKOUT@"
    assert service["KillMode"] == "process" and service["KillSignal"] == "SIGTERM"
    assert service["Restart"] == "on-failure"
    prevent = {int(c) for c in service["RestartPreventExitStatus"].split()}
    assert prevent == {
        supervise.SUPERVISE_COMPLETE,
        supervise.SUPERVISE_ATTENTION,
        supervise.SUPERVISE_BUDGET_SPENT,
        supervise.SUPERVISE_BUDGET_INVALID,
        supervise.SUPERVISE_CRASH_LOOP,
        supervise.SUPERVISE_LOCKED,
        supervise.SUPERVISE_STOPPED,
    }
    # An unexpected error of the supervisor itself is restarted.
    assert supervise.SUPERVISE_INTERNAL_ERROR not in prevent
    # The drain (--drain-timeout 180) fits in the stop timeout.
    assert int(service["TimeoutStopSec"]) > 180
    assert unit["Install"]["WantedBy"] == "default.target"
    assert os.access(SCRIPT, os.X_OK)


def checkout_copy(where: Path) -> Path:
    """The launcher and the unit template at their places in a checkout at ``where``;
    the copied launcher."""
    script = where / "evals" / "skill_bench" / "tools" / "phase16.sh"
    script.parent.mkdir(parents=True)
    shutil.copy2(SCRIPT, script)
    template = where / "evals" / "skill_bench" / "infra" / UNIT_TEMPLATE.name
    template.parent.mkdir(parents=True)
    shutil.copy2(UNIT_TEMPLATE, template)
    return script


def test_install_unit_renders_the_template_for_its_own_checkout(tmp_path):
    """``install-unit``: the template with the checkout's path filled in, written to
    ``$XDG_CONFIG_HOME/systemd/user/skill-bench@.service`` (``~/.config`` without it),
    whatever checkout the launcher lives in."""
    checkout = (tmp_path / "checkout").resolve()
    script = checkout_copy(checkout)
    for extra, unit_dir in (
        ({}, tmp_path / "home" / ".config" / "systemd" / "user"),
        (
            {"XDG_CONFIG_HOME": str(tmp_path / "xdg")},
            tmp_path / "xdg" / "systemd" / "user",
        ),
    ):
        done = run(tmp_path, "install-unit", script=script, **extra)
        assert done.returncode == 0, done.stderr
        written = unit_dir / "skill-bench@.service"
        assert done.stdout.splitlines()[0] == (
            f"wrote {written} for the checkout {checkout}; then:"
        )
        assert "systemctl --user daemon-reload" in done.stdout
        text = written.read_text()
        assert "@CHECKOUT@" not in text
        service = read_unit(text)["Service"]
        assert service["WorkingDirectory"] == str(checkout)
        assert service["ExecStart"] == (
            f"{checkout}/evals/skill_bench/tools/phase16.sh exec %i"
        )
        # Everything else is the template's.
        assert text == UNIT_TEMPLATE.read_text().replace("@CHECKOUT@", str(checkout))


def test_install_unit_refuses_a_checkout_path_a_unit_cannot_take(tmp_path):
    script = checkout_copy(tmp_path / "my checkout")
    done = run(tmp_path, "install-unit", script=script)
    assert done.returncode == 2
    assert "the unit's ExecStart cannot take as it is" in done.stderr
    assert not (tmp_path / "home" / ".config" / "systemd").exists()


def fake_python(tmp_path: Path) -> Path:
    """A Python stand-in that logs its arguments (and ``ALFWORLD_DATA``); the lock check
    (``-c``) finds no supervisor."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    python = bin_dir / "fake-python"
    python.write_text(
        "#!/bin/sh\n"
        'args=$(printf "%s " "$*" | tr "\\n" " ")\n'
        'printf "%s|%s\\n" "${ALFWORLD_DATA:-}" "${args% }" >> "$PY_LOG"\n'
        '[ "$1" = -c ] && exit 1\n'
        "exit 0\n"
    )
    python.chmod(0o755)
    return python


def test_exec_runs_the_supervisor_in_the_foreground(tmp_path):
    python = fake_python(tmp_path)
    log = tmp_path / "py.log"
    done = run(
        tmp_path,
        "exec",
        "r1",
        SKILL_BENCH_ENV_FILE="/keys/skill-bench.env",
        P16_BUDGET=BUDGET,
        P16_PYTHON=str(python),
        PY_LOG=str(log),
    )
    assert done.returncode == 0, done.stderr
    check, line = log.read_text().splitlines()
    assert check.split("|", 1)[1].split()[0] == "-c"  # the supervisors' locks first
    data, argv = line.split("|", 1)
    assert data == str(tmp_path / "home" / ".cache" / "alfworld")
    assert argv.split() == expected("r1", budget=BUDGET, port="18541")[1:]


def test_recover_brings_back_postgres_and_webshop_then_the_supervisors(tmp_path):
    """After a reboot: the compose Postgres (``up -d``), the WebShop server (``--start``
    keeps a running one and replaces the one a reboot left exited), then each named
    step's supervisor, in that order."""
    tmux_log = fake_tmux(tmp_path)
    python = fake_python(tmp_path)
    py_log = tmp_path / "py.log"
    docker_log = tmp_path / "docker.log"
    docker = tmp_path / "bin" / "docker"
    docker.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$DOCKER_LOG"\nexit 0\n')
    docker.chmod(0o755)
    done = run(
        tmp_path,
        "recover",
        "r1",
        "r2",
        SKILL_BENCH_ENV_FILE="/k",
        P16_BUDGET=BUDGET,
        P16_PYTHON=str(python),
        PY_LOG=str(py_log),
        DOCKER_LOG=str(docker_log),
    )
    assert done.returncode == 0, done.stderr
    assert docker_log.read_text().splitlines() == [
        "compose -f evals/skill_bench/infra/compose.yaml up -d"
    ]
    calls = [line.split("|", 1)[1] for line in py_log.read_text().splitlines()]
    assert calls[0] == "-m evals.skill_bench.envs.webshop --start --wait-s 1800"
    assert [c.split()[0] for c in calls[1:]] == ["-c"]  # one lock check of every step
    assert [line.split()[0] for line in tmux_log.read_text().splitlines()] == [
        "has-session",
        "new-session",
        "has-session",
        "new-session",
    ]


def hold(runs: Path, step: str) -> Any:
    """Hold STEP's supervisor lock as a live supervisor does (close the handle to let
    it go)."""
    lock = runs / f"phase16-{step}" / "_supervise" / "supervisor.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock, "w")  # noqa: SIM115 - held across the caller's launches
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def test_one_budget_at_a_time(tmp_path):
    """Every step's budget counts the key's whole spend since its start, so a step
    whose budget is not the running one's is refused, by ``up`` and by ``exec`` (the
    systemd unit's command; exit 2, which the unit does not restart), before any tmux
    or supervisor call; so is one ``up`` naming two budgets. The replicates share one
    budget start: r2 and r3 start beside r1. P16_ALLOW_OVERLAP=1 starts it anyway."""
    log = fake_tmux(tmp_path)
    runs = tmp_path / "runs"
    env = {
        "SKILL_BENCH_ENV_FILE": "/k",
        "P16_BUDGET": BUDGET,
        "P16_DEV_BUDGET": DEV_BUDGET,
        "P16_RUNS": str(runs),
    }
    dev = hold(runs, "dev")
    try:
        for args in (("up", "r1", "r2", "r3"), ("exec", "r1")):
            done = run(tmp_path, *args, **env)
            assert done.returncode == 2, (args, done.stdout, done.stderr)
            step = args[1]
            assert done.stderr.startswith(
                f"phase16.sh: {step}: the dev supervisor runs ({runs}/phase16-dev), "
            ), done.stderr
            assert "P16_ALLOW_OVERLAP=1" in done.stderr
        assert not log.exists()  # nothing started
        # Its own step is left as it is (the supervisor holds it).
        done = run(tmp_path, "up", "dev", **env)
        assert done.returncode == 0, done.stderr
        assert (
            done.stdout
            == f"dev: a supervisor holds {runs}/phase16-dev; left as it is\n"
        )
        # Knowingly overlapping.
        done = run(tmp_path, "up", "r1", P16_ALLOW_OVERLAP="1", **env)
        assert done.returncode == 0, done.stderr
        assert (
            done.stdout == "r1: started its supervisor in tmux session p16-r1\n"
        )
    finally:
        dev.close()
    log.unlink()
    r1 = hold(runs, "r1")
    try:
        done = run(tmp_path, "up", "r1", "r2", "r3", **env)
        assert done.returncode == 0, done.stderr
        assert done.stdout.splitlines()[1:] == [
            "r2: started its supervisor in tmux session p16-r2",
            "r3: started its supervisor in tmux session p16-r3",
        ]
        done = run(tmp_path, "up", "dev", **env)
        assert done.returncode == 2 and "the r1 supervisor runs" in done.stderr
    finally:
        r1.close()
    # Nothing runs: still one budget per launch.
    log.unlink()
    done = run(tmp_path, "up", "r1", "dev", **env)
    assert done.returncode == 2
    assert "r1 and dev have budgets of their own" in done.stderr
    assert not log.exists()
    done = run(tmp_path, "up", "dev", **env)
    assert done.returncode == 0, done.stderr


def test_p16_methods_selects_the_replicates_methods_on_every_path(tmp_path):
    """r1-r3 take their method selection from P16_METHODS (the launch config or the
    environment) on every launch path, the systemd unit's ``exec`` included; the
    development pass keep their study's methods."""
    key = tmp_path / "key.env"
    key.write_text("OPENROUTER_API_KEY=not-a-key\n")
    for n in (1, 2, 3):
        argv = printed(
            tmp_path, f"r{n}", P16_BUDGET=BUDGET, P16_METHODS="none reasoningbank awm"
        )
        at = argv.index("--method")
        assert argv[at - 2 : at + 4] == [
            "--study",
            "phase16",
            "--method",
            "none",
            "reasoningbank",
            "awm",
        ]
        argv = printed(
            tmp_path,
            f"r{n}",
            P16_BUDGET=BUDGET,
            P16_METHODS="mirix",
            SKILL_BENCH_ENV_FILE=str(key),
            P16_RUNS=str(tmp_path / "runs"),
        )
        options, _ = supervise.parse_args(argv[3:])
        args = list(options.scheduler_args)
        assert args[args.index("--method") + 1 : args.index("--method") + 2] == [
            "mirix"
        ]
    for step in ("dev",):
        assert "--method" not in printed(
            tmp_path, step, P16_BUDGET=BUDGET, P16_METHODS="mirix"
        )
    config = tmp_path / "launch.env"
    config.write_text(
        "SKILL_BENCH_ENV_FILE=/k\nP16_BUDGET=80\nP16_METHODS='none awm'\n"
    )
    done = run(tmp_path, "print", "r2", P16_CONFIG=str(config))
    argv = shlex.split(done.stdout)
    assert argv[argv.index("--method") + 1 : argv.index("--method") + 3] == [
        "none",
        "awm",
    ]
    # exec (the unit) starts the supervisor with the same selection.
    python = fake_python(tmp_path)
    log = tmp_path / "py.log"
    done = run(
        tmp_path,
        "exec",
        "r3",
        SKILL_BENCH_ENV_FILE="/k",
        P16_BUDGET=BUDGET,
        P16_METHODS="mirix",
        P16_PYTHON=str(python),
        PY_LOG=str(log),
    )
    assert done.returncode == 0, done.stderr
    supervisor = log.read_text().splitlines()[-1].split("|", 1)[1]
    assert "--study phase16 --method mirix --concurrency" in supervisor


def test_p16_benchmarks_narrows_the_replicates(tmp_path):
    """r1-r3 take their benchmarks from P16_BENCHMARKS (empty: the study's) on every
    launch path: the supervisor gets ``--benchmark`` (its own parser and the
    scheduler's accept it). The development pass keep their study's
    benchmarks."""
    key = tmp_path / "key.env"
    key.write_text("OPENROUTER_API_KEY=not-a-key\n")
    runs = tmp_path / "runs"
    env = {
        "SKILL_BENCH_ENV_FILE": str(key),
        "P16_BUDGET": BUDGET,
        "P16_METHODS": "mirix",
        "P16_BENCHMARKS": "alfworld",
        "P16_RUNS": str(runs),
    }
    for n in (1, 2, 3):
        argv = printed(tmp_path, f"r{n}", **env)
        at = argv.index("--method")
        assert argv[at - 2 : at + 4] == [
            "--study",
            "phase16",
            "--method",
            "mirix",
            "--benchmark",
            "alfworld",
        ]
        options, _ = supervise.parse_args(argv[3:])
        args = list(options.scheduler_args)
        assert args[args.index("--benchmark") + 1 :][:1] == ["alfworld"]
    two = printed(tmp_path, "r1", **{**env, "P16_BENCHMARKS": "alfworld webshop"})
    assert two[two.index("--benchmark") + 1 : two.index("--benchmark") + 3] == [
        "alfworld",
        "webshop",
    ]
    for step in ("dev",):
        assert "--benchmark" not in printed(tmp_path, step, **env)
    assert "--benchmark" not in printed(tmp_path, "r1", **{**env, "P16_BENCHMARKS": ""})
    # exec (the unit) and up start the supervisor with the same selection.
    python = fake_python(tmp_path)
    log = tmp_path / "py.log"
    tmux_log = fake_tmux(tmp_path)
    done = run(tmp_path, "exec", "r1", P16_PYTHON=str(python), PY_LOG=str(log), **env)
    assert done.returncode == 0, done.stderr
    supervisor = log.read_text().splitlines()[-1].split("|", 1)[1]
    assert supervisor.split()[:2] == ["-m", "evals.skill_bench.tools.supervise"]
    assert "--method mirix --benchmark alfworld --concurrency" in supervisor
    done = run(tmp_path, "up", "r2", P16_PYTHON=str(python), PY_LOG=str(log), **env)
    assert done.returncode == 0, done.stderr
    assert tmux_log.read_text().splitlines()[-1].startswith("new-session -d -s p16-r2 ")


def test_status_needs_no_budget_or_key_file(tmp_path):
    """``status`` only reads: ``supervise --status`` of each step (default: all five),
    with no budget and no key file set."""
    python = fake_python(tmp_path)
    log = tmp_path / "py.log"
    runs = tmp_path / "runs"
    env = {"P16_PYTHON": str(python), "PY_LOG": str(log), "P16_RUNS": str(runs)}
    for args, steps in (
        ((), ("dev", "r1", "r2", "r3")),
        (("dev", "r2"), ("dev", "r2")),
    ):
        log.unlink(missing_ok=True)
        done = run(tmp_path, "status", *args, **env)
        assert done.returncode == 0, done.stderr
        assert done.stdout.splitlines() == [
            f"== {step} ({runs}/phase16-{step})" for step in steps
        ]
        calls = [line.split("|", 1)[1] for line in log.read_text().splitlines()]
        assert calls == [
            "-m evals.skill_bench.tools.supervise --status --runs-dir "
            f"{runs}/phase16-{step}"
            for step in steps
        ]
    done = run(tmp_path, "status", "r4", **env)
    assert done.returncode == 2 and "unknown step 'r4'" in done.stderr


def test_up_needs_the_budget_of_each_step_it_starts(tmp_path):
    """No USD default: ``up`` refuses a step without its budget before any tmux call. A
    step whose supervisor runs starts nothing, so it needs none."""
    log = fake_tmux(tmp_path)
    runs = tmp_path / "runs"
    env = {"SKILL_BENCH_ENV_FILE": "/k", "P16_RUNS": str(runs)}
    for step, name in (
        ("dev", "P16_DEV_BUDGET"),
        ("r1", "P16_BUDGET"),
    ):
        done = run(tmp_path, "up", step, **env)
        assert done.returncode == 2, (step, done.stdout, done.stderr)
        assert done.stderr.startswith(f"phase16.sh: set {name}, the USD budget for ")
        assert done.stdout == ""
    assert not log.exists()
    dev = hold(runs, "dev")
    try:
        done = run(tmp_path, "up", "dev", **env)
        assert done.returncode == 0, done.stderr
        assert (
            done.stdout
            == f"dev: a supervisor holds {runs}/phase16-dev; left as it is\n"
        )
    finally:
        dev.close()


def test_recover_brings_back_postgres_and_webshop_without_a_budget(tmp_path):
    """``recover`` brings back the compose Postgres and the WebShop server whatever is
    set; starting a step's supervisor then needs its budget, as ``up`` does."""
    tmux_log = fake_tmux(tmp_path)
    python = fake_python(tmp_path)
    py_log = tmp_path / "py.log"
    docker_log = tmp_path / "docker.log"
    docker = tmp_path / "bin" / "docker"
    docker.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$DOCKER_LOG"\nexit 0\n')
    docker.chmod(0o755)
    done = run(
        tmp_path,
        "recover",
        "dev",
        SKILL_BENCH_ENV_FILE="/k",
        P16_PYTHON=str(python),
        P16_RUNS=str(tmp_path / "runs"),
        PY_LOG=str(py_log),
        DOCKER_LOG=str(docker_log),
    )
    assert done.returncode == 2
    assert "phase16.sh: set P16_DEV_BUDGET, the USD budget for dev" in done.stderr
    assert docker_log.read_text().splitlines() == [
        "compose -f evals/skill_bench/infra/compose.yaml up -d"
    ]
    calls = [line.split("|", 1)[1] for line in py_log.read_text().splitlines()]
    assert calls[0] == "-m evals.skill_bench.envs.webshop --start --wait-s 1800"
    assert not tmux_log.exists()


def test_up_and_recover_refuse_without_tmux(tmp_path):
    """Without tmux, ``up`` exits 2 and says what to do instead, before any start and
    without claiming one; ``recover`` refuses before Postgres and WebShop."""
    bin_dir = tmp_path / "bin-without-tmux"
    bin_dir.mkdir()
    for tool in ("dirname", "tr"):
        found = shutil.which(tool)
        assert found, tool
        (bin_dir / tool).symlink_to(found)
    docker_log = tmp_path / "docker.log"
    docker = bin_dir / "docker"
    docker.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$DOCKER_LOG"\nexit 0\n')
    docker.chmod(0o755)
    python = fake_python(tmp_path)
    py_log = tmp_path / "py.log"
    env = {
        "PATH": str(bin_dir),
        "SKILL_BENCH_ENV_FILE": "/k",
        "P16_BUDGET": BUDGET,
        "P16_RUNS": str(tmp_path / "runs"),
        "P16_PYTHON": str(python),
        "PY_LOG": str(py_log),
        "DOCKER_LOG": str(docker_log),
    }
    for command in ("up", "recover"):
        done = run(tmp_path, command, "r1", **env)
        assert done.returncode == 2, (command, done.stdout, done.stderr)
        assert done.stderr.startswith(
            "phase16.sh: tmux is not installed (not on PATH)"
        ), done.stderr
        assert "phase16.sh exec STEP" in done.stderr
        assert "command not found" not in done.stderr
        assert done.stdout == ""
    assert not docker_log.exists()
    assert "envs.webshop" not in (py_log.read_text() if py_log.exists() else "")


def test_up_stops_when_tmux_cannot_start_a_session(tmp_path):
    """A session tmux could not start is an error (exit 2), never reported as started."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    tmux = bin_dir / "tmux"
    tmux.write_text(
        '#!/bin/sh\n[ "$1" = new-session ] && exit 1\n[ "$1" = has-session ] && exit 1\n'
        "exit 0\n"
    )
    tmux.chmod(0o755)
    done = run(
        tmp_path,
        "up",
        "r1",
        "r2",
        SKILL_BENCH_ENV_FILE="/k",
        P16_BUDGET=BUDGET,
        P16_RUNS=str(tmp_path / "runs"),
    )
    assert done.returncode == 2
    assert done.stdout == ""
    assert done.stderr == (
        "phase16.sh: r1: tmux could not start the session p16-r1 (exit 1); its "
        "supervisor is not running\n"
    )
