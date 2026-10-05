"""Host resources for the scheduler's resource guard: keep the host's memory healthy.

Standard library and system binaries only, so the scheduler and the supervisor can sample
on any host the harness runs on:

* **macOS:** ``sysctl -n hw.memsize`` (total), ``sysctl -n kern.memorystatus_level``
  (the "free percentage" ``memory_pressure`` prints, the kernel's own measure of the
  memory it can hand out before it is under pressure: available = that share of the
  total), ``vm_stat`` (free + inactive + speculative + purgeable pages, times the page
  size its header names: 16 KiB on Apple silicon; kept as ``mem_available_vm_stat_gb``,
  and the available memory when the level cannot be read) and ``sysctl vm.swapusage``.
  The ``vm_stat`` sum leaves out what the compressor can reclaim, so it understates what
  macOS reports free and would admit fewer cells than the host can run.
* **Linux:** ``/proc/meminfo`` (``MemTotal``, ``MemAvailable``, ``SwapTotal`` and
  ``SwapFree``).
* **Both:** free disk under the runs directory (``shutil.disk_usage``), the load averages
  (``os.getloadavg``) and the CPU count, and the resident memory of each cell's process
  groups from one ``ps -A -o pid=,pgid=,rss=`` (``processes.process_group_rss``): its
  ``run.py`` worker (its own session, with its ALFWorld games) and its MIRIX server.

:func:`sample` never raises for a source it cannot read: the field stays None and the
source is named in ``errors``. :func:`admit` is the launch gate (a cell starts only while
the available memory minus its estimated peak, :data:`CELL_MEMORY_GB`, and minus the
headroom still reserved for the workers that have not reached theirs, stays at or above
the floor, and the disk has its minimum free) and :func:`emergency` the scheduler's memory
emergency test. :func:`docker_memory` reads the containers' memory (``docker stats``;
optional, every ten minutes).

Everything here is blocking (subprocesses and file reads): async callers use
``asyncio.to_thread``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from evals.skill_bench.infra.processes import PS_TABLE_ARGV, process_group_rss

GIB = 1024**3

#: Admission estimates of one cell's peak memory (GB) by benchmark and method: the
#: ``run.py`` worker (0.2–0.4 GB), its ALFWorld games, and the MIRIX server (0.4–1.0 GB).
#: The shared gateway and Postgres are not included. From the import footprints plus
#: allowances for batches and sandboxes; ``resources.jsonl`` (per-cell RSS) can
#: recalibrate them.
CELL_MEMORY_GB: Mapping[str, Mapping[str, float]] = {
    "alfworld": {
        "none": 0.5,
        "reasoningbank": 0.6,
        "awm": 0.6,
        "mirix": 1.4,
    },
    # The worker holds only an HTTP client per episode; the
    # shared WebShop server (one container, --memory 24g) is outside every cell's
    # accounting and is started, and ready, before any WebShop cell is admitted
    # (schedule.py's webshop gate). Text-only like ALFWorld without its game processes;
    # an estimate until resources.jsonl has real numbers.
    "webshop": {
        "none": 0.35,
        "reasoningbank": 0.45,
        "awm": 0.5,
        "mirix": 1.4,
    },
}
#: The estimate of a benchmark or method the table does not name (the largest entry).
DEFAULT_CELL_MEMORY_GB = max(v for row in CELL_MEMORY_GB.values() for v in row.values())
#: How long one system binary may take before its source counts as unreadable.
COMMAND_TIMEOUT_S = 10.0
DOCKER_TIMEOUT_S = 30.0

#: ``subprocess`` runner: argv -> stdout, or None when the command failed.
Runner = Callable[[Sequence[str]], "str | None"]


class _Cell(Protocol):
    @property
    def benchmark(self) -> str: ...

    @property
    def method(self) -> str: ...


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_command(
    argv: Sequence[str], timeout_s: float = COMMAND_TIMEOUT_S
) -> str | None:
    """stdout of ``argv`` in the C locale, or None when it cannot run or fails (blocking)."""
    try:
        result = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
            env={**os.environ, "LC_ALL": "C", "LANG": "C"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Parsers (pure; tested on fixtures of both platforms' outputs)
# ---------------------------------------------------------------------------

_VM_PAGE_SIZE = re.compile(r"page size of (\d+) bytes")
_VM_LINE = re.compile(r'^"?([^":]+?)"?:\s+(\d+)\.?\s*$')


def parse_vm_stat(text: str) -> dict[str, int]:
    """``vm_stat``'s counters by name (``"Pages free"`` ...), plus ``page_size`` from its
    header line (``Mach Virtual Memory Statistics: (page size of 16384 bytes)``)."""
    found: dict[str, int] = {}
    for line in text.splitlines():
        header = _VM_PAGE_SIZE.search(line)
        if header:
            found["page_size"] = int(header.group(1))
            continue
        match = _VM_LINE.match(line.strip())
        if match:
            found[match.group(1).strip()] = int(match.group(2))
    return found


#: The ``vm_stat`` pages macOS can hand to a new process without paging anything out.
VM_AVAILABLE_PAGES: tuple[str, ...] = (
    "Pages free",
    "Pages inactive",
    "Pages speculative",
    "Pages purgeable",
)


def vm_stat_available_bytes(text: str) -> int | None:
    """Available memory from ``vm_stat``: (free + inactive + speculative + purgeable) x the
    header's page size; None without a page size or a free count."""
    counters = parse_vm_stat(text)
    page = counters.get("page_size")
    if not page or "Pages free" not in counters:
        return None
    return sum(counters.get(name, 0) for name in VM_AVAILABLE_PAGES) * page


_SWAP_FIELD = re.compile(r"(total|used|free)\s*=\s*([\d.]+)([KMGT]?)", re.IGNORECASE)
_UNIT = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}


def parse_swapusage(text: str) -> dict[str, int]:
    """``sysctl vm.swapusage`` (``total = 5120.00M  used = 3832.62M  free = 1287.38M``) in
    bytes by field."""
    return {
        name.lower(): int(float(value) * _UNIT[unit.upper()])
        for name, value, unit in _SWAP_FIELD.findall(text)
    }


def parse_meminfo(text: str) -> dict[str, int]:
    """``/proc/meminfo`` in bytes by key (its ``kB`` are KiB)."""
    found: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        fields = rest.split()
        if not fields or not fields[0].isdigit():
            continue
        scale = 1024 if len(fields) > 1 and fields[1].lower() == "kb" else 1
        found[key.strip()] = int(fields[0]) * scale
    return found


_SIZE = re.compile(r"([\d.]+)\s*([KMGT]?i?B)", re.IGNORECASE)
_DOCKER_UNIT = {
    "B": 1,
    "KB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "TB": 1000**4,
    "KIB": 1024,
    "MIB": 1024**2,
    "GIB": 1024**3,
    "TIB": 1024**4,
}


def parse_docker_stats(text: str) -> dict[str, float]:
    """``docker stats --no-stream --format '{{.Name}} {{.MemUsage}}'`` -> GB used per
    container (``mirix-skill-bench-postgres-1 229.8MiB / 5.785GiB``)."""
    found: dict[str, float] = {}
    for line in text.splitlines():
        name, _, usage = line.strip().partition(" ")
        match = _SIZE.search(usage.split("/")[0]) if usage else None
        if not name or match is None:
            continue
        unit = _DOCKER_UNIT.get(match.group(2).upper())
        if unit is None:
            continue
        found[name] = round(float(match.group(1)) * unit / GIB, 3)
    return found


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HostResources:
    """One reading of the host (GB are GiB). A field is None when its source could not
    be read; ``errors`` names those sources."""

    at: str
    system: str
    mem_total_gb: float | None = None
    mem_available_gb: float | None = None
    swap_total_gb: float | None = None
    swap_used_gb: float | None = None
    #: macOS ``kern.memorystatus_level``: the share of memory free, in percent.
    mem_pressure_level: int | None = None
    #: macOS: the ``vm_stat`` sum (information; ``mem_available_gb`` is the level's).
    mem_available_vm_stat_gb: float | None = None
    #: Where ``mem_available_gb`` came from: ``memorystatus_level``, ``vm_stat`` or
    #: ``meminfo``.
    mem_available_source: str | None = None
    disk_free_gb: float | None = None
    disk_total_gb: float | None = None
    load1: float | None = None
    load5: float | None = None
    load15: float | None = None
    cpus: int | None = None
    #: Resident memory per cell, summed over its process groups (worker, MIRIX server).
    workers_rss_gb: Mapping[str, float] = field(default_factory=dict)
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": self.at,
            "system": self.system,
            "mem_total_gb": self.mem_total_gb,
            "mem_available_gb": self.mem_available_gb,
            "swap_total_gb": self.swap_total_gb,
            "swap_used_gb": self.swap_used_gb,
            "mem_pressure_level": self.mem_pressure_level,
            "mem_available_vm_stat_gb": self.mem_available_vm_stat_gb,
            "mem_available_source": self.mem_available_source,
            "disk_free_gb": self.disk_free_gb,
            "disk_total_gb": self.disk_total_gb,
            "load": [self.load1, self.load5, self.load15],
            "cpus": self.cpus,
            "workers_rss_gb": dict(sorted(self.workers_rss_gb.items())),
            "workers_rss_total_gb": round(sum(self.workers_rss_gb.values()), 3),
            "errors": list(self.errors),
        }


def _gb(value: float | int | None) -> float | None:
    return None if value is None else round(float(value) / GIB, 3)


def _darwin_memory(run: Runner, errors: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    total = run(["sysctl", "-n", "hw.memsize"])
    if total and total.strip().isdigit():
        out["mem_total_gb"] = _gb(int(total.strip()))
    else:
        errors.append("sysctl hw.memsize")
    vm = run(["vm_stat"])
    available = vm_stat_available_bytes(vm) if vm else None
    if available is None:
        errors.append("vm_stat")
    else:
        out["mem_available_vm_stat_gb"] = _gb(available)
        out["mem_available_gb"] = _gb(available)
        out["mem_available_source"] = "vm_stat"
    level = run(["sysctl", "-n", "kern.memorystatus_level"])
    if level and level.strip().isdigit():
        out["mem_pressure_level"] = int(level.strip())
        if out.get("mem_total_gb") is not None and 0 <= int(level.strip()) <= 100:
            # The kernel's own measure: what it can hand out before it is
            # under pressure, the compressor's reclaimable pages included.
            out["mem_available_gb"] = round(
                out["mem_total_gb"] * int(level.strip()) / 100, 3
            )
            out["mem_available_source"] = "memorystatus_level"
    else:
        errors.append("sysctl kern.memorystatus_level")
    swap = parse_swapusage(run(["sysctl", "vm.swapusage"]) or "")
    if "total" in swap and "used" in swap:
        out["swap_total_gb"] = _gb(swap["total"])
        out["swap_used_gb"] = _gb(swap["used"])
    else:
        errors.append("sysctl vm.swapusage")
    return out


def _linux_memory(
    read_text: Callable[[str], str | None], errors: list[str]
) -> dict[str, Any]:
    info = parse_meminfo(read_text("/proc/meminfo") or "")
    out: dict[str, Any] = {}
    if "MemTotal" in info and "MemAvailable" in info:
        out["mem_total_gb"] = _gb(info["MemTotal"])
        out["mem_available_gb"] = _gb(info["MemAvailable"])
        out["mem_available_source"] = "meminfo"
    else:
        errors.append("/proc/meminfo")
    if "SwapTotal" in info and "SwapFree" in info:
        out["swap_total_gb"] = _gb(info["SwapTotal"])
        out["swap_used_gb"] = _gb(info["SwapTotal"] - info["SwapFree"])
    return out


def sample(
    runs_dir: Path,
    *,
    workers: Mapping[str, Collection[int]] | None = None,
    system: str | None = None,
    run: Runner = run_command,
    read_text: Callable[[str], str | None] = _read_text,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
    loadavg: Callable[[], tuple[float, float, float]] = os.getloadavg,
    cpu_count: Callable[[], int | None] = os.cpu_count,
) -> HostResources:
    """Read the host once (blocking; the readers are injectable for tests).

    ``workers`` maps a cell id to its process group ids (its worker's and its MIRIX
    server's); their resident memory comes from one ``ps`` call.
    """
    system = system or sys.platform
    errors: list[str] = []
    fields: dict[str, Any] = {}
    if system == "darwin":
        fields.update(_darwin_memory(run, errors))
    elif system.startswith("linux"):
        fields.update(_linux_memory(read_text, errors))
    else:
        errors.append(f"memory: unsupported platform {system}")
    try:
        usage = disk_usage(runs_dir if runs_dir.exists() else runs_dir.parent)
        fields["disk_free_gb"] = _gb(usage.free)
        fields["disk_total_gb"] = _gb(usage.total)
    except OSError:
        errors.append("disk_usage")
    try:
        fields["load1"], fields["load5"], fields["load15"] = (
            round(v, 2) for v in loadavg()
        )
    except OSError:
        errors.append("loadavg")
    fields["cpus"] = cpu_count()
    rss: dict[str, float] = {}
    if workers:
        groups = {pgid for pgids in workers.values() for pgid in pgids}
        table = run(PS_TABLE_ARGV)
        if table is None:
            errors.append("ps")
        else:
            by_group = process_group_rss(groups, ps_output=table)
            for cell, pgids in workers.items():
                rss[cell] = round(sum(by_group.get(g, 0) for g in set(pgids)) / GIB, 3)
    return HostResources(
        at=_now(), system=system, workers_rss_gb=rss, errors=tuple(errors), **fields
    )


def docker_memory(run: Runner | None = None) -> dict[str, float] | None:
    """GB of memory each running container uses (``docker stats --no-stream``; blocking),
    None when Docker cannot be asked. Optional: the scheduler reads it every ten minutes
    for the compose Postgres and the WebShop server, and ignores a failure."""
    argv = ["docker", "stats", "--no-stream", "--format", "{{.Name}} {{.MemUsage}}"]
    text = (
        run(argv) if run is not None else run_command(argv, timeout_s=DOCKER_TIMEOUT_S)
    )
    return None if text is None else parse_docker_stats(text)


# ---------------------------------------------------------------------------
# Admission and emergency
# ---------------------------------------------------------------------------


def estimate_gb(benchmark: str, method: str) -> float:
    """The admission estimate of one cell's peak memory (:data:`CELL_MEMORY_GB`)."""
    return CELL_MEMORY_GB.get(benchmark, {}).get(method, DEFAULT_CELL_MEMORY_GB)


def admit(
    reading: HostResources,
    cell: _Cell,
    *,
    min_free_memory_gb: float | None = None,
    min_free_disk_gb: float | None = None,
    reserved_gb: float = 0.0,
) -> tuple[bool, str]:
    """May ``cell`` start now? (admitted, why).

    Admitted while ``mem_available - estimate(cell) - reserved_gb >= min_free_memory_gb``
    and ``disk_free >= min_free_disk_gb``. ``reserved_gb`` is the headroom the running
    workers have not taken yet (their estimates minus their resident memory): a worker
    launched a moment ago does not show in the available memory. A reading that could
    not be taken does not hold a launch back (the reason says so): a gate that never
    opens would stop the matrix silently.
    """
    estimate = estimate_gb(cell.benchmark, cell.method)
    notes: list[str] = []
    if min_free_disk_gb is not None:
        if reading.disk_free_gb is None:
            notes.append("disk unknown")
        elif reading.disk_free_gb < min_free_disk_gb:
            return False, (
                f"disk: {reading.disk_free_gb:.1f} GB free < {min_free_disk_gb:g} GB"
            )
    if min_free_memory_gb is not None:
        if reading.mem_available_gb is None:
            notes.append("memory unknown")
        else:
            left = reading.mem_available_gb - estimate - reserved_gb
            if left < min_free_memory_gb:
                return False, (
                    f"memory: {reading.mem_available_gb:.2f} GB available - "
                    f"{estimate:g} GB ({cell.benchmark}/{cell.method}) - "
                    f"{reserved_gb:.2f} GB reserved = {left:.2f} GB < "
                    f"{min_free_memory_gb:g} GB"
                )
    return True, "; ".join(notes) or f"ok ({estimate:g} GB estimated)"


def reserved_gb(
    cells: Mapping[str, _Cell], workers_rss_gb: Mapping[str, float]
) -> float:
    """Headroom still reserved for running cells: each one's estimate minus the resident
    memory it already has (never below 0)."""
    return round(
        sum(
            max(
                0.0,
                estimate_gb(cell.benchmark, cell.method)
                - workers_rss_gb.get(cell_id, 0.0),
            )
            for cell_id, cell in cells.items()
        ),
        3,
    )


def emergency(reading: HostResources, floor_gb: float | None) -> bool:
    """True when the available memory is below ``floor_gb`` (an unknown reading is not)."""
    return (
        floor_gb is not None
        and reading.mem_available_gb is not None
        and reading.mem_available_gb < floor_gb
    )
