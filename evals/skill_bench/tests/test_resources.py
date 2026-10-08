"""The resource guard's host readings, admission and emergency test.

The parsers run on fixture outputs of both platforms (macOS ``vm_stat`` with Apple silicon's
16 KiB page header, ``sysctl``, ``vm.swapusage``; Linux ``/proc/meminfo``; ``ps``;
``docker stats``), :func:`resources.sample` on injected readers for both platforms and
once on the real host.
"""

from __future__ import annotations

import os
import sys
from collections import namedtuple
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from evals.skill_bench import protocol
from evals.skill_bench.infra import resources
from evals.skill_bench.infra.processes import PS_TABLE_ARGV
from evals.skill_bench.infra.resources import (
    CELL_MEMORY_GB,
    GIB,
    HostResources,
    admit,
    docker_memory,
    emergency,
    estimate_gb,
    parse_docker_stats,
    parse_meminfo,
    parse_swapusage,
    parse_vm_stat,
    reserved_gb,
    sample,
    vm_stat_available_bytes,
)
from evals.skill_bench.run import CellSpec

VM_STAT = """\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               21283.
Pages active:                            400632.
Pages inactive:                          399141.
Pages speculative:                          766.
Pages throttled:                              0.
Pages wired down:                        261618.
Pages purgeable:                           1701.
"Translation faults":               67617187365.
Pages copy-on-write:                 1287185712.
File-backed pages:                       211549.
Pages stored in compressor:             1635508.
"""
SWAPUSAGE = (
    "vm.swapusage: total = 5120.00M  used = 3832.62M  free = 1287.38M  (encrypted)\n"
)
MEMINFO = """\
MemTotal:       63999488 kB
MemFree:         1024000 kB
MemAvailable:   52428800 kB
Buffers:          204800 kB
SwapTotal:       2097148 kB
SwapFree:        1048574 kB
HugePages_Total:       0
"""
PS = """\
    1     1  18896
  501   501 204800
  502   501 102400
  777   777  51200
  900   900
garbage line
"""
DOCKER = """\
mirix-skill-bench-postgres-1 229.8MiB / 5.785GiB
skill-bench-sandbox-ab12 1.5GiB / 1GiB
tiny 512kB / 1GiB
bytes 2048B / 1GiB
broken line
"""
Usage = namedtuple("Usage", "total used free")


def fake_runner(outputs: Mapping[tuple[str, ...], str | None]) -> Any:
    calls: list[tuple[str, ...]] = []

    def run(argv: Any) -> str | None:
        key = tuple(argv)
        calls.append(key)
        return outputs.get(key)

    run.calls = calls  # type: ignore[attr-defined]
    return run


DARWIN: dict[tuple[str, ...], str | None] = {
    ("sysctl", "-n", "hw.memsize"): "25769803776\n",
    ("vm_stat",): VM_STAT,
    ("sysctl", "-n", "kern.memorystatus_level"): "52\n",
    ("sysctl", "vm.swapusage"): SWAPUSAGE,
    PS_TABLE_ARGV: PS,
}


def reader(**files: str) -> Any:
    def read(path: str) -> str | None:
        return files.get(path)

    return read


def fixed(**kw: Any) -> dict[str, Any]:
    return {
        "disk_usage": lambda _p: Usage(100 * GIB, 40 * GIB, 60 * GIB),
        "loadavg": lambda: (1.234, 2.0, 3.0),
        "cpu_count": lambda: 8,
        **kw,
    }


# --- parsers --------------------------------------------------------------------------------


def test_vm_stat_is_read_with_the_page_size_of_its_header():
    counters = parse_vm_stat(VM_STAT)
    assert counters["page_size"] == 16384
    assert counters["Pages free"] == 21283 and counters["Pages purgeable"] == 1701
    assert counters["Translation faults"] == 67617187365
    expected = (21283 + 399141 + 766 + 1701) * 16384
    assert vm_stat_available_bytes(VM_STAT) == expected
    # An Intel Mac's 4 KiB page: the header decides, never an assumed size.
    intel = VM_STAT.replace("16384", "4096")
    assert vm_stat_available_bytes(intel) == expected // 4
    assert vm_stat_available_bytes("Pages free: 12.\n") is None  # no page size
    assert vm_stat_available_bytes("") is None


def test_swap_and_meminfo_are_read_in_bytes():
    swap = parse_swapusage(SWAPUSAGE)
    assert swap["total"] == 5120 * 1024**2
    assert swap["used"] == int(3832.62 * 1024**2)
    assert parse_swapusage("vm.swapusage: total = 2.00G  used = 0.50G  free = 1.50G")[
        "used"
    ] == int(0.5 * 1024**3)
    assert parse_swapusage("nonsense") == {}
    info = parse_meminfo(MEMINFO)
    assert info["MemTotal"] == 63999488 * 1024
    assert info["MemAvailable"] == 52428800 * 1024
    assert info["HugePages_Total"] == 0  # no unit: a count


def test_docker_stats_give_gb_per_container():
    found = parse_docker_stats(DOCKER)
    assert found == {
        "mirix-skill-bench-postgres-1": round(229.8 * 1024**2 / GIB, 3),
        "skill-bench-sandbox-ab12": 1.5,
        "tiny": round(512 * 1000 / GIB, 3),
        "bytes": round(2048 / GIB, 3),
    }
    assert docker_memory(fake_runner({})) is None  # Docker cannot be asked
    run = fake_runner(
        {
            (
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "{{.Name}} {{.MemUsage}}",
            ): DOCKER
        }
    )
    assert docker_memory(run)["skill-bench-sandbox-ab12"] == 1.5


# --- sampling on both platforms -------------------------------------------------------------


def test_a_macos_reading_from_vm_stat_sysctl_and_ps(tmp_path):
    run = fake_runner(DARWIN)
    reading = sample(
        tmp_path,
        workers={"webshop-luna-mirix": [501, 777], "webshop-luna56-none": [900]},
        system="darwin",
        run=run,
        **fixed(),
    )
    assert reading.system == "darwin" and reading.errors == ()
    assert reading.mem_total_gb == 24.0
    # macOS's own free percentage names the available memory; the vm_stat
    # sum (which leaves the compressor's reclaimable pages out) is kept for information.
    assert reading.mem_available_gb == round(24.0 * 52 / 100, 3)
    assert reading.mem_available_source == "memorystatus_level"
    assert reading.mem_available_vm_stat_gb == round(
        (21283 + 399141 + 766 + 1701) * 16384 / GIB, 3
    )
    assert reading.mem_pressure_level == 52
    assert reading.swap_total_gb == 5.0
    assert reading.swap_used_gb == round(3832.62 * 1024**2 / GIB, 3)
    assert (reading.disk_free_gb, reading.disk_total_gb) == (60.0, 100.0)
    assert (reading.load1, reading.load5, reading.load15) == (1.23, 2.0, 3.0)
    assert reading.cpus == 8
    # Group 501 (two members) + group 777; group 900 has no rss column: nothing.
    assert reading.workers_rss_gb == {
        "webshop-luna-mirix": round((204800 + 102400 + 51200) * 1024 / GIB, 3),
        "webshop-luna56-none": 0.0,
    }
    assert sum(1 for c in run.calls if c == PS_TABLE_ARGV) == 1  # one ps per sample
    payload = reading.to_dict()
    assert payload["load"] == [1.23, 2.0, 3.0]
    assert payload["mem_available_source"] == "memorystatus_level"
    # Without the level, the vm_stat sum is the available memory.
    no_level = sample(
        tmp_path,
        system="darwin",
        run=fake_runner({**DARWIN, ("sysctl", "-n", "kern.memorystatus_level"): None}),
        **fixed(),
    )
    assert no_level.mem_available_gb == reading.mem_available_vm_stat_gb
    assert no_level.mem_available_source == "vm_stat"
    assert "sysctl kern.memorystatus_level" in no_level.errors
    assert (
        payload["workers_rss_total_gb"] == reading.workers_rss_gb["webshop-luna-mirix"]
    )


def test_a_linux_reading_from_proc_meminfo(tmp_path):
    run = fake_runner({PS_TABLE_ARGV: PS})
    reading = sample(
        tmp_path,
        workers={"alfworld-luna56-awm": [501]},
        system="linux",
        run=run,
        read_text=reader(**{"/proc/meminfo": MEMINFO}),
        **fixed(),
    )
    assert reading.errors == ()
    assert reading.mem_total_gb == round(63999488 * 1024 / GIB, 3)
    assert reading.mem_available_gb == 50.0
    assert reading.mem_available_source == "meminfo"
    assert reading.swap_used_gb == round((2097148 - 1048574) * 1024 / GIB, 3)
    assert reading.mem_pressure_level is None  # macOS only
    assert reading.workers_rss_gb == {
        "alfworld-luna56-awm": round(307200 * 1024 / GIB, 3)
    }
    assert ("vm_stat",) not in run.calls


def test_a_source_that_cannot_be_read_is_named_not_raised(tmp_path):
    def broken_disk(_path: Path) -> Any:
        raise OSError("gone")

    reading = sample(
        tmp_path / "missing" / "runs",
        workers={"c": [1]},
        system="darwin",
        run=fake_runner({}),
        disk_usage=broken_disk,
        loadavg=lambda: (_ for _ in ()).throw(OSError("no loadavg")),
        cpu_count=lambda: None,
    )
    assert reading.mem_available_gb is None and reading.mem_total_gb is None
    assert set(reading.errors) >= {
        "sysctl hw.memsize",
        "vm_stat",
        "sysctl kern.memorystatus_level",
        "sysctl vm.swapusage",
        "disk_usage",
        "loadavg",
        "ps",
    }
    assert reading.workers_rss_gb == {}
    linux = sample(
        tmp_path, system="linux", read_text=reader(), run=fake_runner({}), **fixed()
    )
    assert "/proc/meminfo" in linux.errors and linux.mem_available_gb is None
    other = sample(tmp_path, system="sunos5", run=fake_runner({}), **fixed())
    assert other.errors == ("memory: unsupported platform sunos5",)


@pytest.mark.skipif(
    not (sys.platform == "darwin" or sys.platform.startswith("linux")),
    reason="macOS and Linux only",
)
def test_the_real_host_reads_sane_numbers(tmp_path):
    reading = sample(tmp_path, workers={"self": [os.getpgid(os.getpid())]})
    assert reading.errors == ()
    assert reading.mem_total_gb and reading.mem_total_gb > 0.5
    assert reading.mem_available_gb is not None
    assert 0 < reading.mem_available_gb <= reading.mem_total_gb
    assert reading.disk_free_gb is not None and reading.disk_total_gb is not None
    assert 0 < reading.disk_free_gb <= reading.disk_total_gb
    assert reading.cpus and reading.cpus >= 1
    assert reading.swap_total_gb is not None and reading.swap_used_gb is not None
    assert reading.workers_rss_gb["self"] > 0.01  # this pytest process at least
    if sys.platform == "darwin":
        assert reading.mem_pressure_level is not None
        assert 0 <= reading.mem_pressure_level <= 100
        assert reading.mem_available_source == "memorystatus_level"
        assert reading.mem_available_vm_stat_gb is not None


# --- admission and emergency ------------------------------------------------------------------


def reading(mem: float | None, disk: float | None = 50.0, **kw: Any) -> HostResources:
    return HostResources(
        at="now", system="darwin", mem_available_gb=mem, disk_free_gb=disk, **kw
    )


def test_every_matrix_cell_has_a_memory_estimate():
    assert set(CELL_MEMORY_GB) == set(protocol.BENCHMARKS)
    for benchmark in protocol.BENCHMARKS:
        assert set(CELL_MEMORY_GB[benchmark]) == set(protocol.METHODS), benchmark
    # The design's table.
    assert CELL_MEMORY_GB["alfworld"] == {
        "none": 0.5,
        "reasoningbank": 0.6,
        "awm": 0.6,
        "mirix": 1.4,
    }
    # WebShop's cells are HTTP clients of one shared server (its 24 GB are the server
    # container's own cap, not a cell's).
    assert CELL_MEMORY_GB["webshop"] == {
        "none": 0.35,
        "reasoningbank": 0.45,
        "awm": 0.5,
        "mirix": 1.4,
    }
    assert estimate_gb("webarena", "mirix") == resources.DEFAULT_CELL_MEMORY_GB == 1.4


def test_admission_needs_the_floor_after_the_estimate_and_the_reserve():
    cell = CellSpec("webshop", "luna", "mirix")  # 1.4 GB
    ok, why = admit(reading(6.0), cell, min_free_memory_gb=4)
    assert ok and "1.4 GB estimated" in why
    held, why = admit(reading(5.0), cell, min_free_memory_gb=4)
    assert not held and why.startswith("memory:") and "5.00 GB available" in why
    # A worker launched a moment ago does not show yet: its headroom is reserved.
    held, why = admit(reading(6.0), cell, min_free_memory_gb=4, reserved_gb=1.0)
    assert not held and "1.00 GB reserved" in why
    held, why = admit(reading(20.0, disk=8.0), cell, min_free_disk_gb=10)
    assert not held and why == "disk: 8.0 GB free < 10 GB"
    assert admit(reading(20.0, disk=10.0), cell, min_free_disk_gb=10)[0]
    # No gate: always admitted. An unreadable host never holds a launch back.
    assert admit(reading(0.1, disk=0.1), cell)[0]
    ok, why = admit(
        reading(None, disk=None), cell, min_free_memory_gb=4, min_free_disk_gb=1
    )
    assert ok and why == "disk unknown; memory unknown"


def test_the_reserve_is_what_running_cells_have_not_taken_yet():
    cells = {
        "webshop-luna-mirix": CellSpec("webshop", "luna", "mirix"),  # 1.4
        "webshop-luna56-none": CellSpec("webshop", "luna56", "none"),  # 0.35
    }
    assert reserved_gb(cells, {}) == 1.75
    assert reserved_gb(cells, {"webshop-luna-mirix": 1.0}) == 0.75
    # A cell above its estimate reserves nothing (never negative).
    assert (
        reserved_gb(cells, {"webshop-luna-mirix": 3.0, "webshop-luna56-none": 0.35})
        == 0.0
    )


def test_an_emergency_needs_a_known_reading_below_the_floor():
    assert emergency(reading(1.0), 1.5)
    assert not emergency(reading(1.5), 1.5)
    assert not emergency(reading(None), 1.5)
    assert not emergency(reading(0.1), None)
