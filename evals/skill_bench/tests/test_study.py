"""study.py: the methods the study removed (none in this release; the mechanism is tested
with a synthetic one, conftest's ``removed_method``) and the helpers every study tool sets
their leftover cells aside with; the named studies."""

from __future__ import annotations

import argparse
import importlib
import types
from dataclasses import dataclass

import pytest

from evals.skill_bench import protocol, study
from evals.skill_bench.tests.conftest import REMOVED_METHOD

REMOVED = REMOVED_METHOD


def test_this_release_removes_no_method():
    assert dict(study.REMOVED_METHODS) == {}
    assert study.removed_methods_record() == {}
    assert not study.is_removed("dynamic_cheatsheet")
    with pytest.raises(TypeError):
        study.REMOVED_METHODS["x"] = study.Removal("d", "r")  # type: ignore[index]


def test_a_removed_method_has_left_the_protocol_and_the_tree(removed_method):
    for method in study.REMOVED_METHODS:
        assert method not in protocol.METHODS
        assert method not in protocol.protocol_constants()["methods"]
        assert not (protocol.SKILL_BENCH_DIR / "methods" / f"{method}.py").exists()
        assert not (protocol.SKILL_BENCH_DIR / "methods" / "prompts" / method).exists()
        assert not [
            p for p in protocol.fingerprint_manifest()["files"] if f"/{method}/" in p
        ]


def test_the_study_record_is_in_neither_fingerprint():
    """Recording a removal changes neither the protocol fingerprint nor the harness code
    digest a learning resume compares."""
    assert not protocol.is_fingerprinted("study.py")
    assert "study.py" not in protocol.CODE_FINGERPRINT_FILES
    path = protocol.SKILL_BENCH_DIR / "study.py"
    assert path.parent == protocol.SKILL_BENCH_DIR
    assert path.parent.name not in protocol.CODE_FINGERPRINT_DIRS


def test_the_import_refuses_a_removed_method_still_in_the_protocol(monkeypatch):
    """A method recorded as removed while protocol.METHODS still lists it would be run
    and refused at once: the module refuses to load instead (never an assert). The
    release's mapping is empty, so the reload records ``none`` as removed through the
    mapping type the module builds it with."""
    real = types.MappingProxyType

    def removing_none(mapping):
        if not mapping:  # REMOVED_METHODS; the studies' mapping is never empty
            return real({"none": study.Removal("example removal", "a reason")})
        return real(mapping)

    monkeypatch.setattr(types, "MappingProxyType", removing_none)
    with pytest.raises(RuntimeError, match="still in protocol.METHODS"):
        importlib.reload(study)
    monkeypatch.undo()
    importlib.reload(study)
    assert dict(study.REMOVED_METHODS) == {}


def test_refusals_name_the_decision(removed_method):
    message = study.removed_message(REMOVED)
    assert REMOVED in message and "example removal" in message
    assert "An example reason" in message and "stay there" in message
    assert study.is_removed(REMOVED) and not study.is_removed("awm")
    assert not study.is_removed(None) and not study.is_removed(["x"])
    study.refuse_removed(None)
    study.refuse_removed(["none", "awm"])
    with pytest.raises(ValueError, match="example removal"):
        study.refuse_removed(["none", REMOVED])
    assert study.method_argument("awm") == "awm"
    assert study.method_argument("exprag") == "exprag"  # the option's choices decide
    with pytest.raises(argparse.ArgumentTypeError, match="removed from the study"):
        study.method_argument(REMOVED)


def test_cell_ids_are_read_for_their_method():
    assert study.cell_id_method(f"webshop-luna56-{REMOVED}") == REMOVED
    assert study.cell_id_method(f"alfworld-luna-{REMOVED}-smoke-l5-t2") == REMOVED
    assert study.cell_id_method("alfworld-luna-none") == "none"
    # luna56: the key whose prefix is another key is read whole.
    assert study.cell_id_method("alfworld-luna56-mirix") == "mirix"
    assert study.cell_id_method("webshop-luna56-none-pilot") == "none"
    assert study.cell_id_method("alfworld-luna56-awm-smoke-l5-t2") == "awm"
    for bad in (
        "alfworld-luna5-none",
        "alfworld-luna-56-none",
        "alfworld-luna56none",
        "webshop-luna56",
        "webarena-luna-awm",
        "webshop-gpt-awm",
        f"webshop-luna56-{REMOVED}-a01-0a8507e4",  # a gateway run id, not a cell id
        "_schedule",
        f"webshop-luna56-{REMOVED}\n",  # ``$`` would match before a trailing newline
    ):
        assert study.cell_id_method(bad) is None, bad


def test_cells_split_into_the_study_and_the_removed_ones(removed_method):
    @dataclass
    class Cell:
        cell_id: str
        method: str

    cells = [
        Cell("webshop-luna56-awm", "awm"),
        Cell(f"webshop-luna56-{REMOVED}", REMOVED),
        Cell("webshop-luna56-none", "none"),
    ]
    kept, removed = study.split_cells(cells)
    assert [c.cell_id for c in kept] == ["webshop-luna56-awm", "webshop-luna56-none"]
    assert [c.cell_id for c in removed] == [f"webshop-luna56-{REMOVED}"]
    ids = [c.cell_id for c in cells]
    assert study.split_cells(ids, study.cell_id_method) == (
        ["webshop-luna56-awm", "webshop-luna56-none"],
        [f"webshop-luna56-{REMOVED}"],
    )


def test_records_that_carry_only_a_cell_id_are_recognised(removed_method):
    """Some records name a cell by id only."""
    assert study.removed_cell_id(f"webshop-luna-{REMOVED}")
    assert study.removed_cell_id(f"alfworld-luna56-{REMOVED}-smoke-l5-t2")
    for other in ("webshop-luna-awm", f"webshop-luna-{REMOVED}-a01-0a8507e4", None, 3):
        assert not study.removed_cell_id(other), other


def test_the_removed_listing_every_tool_prints(removed_method):
    """Every removed method, with the decision and reason, and its cells found, sorted
    and once each; a method without a cell there is listed too."""
    removal = study.REMOVED_METHODS[REMOVED]
    empty = study.removed_methods_record()
    assert empty == {
        REMOVED: {"decision": removal.decision, "reason": removal.reason, "cells": []}
    }
    listed = study.removed_methods_record(
        [
            (REMOVED, f"webshop-luna-{REMOVED}"),
            ("awm", "webshop-luna-awm"),  # not a removed method: not listed
            (REMOVED, f"alfworld-luna56-{REMOVED}"),
            (REMOVED, f"webshop-luna-{REMOVED}"),  # a second directory of one id
        ]
    )
    assert listed[REMOVED]["cells"] == [
        f"alfworld-luna56-{REMOVED}",
        f"webshop-luna-{REMOVED}",
    ]
    assert set(listed) == {REMOVED}


def test_a_cells_method_is_its_definitions_else_its_ids(removed_method):
    """A removed method's cell whose cell.json lost its method must not go through the
    tools as a study cell (or crash them). The definition's method decides; without one,
    the id's (``run.CellSpec.cell_id`` is built from it)."""
    removed_id = f"webshop-luna56-{REMOVED}"
    assert study.cell_method({"method": "awm"}, removed_id) == "awm"
    assert study.cell_method({"method": REMOVED}, "legacy-cell") == REMOVED
    for definition in ({}, {"method": None}, {"method": ""}, None, "cell"):
        assert study.cell_method(definition, removed_id) == REMOVED, definition
        assert study.cell_method(definition, "webshop-luna56-none") == "none"
        assert study.cell_method(definition, "not-a-cell-id") is None
        assert study.cell_method(definition, None) is None

    @dataclass
    class Loaded:  # as report.CellData: a definition and an id
        cell_id: str
        definition: dict

        @property
        def method(self) -> str:
            return str(self.definition["method"])  # raises without one

    cells = [
        Loaded("webshop-luna56-awm", {"method": "awm"}),
        Loaded(removed_id, {}),
        Loaded("webshop-luna56-none", {}),
    ]
    assert [study.method_of(c) for c in cells] == ["awm", REMOVED, "none"]
    kept, removed = study.split_cells(cells)
    assert [c.cell_id for c in kept] == ["webshop-luna56-awm", "webshop-luna56-none"]
    assert [c.cell_id for c in removed] == [removed_id]


# --- named studies ---------------------------------------------------------------------


def _split_items(benchmark: str, split: str) -> int:
    import json

    path = protocol.SKILL_BENCH_DIR / "envs" / "splits" / benchmark / f"{split}.json"
    return len(json.loads(path.read_text(encoding="utf-8")))


def test_the_phase16_studies():
    assert list(study.STUDIES) == ["phase16-pilot", "phase16-dev", "phase16"]
    pilot, dev, final = (study.STUDIES[n] for n in study.STUDIES)
    for entry in (pilot, dev, final):
        assert entry.decision == "study design"
        assert entry.models == ("luna", "luna56")
    assert pilot.benchmarks == ("webshop",) and pilot.methods == ("none",)
    assert pilot.split_mode == "pilot" and pilot.sizes("webshop") == (None, None)
    assert final.benchmarks == dev.benchmarks == ("alfworld", "webshop")
    assert final.methods == dev.methods == ("mirix", "reasoningbank", "awm", "none")
    assert final.split_mode == "protocol" and final.sizes("alfworld") == (None, None)
    # Development on train and val only: the whole train split to learn on, the whole
    # val split to test on, as the frozen split files hold them.
    assert dev.split_mode == "smoke"
    assert dev.sizes("alfworld") == (39, 18) and dev.sizes("webshop") == (40, 20)
    for benchmark in dev.benchmarks:
        assert dev.sizes(benchmark) == (
            _split_items(benchmark, "train"),
            _split_items(benchmark, "val"),
        )
    assert _split_items("webshop", "pilot") == 50
    with pytest.raises(TypeError):
        study.STUDIES["x"] = final  # type: ignore[index]


def test_the_study_split_modes_are_run_pys():
    from evals.skill_bench import run

    assert study.STUDY_SPLIT_MODES == run.SPLIT_MODES
    assert study.STUDY_PILOT_METHOD == run.PILOT_METHOD


def test_a_study_is_narrowed_never_widened(removed_method):
    final = study.STUDIES["phase16"]
    assert study.narrowed(final) == (
        ("alfworld", "webshop"),
        ("luna", "luna56"),
        ("mirix", "reasoningbank", "awm", "none"),
    )
    assert study.narrowed(
        final,
        benchmarks=["webshop"],
        models=["openai/gpt-5.6-luna"],
        methods=["none", "mirix"],
    ) == (("webshop",), ("luna56",), ("mirix", "none"))
    for bad, flag in (
        (dict(benchmarks=["webarena"]), "--benchmark"),
        (dict(models=["luna5"]), "--model"),
        (dict(methods=["exprag"]), "--method"),
    ):
        with pytest.raises(ValueError, match=f"{flag} .* is outside --study phase16"):
            study.narrowed(final, **bad)
    with pytest.raises(ValueError, match="removed from the study"):
        study.narrowed(final, methods=[REMOVED])
    assert study.study_argument("phase16") == "phase16"
    with pytest.raises(argparse.ArgumentTypeError, match="studies: phase16-pilot"):
        study.study_argument("no-such-study")


def test_a_study_that_names_what_the_protocol_lacks_refuses_to_load(monkeypatch):
    bad = study.Study(
        name="broken",
        decision="x",
        description="x",
        benchmarks=("alfworld",),
        models=("gpt9",),
        methods=("none",),
    )
    pilot_elsewhere = study.Study(
        name="pilot-elsewhere",
        decision="x",
        description="x",
        benchmarks=("alfworld",),
        models=("luna",),
        methods=("none",),
        split_mode="pilot",
    )
    for entry, text in ((bad, "models"), (pilot_elsewhere, "a pilot runs")):
        monkeypatch.setattr(study, "STUDIES", {entry.name: entry})
        with pytest.raises(RuntimeError, match=text):
            study._check_studies()
