"""The study's composition beyond the frozen protocol: the methods it removed, and the
named cell selections a launch takes (:data:`STUDIES`).

``protocol.METHODS`` names the methods the study compares, and it is part of the frozen
protocol constants (``protocol.protocol_constants()["methods"]``). A method removed from
the study leaves that list, and its code and prompts leave the tree. Its cells already on
disk stay there, with their gateway runs, ledgers and blobs, until someone deletes them;
:data:`REMOVED_METHODS` records why it went (no method in this release), so that every
study tool can recognise such a cell, set it aside, and name the decision instead of
crashing on a method it no longer knows. ``tools/remove_method_data.py`` deletes such
cells: it first writes a spend manifest, ``<runs>/_removed/<method>/<removal_id>/
manifest.json`` (the method, the decision, each cell's spend, calls, tokens and status,
the deleted paths, the host and the time; its calls in ``calls.jsonl`` beside it) and
``<runs>/_removed/index.json``, then deletes the method's cell directories and gateway
runs (never the shared ``_blobs`` or ``_gateway``), so the report and the budget still
attribute that money:

* ``schedule.py``, ``tools/supervise.py`` and ``run.py`` refuse to run a removed method
  (``--method``, :func:`method_argument`; :func:`refuse_removed` for the Python API);
* the report, the audit and the learning summary set its cells aside
  (:func:`split_cells`, by the method a cell's definition names, else its id's,
  :func:`cell_method`; :func:`removed_cell_id` for a record that carries only a cell
  id), list them under ``removed`` with the decision (:func:`removed_methods_record`),
  and neither compare, audit nor summarize them. Their money stays counted: the
  report shows their ledgers' share (or, once deleted, their removal manifests' spend)
  apart from the study's total, and every gateway spend window still sums every ledger
  of its directory.

Removing a method re-freezes the protocol, but that does not make the other cells an
"earlier version" the audit cannot see past: the record keeps the constants per key, and
``methods`` is a composition constant (``protocol.COMPOSITION_CONSTANTS``,
``protocol.unauditable_changes``). Deleting a method's code does not change how any other
cell runs.

**Named studies** (:data:`STUDIES`). A study names the benchmarks, models, methods and
split mode of a batch of cells, so a launch is one word (``schedule.py --study phase16`` /
``tools/supervise.py --study phase16``, and ``tools/preflight.py --study``) instead of four
lists that must be typed alike on every host and replicate. ``--benchmark``, ``--model``
and ``--method`` beside ``--study`` narrow it (a name outside the study is refused); the
split mode and sizes are the study's own. The studies (ALFWorld and WebShop; luna and
luna56; the four methods):

* ``phase16-dev``: development on train and val only (the test split runs once, at the
  end): cells that learn on the whole train split and test on the whole val split
  (:meth:`Study.sizes`: ALFWorld 39 / 18, WebShop 40 / 20), every method, so MIRIX's val
  numbers have the baselines' beside them; 16 cells.
* ``phase16``: the final protocol cells (whole train, then the test split once), 16
  cells, run three times (:attr:`Study.replicates`) in the replicate runs directories
  ``runs/phase16-r1``, ``-r2``, ``-r3`` (identical cell ids).

A study is no protocol constant: it selects cells, it changes nothing a cell runs.

This module is outside both fingerprints (the protocol's and the harness code digest,
``protocol.CODE_FINGERPRINT_DIRS``), so recording a removal or a study changes neither.
It imports only ``protocol``; method code never imports it.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeVar

from evals.skill_bench import protocol


@dataclass(frozen=True)
class Removal:
    """Why a method left the study: the ``decision`` (a short tag naming who decided,
    shown with every refusal and listing) and the ``reason``."""

    decision: str
    reason: str


#: Methods removed from the study, by their former protocol name. This release removes
#: none; the mechanism stays for a study that drops a method later.
REMOVED_METHODS: Mapping[str, Removal] = MappingProxyType({})

_overlap = sorted(set(REMOVED_METHODS) & set(protocol.METHODS))
if _overlap:  # never an assert: python -O must not drop it
    raise RuntimeError(
        f"removed methods {_overlap} are still in protocol.METHODS: a removed method "
        "leaves the protocol's method list (and its code the tree)"
    )


def is_removed(method: object) -> bool:
    """Whether ``method`` names a method the study removed."""
    return isinstance(method, str) and method in REMOVED_METHODS


def removed_message(method: str) -> str:
    """One sentence for a refusal or a listing: the method, the decision and the reason,
    and what happens to its cells."""
    removal = REMOVED_METHODS[method]
    return (
        f"the method {method!r} was removed from the study ({removal.decision}: "
        f"{removal.reason}) Its cells already on disk stay there, but none is run, "
        "audited, compared or summarized."
    )


def refuse_removed(methods: Iterable[str] | None) -> None:
    """Raise ``ValueError`` naming the decision for the first removed method among
    ``methods`` (a whole selection is refused, never silently narrowed)."""
    for method in methods or ():
        if is_removed(method):
            raise ValueError(removed_message(method))


def method_argument(value: str) -> str:
    """``type=`` of a ``--method`` option: a removed method is refused with the
    decision (exit 2 through the parser), not argparse's bare "invalid choice"; any
    other value goes on to the option's ``choices``."""
    if is_removed(value):
        raise argparse.ArgumentTypeError(removed_message(value))
    return value


#: Longest names first (``luna56`` before ``luna``), so no alternative is a prefix tried
#: before the name it begins.
_CELL_ID = re.compile(
    rf"^(?:{'|'.join(map(re.escape, sorted(protocol.BENCHMARKS, key=len, reverse=True)))})-"
    rf"(?:{'|'.join(map(re.escape, sorted(protocol.MODELS, key=len, reverse=True)))})-"
    r"(?P<method>[a-z0-9_]+)(?:-smoke-l\d+-t\d+)?"
)


def cell_id_method(cell_id: str) -> str | None:
    """The method a cell id names (``<benchmark>-<model>-<method>`` with an optional
    ``-smoke-l<n>-t<n>``; ``run.CellSpec.cell_id``), or None when ``cell_id`` is no cell
    id. Reads the id only: a removed method's cells have no ``CellSpec``."""
    # fullmatch: ``$`` also matches before a trailing newline.
    match = _CELL_ID.fullmatch(cell_id)
    return match.group("method") if match else None


def removed_cell_id(cell_id: object) -> bool:
    """Whether ``cell_id`` names a cell of a removed method (by the id alone,
    :func:`cell_id_method`): for records that carry only an id."""
    return isinstance(cell_id, str) and is_removed(cell_id_method(cell_id))


def cell_method(definition: object, cell_id: object) -> str | None:
    """The method of a cell on disk: its definition's (``cell.json`` ``cell``)
    ``method``, else the method its id names (:func:`cell_id_method`), else None.

    ``run.py`` always writes the method, so the id decides only for a definition that
    names none (a removed method's cell whose ``cell.json`` lost it must still be set
    aside, never judged as a study cell). The id is ``run.CellSpec.cell_id``, built from
    the method."""
    method = definition.get("method") if isinstance(definition, Mapping) else None
    if isinstance(method, str) and method:
        return method
    return cell_id_method(cell_id) if isinstance(cell_id, str) else None


def method_of(cell: object) -> object:
    """A loaded cell's method as :func:`split_cells` reads it: :func:`cell_method` of its
    ``definition`` and ``cell_id`` (``report.CellData``); an object without a
    definition, its ``method``."""
    definition = getattr(cell, "definition", None)
    if isinstance(definition, Mapping):
        return cell_method(definition, getattr(cell, "cell_id", None))
    return getattr(cell, "method", None)


def removed_methods_record(
    cells: Iterable[tuple[str, str]] = (),
) -> dict[str, dict[str, object]]:
    """The ``methods`` block of every tool's listing of removed methods (the report, the
    audit and the learning summary; JSON-ready): each removed method, the
    decision and reason, and the ids of its cells among ``cells``
    (``(method, cell_id)`` pairs), sorted. A method with no cell there is listed too:
    the reader learns why it is missing from the study."""
    ids: dict[str, set[str]] = {method: set() for method in REMOVED_METHODS}
    for method, cell_id in cells:
        if method in ids:
            ids[method].add(cell_id)
    return {
        method: {
            "decision": removal.decision,
            "reason": removal.reason,
            "cells": sorted(ids[method]),
        }
        for method, removal in sorted(REMOVED_METHODS.items())
    }


#: The split modes a study may name (``run.SPLIT_MODES``; ``run.py`` imports this module,
#: so the names are repeated here and a test pins the two together).
STUDY_SPLIT_MODES: tuple[str, ...] = ("protocol", "smoke")


@dataclass(frozen=True)
class Study:
    """A named selection of cells (:data:`STUDIES`): benchmarks, model keys and methods
    in ``schedule.py``'s launch order, and the split mode. A smoke study learns on the
    whole train split and tests on the whole val split of each benchmark
    (:meth:`sizes`)."""

    name: str
    decision: str
    description: str
    benchmarks: tuple[str, ...]
    models: tuple[str, ...]
    methods: tuple[str, ...]
    split_mode: str = "protocol"
    #: How many replicate runs directories the study runs each cell in (three for the
    #: final cells); ``tools/aggregate_replicates.py --study`` expects that many,
    #: whatever the number of directories named.
    replicates: int = 1

    def sizes(self, benchmark: str) -> tuple[int | None, int | None]:
        """``(n_train, n_test)`` of the study's cells of ``benchmark``: the whole train
        and val splits for a smoke study (``protocol.SPLIT_SIZES``, which the frozen
        split files hold), ``(None, None)`` otherwise (whole splits)."""
        if self.split_mode != "smoke":
            return None, None
        sizes = protocol.SPLIT_SIZES[benchmark]
        return int(sizes["train"]), int(sizes["val"])


_PHASE16 = "study design"
_PHASE16_BENCHMARKS = ("alfworld", "webshop")
_PHASE16_MODELS = ("luna", "luna56")
_PHASE16_METHODS = ("mirix", "reasoningbank", "awm", "none")

#: The named studies, by name (module docstring).
STUDIES: Mapping[str, Study] = MappingProxyType(
    {
        "phase16-dev": Study(
            name="phase16-dev",
            decision=_PHASE16,
            description="development on train and val only: learn on the whole "
            "train split, test on the whole val split, every method",
            benchmarks=_PHASE16_BENCHMARKS,
            models=_PHASE16_MODELS,
            methods=_PHASE16_METHODS,
            split_mode="smoke",
        ),
        "phase16": Study(
            name="phase16",
            decision=_PHASE16,
            description="the final protocol cells: ALFWorld and WebShop, luna and "
            "luna56, the four methods; three replicate runs directories",
            benchmarks=_PHASE16_BENCHMARKS,
            models=_PHASE16_MODELS,
            methods=_PHASE16_METHODS,
            split_mode="protocol",
            replicates=3,
        ),
    }
)


def _check_studies() -> None:
    """Every study names protocol benchmarks, model keys and methods (never a removed
    one) and a known split mode (an error at import, never at launch)."""
    for name, entry in STUDIES.items():
        problems = []
        if entry.name != name:
            problems.append(f"named {entry.name!r}")
        for what, values, known in (
            ("benchmarks", entry.benchmarks, protocol.BENCHMARKS),
            ("models", entry.models, tuple(protocol.MODELS)),
            ("methods", entry.methods, protocol.METHODS),
        ):
            unknown = sorted(set(values) - set(known))
            if unknown or not values or len(set(values)) != len(values):
                problems.append(f"{what} {list(values)}")
        if entry.split_mode not in STUDY_SPLIT_MODES:
            problems.append(f"split mode {entry.split_mode!r}")
        if isinstance(entry.replicates, bool) or not (
            isinstance(entry.replicates, int) and entry.replicates >= 1
        ):
            problems.append(f"replicates {entry.replicates!r}")
        if problems:
            raise RuntimeError(f"study {name!r}: {'; '.join(problems)}")


_check_studies()


def study_argument(value: str) -> str:
    """``type=`` of a ``--study`` option: a known study name, else the names there are."""
    if value not in STUDIES:
        raise argparse.ArgumentTypeError(
            f"unknown study {value!r}; studies: {', '.join(STUDIES)}"
        )
    return value


def narrowed(
    entry: Study,
    *,
    benchmarks: Iterable[str] | None = None,
    models: Iterable[str] | None = None,
    methods: Iterable[str] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """The study's benchmarks, model keys and methods, each narrowed to the names given
    (None or empty: all of the study's), in the study's order. A name outside the study
    is refused (``ValueError``): ``--study`` never quietly widens or drops a selection.
    ``models`` takes keys or model ids (``openai/gpt-5.6-luna`` is ``luna56``)."""
    by_id = {v: k for k, v in protocol.MODELS.items()}
    out = []
    for what, given, allowed in (
        ("--benchmark", benchmarks, entry.benchmarks),
        ("--model", [by_id.get(m, m) for m in models or ()], entry.models),
        ("--method", methods, entry.methods),
    ):
        picked = list(given or ())
        refuse_removed(picked if what == "--method" else ())
        outside = sorted(set(picked) - set(allowed))
        if outside:
            raise ValueError(
                f"{what} {outside} is outside --study {entry.name} ({list(allowed)})"
            )
        out.append(tuple(v for v in allowed if not picked or v in set(picked)))
    return out[0], out[1], out[2]


T = TypeVar("T")


def split_cells(
    cells: Iterable[T], method: Callable[[T], object] = method_of
) -> tuple[list[T], list[T]]:
    """``(study cells, removed cells)``, each in the given order; ``method`` reads a
    cell's method (default :func:`method_of`: its definition's, else its id's). Read a
    removed cell's method with :func:`method_of` too: ``report.CellData.method`` raises
    for a definition that names none."""
    kept: list[T] = []
    removed: list[T] = []
    for cell in cells:
        (removed if is_removed(method(cell)) else kept).append(cell)
    return kept, removed
