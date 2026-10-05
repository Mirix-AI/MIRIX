"""The retrieval gate's defaults and the calibration behind them.

The calibration ran against gemini-embedding-001 (through OpenRouter) on a
synthetic set; its data, script and results are committed under
``scripts/calibration/q10/``. These tests are offline:

* the committed results support the code defaults (every relevant skill
  kept, nothing passed for a query no skill matches) and 0.60 did not;
* the calibration set embeds each skill exactly as MIRIX does
  (``skill_retrieval_text``);
* the script, pointed at the offline fake server, fetches every text once in
  batches, reads its key from the environment only, and reproduces its
  results from an embeddings cache with ``--offline`` (no request, no key);
* the key only goes where it belongs: only 429/5xx/network
  failures are retried (a 401 sends the key once), a redirect is never
  followed, and a plain-HTTP endpoint off loopback is refused before any
  request.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from fakes.fake_llm_server import ROUTE_EMBEDDINGS, Fault
from mirix import constants
from mirix.services.skill_history import skill_retrieval_text

CALIBRATION = Path(__file__).resolve().parent.parent / "scripts" / "calibration" / "q10"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "q10_calibrate", CALIBRATION / "q10_calibrate.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def results():
    return json.loads((CALIBRATION / "q10_results.json").read_text())


@pytest.fixture(scope="module")
def data():
    return json.loads((CALIBRATION / "q10_calibration_data.json").read_text())


def _grid_row(results, floor, ratio):
    [row] = [
        r
        for r in results["grid"]
        if math.isclose(r["floor"], floor) and math.isclose(r["ratio"], ratio)
    ]
    return row


class TestTheDefaults:
    def test_the_calibrated_values(self):
        assert constants.SKILL_RETRIEVAL_MIN_SIMILARITY_DEFAULT == 0.62
        assert constants.SKILL_RETRIEVAL_RELATIVE_RATIO_DEFAULT == 0.90

    def test_the_settings_default_to_them(self, monkeypatch):
        for name in (
            "MIRIX_SKILL_RETRIEVAL_MIN_SIMILARITY",
            "MIRIX_SKILL_RETRIEVAL_RELATIVE_RATIO",
        ):
            monkeypatch.delenv(name, raising=False)
        assert constants._cosine_setting(
            "MIRIX_SKILL_RETRIEVAL_MIN_SIMILARITY",
            None,
            constants.SKILL_RETRIEVAL_MIN_SIMILARITY_DEFAULT,
        ) == pytest.approx(0.62)
        assert constants._fraction_setting(
            "MIRIX_SKILL_RETRIEVAL_RELATIVE_RATIO",
            None,
            constants.SKILL_RETRIEVAL_RELATIVE_RATIO_DEFAULT,
        ) == pytest.approx(0.90)

    def test_the_results_support_them(self, results):
        summary = results["summary"]
        floor = constants.SKILL_RETRIEVAL_MIN_SIMILARITY_DEFAULT
        ratio = constants.SKILL_RETRIEVAL_RELATIVE_RATIO_DEFAULT
        assert summary["model"] == "google/gemini-embedding-001"
        assert summary["top1_accuracy"] == 1.0
        # The floor sits between the best unmatched query and the worst
        # relevant skill.
        assert summary["unmatched_max_cos"]["1.0"] < floor
        assert floor < summary["relevant_cos"]["0.0"]
        chosen = _grid_row(results, floor, ratio)
        assert chosen["recall"] == 1.0
        assert chosen["unmatched_queries_items_passed"] == 0
        assert chosen["nonrel_passed_per_query"] == 0.0
        # The old placeholder let a skill through for an unmatched query.
        assert _grid_row(results, 0.60, ratio)["unmatched_queries_items_passed"] > 0
        # The ratio decides nothing between 0.85 and 0.95 at this floor.
        plateau = [_grid_row(results, floor, r) for r in (0.85, 0.88, 0.9, 0.92, 0.95)]
        assert all(
            {k: v for k, v in row.items() if k != "ratio"}
            == {k: v for k, v in chosen.items() if k != "ratio"}
            for row in plateau
        )
        assert summary["nonrelevant_ratio_to_max"]["0.99"] < 0.80

    def test_the_committed_results_cover_the_committed_set(self, results, data):
        summary = results["summary"]
        assert summary["n_skills"] == len(data["skills"]) == 30
        assert summary["n_queries"] == len(data["queries"]) == 60
        assert summary["n_unmatched"] == len(data["unmatched"]) == 16
        assert [q["id"] for q in results["per_query"]] == [
            q["id"] for q in data["queries"]
        ]
        assert all(
            row["top"] == q["rel"]
            for row, q in zip(results["per_query"], data["queries"])
        )


class TestTheSet:
    def test_every_skill_is_embedded_as_mirix_embeds_it(self, data):
        script = _load_script()
        for skill in data["skills"]:
            assert script.retrieval_text(skill) == skill_retrieval_text(skill)

    def test_every_query_has_one_relevant_skill_of_the_set(self, data):
        ids = {s["id"] for s in data["skills"]}
        assert len(ids) == len(data["skills"])
        assert all(q["rel"] in ids for q in data["queries"])
        assert not any("rel" in u for u in data["unmatched"])

    def test_no_benchmark_domain_leaks_in(self, data):
        # The description names the benchmarks it avoids; the set itself
        # must not touch them.
        text = json.dumps({k: v for k, v in data.items() if k != "description"}).lower()
        for word in ("alfworld", "treasury", "docvqa", "officeqa", "countertop"):
            assert word not in text


class TestTheScript:
    def test_fetch_then_reproduce_offline(
        self, fake_llm_server, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
        monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-calibration-fake")
        # Small vectors keep the pure-Python analysis fast; the real run's
        # are 3,072 wide.
        fake_llm_server.embedding_dim = 16
        script = _load_script()
        cache = tmp_path / "vectors.json"
        first = tmp_path / "first.json"
        code = script.main(
            [
                "--endpoint",
                f"{fake_llm_server.base_url}/embeddings",
                "--embeddings-cache",
                str(cache),
                "--out",
                str(first),
            ]
        )
        assert code == 0
        requests = fake_llm_server.embedding_requests()
        # 30 skills + 60 queries + 16 unmatched = 106 texts, 32 per request.
        assert [len(r.json["input"]) for r in requests] == [32, 32, 32, 10]
        assert all(r.json["model"] == "google/gemini-embedding-001" for r in requests)
        assert all(
            r.headers.get("authorization") == "Bearer sk-or-calibration-fake"
            for r in requests
        )
        out = capsys.readouterr().out
        assert "sk-or-calibration-fake" not in out
        assert "sk-or-calibration-fake" not in cache.read_text()
        produced = json.loads(first.read_text())
        assert set(produced) == {"summary", "per_query", "grid"}
        assert len(produced["grid"]) == 16 * 6

        # Offline, from the cache: same results, no request, no key needed.
        monkeypatch.delenv("OPENROUTER_API_KEY")
        second = tmp_path / "second.json"
        code = script.main(
            ["--offline", "--embeddings-cache", str(cache), "--out", str(second)]
        )
        assert code == 0
        assert len(fake_llm_server.embedding_requests()) == 4
        again = json.loads(second.read_text())
        produced["summary"].pop("usage")
        again["summary"].pop("usage")
        assert again == produced

    def test_no_key_means_no_request(self, fake_llm_server, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        script = _load_script()
        out = tmp_path / "out.json"
        code = script.main(
            ["--endpoint", f"{fake_llm_server.base_url}/embeddings", "--out", str(out)]
        )
        assert code == 2
        assert fake_llm_server.embedding_requests() == []
        assert not out.exists()

    def test_offline_without_the_vectors_fails(self, tmp_path):
        script = _load_script()
        out = tmp_path / "out.json"
        assert script.main(["--offline", "--out", str(out)]) == 2
        assert not out.exists()

    def test_the_gate_arithmetic(self):
        script = _load_script()
        data = {
            "skills": [{"id": "a"}, {"id": "b"}],
            "queries": [{"id": "q", "rel": "a"}],
            "unmatched": [{"id": "u"}],
        }
        vectors = {
            "a": [1.0, 0.0],
            "b": [0.0, 1.0],
            "q": [0.8, 0.6],  # cos a = 0.8, cos b = 0.6
            "u": [0.6, -0.8],  # cos a = 0.6, cos b = -0.8
        }
        table = script.score_table(data, vectors)
        assert table["q"] == pytest.approx({"a": 0.8, "b": 0.6})
        row = script.gate_eval(data, table, 0.62, 0.7)
        assert row["recall"] == 1.0 and row["nonrel_passed_per_query"] == 0.0
        assert row["unmatched_queries_items_passed"] == 0
        row = script.gate_eval(data, table, 0.55, 0.7)
        assert row["nonrel_passed_per_query"] == 1.0  # b: 0.6 >= 0.55 and 0.56
        assert row["unmatched_queries_items_passed"] == 1  # u's a: 0.6
        assert script.percentile([1, 2, 3, 4], 0.5) == pytest.approx(2.5)


class TestTheKeyGoesOnlyWhereItBelongs:
    """The real key is sent once per request that can succeed, never
    after a redirect, never in clear text off this machine."""

    TEXTS = {"a": "alpha", "b": "beta"}

    @pytest.fixture
    def fetch(self, fake_llm_server, monkeypatch):
        monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
        monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
        fake_llm_server.embedding_dim = 4
        script = _load_script()
        slept = []

        def fetch():
            return script.fetch_embeddings(
                self.TEXTS,
                list(self.TEXTS),
                api_key="sk-or-i28-fake",
                endpoint=f"{fake_llm_server.base_url}/embeddings",
                sleep=slept.append,
            )

        fetch.slept = slept
        return fetch

    @pytest.mark.parametrize("status", [400, 401, 403, 404])
    def test_a_client_error_is_not_retried(self, fetch, fake_llm_server, status):
        fake_llm_server.inject_fault(
            Fault.http_error(status), route=ROUTE_EMBEDDINGS, times=4
        )
        with pytest.raises(SystemExit, match=f"{status} \\(not retried\\)"):
            fetch()
        assert len(fake_llm_server.embedding_requests()) == 1
        assert fetch.slept == []

    @pytest.mark.parametrize("fault", [Fault.rate_limit(), Fault.http_error(503)])
    def test_a_transient_failure_is_retried(self, fetch, fake_llm_server, fault):
        fake_llm_server.inject_fault(fault, route=ROUTE_EMBEDDINGS, times=2)
        vectors, _usage = fetch()
        assert set(vectors) == {"a", "b"}
        assert len(fake_llm_server.embedding_requests()) == 3
        assert fetch.slept == [1, 2]

    def test_a_dropped_connection_is_retried(self, fetch, fake_llm_server):
        fake_llm_server.inject_fault(
            Fault(disconnect=True), route=ROUTE_EMBEDDINGS, times=1
        )
        vectors, _usage = fetch()
        assert set(vectors) == {"a", "b"}
        assert fetch.slept == [1]

    def test_four_transient_failures_give_up_without_writing(
        self, fetch, fake_llm_server
    ):
        fake_llm_server.inject_fault(
            Fault.http_error(502), route=ROUTE_EMBEDDINGS, times=4
        )
        with pytest.raises(SystemExit, match="failed 4 times"):
            fetch()
        assert len(fake_llm_server.embedding_requests()) == 4
        assert fetch.slept == [1, 2, 4]

    @pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
    def test_a_redirect_is_never_followed(self, fetch, fake_llm_server, status):
        """urllib would re-send the POST (as a GET) with the Authorization header,
        to whatever host the Location names."""
        fake_llm_server.inject_fault(
            Fault(
                status=status,
                body={"error": {"message": "moved"}},
                headers={"Location": f"{fake_llm_server.base_url}/embeddings"},
            ),
            route=ROUTE_EMBEDDINGS,
            times=1,
        )
        with pytest.raises(SystemExit, match="not retried"):
            fetch()
        assert len(fake_llm_server.requests) == 1

    @pytest.mark.parametrize(
        "endpoint",
        [
            "http://openrouter.ai/api/v1/embeddings",
            "http://10.0.0.5:8080/embeddings",
            "ftp://127.0.0.1/embeddings",
            "openrouter.ai/api/v1/embeddings",
            "https:///embeddings",
        ],
    )
    def test_an_endpoint_the_key_would_leak_to_is_refused(
        self, fake_llm_server, tmp_path, monkeypatch, capsys, endpoint
    ):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-i28-fake")
        script = _load_script()
        out = tmp_path / "out.json"
        assert script.main(["--endpoint", endpoint, "--out", str(out)]) == 2
        assert "refusing to send the key" in capsys.readouterr().err
        assert fake_llm_server.requests == []
        assert not out.exists()

    def test_allowed_endpoints_and_the_warning(self):
        script = _load_script()
        assert script.check_endpoint(script.ENDPOINT) is None
        for endpoint in (
            "http://127.0.0.1:9/embeddings",
            "http://localhost:9/embeddings",
            "http://[::1]:9/embeddings",
            "https://example.org/v1/embeddings",
        ):
            warning = script.check_endpoint(endpoint)
            assert warning is not None and endpoint in warning
