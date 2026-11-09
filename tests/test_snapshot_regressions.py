import json
import subprocess
from unittest.mock import MagicMock

import pytest

from evals import memory_snapshot as snapshot


@pytest.fixture
def stored_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot, "SNAP_ROOT", tmp_path)
    path = tmp_path / "saved"
    path.mkdir()
    (path / "meta.json").write_text(json.dumps({"pg_tables": ["semantic_memory"]}))
    (path / "pg_memory.dump").write_bytes(b"dump")
    (path / "neo4j_graph.json").write_text('{"nodes": [], "relationships": []}')
    return path


def test_restore_failure_never_reports_success(stored_snapshot, monkeypatch, capsys):
    graph_import = MagicMock()
    monkeypatch.setattr(snapshot, "_neo4j_import", graph_import)

    def run(args, **kwargs):
        if "pg_restore" in str(args[0]):
            return subprocess.CompletedProcess(args, 1, "", "invalid archive")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(snapshot.subprocess, "run", run)
    with pytest.raises((SystemExit, RuntimeError, subprocess.CalledProcessError)):
        snapshot.load("saved")
    assert "[snapshot] loaded" not in capsys.readouterr().out
    graph_import.assert_not_called()


def test_missing_graph_file_fails_before_database_writes(stored_snapshot, monkeypatch):
    (stored_snapshot / "neo4j_graph.json").unlink()
    run = MagicMock(return_value=subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(snapshot.subprocess, "run", run)
    with pytest.raises((FileNotFoundError, ValueError, SystemExit)):
        snapshot.load("saved")
    run.assert_not_called()


def test_empty_graph_still_clears_previous_graph(monkeypatch):
    neo4j = pytest.importorskip("neo4j")

    transaction = MagicMock()
    session = MagicMock()
    session.__enter__.return_value = session
    session.execute_write.side_effect = lambda operation: operation(transaction)
    driver = MagicMock()
    driver.session.return_value = session
    monkeypatch.setattr(neo4j.GraphDatabase, "driver", lambda *a, **kw: driver)
    snapshot._neo4j_import({"nodes": [], "relationships": []})
    queries = [call.args[0] for call in transaction.run.call_args_list + session.run.call_args_list]
    assert any("DETACH DELETE" in query for query in queries)


def test_export_failure_is_not_an_empty_success(monkeypatch):
    neo4j = pytest.importorskip("neo4j")

    monkeypatch.setattr(
        neo4j.GraphDatabase, "driver",
        MagicMock(side_effect=RuntimeError("database unavailable")),
    )
    with pytest.raises((RuntimeError, SystemExit)):
        snapshot._neo4j_export()


def test_table_probe_failure_is_not_an_absent_table(monkeypatch):
    monkeypatch.setattr(
        snapshot.subprocess, "run",
        lambda *a, **kw: subprocess.CompletedProcess([], 1, "", "connection refused"),
    )
    with pytest.raises((RuntimeError, SystemExit, subprocess.CalledProcessError)):
        snapshot._table_exists("semantic_memory")


def test_snapshot_paths_cannot_escape_root(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot, "SNAP_ROOT", tmp_path / "snapshots")
    victim = tmp_path / "valuable"
    victim.mkdir()
    (victim / "keep").write_text("keep")
    with pytest.raises((ValueError, SystemExit)):
        snapshot.delete("../valuable")
    assert (victim / "keep").exists()


def test_temporal_graph_properties_round_trip_without_type_or_precision_loss():
    import datetime as dt
    pytest.importorskip("neo4j")
    from neo4j.time import Date, DateTime, Duration, Time
    from neo4j.spatial import WGS84Point

    properties = {
        "date": Date(2026, 11, 9),
        "instant": DateTime(2026, 11, 9, 12, 30, 1, 123456789, tzinfo=dt.timezone.utc),
        "time": Time(12, 30, 1, 123456789),
        "duration": Duration(months=2, days=3, seconds=4, nanoseconds=5),
        "location": WGS84Point((1.5, 2.5)),
        "python_datetime": dt.datetime(2026, 11, 9, tzinfo=dt.timezone.utc),
        "python_date": dt.date(2026, 11, 9),
        "python_time": dt.time(12, 30),
        "python_delta": dt.timedelta(days=2, seconds=3, microseconds=4),
        "literal_string": "2026-11-09T12:30:00+00:00",
    }
    encoded = json.dumps(snapshot._encode_snapshot(properties))
    restored = json.loads(encoded, object_hook=snapshot._json_object_hook)
    assert restored == properties
    for key, original in properties.items():
        assert type(restored[key]) is type(original), key


def test_restore_sql_errors_use_one_transaction(stored_snapshot, monkeypatch, capsys):
    from pathlib import Path

    commands = []
    def run(args, **kwargs):
        commands.append(args)
        if "pg_restore" in str(args[0]):
            Path(args[args.index("-f") + 1]).write_text("INVALID SQL;\n")
            return subprocess.CompletedProcess(args, 0, "", "")
        script = Path(args[args.index("-f") + 1]).read_text()
        assert script.startswith('TRUNCATE public."semantic_memory" CASCADE;')
        assert "INVALID SQL" in script
        assert "--single-transaction" in args
        assert "--set=ON_ERROR_STOP=1" in args
        return subprocess.CompletedProcess(args, 1, "", "syntax error")

    monkeypatch.setattr(snapshot.subprocess, "run", run)
    graph_import = MagicMock()
    monkeypatch.setattr(snapshot, "_neo4j_import", graph_import)
    with pytest.raises(RuntimeError, match="syntax error"):
        snapshot.load("saved")
    assert len(commands) == 2
    assert "[snapshot] loaded" not in capsys.readouterr().out
    graph_import.assert_not_called()


def test_failed_graph_export_does_not_publish_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot, "SNAP_ROOT", tmp_path)
    monkeypatch.setattr(snapshot, "_table_exists", lambda table: True)
    monkeypatch.setattr(snapshot, "_run_checked", MagicMock())
    monkeypatch.setattr(snapshot, "_graph_configured", lambda: True)
    monkeypatch.setattr(snapshot, "_neo4j_export", MagicMock(side_effect=RuntimeError("offline")))
    with pytest.raises(RuntimeError, match="offline"):
        snapshot.save("failed", False)
    assert not (tmp_path / "failed").exists()


def test_restore_invalidates_only_caches_for_restored_tables(monkeypatch):
    import redis
    cache = MagicMock()
    cache.scan_iter.side_effect = lambda match, **kw: [match[:-1] + "stale"]
    monkeypatch.setenv("MIRIX_REDIS_ENABLED", "true")
    monkeypatch.delenv("MIRIX_REDIS_URI", raising=False)
    monkeypatch.setattr(redis, "Redis", MagicMock(return_value=cache))
    snapshot._invalidate_redis_cache(["semantic_memory", "block"])
    assert [call.kwargs["match"] for call in cache.scan_iter.call_args_list] == ["semantic:*", "block:*"]
    assert [call.args for call in cache.delete.call_args_list] == [("semantic:stale",), ("block:stale",)]
    cache.flushdb.assert_not_called()
    cache.flushall.assert_not_called()
