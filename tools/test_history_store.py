"""运行历史的事务、关联查询和重启语义，不访问地图或外网。"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "resources" / "bridge"))
from history_store import HistoryStore


@pytest.fixture
def store(tmp_path):
    value = HistoryStore(tmp_path, "model-v1")
    yield value
    value.close()


def snapshot(job_id, status="running", **fields):
    return {"jobId": job_id, "status": status, "stage": "计算中",
            "created_at": "2026-11-19T00:00:00+00:00", "createdEpoch": 1, **fields}


def create(store, job_id="run-1", kind="activity-space", **fields):
    value = snapshot(job_id, **fields)
    store.create_run(job_id, kind, "request-key", {"city": "南京", "stayMin": 20}, value)
    return value


def test_initial_schema_is_migrated_and_reads_joined_result(store):
    create(store)
    result = {"success": True, "hasActivitySpace": True, "categories": [{"id": "park"}]}
    store.save_snapshot(snapshot("run-1", "completed", result=result, progress=100))
    row = store.get_run("run-1")
    assert row["modelVersion"] == "model-v1"
    assert row["parameters"]["city"] == "南京"
    assert row["result"] == result
    item = store.list_runs()["items"][0]
    assert item["summary"] == {"hasActivitySpace": True, "categoryCount": 1}
    assert item["hasResult"] is True
    assert "result" not in item
    with sqlite3.connect(store.work_root / "history.sqlite3") as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone()[0] == "0001_analysis_history"


def test_pagination_filter_and_untrusted_strings_are_bound(store):
    for index in range(5):
        create(store, f"run-{index}", createdEpoch=index)
    create(store, "other", kind="recommendation", createdEpoch=9)
    store.save_snapshot(snapshot("run-2", "failed", error="模拟上游失败"))
    page = store.list_runs(page=2, page_size=2, kind="activity-space")
    assert page["total"] == 5
    assert [item["jobId"] for item in page["items"]] == ["run-2", "run-1"]
    assert store.list_runs(status="failed")["total"] == 1
    assert store.list_runs(kind="' OR 1=1 --")["total"] == 0
    assert store.get_run("' OR 1=1 --") is None
    assert store.list_runs()["total"] == 6


@pytest.mark.parametrize("kwargs", [{"page": 0}, {"page_size": 51}, {"page": True},
                                    {"page_size": 0}, {"status": "unknown"}])
def test_invalid_query_is_rejected(store, kwargs):
    with pytest.raises(ValueError):
        store.list_runs(**kwargs)


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled", "timed_out", "interrupted"])
def test_terminal_state_cannot_be_overwritten_by_late_running_snapshot(store, status):
    create(store)
    assert store.save_snapshot(snapshot("run-1", status))
    assert store.save_snapshot(snapshot("run-1", "running")) is False
    assert store.get_run("run-1")["status"] == status


def test_partial_result_is_preserved_on_cancel_and_complete_result_wins(store):
    create(store)
    partial = {"success": True, "partial": True, "candidates": [{"id": "p1"}]}
    store.save_snapshot(snapshot("run-1", partialResult=partial))
    store.save_snapshot(snapshot("run-1", "cancelled"))
    row = store.get_run("run-1")
    assert row["result"] == partial
    assert row["isPartial"] is True
    assert store.save_snapshot(snapshot("run-1", "completed", result={"success": True})) is False

    create(store, "run-2")
    full = {"success": True, "candidates": [{"id": "p1"}, {"id": "p2"}]}
    store.save_snapshot(snapshot("run-2", "completed", result=full))
    store.save_snapshot(snapshot("run-2", "completed", partialResult=partial))
    assert store.get_run("run-2")["result"] == full


def test_restart_preserves_results_and_marks_only_unfinished_runs(tmp_path):
    original = HistoryStore(tmp_path, "model-v1")
    create(original, "pending")
    create(original, "finished")
    original.save_snapshot(snapshot("finished", "completed", result={"success": True}))
    original.close()

    restarted = HistoryStore(tmp_path, "model-v2")
    try:
        assert restarted.mark_interrupted() == 1
        assert restarted.mark_interrupted() == 0
        assert restarted.get_run("pending")["status"] == "interrupted"
        assert restarted.get_run("finished")["result"]["success"] is True
        assert restarted.get_run("finished")["modelVersion"] == "model-v1"
        create(restarted, "new-run")
        assert restarted.get_run("new-run")["modelVersion"] == "model-v2"
    finally:
        restarted.close()


def test_threads_have_independent_sessions_and_do_not_lose_runs(store):
    def record(index):
        job_id = f"parallel-{index}"
        create(store, job_id)
        store.save_snapshot(snapshot(job_id, "completed", result={"success": True}))
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(record, range(12)))
    assert store.list_runs(status="completed")["total"] == 12


def test_invalid_json_rolls_back_without_creating_run(store):
    with pytest.raises(ValueError):
        store.create_run("bad-json", "activity-space", "key", {"stayMin": float("nan")}, snapshot("bad-json"))
    assert store.get_run("bad-json") is None


def test_duplicate_id_does_not_overwrite_parameters(store):
    create(store)
    with pytest.raises(ValueError):
        store.create_run("run-1", "other-kind", "other-key", {}, snapshot("run-1"))
    assert store.get_run("run-1")["parameters"] == {"city": "南京", "stayMin": 20}
