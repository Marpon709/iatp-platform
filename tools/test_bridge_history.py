"""验证真实任务入口与数据库的衔接；计算引擎用受控替身，避免调用地图外网。"""
import copy
import io
import json
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'resources' / 'bridge'))
import service
from history_store import HistoryStore


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    threads_before = set(threading.enumerate())
    store = HistoryStore(tmp_path, 'test-model')
    monkeypatch.setattr(service, 'HISTORY', store)
    monkeypatch.setattr(service, 'JOBS', {})
    monkeypatch.setattr(service, 'MODEL_VERSION', 'test-model')
    yield service
    for worker in set(threading.enumerate()) - threads_before:
        worker.join(5)
        assert not worker.is_alive(), 'test worker must finish before resetting globals'
    store.close()


def wait_until_finished(bridge, jid):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        record = bridge.HISTORY.get_run(jid)
        if record['status'] != 'running':
            return record
        time.sleep(.01)
    raise AssertionError('mock calculation did not finish')


def get_request(bridge, path):
    """直接调用 HTTP 路由，不依赖运行环境是否允许回环套接字。"""
    handler = object.__new__(bridge.Handler)
    handler.path = path
    handler.respond = lambda value, status=200: (status, value)
    return handler.do_GET()


def test_route_chain_is_saved_and_listed(bridge, monkeypatch):
    result = {'success': True, 'totalCarbonG': 12, 'totalTravelMin': 8, 'legs': []}
    monkeypatch.setattr(bridge, 'normalized_chain', lambda payload: copy.deepcopy(result))
    reply = bridge.recorded_route_chain({'city': '南京', 'stops': []})
    record = bridge.HISTORY.get_run(reply['historyId'])
    assert record['status'] == 'completed'
    assert record['parameters']['city'] == '南京'
    assert record['result'] == result
    status, listing = get_request(bridge, '/api/showcase/history?kind=route-chain&pageSize=1')
    assert status == 200 and listing['total'] == 1
    assert listing['items'][0]['modelVersion'] == 'test-model'
    status, detail = get_request(bridge, '/api/showcase/history/' + reply['historyId'])
    assert status == 200 and detail['run']['jobId'] == reply['historyId']


def test_route_failure_is_saved_without_credentials(bridge, monkeypatch):
    def unavailable(payload):
        raise RuntimeError('map unavailable?key=private-value')
    monkeypatch.setattr(bridge, 'normalized_chain', unavailable)
    with pytest.raises(RuntimeError):
        bridge.recorded_route_chain({'city': '南京'})
    item = bridge.HISTORY.list_runs()['items'][0]
    record = bridge.HISTORY.get_run(item['jobId'])
    assert record['status'] == 'failed'
    assert 'private-value' not in json.dumps(record)


def test_async_job_survives_in_history_after_memory_is_cleared(bridge, monkeypatch):
    monkeypatch.setattr(bridge, 'calculate_activity_space', lambda payload, progress: {'success': True, 'message': 'ok'})
    job = bridge.new_job({'selectedTypes': ['park']}, kind='activity-space')
    record = wait_until_finished(bridge, job['jobId'])
    assert record['status'] == 'completed'
    bridge.JOBS.clear()
    snapshot = bridge.job_snapshot(job['jobId'])
    assert snapshot['persisted'] is True
    assert snapshot['parameters'] == {'selectedTypes': ['park']}


def test_duplicate_running_request_reuses_one_record(bridge, monkeypatch):
    started, release = threading.Event(), threading.Event()
    def calculate(payload, progress):
        started.set()
        assert release.wait(5)
        return {'success': True}
    monkeypatch.setattr(bridge, 'calculate_activity_space', calculate)
    try:
        first = bridge.new_job({'value': 1}, kind='activity-space')
        assert started.wait(5)
        second = bridge.new_job({'value': 1}, kind='activity-space')
        assert second['jobId'] == first['jobId']
        assert second['reusedRunningJob'] is True
        assert bridge.HISTORY.list_runs()['total'] == 1
    finally:
        release.set()
        wait_until_finished(bridge, first['jobId'])


def test_cancelled_job_keeps_partial_result(bridge, monkeypatch):
    started, release, worker_done = threading.Event(), threading.Event(), threading.Event()
    original_persist = bridge.persist_job
    def persist(jid):
        original_persist(jid)
        if threading.current_thread() is not threading.main_thread():worker_done.set()
    monkeypatch.setattr(bridge, 'persist_job', persist)
    def calculate(payload, progress):
        bridge.update_job(partialResult={'success': True, 'partial': True, 'candidates': [{'id': 'one'}]})
        started.set()
        assert release.wait(5)
        return {'success': True, 'candidates': [{'id': 'late'}]}
    monkeypatch.setattr(bridge, 'calculate_activity_space', calculate)
    try:
        job = bridge.new_job({}, kind='activity-space')
        assert started.wait(5)
        assert bridge.cancel_job(job['jobId'])['status'] == 'cancelled'
    finally:
        release.set()
        assert worker_done.wait(5)
    record = bridge.HISTORY.get_run(job['jobId'])
    assert record['status'] == 'cancelled'
    assert record['result']['candidates'][0]['id'] == 'one'


@pytest.mark.parametrize('query', ['page=0', 'pageSize=101', 'page=word', 'status=unknown', 'kind=unknown'])
def test_history_rejects_invalid_filters(bridge, query):
    status, body = get_request(bridge, '/api/showcase/history?' + query)
    assert status == 400 and body['success'] is False


def test_history_detail_not_found(bridge):
    status, body = get_request(bridge, '/api/showcase/history/missing')
    assert status == 404 and body['success'] is False


def test_post_requires_json_object(bridge):
    handler = object.__new__(bridge.Handler)
    handler.path = '/api/showcase/route-chain'
    handler.headers = {'Content-Length': '2'}
    handler.rfile = io.BytesIO(b'[]')
    handler.respond = lambda value, status=200: (status, value)
    status, body = handler.do_POST()
    assert status == 400 and body['success'] is False


def test_database_write_failure_does_not_change_calculation_success(bridge, monkeypatch):
    monkeypatch.setattr(bridge, 'normalized_chain', lambda payload: {'success': True, 'legs': []})
    def unavailable(snapshot):
        raise RuntimeError('database temporarily unavailable')
    monkeypatch.setattr(bridge.HISTORY, 'save_snapshot', unavailable)
    result = bridge.recorded_route_chain({})
    assert result['success'] is True and result['historyWarning']


def test_memory_poll_retries_history_write_and_clears_warning(bridge, monkeypatch):
    jid = 'test-persist-retry'
    snapshot = {'jobId': jid, 'status': 'completed', 'createdEpoch': time.time(), 'result': {'success': True}}
    bridge.HISTORY.create_run(jid, 'activity-space', 'key', {}, snapshot)
    bridge.JOBS[jid] = snapshot
    saved = bridge.HISTORY.save_snapshot
    monkeypatch.setattr(bridge.HISTORY, 'save_snapshot', lambda s: (_ for _ in ()).throw(RuntimeError('busy')))
    assert bridge.job_snapshot(jid)['status'] == 'completed'
    assert bridge.job_snapshot(jid)['historyWarning']
    monkeypatch.setattr(bridge.HISTORY, 'save_snapshot', saved)
    assert 'historyWarning' not in bridge.job_snapshot(jid)


def test_failed_history_initialization_is_not_cached(bridge, monkeypatch):
    class FailedStore:
        closed = False
        def __init__(self, *args):pass
        def mark_interrupted(self):raise RuntimeError('database busy')
        def close(self):FailedStore.closed = True
    monkeypatch.setattr(bridge, 'HISTORY', None)
    monkeypatch.setattr(bridge, 'HistoryStore', FailedStore)
    with pytest.raises(RuntimeError):bridge.get_history()
    assert bridge.HISTORY is None and FailedStore.closed


def test_sample_review_is_real_persistent_and_idempotent(bridge):
    first=bridge.review_sample()
    second=bridge.review_sample()
    assert first['historyId']==second['historyId'] and second['reused']
    run=bridge.HISTORY.get_run(first['historyId'])
    assert run['kind']=='result-review' and run['validation']['status']=='passed'
    assert run['recordSource']=='saved-sample-review'
    assert '未重新查询地图服务' in run['parameters']['处理方式']
    status,body=get_request(bridge,'/api/showcase/history?kind=result-review')
    assert status==200 and body['total']==1


def test_review_sample_post_endpoint(bridge):
    handler=object.__new__(bridge.Handler)
    handler.path='/api/showcase/review-sample'
    handler.headers={'Content-Length':'2'}
    handler.rfile=io.BytesIO(b'{}')
    handler.respond=lambda value,status=200:(status,value)
    status,body=handler.do_POST()
    assert status==200 and body['historyId'].startswith('review-')
