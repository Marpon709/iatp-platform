import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'resources/bridge'))
from result_checks import validate_result


def sample():
    return json.loads((ROOT / 'editable-app/web/data/default-chain.json').read_text(encoding='utf-8'))


def test_saved_real_route_sample_passes_without_mutation():
    result = sample()
    before = copy.deepcopy(result)
    assert validate_result('route-chain', result)['status'] == 'passed'
    assert result == before


def test_wrong_total_or_swapped_coordinates_fail():
    result = sample()
    result['totalCarbonG'] += 10
    assert not validate_result('route-chain', result)['checks'][0]['passed']
    result['legs'][0]['coordinates'] = [[32.0, 118.0]]
    assert not validate_result('route-chain', result)['checks'][-1]['passed']


def test_nonfinite_and_negative_values_fail():
    result = sample()
    for value in [float('nan'), -1, True, '1']:
        result['legs'][0]['timeMin'] = value
        assert not validate_result('route-chain', result)['checks'][1]['passed']


def test_activity_space_empty_geometry_is_valid_only_without_space():
    result = {'success': True, 'hasActivitySpace': False, 'areas': [], 'categories': [{'count': 0, 'score': 0}]}
    assert validate_result('activity-space', result)['status'] == 'passed'
    result['hasActivitySpace'] = True
    assert validate_result('activity-space', result)['status'] == 'failed'


def test_failed_or_unsupported_result_is_skipped():
    assert validate_result('route-chain', {'success': False})['status'] == 'skipped'
    assert validate_result('recommendation', {'success': True})['status'] == 'skipped'
