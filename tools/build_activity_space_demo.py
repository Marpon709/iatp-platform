"""Recompute the meeting example with the bundled local engine."""
from pathlib import Path
import datetime
import json
import os
import sys

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / 'resources/bridge'), str(root / 'resources/backend')]
os.chdir(root / 'resources/backend')

from service import activity_config
from activity_space import validate_request, engine_request, present_result
from daily_activity_range import compute_daily_activity_range

types = activity_config()['activityTypes']
parameters = validate_request({
    'origin': {'label': '单位 · 新街口', 'coordinate': [118.786379, 32.041372]},
    'destination': {'label': '家 · 上海路', 'coordinate': [118.773708, 32.050131]},
    'departure': '17:00', 'arrival': '18:00', 'stayMin': 20, 'mode': 'walk',
    'selectedTypes': ['bookstore', 'supermarket', 'park'],
}, types)
request = engine_request(parameters, types)
raw = compute_daily_activity_range(**request)
result = present_result(raw, parameters, types)
if not result.get('success') or not result['polygons'] or not result['routes']:
    raise RuntimeError(result.get('message') or 'Default example has no activity space or route')
result['computedAt'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
target = root / 'editable-app/web/data/activity-space-demo.json'
target.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':')), encoding='utf-8')
print('DEMO READY', len(result['polygons']), 'polygons,', len(result['pois']), 'facilities')
