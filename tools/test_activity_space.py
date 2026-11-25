"""Checks for real demo geometry, scoring inputs and request validation."""
from pathlib import Path
import json
import math
import sys
import unittest
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'resources/bridge'))
from activity_space import validate_request, engine_request, present_result


class ActivitySpaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = json.loads((ROOT / 'editable-app/web/data/activity-space-demo.json').read_text(encoding='utf-8'))
        cls.types = json.loads((ROOT / 'editable-app/web/data/live-config.json').read_text(encoding='utf-8'))['activityTypes']

    def test_valid_and_invalid_parameters(self):
        parameters = validate_request(self.demo['parameters'], self.types)
        self.assertEqual(parameters['mode'], 'walk')
        for changes in [{'arrival': '16:00'}, {'stayMin': float('nan')},
                        {'selectedTypes': []}, {'selectedTypes': ['unknown']},
                        {'origin': {'coordinate': [32, 118]}}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_request({**parameters, **changes}, self.types)
        request = engine_request(parameters, self.types)
        self.assertTrue(all(value == 20 for value in request['poi_activity_times'].values()))
        self.assertEqual(request['activities'][1]['start_time'], '18:00')

    def test_geometry_coordinates_and_actual_scored_facilities(self):
        from shapely.geometry import shape, Point
        from shapely.ops import unary_union
        area = unary_union([shape(geometry) for geometry in self.demo['areas']])
        self.assertTrue(area.is_valid)
        self.assertGreater(area.bounds[0], 118)
        self.assertLess(area.bounds[3], 33)
        counts = Counter(poi['type'] for poi in self.demo['pois'])
        for category in self.demo['categories']:
            self.assertEqual(counts[category['id']], category['count'])
            total = sum(poi['contribution'] for poi in self.demo['pois'] if poi['type'] == category['id'])
            self.assertAlmostEqual(total, category['score'], delta=.015)
        # The engine simplifies the geometry used for facility lookup slightly.
        for poi in self.demo['pois']:
            self.assertTrue(area.buffer(.0001).covers(Point(*poi['coordinate'])))
            self.assertAlmostEqual(poi['contribution'], math.exp(-.003 * poi['distance']), delta=.0002)

    def test_no_elastic_time(self):
        raw = {'success': True, 'activity_spaces': [{'commute_route': [[32.04, 118.78], [32.05, 118.77]]}],
               'gap_details': [{'elastic_time_min': 0}], 'reachable_polygons': []}
        result = present_result(raw, self.demo['parameters'], self.types)
        self.assertFalse(result['hasActivitySpace'])
        self.assertEqual(result['routes'][0][0], [118.78, 32.04])


if __name__ == '__main__':
    unittest.main()
