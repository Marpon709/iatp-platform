"""城市数据与 CarbonBudgetCalculator 单例缓存。"""
from __future__ import annotations

import os
import threading

import numpy as np
from sklearn.neighbors import BallTree

_ENGINE_LOCK = threading.Lock()
_ENGINE_CACHE: dict = {}
_ENGINE_VERSION = "foot-bike-drive-v2-directed-transit"

_DEFAULT_ACTIVITY_TYPES = [
    'work', 'medical', 'shopping', 'catering',
    'education', 'recreation', 'transportation_station',
]

def _cache_key(city: str) -> str:
    return f"{str(city or '南京').strip()}:{_ENGINE_VERSION}"

def _ensure_backend_cwd():
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(backend_dir)
    return backend_dir

def _build_calculator(city: str):
    from transport_modes import read_all_city_data
    from carbon_budget import CarbonBudgetCalculator, map_poi_categories

    city_name = str(city or '南京').strip()
    dataset = read_all_city_data(city=city_name, use_cache=True)
    networks, poi_data, population_data, pop_transform, nodes_with_population, admin_boundary, pop_raster_crs = dataset

    poi_geojson = poi_data
    valid_count = sum(
        1 for gdf in poi_geojson.values()
        if gdf is not None and hasattr(gdf, 'empty') and not gdf.empty
    )
    if valid_count == 0:
        poi_geojson = map_poi_categories(poi_data)

    calc = CarbonBudgetCalculator(networks, poi_geojson, admin_boundary, city=city_name)
    calc.activity_types = list(_DEFAULT_ACTIVITY_TYPES)
    calc.precompute_indices()

    for activity in ('shopping', 'catering', 'transportation_station'):
        if activity not in calc.poi_trees and activity in poi_geojson:
            gdf = poi_geojson[activity]
            if gdf is not None and hasattr(gdf, 'geometry') and len(gdf) > 0:
                try:
                    coords = np.array([[p.x, p.y] for p in gdf.geometry])
                    tree = BallTree(coords, leaf_size=40)
                    calc.poi_trees[activity] = (tree, coords, gdf)
                except Exception as exc:
                    print(f"[city_engine] 补充 POI 索引失败 {activity}: {exc}")

    return {
        'calc': calc,
        'dataset': dataset,
        'city': city_name,
    }

def get_shared_calculator(city: str = '南京'):
    """获取或初始化指定城市的 CarbonBudgetCalculator（进程内单例）。"""
    _ensure_backend_cwd()
    key = _cache_key(city)
    with _ENGINE_LOCK:
        entry = _ENGINE_CACHE.get(key)
        if entry and entry.get('calc') is not None:
            return entry['calc']
        try:
            entry = _build_calculator(city)
            _ENGINE_CACHE[key] = entry
            print(f"[city_engine] 计算器就绪: {city}", flush=True)
            return entry['calc']
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[city_engine] 计算器初始化失败: {exc}", flush=True)
            return None

def get_city_dataset(city: str = '南京'):
    """返回 read_all_city_data 的完整元组（与计算器共用同一次加载）。"""
    _ensure_backend_cwd()
    key = _cache_key(city)
    with _ENGINE_LOCK:
        entry = _ENGINE_CACHE.get(key)
        if entry and entry.get('dataset') is not None:
            return entry['dataset']
    get_shared_calculator(city=city)
    with _ENGINE_LOCK:
        entry = _ENGINE_CACHE.get(key) or {}
        return entry.get('dataset')

def configure_calc_district(calc, selected_district=None):
    if calc is None:
        return None
    district = selected_district or '全部区域'
    calc.selected_district = district
    if district and district != '全部区域':
        calc.district_boundary = calc._filter_district(district)
    else:
        calc.district_boundary = calc.admin_boundary
    return calc

def prewarm_city_engine(city: str = '南京'):
    """后台预热：加载路网、POI 与索引。"""
    print(f"[city_engine] 开始预热 {city} ...", flush=True)
    calc = get_shared_calculator(city=city)
    if calc is not None:
        print(f"[city_engine] 预热完成: {city}", flush=True)
    else:
        print(f"[city_engine] 预热失败: {city}", flush=True)
    return calc
