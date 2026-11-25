import pandas as pd
from flask import Flask, request, jsonify, send_file, Response
import json
from flask_cors import CORS
import os
import math
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import threading
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECTS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))

INSTALL_DIR = os.environ.get('IATP_INSTALL_DIR', '').strip() or ROOT_DIR

def _dotenv_candidate_paths():
    # Packaged distribution: configuration is supplied only by the desktop shell.
    # Never inspect a developer/user .env file from disk.
    return []


def _load_dotenv_fallback(env_path: str) -> None:

    if not os.path.isfile(env_path):
        return
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, _, value = line.partition('=')
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        pass

try:
    from dotenv import load_dotenv
    _dotenv_loaded = False
    for _env_path in _dotenv_candidate_paths():
        if os.path.isfile(_env_path):
            load_dotenv(_env_path)
            _dotenv_loaded = True
            print(f'[env] 已加载 .env: {_env_path}')
            break
    if not _dotenv_loaded:
        for _env_path in _dotenv_candidate_paths():
            _load_dotenv_fallback(_env_path)
            if os.environ.get('TIANDITU_API_KEY', '').strip() or os.environ.get('VITE_TIANDITU_API_KEY', '').strip():
                print(f'[env] 已加载 .env: {_env_path}')
                break
except ImportError:
    for _env_path in _dotenv_candidate_paths():
        _load_dotenv_fallback(_env_path)
        if os.environ.get('TIANDITU_API_KEY', '').strip() or os.environ.get('VITE_TIANDITU_API_KEY', '').strip():
            print(f'[env] 已加载 .env: {_env_path}')
            break

# 前后端共用一把 Key：若未设 TIANDITU_API_KEY，兼容读取旧的 VITE_TIANDITU_API_KEY
if not os.environ.get('TIANDITU_API_KEY', '').strip():
    _legacy_tk = os.environ.get('VITE_TIANDITU_API_KEY', '').strip()
    if _legacy_tk:
        os.environ['TIANDITU_API_KEY'] = _legacy_tk
        print('[env] 已用 VITE_TIANDITU_API_KEY 作为 TIANDITU_API_KEY')
if not os.environ.get('AMAP_API_KEY', '').strip():
    _legacy_amap = os.environ.get('VITE_AMAP_API_KEY', '').strip()
    if _legacy_amap:
        os.environ['AMAP_API_KEY'] = _legacy_amap
        print('[env] 已用 VITE_AMAP_API_KEY 作为 AMAP_API_KEY')
DATA_DIR = os.path.join(BASE_DIR, 'Nanjin2018')
POI_DIR = os.path.join(BASE_DIR, 'POI', 'Nanjing')
RESIDENTIAL_DIR = os.path.join(PROJECTS_DIR, 'POI')
CITY_DATA_DIRS = {
    '南京': os.path.join(BASE_DIR, 'Nanjin2018'),
    '武汉': os.path.join(BASE_DIR, 'Wuhan'),
    '北京': os.path.join(BASE_DIR, 'Beijing'),
    '上海': os.path.join(BASE_DIR, 'Shanghai'),
}
CITY_POI_DIRS = {
    '南京': os.path.join(BASE_DIR, 'POI', 'Nanjing'),
    '武汉': os.path.join(BASE_DIR, 'POI','Wuhan'),
    '北京': os.path.join(BASE_DIR, 'POI','Beijing'),
    '上海': os.path.join(BASE_DIR, 'POI','Shanghai'),
}
CITY_PREFIXES = {
    '南京': 'nanjing',
    '武汉': 'wuhan',
    '北京': 'beijing',
    '上海': 'shanghai',
}

def get_city_data_dir(city):
    return CITY_DATA_DIRS.get(str(city or '南京').strip(), CITY_DATA_DIRS['南京'])

def get_city_poi_dir(city):
    return CITY_POI_DIRS.get(str(city or '南京').strip(), CITY_POI_DIRS['南京'])

def get_city_prefix(city):
    return CITY_PREFIXES.get(str(city or '南京').strip(), CITY_PREFIXES['南京'])

# 确保后端目录始终在模块搜索路径中（打包到 Electron 后尤其重要）
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from Data_loader import get_poi_subcategories, filter_poi_by_subcategory

calculation_results = {
    'raster_data': None,
    'transform': None,
    'stats': None,
    'calculator': None,
    'last_calculated': None,
    'poi_category': None,  # 当前 POI 类型
    'activity_raster_data': None,
}

equity_analysis_results = {
    'moran_raster': None,
    'cluster_raster': None,
    'z_raster': None,
    'pvalue_raster': None,
    'bounds': None,
    'poi_category': None,
    'subcategory': None,
    'moran_stats': None,
    'color_thresholds': None,
    'resolution': None,
    'search_radius_m': None,
    'raster_data': None,
    'cluster_raster_data': None,
    'z_raster_data': None,
    'pvalue_raster_data': None,
    'fairness_metrics': None,
    'pop_weighted_metrics': None,
    'gwr_meta': None,
    'gwr_gdp_raster_data': None,
    'gwr_pop_density_raster_data': None,
    'gwr_beta_gdp_raster': None,
    'gwr_beta_pop_raster': None,
    'gdp_view_available': False,
    'population_view_available': False,
    'district_stats': None,
}

def _store_equity_analysis_result(result: dict) -> dict:
    #提取 numpy 栅格存入全局状态，并从 API 响应中移除不可 JSON 序列化的字段。
    global equity_analysis_results
    if not result.get('success'):
        return result

    moran_np = result.pop('_moran_raster_np', None)
    cluster_np = result.pop('_cluster_raster_np', None)
    z_np = result.pop('_z_raster_np', None)
    pvalue_np = result.pop('_pvalue_raster_np', None)
    gwr_gdp_np = result.pop('_gwr_beta_gdp_raster_np', None)
    gwr_pop_np = result.pop('_gwr_beta_pop_raster_np', None)

    equity_analysis_results.update({
        'moran_raster': np.asarray(moran_np) if moran_np is not None else None,
        'cluster_raster': np.asarray(cluster_np) if cluster_np is not None else None,
        'z_raster': np.asarray(z_np) if z_np is not None else None,
        'pvalue_raster': np.asarray(pvalue_np) if pvalue_np is not None else None,
        'gwr_beta_gdp_raster': np.asarray(gwr_gdp_np) if gwr_gdp_np is not None else None,
        'gwr_beta_pop_raster': np.asarray(gwr_pop_np) if gwr_pop_np is not None else None,
        'bounds': result.get('equity_bounds'),
        'poi_category': result.get('poi_category'),
        'subcategory': result.get('subcategory'),
        'moran_stats': result.get('moran_stats'),
        'color_thresholds': result.get('color_thresholds'),
        'resolution': result.get('resolution'),
        'search_radius_m': result.get('search_radius_m'),
        'raster_data': result.get('raster_data'),
        'cluster_raster_data': result.get('cluster_raster_data'),
        'z_raster_data': result.get('z_raster_data'),
        'pvalue_raster_data': result.get('pvalue_raster_data'),
        'fairness_metrics': result.get('fairness_metrics'),
        'pop_weighted_metrics': result.get('pop_weighted_metrics'),
        'gwr_meta': result.get('gwr_meta'),
        'gwr_gdp_raster_data': result.get('gwr_gdp_raster_data'),
        'gwr_pop_density_raster_data': result.get('gwr_pop_density_raster_data'),
        'gdp_view_available': result.get('gdp_view_available', False),
        'population_view_available': result.get('population_view_available', False),
        'district_stats': result.get('district_stats'),
    })
    return result

# 不同 POI 类型的颜色配置（9级，与前端图例对应）
ACCESSIBILITY_COLORS = {
    'work': {
        'colors': [
            '#313695',  # 第1段 - 深蓝色
            '#4575b4',  # 第2段 - 蓝色
            '#74add1',  # 第3段 - 浅蓝色
            '#abd9e9',  # 第4段 - 天蓝色
            '#ffffbf',  # 第5段 - 浅黄色
            '#fee090',  # 第6段 - 黄色
            '#fdae61',  # 第7段 - 橙色
            '#f46d43',  # 第8段 - 橙红色
            '#d73027'   # 第9段 - 红色
        ]
    },
    'transportation_station': {
        'colors': [
            '#313695', '#4575b4', '#74add1', '#abd9e9',
            '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027'
        ]
    },
    'education': {
        'colors': [
            '#313695', '#4575b4', '#74add1', '#abd9e9',
            '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027'
        ]
    },
    'recreation': {
        'colors': [
            '#313695', '#4575b4', '#74add1', '#abd9e9',
            '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027'
        ]
    },
    'medical': {
        'colors': [
            '#313695', '#4575b4', '#74add1', '#abd9e9',
            '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027'
        ]
    },
    'shopping': {
        'colors': [
            '#313695', '#4575b4', '#74add1', '#abd9e9',
            '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027'
        ]
    },
    'catering': {
        'colors': [
            '#313695', '#4575b4', '#74add1', '#abd9e9',
            '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027'
        ]
    }
}

# 可达性为 0 时的固定灰色（与前端图例一致）
ACCESSIBILITY_ZERO_COLOR = '#bdbdbd'

def _ensure_strictly_increasing_bounds(bounds, decimals=4):
    """将 display_bounds 调整为严格递增，供 BoundaryNorm 使用。"""
    if not bounds:
        return bounds
    out = [float(bounds[0])]
    for i in range(1, len(bounds)):
        b = float(bounds[i])
        span = max(abs(out[-1]), abs(b), 1.0)
        step = max(10 ** (-decimals), span * 1e-6, abs(out[-1]) * 1e-9 + 1e-12)
        if b <= out[-1]:
            b = out[-1] + step
        if i < len(bounds) - 1:
            rounded = round(b, decimals)
            if rounded <= out[-1] and b > out[-1]:
                out.append(b)
            else:
                out.append(rounded)
        else:
            out.append(b)
    return out

def compute_accessibility_color_thresholds(raster_data, colors_nine, num_levels=9, num_samples=50000):

    colors_nine = list(colors_nine)
    if len(colors_nine) != num_levels:
        raise ValueError(f'colors_nine 需要 {num_levels} 种颜色，当前 {len(colors_nine)}')

    valid_full = raster_data[~np.isnan(raster_data)]
    base = {
        'zero_color': ACCESSIBILITY_ZERO_COLOR,
        'has_zero_bucket': True,
        'colors': [ACCESSIBILITY_ZERO_COLOR] + colors_nine,
    }

    if valid_full.size == 0:
        display_bounds = _ensure_strictly_increasing_bounds(
            [0.0, 1e-12, 2e-12, 3e-12, 4e-12, 5e-12, 6e-12, 7e-12, 8e-12, 9e-12, 1e-11]
        )
        return {
            **base,
            'thresholds': [0.0] * num_levels,
            'display_bounds': display_bounds,
            'data_max': 0.0,
            'vmax': 0.0,
            'zero_max': 0.0,
            'p90': None,
        }

    pos_full = valid_full[valid_full > 0]
    pos_max = float(np.max(pos_full)) if pos_full.size else 0.0
    p90_pos = float(np.percentile(pos_full, 90.0)) if pos_full.size else 0.0

    if pos_full.size == 0:
        display_bounds = _ensure_strictly_increasing_bounds(
            [0.0, 1e-12, 2e-12, 3e-12, 4e-12, 5e-12, 6e-12, 7e-12, 8e-12, 9e-12, 1e-11]
        )
        return {
            **base,
            'thresholds': [0.0] * num_levels,
            'display_bounds': display_bounds,
            'data_max': 0.0,
            'vmax': 0.0,
            'zero_max': 0.0,
            'p90': None,
        }

    pos_min = float(np.min(pos_full))
    zero_eps = max(1e-12, pos_min * 1e-9)
    if zero_eps >= pos_min:
        zero_eps = pos_min / 2.0 if pos_min > 0 else 1e-12

    valid_data = pos_full
    if valid_data.size > num_samples:
        idx = np.random.choice(valid_data.size, num_samples, replace=False)
        valid_data = valid_data[idx]

    upper = pos_max
    if upper <= zero_eps:
        upper = zero_eps + max(1e-6, abs(zero_eps) * 1e-9 + 1e-6)
    else:
        upper = upper + max(1e-9, abs(upper) * 1e-12)

    # 9 档正值分位：zero_eps → P11.1 … P88.9 → upper（共 9 个正值区间）
    edges_pos = [zero_eps]
    for k in range(1, num_levels):
        q = 100.0 * k / num_levels
        edges_pos.append(float(np.percentile(valid_data, q)))
    edges_pos[-1] = min(edges_pos[-1], p90_pos if p90_pos > zero_eps else pos_max)
    edges_pos.append(upper)
    edges_pos = _ensure_strictly_increasing_bounds(edges_pos, decimals=4)
    if edges_pos[-1] < upper:
        edges_pos[-1] = upper
    if edges_pos[0] <= 0.0:
        edges_pos[0] = zero_eps

    display_bounds = [0.0] + edges_pos
    if display_bounds[1] <= display_bounds[0]:
        display_bounds[1] = zero_eps
    thresholds = [float(edges_pos[i]) for i in range(len(edges_pos) - 1)]

    return {
        **base,
        'thresholds': thresholds,
        'display_bounds': [float(b) for b in display_bounds],
        'data_max': pos_max,
        'vmax': pos_max,
        'zero_max': 0.0,
        'p90': float(edges_pos[-2]) if len(edges_pos) >= 2 else None,
    }

def _strictly_increasing_thresholds(raw_list, vmin, vmax, decimals=4):
    #将阈值序列调整为严格递增并四舍五入，避免图例出现多段 0~0。
    span = float(vmax - vmin)
    step = max(10 ** (-decimals), span * 1e-6 if span > 0 else 10 ** (-decimals))
    out = []
    prev = None
    for t in raw_list:
        t = float(np.clip(t, vmin, vmax))
        if prev is not None and t <= prev:
            t = min(prev + step, vmax)
        if prev is not None and t <= prev:
            t = min(prev + step * 10, vmax)
        out.append(round(t, decimals))
        prev = out[-1]
    # 四舍五入后再次消除并列
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            bump = round(out[i - 1] + step, decimals)
            out[i] = min(bump, vmax) if vmax > out[i - 1] else out[i - 1]
    return out

def compute_quantile_thresholds(raster_data, num_levels=9, num_samples=50000):

    valid_full = raster_data[~np.isnan(raster_data)]
    if valid_full.size == 0:
        return [0.0] * num_levels

    # 分位数可在大栅格上抽样，但上下界必须用全量数据，避免图例最高档超过真实 max
    vmin_full = float(np.min(valid_full))
    vmax_full = float(np.max(valid_full))

    valid_data = valid_full
    if valid_data.size > num_samples:
        indices = np.random.choice(valid_data.size, num_samples, replace=False)
        valid_data = valid_data[indices]

    vmin = vmin_full
    vmax = vmax_full
    if vmax <= vmin + 1e-18:
        return [round(vmin, 4)] * num_levels

    p90_full = float(np.percentile(valid_full, 90.0))

    # 与全局最小值几乎相等的像元视为「零档」，其余为正值子集
    tol = max(1e-12, abs(vmin) * 1e-12 + 1e-15 * (vmax - vmin + 1.0))
    pos = valid_data[valid_data > vmin + tol]
    pos_full = valid_full[valid_full > vmin_full + tol]
    zero_like_frac = 1.0 - (pos.size / float(valid_data.size))
    min_pos = max(50, int(valid_data.size * 0.001))

    pct_lowers = np.linspace(0.0, 90.0, num_levels)
    raw = [float(np.percentile(valid_data, q)) for q in pct_lowers]
    raw[-1] = p90_full

    if pos.size >= min_pos and zero_like_frac >= 0.04:
        # 第一档固定 vmin；中间档在正值子集上拉开（避免多段 0~0）；末档仍用全栅格 P90
        raw[0] = vmin_full
        if num_levels > 2:
            src = pos_full if pos_full.size >= min_pos else pos
            mid_pct = np.linspace(
                100.0 / num_levels,
                90.0 * (num_levels - 2) / num_levels,
                num_levels - 2,
            )
            for i, q in enumerate(mid_pct, start=1):
                raw[i] = min(float(np.percentile(src, q)), p90_full)
        raw[-1] = p90_full

    raw = _strictly_increasing_thresholds(raw, vmin_full, vmax_full, decimals=4)
    if len(raw) > num_levels:
        raw = raw[:num_levels]
    elif len(raw) < num_levels:
        last = raw[-1] if raw else vmin_full
        span = max(p90_full - last, (vmax_full - vmin_full) * 1e-3, 1e-6)
        for _ in range(num_levels - len(raw)):
            last = min(last + span / (num_levels - len(raw) + 1), p90_full)
            raw.append(round(last, 4))
        raw = _strictly_increasing_thresholds(raw, vmin_full, vmax_full, decimals=4)[:num_levels]

    # 末档下界固定为全栅格 P90；若中间档被抬高则回推，保证最高档覆盖 90%–100% 分位
    if raw:
        raw[-1] = round(min(p90_full, vmax_full), 4)
        span = max(vmax_full - vmin_full, 1e-6)
        step = max(10 ** -4, span * 1e-6)
        for i in range(len(raw) - 2, -1, -1):
            if raw[i] >= raw[i + 1]:
                raw[i] = round(max(vmin_full, raw[i + 1] - step), 4)

    return [float(t) for t in raw]

# 可达性分析：未传出行时间时的默认阈值（分钟）
LOCATION_ACCESSIBILITY_TRAVEL_TIME_DEFAULT = 30
# 碳预算多活动可达性：与「基于位置」默认同出行时间、默认交通方式（骑行）
CARBON_BUDGET_TRAVEL_TIME_DEFAULT = LOCATION_ACCESSIBILITY_TRAVEL_TIME_DEFAULT
CARBON_BUDGET_TRANSPORT_MODE_DEFAULT = 'bike'
from CalculateDistance import CARBON_DEFAULT_RCB_TONS_YEAR

# 可达性计算缓存：城市 + POI 类型 + 子类别 + 出行时间 + 交通方式
accessibility_cache = {}  # key: (city, poi_category, subcategory, travel_time, transport_mode)

def _safe_json(val):
    #将 nan/inf 转为 None；数值转 Python 标量；保留 str/bool 等可直接 JSON 的类型。
    if val is None:
        return None
    if isinstance(val, str):
        return val
    if isinstance(val, bool):
        return val
    if isinstance(val, (np.integer, int)) and not isinstance(val, bool):
        return int(val)
    if isinstance(val, (np.floating, float)):
        x = float(val)
        if np.isnan(x) or np.isinf(x):
            return None
        return x
    try:
        x = float(val)
        if np.isnan(x) or np.isinf(x):
            return None
        return x
    except (TypeError, ValueError):
        return val

def _safe_json_stats(stats):
    if not isinstance(stats, dict):
        return _safe_json(stats)
    result = {}
    for key, val in stats.items():
        if isinstance(val, dict):
            result[key] = {k: _safe_json(v) for k, v in val.items()}
        else:
            result[key] = _safe_json(val)
    return result

calculation_progress = {
    'is_running': False,
    'progress': 0,
    'status': '等待计算',
    'error': None
}

carbon_budget_results = {
    'icb_raster': None,
    'acc_raster': None,
    'min_cb_raster': None,
    'grid_inside_mask': None,
    'selected_district': None,
    'city': None,
    'transform': None,
    'stats': None,
    'resolution': None,
    'search_radius_m': None,
    'travel_time_min': None,
    'transport_mode': None,
    'last_calculated': None
}

carbon_budget_progress = {
    'is_running': False,
    'progress': 0,
    'status': '等待计算',
    'error': None,
    'last_progress_at': None,
}

# 公平性分析进度与最近一次 API 结果（不含 numpy 栅格）
equity_analysis_progress = {
    'is_running': False,
    'progress': 0,
    'status': '等待计算',
    'error': None,
    'last_progress_at': None,
}
equity_analysis_last_result = None

# 超过该秒数无进度更新则视为任务已失效（进程中断或状态卡住）
CARBON_BUDGET_STALE_SECONDS = 1800

app = Flask(__name__)
CORS(app, origins="*")  # 允许所有来源的跨域请求

def _prewarm_travel_recommendation_engine():
    try:
        from tianditu_poi import require_tianditu_api_key
        require_tianditu_api_key()
        print("[出行推荐] 天地图 API Key 已配置")
    except Exception as exc:
        print(f"[出行推荐] 预热提示: {exc}")

def _prewarm_city_engine():
    #后台预热路网、POI 与 CarbonBudgetCalculator，缩短首次分析等待。
    try:
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(backend_dir)
        from city_engine import prewarm_city_engine
        prewarm_city_engine(city='南京')
    except Exception as exc:
        print(f"[城市引擎] 预热失败: {exc}")

threading.Thread(target=_prewarm_travel_recommendation_engine, daemon=True).start()
threading.Thread(target=_prewarm_city_engine, daemon=True).start()

def run_calculation_background(resolution, group_config=None, city='南京'):
    try:
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(backend_dir)
        print(f"[可达性计算] 工作目录: {os.getcwd()}")

        # 延迟导入以避免应用启动时缺依赖报错
        from AccessibilityAnalysis import read_all_city_data, NanjingAccessibilityCalculator, _CITY_DATA_CACHE

        if city in _CITY_DATA_CACHE:
            print(f"[可达性计算] 清除 {city} 的旧缓存")
            del _CITY_DATA_CACHE[city]

        calculation_progress.update({
            'is_running': True,
            'progress': 5,
            'status': '正在准备数据',
            'error': None
        })

        start_time = time.time()

        calculation_progress.update({
            'progress': 15,
            'status': '正在加载网络数据'
        })

        time.sleep(1)
        calculation_progress.update({
            'progress': 25,
            'status': '正在加载POI数据'
        })

        time.sleep(1)
        calculation_progress.update({
            'progress': 35,
            'status': '正在初始化计算器'
        })

        # 统一使用 read_all_city_data 加载数据
        networks, poi_data, population_data, pop_transform, nodes_with_population, admin_boundary, pop_raster_crs = read_all_city_data(city=city, use_cache=True)

        if not poi_data or all(gdf.empty for gdf in poi_data.values() if gdf is not None):
            print(f"[可达性计算] 错误: {city} POI 数据为空")
            calculation_progress.update({'progress': -1, 'status': 'POI 数据加载失败'})
            return

        calculator = NanjingAccessibilityCalculator(
            networks,
            poi_data,
            admin_boundary,
            population_data=population_data,
            pop_transform=pop_transform,
            city=city,
            pop_raster_crs=pop_raster_crs,
            nodes_with_population=nodes_with_population
        )

        if isinstance(group_config, dict):
            selected_poi_category = str(group_config.get('poi_category', '')).strip()
            subcategory_keyword = str(group_config.get('subcategory', '')).strip()
            poi_gdf_filtered = None

            if selected_poi_category:
                poi_gdf = calculator.poi_data.get(selected_poi_category)
                if poi_gdf is not None and not getattr(poi_gdf, 'empty', True):
                    calculator.activities = [selected_poi_category]
                    calculator.activity_weights = {selected_poi_category: 1.0}
                    if subcategory_keyword:
                        poi_gdf_filtered = filter_poi_by_subcategory(poi_gdf, subcategory_keyword, selected_poi_category)
                        poi_count = len(poi_gdf_filtered) if poi_gdf_filtered is not None and not poi_gdf_filtered.empty else 0
                        calculation_progress.update({
                            'progress': 38,
                            'status': f'已选择POI类型: {selected_poi_category}，子类别: {subcategory_keyword}（{poi_count}个POI）'
                        })
                    else:
                        calculation_progress.update({
                            'progress': 38,
                            'status': f'已选择POI类型: {selected_poi_category}'
                        })
                    calculation_progress['poi_category'] = selected_poi_category
                    calculation_progress['subcategory'] = subcategory_keyword

            transport_mode = str(group_config.get('transport_mode', 'foot')).strip() or 'foot'
            travel_time_min = int(float(group_config.get('travel_time', LOCATION_ACCESSIBILITY_TRAVEL_TIME_DEFAULT)))
            calculator.transport_mode = transport_mode
            calculator.travel_time_minutes = travel_time_min
            mode_weights = group_config.get('mode_weights')
            if isinstance(mode_weights, dict) and mode_weights:
                calculator.mode_weights = mode_weights
            custom_times = group_config.get('custom_times')
            if isinstance(custom_times, dict) and custom_times:
                calculator.custom_activity_times = custom_times
            print(
                f"[可达性计算] 出行时间={travel_time_min}min，"
                f"交通方式加权={calculator._normalize_mode_weights()}"
            )

        def update_calc_progress(p, status):
            # 进度百分比只增不减；状态文案始终更新（地铁等长耗时步骤需刷新 status）
            current = int(calculation_progress.get('progress', 0) or 0)
            target = int(p)
            updates = {'status': status}
            if target > current:
                updates['progress'] = target
            calculation_progress.update(updates)

        # 计算 t0 距离阈值（米）：由出行时间 × 交通方式速度得出
        t0_distance_m = None
        if isinstance(group_config, dict):
            travel_minutes = float(group_config.get('travel_time', LOCATION_ACCESSIBILITY_TRAVEL_TIME_DEFAULT))
            transport_mode = str(group_config.get('transport_mode', '')).strip()
            from transport_modes import speed_ms as get_speed_ms
            t0_distance_m = travel_minutes * 60.0 * get_speed_ms(transport_mode)

        print(f"开始计算高分辨率可达性栅格，分辨率: {resolution}")
        raster_data, transform, stats, activity_raster_data = calculator.calculate_composite_accessibility(
            resolution=float(resolution),
            progress_callback=update_calc_progress,
            subcategory_keyword=subcategory_keyword,
            subcategory_poi_gdf=poi_gdf_filtered,
            subcategory_poi_activity=selected_poi_category,
            t0_distance_m=t0_distance_m
        )

        calculation_progress.update({
            'progress': 85,
            'status': '正在处理结果'
        })

        print(f"[可达性计算完成] raster_data shape: {raster_data.shape}, transform: {transform}")
        calculation_results.update({
            'raster_data': raster_data,
            'transform': transform,
            'stats': stats,
            'calculator': calculator,
            'last_calculated': datetime.now().isoformat(),
            'poi_category': selected_poi_category,
            'activity_raster_data': activity_raster_data,
            'city': city,
        })

        if transform is not None and len(raster_data.shape) == 2:
            tm_cache = str(getattr(calculator, 'transport_mode', 'foot'))
            tt_cache = int(getattr(calculator, 'travel_time_minutes', LOCATION_ACCESSIBILITY_TRAVEL_TIME_DEFAULT))
            from fairness_analysis import store_accessibility_cache_entry
            store_accessibility_cache_entry(
                accessibility_cache,
                city,
                selected_poi_category or 'work',
                subcategory_keyword,
                tt_cache,
                tm_cache,
                raster_data,
                transform,
                stats=stats,
                activity_raster_data=activity_raster_data,
                resolution=resolution,
            )

        calculation_progress.update({
            'progress': 100,
            'status': f'高分辨率计算完成，耗时 {time.time() - start_time:.2f} 秒'
        })

        time.sleep(1)
        calculation_progress.update({
            'is_running': False
        })

    except Exception as e:
        calculation_progress.update({
            'is_running': False,
            'status': '计算失败',
            'error': str(e)
        })
        print(f"计算过程中发生错误: {e}")

@app.route('/api/calculate_accessibility', methods=['POST'])
def calculate_accessibility():
    try:
        if calculation_progress['is_running']:
            return jsonify({
                'success': False,
                'message': '已有计算任务在运行，请等待完成后再试'
            }), 400

        params = request.json or {}
        resolution = params.get('resolution', 0.009)
        group_config = params.get('group_config')
        city = params.get('city', '南京')

        selected_poi_category = None
        selected_group_type = None
        if isinstance(group_config, dict):
            selected_poi_category = str(group_config.get('poi_category', '')).strip()
            selected_group_type = str(group_config.get('group_type', '')).strip()

        subcategory = None
        transport_mode = 'foot'
        travel_time = LOCATION_ACCESSIBILITY_TRAVEL_TIME_DEFAULT
        if isinstance(group_config, dict):
            subcategory = str(group_config.get('subcategory', '')).strip()
            transport_mode = str(group_config.get('transport_mode', 'foot')).strip() or 'foot'
            travel_time = int(float(group_config.get('travel_time', LOCATION_ACCESSIBILITY_TRAVEL_TIME_DEFAULT)))
        if resolution > 0:
            from fairness_analysis import build_accessibility_cache_key
            cache_key = build_accessibility_cache_key(
                city,
                selected_poi_category or 'work',
                subcategory or '',
                travel_time,
                transport_mode,
                resolution,
            )
            cached = accessibility_cache.get(cache_key)
            if cached is not None:
                activity_raster = cached.get('activity_raster_data')
                activity_raster_response = {}
                if activity_raster:
                    for act, act_raster in activity_raster.items():
                        flat = act_raster.flatten()
                        activity_raster_response[act] = {
                            'width': act_raster.shape[1],
                            'height': act_raster.shape[0],
                            'values': [float(x) if not np.isnan(x) else None for x in flat],
                        }
                calculation_results.update({
                    'raster_data': cached['raster_data'],
                    'transform': cached['transform'],
                    'stats': cached['stats'],
                    'calculator': None,
                    'last_calculated': cached['last_calculated'],
                    'poi_category': selected_poi_category,
                    'activity_raster_data': activity_raster,
                    'city': city,
                })
                calculation_progress.update({
                    'is_running': False,
                    'progress': 100,
                    'status': f'使用缓存的 {selected_poi_category} 可达性结果（{city}）',
                    'error': None
                })
                return jsonify({
                    'success': True,
                    'message': f'使用已缓存的可达性结果（{city}，POI类型: {selected_poi_category}）',
                    'cached': True,
                    'progress': 100,
                    'activity_raster_data': activity_raster_response
                })

        calculation_progress.update({
            'is_running': False,
            'progress': 0,
            'status': '等待计算',
            'error': None
        })

        thread = threading.Thread(target=run_calculation_background, args=(resolution, group_config, city))
        thread.daemon = True
        thread.start()

        return jsonify({
            'success': True,
            'message': '高分辨率可达性计算已开始，请通过 /api/get_progress 端点查询进度',
            'progress_endpoint': '/api/get_progress'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': '启动计算时发生错误'
        }), 500

@app.route('/api/get_accessibility_map', methods=['GET'])
def get_accessibility_map():
    try:
        if calculation_results['raster_data'] is None:
            return jsonify({
                'success': False,
                'message': '请先运行可达性计算'
            }), 400

        raster_data = calculation_results['raster_data']
        transform = calculation_results['transform']

        if transform is None:
            return jsonify({
                'success': False,
                'message': '计算结果无效（transform为None），请重新计算'
            }), 400

        buf = io.BytesIO()

        plt.style.use('default')

        fig = plt.figure(figsize=(10, 8), dpi=150, facecolor='none')
        ax = fig.add_axes([0, 0, 1, 1])  # 全尺寸，无边框

        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        ax.set_axis_off()

        raster_data = calculation_results['raster_data']
        transform = calculation_results['transform']

        if transform is None:
            return jsonify({
                'success': False,
                'message': '计算结果无效（transform为None），请重新计算'
            }), 400

        left = transform[2]  # 左上角x坐标
        right = transform[2] + transform[0] * raster_data.shape[1]
        top = transform[5]  # 左上角y坐标
        bottom = transform[5] + transform[4] * raster_data.shape[0]
        extent = [left, right, bottom, top]

        poi_category = calculation_results.get('poi_category', 'work')

        color_config = ACCESSIBILITY_COLORS.get(poi_category, ACCESSIBILITY_COLORS['work'])
        colors_list = color_config['colors']

        # 0 值灰色；非 0 正值按分位数 9 档着色
        color_threshold_cfg = compute_accessibility_color_thresholds(
            raster_data, colors_list, num_levels=9
        )
        colors_list = color_threshold_cfg['colors']
        display_bounds = color_threshold_cfg['display_bounds']
        thresholds = color_threshold_cfg['thresholds']
        vmin = 0.0
        vmax_data = color_threshold_cfg['data_max']
        vmax = color_threshold_cfg['vmax']

        raster_display = raster_data.copy()

        from matplotlib.colors import ListedColormap, BoundaryNorm
        cmap = ListedColormap(colors_list)
        norm = BoundaryNorm(display_bounds, len(colors_list))

        im = ax.imshow(
            raster_display,
            extent=extent,
            cmap=cmap,
            norm=norm,  # 使用分位数分段标准化
            origin='lower',
            interpolation='nearest'
        )

        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    pad_inches=0, transparent=True)
        buf.seek(0)

        image_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)

        stats = calculation_results['stats'].copy() if calculation_results['stats'] else {}
        for key, value in stats.items():
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                stats[key] = None

        bounds = {
            'north': float(top),  # 北边界
            'south': float(bottom),  # 南边界
            'west': float(left),  # 西边界
            'east': float(right)  # 东边界
        }

        raster_data_flat = raster_data.flatten()
        raster_data_response = {
            'width': raster_data.shape[1],
            'height': raster_data.shape[0],
            'values': [float(x) if not np.isnan(x) else None for x in raster_data_flat],
            'bounds': bounds
        }

        activity_raster_data = calculation_results.get('activity_raster_data') or {}
        activity_raster_response = {}
        for act, act_raster in activity_raster_data.items():
            if act_raster is None:
                continue
            flat = act_raster.flatten()
            activity_raster_response[act] = {
                'width': act_raster.shape[1],
                'height': act_raster.shape[0],
                'values': [float(x) if not np.isnan(x) else None for x in flat],
            }

        return jsonify({
            'success': True,
            'image_data': image_data,
            'stats': stats,
            'last_calculated': calculation_results['last_calculated'],
            'bounds': bounds,
            'crs': 'EPSG:4326',
            'color_range': {
                'min': float(vmin),
                'max': float(vmax)
            },
            'poi_category': poi_category,
            'threshold_config': color_threshold_cfg,
            'raster_data': raster_data_response,  # 添加栅格数据用于悬停显示
            'activity_raster_data': activity_raster_response
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': '生成可视化图像失败'
        }), 500

@app.route('/api/get_accessibility_stats', methods=['GET'])
def get_accessibility_stats():
    if calculation_results['stats'] is None:
        return jsonify({
            'success': False,
            'message': '请先运行可达性计算'
        }), 400

    stats = calculation_results['stats'].copy()
    for key, value in stats.items():
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            stats[key] = None

    return jsonify({
        'success': True,
        'stats': stats,
        'last_calculated': calculation_results['last_calculated']
    })

@app.route('/api/get_poi_subcategories', methods=['GET'])
def get_poi_subcategories_api():
    city = request.args.get('city', '南京')
    poi_dir = get_city_poi_dir(city)
    print(f"[子类别API] 收到请求, city={city}, poi_dir={poi_dir}, 目录存在={os.path.exists(poi_dir)}")
    try:
        subcategories = get_poi_subcategories(poi_dir=poi_dir, city=city)
        result_keys = list(subcategories.keys())
        print(f"[子类别API] 返回成功, 共 {len(result_keys)} 个大类")
        return jsonify({
            'success': True,
            'subcategories': subcategories
        })
    except Exception as e:
        print(f"[子类别API] 异常: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/get_progress', methods=['GET'])
def get_progress():
    return jsonify({
        'success': True,
        'progress': calculation_progress['progress'],
        'status': calculation_progress['status'],
        'is_running': calculation_progress['is_running'],
        'error': calculation_progress['error']
    })

def _equity_travel_params_from_request(req_data):
    travel_time = req_data.get('travel_time')
    transport_mode = req_data.get('transport_mode')
    legacy_radius = req_data.get('search_radius_m')

    if travel_time is not None or transport_mode is not None:
        tt, mode, radius_m = _carbon_budget_travel_time_to_radius_m(
            travel_time if travel_time is not None else LOCATION_ACCESSIBILITY_TRAVEL_TIME_DEFAULT,
            transport_mode if transport_mode is not None else 'bike',
        )
        return tt, mode, radius_m, True

    if legacy_radius is not None:
        try:
            radius_m = float(legacy_radius)
            radius_m = max(500.0, min(radius_m, 100000.0))
        except (TypeError, ValueError):
            radius_m = None
        return None, None, radius_m, False

    tt, mode, radius_m = _carbon_budget_travel_time_to_radius_m(
        LOCATION_ACCESSIBILITY_TRAVEL_TIME_DEFAULT,
        'bike',
    )
    return tt, mode, radius_m, True

def _touch_equity_progress(**fields):
    equity_analysis_progress.update(fields)
    equity_analysis_progress['last_progress_at'] = time.time()

def _start_equity_analysis_job(analyze_kwargs: dict) -> dict:
    global equity_analysis_last_result

    if equity_analysis_progress.get('is_running'):
        return {
            'success': False,
            'message': '已有公平性分析任务在运行，请等待完成后再试',
            'progress': equity_analysis_progress.get('progress', 0),
            'status': equity_analysis_progress.get('status', ''),
        }

    def progress_cb(pct, status):
        _touch_equity_progress(progress=int(pct), status=str(status))

    def run_job():
        global equity_analysis_last_result
        try:
            from fairness_analysis import analyze_fairness_from_poi
            _touch_equity_progress(
                is_running=True,
                progress=5,
                status='正在启动公平性分析...',
                error=None,
            )
            result = analyze_fairness_from_poi(
                **analyze_kwargs,
                progress_callback=progress_cb,
            )
            result = _store_equity_analysis_result(result)
            if not result.get('success'):
                _touch_equity_progress(
                    is_running=False,
                    progress=0,
                    status=result.get('message') or '分析失败',
                    error=result.get('message') or '分析失败',
                )
                equity_analysis_last_result = result
                return
            equity_analysis_last_result = result
            _touch_equity_progress(
                is_running=False,
                progress=100,
                status=result.get('message') or '分析完成',
                error=None,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            equity_analysis_last_result = {'success': False, 'message': str(e)}
            _touch_equity_progress(
                is_running=False,
                progress=0,
                status='分析失败',
                error=str(e),
            )

    _touch_equity_progress(is_running=False, progress=0, status='等待计算', error=None)
    thread = threading.Thread(target=run_job, daemon=True)
    thread.start()

    return {
        'success': True,
        'message': '公平性分析已开始，请通过进度接口查询状态',
        'progress_endpoint': '/api/get_equity_progress',
    }

@app.route('/api/equity_analysis', methods=['POST'])
def equity_analysis():
    try:
        from fairness_analysis import analyze_fairness_from_poi, DEFAULT_RESOLUTION

        req_data = request.get_json() or {}
        poi_category = req_data.get('poi_category', 'work')
        subcategory = req_data.get('subcategory') or None
        city = req_data.get('city', '南京')
        resolution = float(req_data.get('resolution', DEFAULT_RESOLUTION))
        travel_time_min, transport_mode, search_radius_m, use_travel_params = _equity_travel_params_from_request(req_data)
        mix_lam = req_data.get('spatial_weight_mix_lambda', None)
        if mix_lam is not None:
            try:
                mix_lam = float(mix_lam)
                mix_lam = max(0.0, min(1.0, mix_lam))
            except (TypeError, ValueError):
                mix_lam = None

        analyze_kwargs = dict(
            poi_category=poi_category,
            subcategory=subcategory,
            city=city,
            resolution=resolution,
            spatial_weight_mix_lambda=mix_lam,
        )
        if use_travel_params:
            analyze_kwargs['travel_time_min'] = travel_time_min
            analyze_kwargs['transport_mode'] = transport_mode
            analyze_kwargs['accessibility_cache'] = accessibility_cache
        elif search_radius_m is not None:
            analyze_kwargs['search_radius_m'] = search_radius_m

        return jsonify(_start_equity_analysis_job(analyze_kwargs))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/equity_population_weighted', methods=['POST'])
def equity_population_weighted():
    #公平性分析 + 人口加权指标（与 equity_analysis 参数相同）。
    try:
        from fairness_analysis import analyze_fairness_from_poi, DEFAULT_RESOLUTION

        req_data = request.get_json() or {}
        poi_category = req_data.get('poi_category', 'work')
        subcategory = req_data.get('subcategory') or None
        city = req_data.get('city', '南京')
        resolution = float(req_data.get('resolution', DEFAULT_RESOLUTION))
        travel_time_min, transport_mode, search_radius_m, use_travel_params = _equity_travel_params_from_request(req_data)
        mix_lam = req_data.get('spatial_weight_mix_lambda', None)
        if mix_lam is not None:
            try:
                mix_lam = float(mix_lam)
                mix_lam = max(0.0, min(1.0, mix_lam))
            except (TypeError, ValueError):
                mix_lam = None

        analyze_kwargs = dict(
            poi_category=poi_category,
            subcategory=subcategory,
            city=city,
            resolution=resolution,
            spatial_weight_mix_lambda=mix_lam,
            population_weighted=True,
        )
        if use_travel_params:
            analyze_kwargs['travel_time_min'] = travel_time_min
            analyze_kwargs['transport_mode'] = transport_mode
            analyze_kwargs['accessibility_cache'] = accessibility_cache
        elif search_radius_m is not None:
            analyze_kwargs['search_radius_m'] = search_radius_m

        return jsonify(_start_equity_analysis_job(analyze_kwargs))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/get_equity_progress', methods=['GET'])
def get_equity_progress():
    return jsonify({
        'success': True,
        'progress': equity_analysis_progress.get('progress', 0),
        'status': equity_analysis_progress.get('status', ''),
        'is_running': equity_analysis_progress.get('is_running', False),
        'error': equity_analysis_progress.get('error'),
    })

@app.route('/api/get_equity_results', methods=['GET'])
def get_equity_results():
    """获取最近一次公平性分析结果（JSON，不含 numpy 栅格）。"""
    global equity_analysis_last_result
    if not equity_analysis_last_result:
        return jsonify({
            'success': False,
            'message': '暂无公平性分析结果，请先运行分析',
        }), 404
    if equity_analysis_progress.get('is_running'):
        return jsonify({
            'success': False,
            'message': '公平性分析仍在进行中',
            'progress': equity_analysis_progress.get('progress', 0),
            'status': equity_analysis_progress.get('status', ''),
        }), 409
    return jsonify(equity_analysis_last_result)

@app.route('/api/get_equity_heatmap_image', methods=['GET'])
def get_equity_heatmap_image():
    """获取公平性分析热力图 PNG（与可达性/碳预算模块一致），并返回扁平栅格数据供点击查询。"""
    try:
        if equity_analysis_results.get('moran_raster') is None:
            return jsonify({
                'success': False,
                'message': '请先运行公平性分析'
            }), 400

        mode = request.args.get('mode', 'cluster')
        if mode == 'moran':
            mode = 'significance'
        bounds = equity_analysis_results.get('bounds')
        if bounds is None:
            return jsonify({
                'success': False,
                'message': '公平性栅格数据不完整，请重新分析'
            }), 400

        if mode in ('gdp', 'population'):
            from fairness_analysis import generate_gwr_heatmap_image
            coef_key = 'gwr_beta_gdp_raster' if mode == 'gdp' else 'gwr_beta_pop_raster'
            coef_raster = equity_analysis_results.get(coef_key)
            if coef_raster is None:
                return jsonify({
                    'success': False,
                    'message': f'GWR {"GDP 系数" if mode == "gdp" else "人口密度系数"}栅格未生成。'
                               f'请确认该城市{"GDP" if mode == "gdp" else "人口"}栅格数据可用并重新分析。',
                    'gdp_view_available': equity_analysis_results.get('gdp_view_available', False),
                    'population_view_available': equity_analysis_results.get('population_view_available', False),
                    'district_stats': equity_analysis_results.get('district_stats'),
                }), 400
            title = 'GWR · GDP 系数 β₁' if mode == 'gdp' else 'GWR · 人口密度系数 β₂'
            img_base64, meta = generate_gwr_heatmap_image(coef_raster, bounds, title=title)
            flat_key = 'gwr_gdp_raster_data' if mode == 'gdp' else 'gwr_pop_density_raster_data'
            flat = equity_analysis_results.get(flat_key) or {}
            return jsonify({
                'success': True,
                'mode': mode,
                'image_data': img_base64,
                'bounds': meta.get('bounds', bounds),
                'width': meta.get('width', flat.get('width')),
                'height': meta.get('height', flat.get('height')),
                'raster_data': flat.get('values', []),
                'gwr_meta': equity_analysis_results.get('gwr_meta'),
                'moran_stats': equity_analysis_results.get('moran_stats'),
                'fairness_metrics': equity_analysis_results.get('fairness_metrics'),
                'poi_category': equity_analysis_results.get('poi_category'),
                'subcategory': equity_analysis_results.get('subcategory'),
                'gdp_view_available': equity_analysis_results.get('gdp_view_available', False),
                'population_view_available': equity_analysis_results.get('population_view_available', False),
                'district_stats': equity_analysis_results.get('district_stats'),
            })

        moran = equity_analysis_results['moran_raster']
        cluster = equity_analysis_results['cluster_raster']
        pvalue = equity_analysis_results.get('pvalue_raster')
        if moran is None or cluster is None:
            return jsonify({
                'success': False,
                'message': '公平性栅格数据不完整，请重新分析'
            }), 400
        if mode == 'significance' and pvalue is None:
            return jsonify({
                'success': False,
                'message': '显著性 P 值栅格未生成，请重新运行公平性分析'
            }), 400

        from fairness_analysis import generate_equity_heatmap_image

        img_base64, meta = generate_equity_heatmap_image(
            moran_raster=moran,
            cluster_raster=cluster,
            bounds=bounds,
            mode=mode,
            pvalue_raster=pvalue,
        )

        raster_data = equity_analysis_results.get('raster_data') or {}
        cluster_raster_data = equity_analysis_results.get('cluster_raster_data') or {}
        z_raster_data = equity_analysis_results.get('z_raster_data') or {}
        pvalue_raster_data = equity_analysis_results.get('pvalue_raster_data') or {}

        return jsonify({
            'success': True,
            'mode': meta.get('mode', mode),
            'image_data': img_base64,
            'bounds': meta.get('bounds', bounds),
            'width': meta.get('width', raster_data.get('width')),
            'height': meta.get('height', raster_data.get('height')),
            'raster_data': raster_data.get('values', []),
            'cluster_raster_data': cluster_raster_data.get('values', []),
            'z_raster_data': z_raster_data.get('values', []),
            'pvalue_raster_data': pvalue_raster_data.get('values', []),
            'moran_stats': equity_analysis_results.get('moran_stats'),
            'fairness_metrics': equity_analysis_results.get('fairness_metrics'),
            'poi_category': equity_analysis_results.get('poi_category'),
            'subcategory': equity_analysis_results.get('subcategory'),
            'gdp_view_available': equity_analysis_results.get('gdp_view_available', False),
            'population_view_available': equity_analysis_results.get('population_view_available', False),
            'district_stats': equity_analysis_results.get('district_stats'),
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e),
            'error': traceback.format_exc(),
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'success': True,
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'calculation_ready': calculation_results['raster_data'] is not None
    })

@app.route('/api/get_population_map', methods=['GET'])
def get_population_map():
    try:
        city = request.args.get('city', '南京')
        data_dir = get_city_data_dir(city)
        prefix = get_city_prefix(city)
        population_raster_path = os.path.join(data_dir, f"{prefix}_population.tif")

        if not os.path.exists(population_raster_path):
            return jsonify({
                'success': False,
                'message': f'人口栅格文件不存在: {population_raster_path}'
            }), 404

        import rasterio
        with rasterio.open(population_raster_path) as src:
            population_data = src.read(1)
            transform = src.transform
            bounds = src.bounds

        buf = io.BytesIO()
        plt.style.use('default')
        fig = plt.figure(figsize=(10, 8), dpi=150, facecolor='none')
        ax = fig.add_axes([0, 0, 1, 1])
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        ax.set_axis_off()

        # 计算extent (注意：栅格通常是[bottom, top]顺序)
        left = bounds.left
        right = bounds.right
        bottom = bounds.bottom
        top = bounds.top

        pop_log = np.log1p(population_data)

        from matplotlib.colors import LinearSegmentedColormap
        colors_list = [
            '#1a1a1a',  # 极低 - 深色背景
            '#4a4a4a',  # 很低
            '#8b7355',  # 低 - 棕色
            '#c9a86c',  # 中低
            '#e6c88c',  # 中等
            '#f0d890',  # 中高
            '#ffe680',  # 高
            '#ff9933'   # 极高
        ]
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('population_custom', colors_list, N=n_bins)
        cmap.set_bad(alpha=0)

        valid_data = pop_log[~np.isnan(pop_log)]
        if len(valid_data) > 0:
            vmin = np.percentile(valid_data, 2)
            vmax = np.percentile(valid_data, 98)
        else:
            vmin, vmax = 0, 10

        im = ax.imshow(
            pop_log,
            extent=[left, right, bottom, top],
            cmap=cmap,
            origin='upper',
            vmin=vmin,
            vmax=vmax,
            interpolation='bilinear'
        )

        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    pad_inches=0, transparent=True)
        buf.seek(0)
        image_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)

        raster_data_flat = population_data.flatten()
        raster_data_response = {
            'width': population_data.shape[1],
            'height': population_data.shape[0],
            'values': [float(x) if not (np.isnan(x) or x < 0) else None for x in raster_data_flat],
            'bounds': {
                'north': float(top),
                'south': float(bottom),
                'west': float(left),
                'east': float(right)
            }
        }

        valid_pop = population_data[~np.isnan(population_data) & (population_data > 0)]
        total_population = float(np.sum(valid_pop)) if len(valid_pop) > 0 else 0
        mean_pop = float(np.mean(valid_pop)) if len(valid_pop) > 0 else 0
        std_pop = float(np.std(valid_pop)) if len(valid_pop) > 0 else 0
        min_pop = float(np.min(valid_pop)) if len(valid_pop) > 0 else 0
        max_pop = float(np.max(valid_pop)) if len(valid_pop) > 0 else 0
        median_pop = float(np.median(valid_pop)) if len(valid_pop) > 0 else 0

        valid_count = int(np.sum(~np.isnan(population_data) & (population_data > 0)))
        total_pixels = int(population_data.size)

        pop_quantiles = {}
        for q in [10, 25, 50, 75, 90]:
            pop_quantiles[f'p{q}'] = float(np.percentile(valid_pop, q)) if len(valid_pop) > 0 else 0

        if len(valid_pop) > 0:
            low_threshold = pop_quantiles['p25']
            mid_low_threshold = pop_quantiles['p50']
            mid_high_threshold = pop_quantiles['p75']
            high_threshold = pop_quantiles['p90']

            low_count = int(np.sum(valid_pop <= low_threshold))
            mid_low_count = int(np.sum((valid_pop > low_threshold) & (valid_pop <= mid_low_threshold)))
            mid_high_count = int(np.sum((valid_pop > mid_low_threshold) & (valid_pop <= mid_high_threshold)))
            high_count = int(np.sum((valid_pop > mid_high_threshold) & (valid_pop <= high_threshold)))
            very_high_count = int(np.sum(valid_pop > high_threshold))
        else:
            low_count = mid_low_count = mid_high_count = high_count = very_high_count = 0
            low_threshold = mid_low_threshold = mid_high_threshold = high_threshold = 0

        population_stats = {
            'total_population': total_population,
            'mean_population': mean_pop,
            'std_population': std_pop,
            'min_population': min_pop,
            'max_population': max_pop,
            'median_population': median_pop,
            'valid_pixel_count': valid_count,
            'total_pixel_count': total_pixels,
            'coverage_rate': round(valid_count / total_pixels * 100, 2) if total_pixels > 0 else 0,
            'quantiles': pop_quantiles,
            'density_distribution': {
                '极低 (0-25%)': low_count,
                '低 (25-50%)': mid_low_count,
                '中 (50-75%)': mid_high_count,
                '高 (75-90%)': high_count,
                '极高 (>90%)': very_high_count
            },
            'thresholds': {
                'low': low_threshold,
                'mid_low': mid_low_threshold,
                'mid_high': mid_high_threshold,
                'high': high_threshold
            }
        }

        return jsonify({
            'success': True,
            'image_data': image_data,
            'bounds': raster_data_response['bounds'],
            'crs': str(src.crs) if src.crs else 'EPSG:4326',
            'raster_data': raster_data_response,
            'population_stats': population_stats
        })

    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'message': str(e),
            'error': traceback.format_exc()
        }), 500

def _district_name_field(admin_boundary):
    for field in ['县', '区', 'NAME', 'name', 'district', 'QHMC', '区县', '名称']:
        if field in admin_boundary.columns:
            return field
    return None

def _load_admin_boundary_gdf(city='南京'):
    """仅读取行政边界矢量，避免为区域列表/边界接口加载整套路网 POI。"""
    import geopandas as gpd
    from transport_modes import get_city_data_config

    city_name = str(city or '南京').strip() or '南京'
    cfg = get_city_data_config(city_name)
    boundary_path = os.path.join(BASE_DIR, cfg['boundary_file'])
    if not os.path.isfile(boundary_path):
        raise FileNotFoundError(f'行政边界文件不存在: {boundary_path}')
    return gpd.read_file(boundary_path)

def _filter_admin_boundary_by_district(admin_boundary, selected_district):
    district = str(selected_district or '全部区域').strip() or '全部区域'
    if district == '全部区域' or admin_boundary is None or admin_boundary.empty:
        return admin_boundary
    field = _district_name_field(admin_boundary)
    if not field:
        return admin_boundary
    names = admin_boundary[field].astype(str).str.strip()
    matched = admin_boundary[names == district]
    if matched.empty:
        matched = admin_boundary[names.str.contains(district, na=False)]
    return matched if not matched.empty else admin_boundary

def _build_carbon_budget_inside_mask(raster_shape, transform, selected_district=None, city='南京'):
    """栅格像元是否落在计算区域内（界外须透明，不可当作未达标灰色）。"""
    try:
        from rasterio.features import geometry_mask
        from rasterio.transform import Affine
        from shapely.geometry import mapping
        from shapely.ops import unary_union

        admin = _load_admin_boundary_gdf(city)
        boundary = _filter_admin_boundary_by_district(admin, selected_district)
        if boundary is None or boundary.empty:
            return np.ones(raster_shape, dtype=bool)

        geom = unary_union(boundary.geometry)
        if geom is None or getattr(geom, 'is_empty', False):
            return np.ones(raster_shape, dtype=bool)

        aff = Affine(
            transform.a, transform.b, transform.c,
            transform.d, transform.e, transform.f,
        )
        # rasterio 行 0 在北；本项目的栅格/imshow(origin=lower) 行 0 在南，须翻转
        inside = geometry_mask(
            [mapping(geom)],
            transform=aff,
            invert=True,
            out_shape=raster_shape,
            all_touched=False,
        )
        return np.flipud(inside)
    except Exception as ex:
        print(f'[碳预算热力图] 重建界内掩膜失败: {ex}')
        stored = carbon_budget_results.get('grid_inside_mask')
        if stored is not None:
            arr = np.asarray(stored, dtype=bool)
            if arr.shape == raster_shape:
                return arr
        return np.ones(raster_shape, dtype=bool)

def _render_min_cb_rgba(raster, inside_mask, cmap, norm):
    from matplotlib.cm import ScalarMappable

    h, w = raster.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    inside = np.asarray(inside_mask, dtype=bool)
    if inside.shape != raster.shape:
        inside = np.ones(raster.shape, dtype=bool)

    finite_inside = inside & np.isfinite(raster)
    nan_inside = inside & ~np.isfinite(raster)

    if np.any(finite_inside):
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        rgba[finite_inside] = sm.to_rgba(raster[finite_inside].astype(np.float64))

    rgba[nan_inside, 0] = 0.82
    rgba[nan_inside, 1] = 0.82
    rgba[nan_inside, 2] = 0.82
    rgba[nan_inside, 3] = 0.65
    return rgba

def _districts_from_boundary(admin_boundary):
    districts = ['全部区域']
    if admin_boundary is None or admin_boundary.empty:
        return districts
    district_field = _district_name_field(admin_boundary)
    if not district_field:
        return districts
    vals = (
        admin_boundary[district_field]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )
    seen = set()
    for v in vals:
        if v and v not in seen:
            districts.append(v)
            seen.add(v)
    return districts

@app.route('/api/get_districts', methods=['GET'])
def get_districts():
    try:
        city = request.args.get('city', '南京')
        admin_boundary = _load_admin_boundary_gdf(city)
        return jsonify({'success': True, 'districts': _districts_from_boundary(admin_boundary)})
    except Exception:
        from carbon_budget import NANJING_DISTRICTS
        return jsonify({'success': True, 'districts': NANJING_DISTRICTS})

@app.route('/api/get_district_boundary', methods=['GET'])
def get_district_boundary():
    """获取指定区域的边界GeoJSON"""
    try:
        district_name = request.args.get('district', '全部区域')
        city = request.args.get('city', '南京')

        admin_boundary = _load_admin_boundary_gdf(city)

        if admin_boundary.empty:
            return jsonify({
                'success': False,
                'message': '无法加载行政边界数据'
            }), 404

        if district_name == '全部区域':
            union_geometry = admin_boundary.geometry.unary_union
            from shapely.geometry import mapping
            geojson_data = mapping(union_geometry)

            return jsonify({
                'success': True,
                'geojson': {
                    'type': 'Feature',
                    'geometry': geojson_data,
                    'properties': {'name': '全部区域'}
                }
            })
        else:
            district_field = _district_name_field(admin_boundary)

            if not district_field:
                return jsonify({
                    'success': False,
                    'message': '边界数据中找不到区域名称字段'
                }), 404

            names = admin_boundary[district_field].astype(str).str.strip()
            matched = admin_boundary[names == district_name]
            if matched.empty:
                matched = admin_boundary[names.str.contains(district_name, na=False)]

            if matched.empty:
                return jsonify({
                    'success': False,
                    'message': f'未找到区域: {district_name}'
                }), 404

            district_geom = matched.iloc[0].geometry

            from shapely.geometry import mapping
            geojson_data = mapping(district_geom)

            return jsonify({
                'success': True,
                'geojson': {
                    'type': 'Feature',
                    'geometry': geojson_data,
                    'properties': {
                        'name': matched.iloc[0].get(district_field, district_name)
                    }
                }
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'message': f'获取区域边界失败: {str(e)}'
        }), 500

def _road_network_mode_key(mode: str) -> str:
    m = str(mode or 'drive').strip().lower()
    if m in ('walk', 'foot'):
        return 'foot'
    if m in ('bike', 'ebike'):
        return 'bike'
    return 'drive'

def _edge_coords_lonlat(G, u, v, data) -> list:
    """从边属性或端点解析 [lng, lat] 折线。"""
    geom = data.get('geometry') if isinstance(data, dict) else None
    if geom is not None:
        try:
            from shapely import wkt
            from shapely.geometry import LineString, MultiLineString

            parsed = None
            if isinstance(geom, str):
                parsed = wkt.loads(geom)
            elif hasattr(geom, 'coords'):
                parsed = geom
            if parsed is not None and not getattr(parsed, 'is_empty', True):
                if parsed.geom_type == 'LineString':
                    return [[float(x), float(y)] for x, y in parsed.coords]
                if parsed.geom_type == 'MultiLineString':
                    out = []
                    for part in parsed.geoms:
                        out.extend([[float(x), float(y)] for x, y in part.coords])
                    if out:
                        return out
            if isinstance(geom, (list, tuple)) and geom:
                first = geom[0]
                if isinstance(first, (list, tuple)) and len(first) >= 2:
                    out = []
                    for pt in geom:
                        if len(pt) < 2:
                            continue
                        x, y = float(pt[0]), float(pt[1])
                        if abs(x) <= 180 and abs(y) <= 90:
                            out.append([x, y])
                        else:
                            out.append([y, x])
                    if out:
                        return out
        except Exception:
            pass

    nu, nv = G.nodes.get(u, {}), G.nodes.get(v, {})
    try:
        x1, y1 = float(nu.get('x', nu.get('lon'))), float(nu.get('y', nu.get('lat')))
        x2, y2 = float(nv.get('x', nv.get('lon'))), float(nv.get('y', nv.get('lat')))
        if all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            return [[x1, y1], [x2, y2]]
    except (TypeError, ValueError):
        pass
    return []

def _finite_lonlat_coords(coords) -> list:
    """过滤非有限坐标，避免 json 写出 NaN 导致缓存无法解析。"""
    out = []
    for pt in coords or []:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        try:
            x, y = float(pt[0]), float(pt[1])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        if abs(x) > 180 or abs(y) > 90:
            continue
        out.append([x, y])
    return out

def _edge_road_class(data) -> str:
    if not isinstance(data, dict):
        return ''
    hw = data.get('highway') or data.get('fclass') or data.get('road_type') or data.get('type')
    return str(hw or '').strip().lower()

def _build_road_network_geojson(city: str, mode: str = 'drive', highway_filter=None) -> dict:
    """将 GraphML 路网转为 GeoJSON FeatureCollection（无向去重）。

    highway_filter: 可选，如 {'primary', 'secondary', 'primary_link', 'secondary_link'}。
    """
    from transport_modes import load_road_networks

    city_name = str(city or '南京').strip() or '南京'
    net_key = _road_network_mode_key(mode)
    networks = load_road_networks(city_name)
    G = networks.get(net_key) or networks.get('drive')
    if G is None or G.number_of_nodes() == 0:
        raise FileNotFoundError(f'未找到城市路网: {city_name}/{net_key}')

    allow = None
    if highway_filter:
        allow = {str(x).strip().lower() for x in highway_filter if str(x).strip()}

    features = []
    seen = set()
    edge_iter = G.edges(keys=True, data=True) if G.is_multigraph() else (
        (u, v, 0, d) for u, v, d in G.edges(data=True)
    )
    for u, v, _k, data in edge_iter:
        road_cls = _edge_road_class(data or {})
        if allow is not None and road_cls not in allow:
            continue
        a, b = (u, v) if str(u) <= str(v) else (v, u)
        undirected_key = (a, b)
        if undirected_key in seen:
            continue
        seen.add(undirected_key)
        coords = _finite_lonlat_coords(_edge_coords_lonlat(G, u, v, data or {}))
        if len(coords) < 2:
            continue
        # 过密折线抽稀，减轻前端压力
        if len(coords) > 24:
            step = max(1, len(coords) // 16)
            kept = coords[::step]
            if kept[-1] != coords[-1]:
                kept.append(coords[-1])
            coords = kept
        props = {}
        if road_cls:
            props['highway'] = road_cls[:64]
            props['fclass'] = road_cls[:64]
        if isinstance(data, dict):
            name = data.get('name')
            if name:
                props['name'] = str(name)[:80]
        features.append({
            'type': 'Feature',
            'geometry': {'type': 'LineString', 'coordinates': coords},
            'properties': props,
        })

    return {
        'type': 'FeatureCollection',
        'features': features,
        'properties': {
            'city': city_name,
            'mode': net_key,
            'edge_count': len(features),
            'node_count': int(G.number_of_nodes()),
            'highway_filter': sorted(allow) if allow else None,
        },
    }

@app.route('/api/get_road_network', methods=['GET'])
def get_road_network():
    """获取城市路网 GeoJSON，供地图叠加显示（默认驾车网）。

    查询参数:
      city, mode
      classes: 逗号分隔道路等级，默认 primary,secondary,primary_link,secondary_link
               传 all 则返回全量
    """
    try:
        city = request.args.get('city', '南京')
        mode = request.args.get('mode', 'drive')
        net_key = _road_network_mode_key(mode)
        city_name = str(city or '南京').strip() or '南京'
        classes_raw = str(request.args.get('classes', 'primary,secondary,primary_link,secondary_link')).strip()
        if classes_raw.lower() in ('all', '*', ''):
            highway_filter = None
            cache_tag = 'all'
        else:
            highway_filter = {c.strip().lower() for c in classes_raw.split(',') if c.strip()}
            cache_tag = '_'.join(sorted(highway_filter)) or 'all'

        from transport_modes import get_city_data_config, webgis_base_dir
        cfg = get_city_data_config(city_name)
        base = webgis_base_dir()
        # 优先放到城市子目录（Network/Nanjing）
        city_folder = 'Nanjing' if city_name == '南京' else city_name
        cache_dir = os.path.join(base, 'Network', city_folder)
        if not os.path.isdir(cache_dir):
            cache_dir = os.path.join(base, cfg.get('folder') or 'Network')
        os.makedirs(cache_dir, exist_ok=True)
        # v3：过滤主干/次干 + 禁止 NaN + 原子写入
        cache_path = os.path.join(cache_dir, f'road_display_{net_key}_{cache_tag}_v3.geojson')

        # 清理历史损坏/过期缓存（v2 曾因中断写入或 NaN 导致非法 JSON）
        for legacy_dir in {cache_dir, os.path.join(base, 'Network')}:
            try:
                if not os.path.isdir(legacy_dir):
                    continue
                for name in os.listdir(legacy_dir):
                    if not name.startswith('road_display_'):
                        continue
                    if name.endswith('_v3.geojson'):
                        continue
                    legacy_path = os.path.join(legacy_dir, name)
                    if os.path.isfile(legacy_path):
                        try:
                            os.remove(legacy_path)
                            print(f'[路网显示] 已删除旧缓存: {legacy_path}')
                        except OSError:
                            pass
            except OSError:
                pass

        # 源 GraphML 更新则刷新缓存
        road_files = cfg.get('road_files') or {}
        src_rel = road_files.get(net_key) or cfg.get('road_file')
        src_mtime = 0.0
        if src_rel:
            src_path = os.path.join(base, src_rel)
            if os.path.isfile(src_path):
                src_mtime = os.path.getmtime(src_path)

        use_cache = (
            os.path.isfile(cache_path)
            and os.path.getsize(cache_path) > 100
            and (src_mtime <= 0 or os.path.getmtime(cache_path) >= src_mtime)
        )
        if use_cache:
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    geojson = json.load(f)
                if not isinstance(geojson, dict) or geojson.get('type') != 'FeatureCollection':
                    raise ValueError('缓存不是有效 FeatureCollection')
                return jsonify({
                    'success': True,
                    'geojson': geojson,
                    'from_cache': True,
                    'edge_count': len(geojson.get('features') or []),
                    'city': city_name,
                    'mode': net_key,
                })
            except Exception as cache_ex:
                print(f'[路网显示] 缓存损坏，将重建: {cache_path} ({cache_ex})')
                try:
                    os.remove(cache_path)
                except OSError:
                    pass

        geojson = _build_road_network_geojson(city_name, net_key, highway_filter=highway_filter)
        try:
            tmp_path = cache_path + '.tmp'
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(geojson, f, ensure_ascii=False, allow_nan=False)
            os.replace(tmp_path, cache_path)
        except Exception as ex:
            print(f'[路网显示] 缓存写入失败: {ex}')
            try:
                if os.path.isfile(cache_path + '.tmp'):
                    os.remove(cache_path + '.tmp')
            except OSError:
                pass

        return jsonify({
            'success': True,
            'geojson': geojson,
            'from_cache': False,
            'edge_count': len(geojson.get('features') or []),
            'city': city_name,
            'mode': net_key,
        })
    except FileNotFoundError as e:
        return jsonify({'success': False, 'message': str(e)}), 404
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'message': f'获取路网失败: {str(e)}',
        }), 500

@app.route('/api/get_nanjing_boundary', methods=['GET'])
def get_nanjing_boundary():
    """获取城市行政边界GeoJSON"""
    try:
        import geopandas as gpd
        from shapely.geometry import mapping

        backend_dir = os.path.dirname(os.path.abspath(__file__))
        boundary_dir = os.path.join(backend_dir, 'Nanjin2018')
        boundary_file = os.path.join(boundary_dir, '南京行政边界.shp')

        if not os.path.exists(boundary_file):
            possible_files = ['南京行政边界.shp', '南京市2018.shp', '南京市分区边界图.shp']
            found = False
            for f in possible_files:
                path = os.path.join(boundary_dir, f)
                if os.path.exists(path):
                    boundary_file = path
                    found = True
                    break

            if not found:
                return jsonify({
                    'success': False,
                    'message': '未找到南京行政边界文件'
                }), 404

        gdf = gpd.read_file(boundary_file)

        geojson_data = {
            'type': 'FeatureCollection',
            'features': []
        }

        for _, row in gdf.iterrows():
            feature = {
                'type': 'Feature',
                'geometry': mapping(row.geometry),
                'properties': {}
            }
            geojson_data['features'].append(feature)

        return jsonify({
            'success': True,
            'geojson': geojson_data
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': f'获取南京行政边界失败: {str(e)}'
        }), 500

@app.route('/api/check_working_directory', methods=['GET'])
def check_working_directory():
    """检查当前工作目录和文件路径"""
    import os
    cwd = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    city = request.args.get('city', '南京')
    city_prefix = get_city_prefix(city)
    poi_files = {}
    for category in ['work', 'education', 'shopping', 'errand', 'leisure']:
        filename = f"{city_prefix}_{category}_poi.geojson"
        path_from_cwd = os.path.join(cwd, filename)
        path_from_script = os.path.join(get_city_poi_dir(city), filename)

        poi_files[category] = {
            'filename': filename,
            'path_from_cwd': path_from_cwd,
            'exists_from_cwd': os.path.exists(path_from_cwd),
            'path_from_script': path_from_script,
            'exists_from_script': os.path.exists(path_from_script)
        }

    return jsonify({
        'success': True,
        'current_working_directory': cwd,
        'script_directory': script_dir,
        'poi_files': poi_files
    })

@app.route('/api/get_residential_locations', methods=['GET'])
def get_residential_locations():
    try:
        from Data_loader import load_residential_locations_from_excel
        from shapely.geometry import mapping
        import json

        backend_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(backend_dir)

        district_name = request.args.get('district', None)
        limit = request.args.get('limit', type=int)  # 可选：限制返回的数量

        locations = load_residential_locations_from_excel(os.path.join(RESIDENTIAL_DIR, 'Residential_location1000.xlsx'))

        if district_name and district_name != '全部区域':
            try:
                from AccessibilityAnalysis import read_all_city_data
                city = request.args.get('city', '南京')
                _, _, _, _, _, admin_boundary, _ = read_all_city_data(city=city)

                possible_fields = ['县', 'name', 'NAME', 'Name', '名称', 'district', 'DISTRICT', 'District',
                                  'QHMC', 'qhmc', 'xzqmc', 'XZQMC', '区县', '区', '市县']

                matched = None
                for field in possible_fields:
                    if field in admin_boundary.columns:
                        matched = admin_boundary[admin_boundary[field] == district_name]
                        if not matched.empty:
                            break

                if matched is not None and not matched.empty:
                    district_geom = matched.iloc[0].geometry
                    filtered_locations = []
                    for loc in locations:
                        if district_geom.contains(loc):
                            filtered_locations.append(loc)
                    locations = filtered_locations
            except Exception as e:
                print(f"区域过滤失败: {e}")
                pass

        if limit and len(locations) > limit:
            import random
            locations = random.sample(locations, limit)

        features = []
        print(f"开始转换 {len(locations)} 个居住地点为GeoJSON格式...")
        for idx, loc in enumerate(locations):
            try:
                if not hasattr(loc, 'x') or not hasattr(loc, 'y'):
                    continue

                lon = float(loc.x)
                lat = float(loc.y)
                if not (-180 <= lon <= 180) or not (-90 <= lat <= 90):
                    continue

                feature = {
                    'type': 'Feature',
                    'geometry': mapping(loc),
                    'properties': {
                        'id': idx,
                        'longitude': lon,
                        'latitude': lat
                    }
                }
                features.append(feature)
            except Exception as e:
                if idx < 10:  # 只打印前10个错误
                    print(f"转换居住地点 {idx} 失败: {e}")
                continue

        print(f"成功转换 {len(features)} 个居住地点")

        if len(features) == 0:
            print("警告: 没有有效的居住地点数据")
            return jsonify({
                'success': False,
                'error': '没有有效的居住地点数据',
                'message': '未能从数据中提取有效的居住地点坐标',
                'total_locations': len(locations)
            }), 404

        return jsonify({
            'success': True,
            'type': 'FeatureCollection',
            'features': features,
            'count': len(features)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'message': f'获取居住地点数据失败: {str(e)}'
        }), 500

@app.route('/api/check_data_status', methods=['GET'])
def check_data_status():
    """检查POI和路网数据加载状态"""
    try:
        from AccessibilityAnalysis import read_all_city_data
        from carbon_budget import map_poi_categories
        city = request.args.get('city', '南京')

        networks, poi_data, population_data, pop_transform, nodes_with_population, admin_boundary, _ = read_all_city_data(city=city)

        network_status = {}
        for network_type in ['drive', 'walk', 'bike']:
            if networks.get(network_type) is not None:
                network_status[network_type] = {
                    'loaded': True,
                    'nodes_count': len(networks[network_type].nodes),
                    'edges_count': len(networks[network_type].edges)
                }
            else:
                network_status[network_type] = {
                    'loaded': False,
                    'nodes_count': 0,
                    'edges_count': 0
                }

        poi_status = {}
        for category in ['work', 'education', 'shopping', 'errand', 'leisure']:
            if category in poi_data and not poi_data[category].empty:
                poi_status[category] = {
                    'loaded': True,
                    'count': len(poi_data[category])
                }
            else:
                poi_status[category] = {
                    'loaded': False,
                    'count': 0
                }

        mapped_poi_data = map_poi_categories(poi_data)
        mapped_poi_status = {}
        for category in ['work', 'medical', 'shopping', 'catering', 'education']:
            if category in mapped_poi_data and not mapped_poi_data[category].empty:
                mapped_poi_status[category] = {
                    'loaded': True,
                    'count': len(mapped_poi_data[category])
                }
            else:
                mapped_poi_status[category] = {
                    'loaded': False,
                    'count': 0
                }

        other_data_status = {
            'population_raster': population_data is not None,
            'nodes_with_population': {
                'loaded': not nodes_with_population.empty,
                'count': len(nodes_with_population) if not nodes_with_population.empty else 0
            },
            'admin_boundary': {
                'loaded': not admin_boundary.empty,
                'count': len(admin_boundary) if not admin_boundary.empty else 0
            }
        }

        return jsonify({
            'success': True,
            'networks': network_status,
            'poi_original': poi_status,
            'poi_mapped': mapped_poi_status,
            'other_data': other_data_status,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': '检查数据状态时发生错误'
        }), 500

def _carbon_budget_travel_time_to_radius_m(travel_time_min, transport_mode=None):
    """出行时间 × 交通方式速度 → G2SFCA 距离阈值 d0（米），与基于位置模块一致。"""
    from transport_modes import speed_ms
    tt = int(float(travel_time_min))
    tt = max(5, min(tt, 180))
    mode = str(transport_mode or CARBON_BUDGET_TRANSPORT_MODE_DEFAULT).strip() or CARBON_BUDGET_TRANSPORT_MODE_DEFAULT
    radius_m = tt * 60.0 * speed_ms(mode)
    return tt, mode, max(500.0, min(float(radius_m), 100000.0))

def _touch_carbon_budget_progress(**fields):
    carbon_budget_progress.update(fields)
    carbon_budget_progress['last_progress_at'] = time.time()

def _carbon_budget_job_is_stale():
    if not carbon_budget_progress.get('is_running'):
        return False
    if carbon_budget_progress.get('error'):
        return True
    if float(carbon_budget_progress.get('progress') or 0) >= 100:
        return True
    last = carbon_budget_progress.get('last_progress_at')
    if last is None:
        return True
    return (time.time() - float(last)) > CARBON_BUDGET_STALE_SECONDS

def _clear_stale_carbon_budget_job_if_needed():
    if _carbon_budget_job_is_stale():
        _touch_carbon_budget_progress(
            is_running=False,
            status='上次计算任务已结束或超时，可重新计算',
        )
        return True
    return False

def _ensure_carbon_budget_for_travel(city='南京', max_wait_s=45):
    """出行推荐前尽量获取 ICB 栅格；超时则不阻塞推荐（后台继续算）。"""
    import time

    def _ready():
        return (
            carbon_budget_results.get('icb_raster') is not None
            and carbon_budget_results.get('transform') is not None
        )

    if _ready():
        return True, False

    auto_started = False
    if not carbon_budget_progress.get('is_running'):
        travel_time_min, transport_mode, search_radius_m = _carbon_budget_travel_time_to_radius_m(
            CARBON_BUDGET_TRAVEL_TIME_DEFAULT,
            CARBON_BUDGET_TRANSPORT_MODE_DEFAULT,
        )
        city_name = str(city or '南京').strip() or '南京'
        print(f'[出行推荐] 碳预算栅格未就绪，后台启动计算（{city_name}）…')
        thread = threading.Thread(
            target=run_carbon_budget_calculation_background,
            kwargs={
                'rcb_tons': CARBON_DEFAULT_RCB_TONS_YEAR,
                'resolution_m': 1000,
                'selected_district': '全部区域',
                'city': city_name,
                'search_radius_m': search_radius_m,
                'travel_time_min': travel_time_min,
                'transport_mode': transport_mode,
            },
            daemon=True,
        )
        thread.start()
        auto_started = True
    else:
        print('[出行推荐] 碳预算计算进行中，短暂等待 ICB 栅格…')

    deadline = time.time() + float(max_wait_s)
    while time.time() < deadline:
        if _ready():
            return True, auto_started
        if not carbon_budget_progress.get('is_running') and not _ready():
            break
        time.sleep(0.5)

    if _ready():
        return True, auto_started
    print('[出行推荐] 碳预算尚未就绪，将在无 ICB 配额前提下继续推荐')
    return False, auto_started

def run_carbon_budget_calculation_background(
    rcb_tons=None,
    resolution_m=1000,
    selected_district=None,
    city='南京',
    search_radius_m=None,
    travel_time_min=None,
    transport_mode=None,
):
    """在后台线程中运行碳预算热力图计算（网格化，基于多活动可达性）"""
    if rcb_tons is None:
        rcb_tons = CARBON_DEFAULT_RCB_TONS_YEAR
    try:
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(backend_dir)
        print(f"[碳预算热力图] 工作目录: {os.getcwd()}")

        from city_engine import get_shared_calculator, get_city_dataset, configure_calc_district
        from shapely.geometry import Point
        import geopandas as gpd
        import rasterio

        _touch_carbon_budget_progress(
            is_running=True, progress=2,
            status='正在加载城市数据', error=None
        )

        dataset = get_city_dataset(city=city)
        if not dataset:
            raise ValueError("城市数据加载失败")
        _, poi_data, population_data, pop_transform, nodes_with_pop, admin_boundary, _ = dataset

        valid_poi_count = sum(
            1 for gdf in poi_data.values()
            if gdf is not None and hasattr(gdf, 'empty') and not gdf.empty and len(gdf) > 0
        )
        if valid_poi_count == 0:
            raise ValueError("所有POI数据都为空！请检查POI数据文件。")
        print(f"[碳预算热力图] POI 类型有效: {valid_poi_count}/{len(poi_data)}")

        _touch_carbon_budget_progress(progress=10, status='正在初始化碳预算计算器')

        calc = get_shared_calculator(city=city)
        if calc is None:
            raise ValueError("碳预算计算器初始化失败")
        configure_calc_district(calc, selected_district)

        def _report(pct, status):
            _touch_carbon_budget_progress(progress=pct, status=status)

        result = calc.calculate_carbon_budget_heatmap(
            rcb_tons=rcb_tons,
            population_data=population_data,
            pop_transform=pop_transform,
            resolution_m=resolution_m,
            progress_callback=_report,
            search_radius_m=search_radius_m,
            travel_time_min=travel_time_min,
            transport_mode=transport_mode,
        )

        carbon_budget_results.update({
            'icb_raster': result['icb_raster'],
            'acc_raster': result['acc_raster'],
            'min_cb_raster': result.get('min_cb_raster'),
            'grid_inside_mask': result.get('grid_inside_mask'),
            'selected_district': selected_district or '全部区域',
            'city': city or '南京',
            'transform': result['transform'],
            'stats': result['stats'],
            'resolution': resolution_m,
            'search_radius_m': float(search_radius_m) if search_radius_m is not None else None,
            'travel_time_min': int(travel_time_min) if travel_time_min is not None else None,
            'transport_mode': transport_mode,
            'last_calculated': datetime.now().isoformat(),
        })

        _touch_carbon_budget_progress(progress=100, status='碳预算热力图计算完成')
        time.sleep(1)
        _touch_carbon_budget_progress(is_running=False)

    except Exception as e:
        _touch_carbon_budget_progress(
            is_running=False, status='计算失败', error=str(e)
        )
        print(f"碳预算热力图计算过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

@app.route('/api/calculate_carbon_budget', methods=['POST'])
def calculate_carbon_budget():
    try:
        _clear_stale_carbon_budget_job_if_needed()
        if carbon_budget_progress['is_running']:
            return jsonify({
                'success': False,
                'message': '已有计算任务在运行，请等待完成后再试',
                'progress': carbon_budget_progress['progress'],
                'status': carbon_budget_progress['status'],
            })

        params = request.json or {}
        rcb_tons = float(params.get('rcb_tons', CARBON_DEFAULT_RCB_TONS_YEAR))
        resolution_m = int(params.get('resolution_m', 1000))
        selected_district = params.get('district', '全部区域')
        city = params.get('city', '南京')
        # 出行时间（分钟）→ d0（米），与基于位置默认同参数；兼容旧版 search_radius_m
        travel_time_min = params.get('travel_time')
        transport_mode = params.get('transport_mode', CARBON_BUDGET_TRANSPORT_MODE_DEFAULT)
        if travel_time_min is not None:
            travel_time_min, transport_mode, search_radius_m = _carbon_budget_travel_time_to_radius_m(
                travel_time_min, transport_mode
            )
        elif params.get('search_radius_m') is not None:
            search_radius_m = float(params.get('search_radius_m', 10000.0))
            search_radius_m = max(500.0, min(search_radius_m, 100000.0))
            from transport_modes import speed_ms
            transport_mode = str(transport_mode or CARBON_BUDGET_TRANSPORT_MODE_DEFAULT).strip() or CARBON_BUDGET_TRANSPORT_MODE_DEFAULT
            travel_time_min = max(
                5,
                int(round(search_radius_m / (60.0 * speed_ms(transport_mode))))
            )
        else:
            travel_time_min, transport_mode, search_radius_m = _carbon_budget_travel_time_to_radius_m(
                CARBON_BUDGET_TRAVEL_TIME_DEFAULT, CARBON_BUDGET_TRANSPORT_MODE_DEFAULT
            )
        _touch_carbon_budget_progress(
            is_running=False, progress=0,
            status='等待计算', error=None
        )

        thread = threading.Thread(
            target=run_carbon_budget_calculation_background,
            args=(
                rcb_tons, resolution_m, selected_district, city,
                search_radius_m, travel_time_min, transport_mode,
            )
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            'success': True,
            'message': f'碳预算热力图计算已开始 (区域: {selected_district})',
            'progress_endpoint': '/api/get_carbon_budget_progress',
            'district': selected_district,
            'rcb_tons': rcb_tons,
            'resolution_m': resolution_m,
            'search_radius_m': search_radius_m,
            'travel_time_min': travel_time_min,
            'transport_mode': transport_mode,
        })

    except Exception as e:
        return jsonify({
            'success': False, 'error': str(e),
            'message': '启动碳预算计算时发生错误'
        }), 500

@app.route('/api/get_carbon_budget_progress', methods=['GET'])
def get_carbon_budget_progress():
    return jsonify({
        'success': True,
        'progress': carbon_budget_progress['progress'],
        'status': carbon_budget_progress['status'],
        'is_running': carbon_budget_progress['is_running'],
        'error': carbon_budget_progress['error']
    })

@app.route('/api/ensure_carbon_budget_for_travel', methods=['POST'])
def ensure_carbon_budget_for_travel_api():
    """出行推荐前确保 ICB 栅格可用：复用后端碳预算缓存，必要时后台自动计算。"""
    try:
        data = request.get_json(silent=True) or {}
        city = str(data.get('city') or '南京').strip() or '南京'
        max_wait_s = data.get('max_wait_s', 45)
        try:
            max_wait_s = float(max_wait_s)
        except (TypeError, ValueError):
            max_wait_s = 45.0
        max_wait_s = max(5.0, min(max_wait_s, 120.0))

        ready, auto_prepared = _ensure_carbon_budget_for_travel(
            city=city,
            max_wait_s=max_wait_s,
        )
        last_calculated = carbon_budget_results.get('last_calculated')
        return jsonify({
            'success': True,
            'ready': bool(ready),
            'auto_prepared': bool(auto_prepared),
            'city': city,
            'last_calculated': last_calculated,
            'message': '碳预算栅格已就绪' if ready else '碳预算尚未就绪，请稍后在碳预算分析中运行计算',
        }), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'ready': False,
            'message': str(e),
            'error': str(e),
        }), 500

@app.route('/api/get_carbon_budget_results', methods=['GET'])
def get_carbon_budget_results():
    try:
        if carbon_budget_results['icb_raster'] is None:
            return jsonify({
                'success': False,
                'message': '请先运行碳预算计算'
            }), 400

        icb = carbon_budget_results['icb_raster']
        acc = carbon_budget_results['acc_raster']
        min_cb = carbon_budget_results.get('min_cb_raster')
        transform = carbon_budget_results.get('transform')

        def _raster_to_list(raster):
            flat = raster.flatten()
            return [_safe_json(v) for v in flat]

        transform_data = None
        bounds_data = None
        if transform is not None:
            transform_data = {
                'a': float(transform.a), 'b': float(transform.b),
                'c': float(transform.c), 'd': float(transform.d),
                'e': float(transform.e), 'f': float(transform.f),
                'width': int(icb.shape[1]),
                'height': int(icb.shape[0]),
            }
            bounds_data = {
                'west':  float(transform.c),
                'east':  float(transform.c + transform.a * icb.shape[1]),
                'south': float(transform.e * icb.shape[0] + transform.f),
                'north': float(transform.f),
            }

        stats = carbon_budget_results.get('stats', {})
        stats_clean = {}
        for key, val in stats.items():
            if isinstance(val, dict):
                stats_clean[key] = {k: _safe_json(v) for k, v in val.items()}
            else:
                stats_clean[key] = _safe_json(val)

        payload = {
            'success': True,
            'icb_raster': _raster_to_list(icb),
            'acc_raster': _raster_to_list(acc),
            'min_cb_raster': _raster_to_list(min_cb) if min_cb is not None else None,
            'transform': transform_data,
            'bounds': bounds_data,
            'stats': stats_clean,
            'resolution': carbon_budget_results.get('resolution'),
            'last_calculated': carbon_budget_results.get('last_calculated'),
        }
        return jsonify(payload)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': '获取碳预算热力图结果失败'
        }), 500

@app.route('/api/get_carbon_budget_heatmap_image', methods=['GET'])
def get_carbon_budget_heatmap_image():
    """获取碳预算热力图PNG图像（透明背景，供 Leaflet L.imageOverlay 使用）
    同时返回栅格数据用于点击交互"""
    try:
        if carbon_budget_results['icb_raster'] is None:
            return jsonify({'success': False, 'message': '请先运行碳预算计算'}), 400

        mode = request.args.get('mode', 'icb')  # 'icb' | 'acc' | 'min_cb'
        mode_map = {'icb': 'icb_raster', 'acc': 'acc_raster', 'min_cb': 'min_cb_raster'}
        raster_key = mode_map.get(mode, 'icb_raster')
        raster = carbon_budget_results.get(raster_key)
        if raster is None:
            return jsonify({
                'success': False,
                'message': f'暂无 {mode} 图层数据，请先完成碳预算计算'
            }), 400
        transform = carbon_budget_results.get('transform')

        if transform is None:
            return jsonify({'success': False, 'message': '坐标变换信息缺失'}), 400

        bounds = {
            'west':  float(transform.c),
            'east':  float(transform.c + transform.a * raster.shape[1]),
            'south': float(transform.e * raster.shape[0] + transform.f),
            'north': float(transform.f),
        }
        extent = [bounds['west'], bounds['east'], bounds['south'], bounds['north']]

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 8), dpi=150, facecolor='none')
        ax = fig.add_axes([0, 0, 1, 1])
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        ax.set_axis_off()

        valid = raster[~np.isnan(raster)]
        if len(valid) > 0:
            vmin_data, vmax_data = float(np.min(valid)), float(np.max(valid))
        else:
            vmin_data, vmax_data = 0.0, 1.0

        from matplotlib.colors import ListedColormap, BoundaryNorm

        color_cfg = ACCESSIBILITY_COLORS.get('work', next(iter(ACCESSIBILITY_COLORS.values())))
        colors_nine = list(color_cfg['colors'])

        if mode == 'acc':
            acc_color_cfg = compute_accessibility_color_thresholds(raster, colors_nine, num_levels=9)
            colors_list = acc_color_cfg['colors']
            display_bounds = acc_color_cfg['display_bounds']
            percentile_thresholds = acc_color_cfg['thresholds']
            heatmap_color_config = {**acc_color_cfg, 'mode': mode}
        else:
            colors_list = colors_nine
            percentile_thresholds = compute_quantile_thresholds(raster, num_levels=9)
            if len(valid) > 0:
                data_max = float(np.max(valid))
                last_t = percentile_thresholds[-1] if percentile_thresholds else 0.0
                upper = max(data_max, last_t)
                if upper <= last_t:
                    upper = last_t + max(1e-6, abs(last_t) * 1e-9 + 1e-6)
                else:
                    upper = upper + max(1e-9, abs(upper) * 1e-12)
            else:
                upper = (percentile_thresholds[-1] + 1.0) if percentile_thresholds else 1.0
            display_bounds = percentile_thresholds + [upper]
            heatmap_color_config = {
                'thresholds': percentile_thresholds,
                'display_bounds': display_bounds,
                'colors': colors_list,
                'mode': mode,
            }

        cmap = ListedColormap(colors_list)
        norm = BoundaryNorm(display_bounds, len(colors_list))

        if mode == 'min_cb':
            stats = carbon_budget_results.get('stats') or {}
            selected_district = (
                carbon_budget_results.get('selected_district')
                or stats.get('selected_district')
                or '全部区域'
            )
            city = carbon_budget_results.get('city') or '南京'
            inside_mask = _build_carbon_budget_inside_mask(
                raster.shape, transform, selected_district, city,
            )
            rgba = _render_min_cb_rgba(raster, inside_mask, cmap, norm)
            ax.imshow(
                rgba, extent=extent,
                origin='lower', interpolation='nearest',
            )
        else:
            ax.imshow(
                raster, extent=extent, cmap=cmap,
                norm=norm,
                origin='lower', interpolation='nearest',
            )

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    pad_inches=0, transparent=True)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)

        flat_raster = raster.flatten()
        flat_list = [_safe_json(v) for v in flat_raster]

        return jsonify({
            'success': True,
            'mode': mode,
            'image_data': img_base64,
            'bounds': bounds,
            'width': int(raster.shape[1]),
            'height': int(raster.shape[0]),
            'raster_data': flat_list,
            'vmin': vmin_data,
            'vmax': vmax_data,
            'stats': _safe_json_stats(carbon_budget_results.get('stats', {})),
            'transform': {
                'a': float(transform.a), 'b': float(transform.b),
                'c': float(transform.c), 'd': float(transform.d),
                'e': float(transform.e), 'f': float(transform.f),
            },
            'color_config': heatmap_color_config
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'message': '生成热力图图像失败'}), 500


def _call_view_json(view_fn, query_string=''):
    """在请求上下文中调用返回 jsonify 的视图，解析 JSON 与状态码。"""
    path = f'/?{query_string}' if query_string else '/'
    with app.test_request_context(path):
        result = view_fn()
    if isinstance(result, tuple):
        resp = result[0]
        status = int(result[1]) if len(result) > 1 else 200
    else:
        resp = result
        status = 200
    if hasattr(resp, 'get_json'):
        data = resp.get_json(silent=True)
    else:
        data = resp
    return data or {}, status


def _heatmap_bounds_from_transform(transform, height, width):
    """从 Affine / GDAL 六参数得到 west/east/south/north（与 origin=lower 一致）。"""
    if transform is None:
        return None
    try:
        a = float(transform.a)
        c = float(transform.c)
        e = float(transform.e)
        f = float(transform.f)
    except AttributeError:
        a = float(transform[0])
        c = float(transform[2])
        e = float(transform[4])
        f = float(transform[5])
    return {
        'west': c,
        'east': c + a * width,
        'north': f,
        'south': f + e * height,
    }


def _resolve_heatmap_export_raster(layer, mode):
    """
    解析当前可导出的热力栅格。
    返回 (raster_2d, bounds_dict, label) 或 (None, None, error_message)。
    """
    layer = (layer or '').strip().lower()
    mode = (mode or '').strip().lower()

    if layer == 'accessibility':
        raster = calculation_results.get('raster_data')
        transform = calculation_results.get('transform')
        if raster is None or transform is None:
            return None, None, '请先完成可达性计算'
        h, w = raster.shape
        bounds = _heatmap_bounds_from_transform(transform, h, w)
        return np.asarray(raster, dtype=np.float64), bounds, 'accessibility'

    if layer == 'carbon':
        mode_map = {'icb': 'icb_raster', 'acc': 'acc_raster', 'min_cb': 'min_cb_raster'}
        if mode not in mode_map:
            mode = 'icb'
        raster = carbon_budget_results.get(mode_map[mode])
        transform = carbon_budget_results.get('transform')
        if raster is None or transform is None:
            return None, None, '请先完成碳预算计算'
        h, w = raster.shape
        bounds = _heatmap_bounds_from_transform(transform, h, w)
        return np.asarray(raster, dtype=np.float64), bounds, f'carbon_{mode}'

    if layer == 'equity':
        if mode in ('moran',):
            mode = 'significance'
        if mode not in ('significance', 'cluster', 'gdp', 'population'):
            mode = 'cluster'
        key_map = {
            # 显著性视图与前端一致，导出 P 值栅格
            'significance': 'pvalue_raster',
            'cluster': 'cluster_raster',
            'gdp': 'gwr_beta_gdp_raster',
            'population': 'gwr_beta_pop_raster',
        }
        if equity_analysis_results.get('moran_raster') is None:
            return None, None, '请先运行公平性分析'
        raster = equity_analysis_results.get(key_map[mode])
        bounds = equity_analysis_results.get('bounds')
        if raster is None or bounds is None:
            return None, None, '公平性图层数据不可用，请重新分析'
        return np.asarray(raster, dtype=np.float64), {
            'west': float(bounds['west']),
            'east': float(bounds['east']),
            'south': float(bounds['south']),
            'north': float(bounds['north']),
        }, f'equity_{mode}'

    return None, None, f'不支持的图层类型: {layer}'


def _raster_to_shapefile_zip(raster, bounds, layer_label='heatmap', max_features=80000):
    """
    将栅格有效像元导出为多边形 Shapefile（ZIP）。
    坐标系 WGS84；像元过多时按步长抽样以保证可下载体积。
    """
    import tempfile
    import zipfile
    import shutil
    from shapely.geometry import box
    import geopandas as gpd

    arr = np.asarray(raster, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError('栅格必须为二维数组')
    h, w = arr.shape
    west = float(bounds['west'])
    east = float(bounds['east'])
    south = float(bounds['south'])
    north = float(bounds['north'])
    cell_w = (east - west) / float(w)
    cell_h = (north - south) / float(h)

    valid = np.isfinite(arr)
    rows, cols = np.where(valid)
    n = int(rows.size)
    if n == 0:
        raise ValueError('当前热力图没有可导出的有效像元')

    stride = max(1, int(math.ceil(n / float(max_features))))
    if stride > 1:
        rows = rows[::stride]
        cols = cols[::stride]

    geoms = []
    values = []
    row_ids = []
    col_ids = []
    for i, j in zip(rows.tolist(), cols.tolist()):
        x0 = west + j * cell_w
        y0 = south + i * cell_h
        geoms.append(box(x0, y0, x0 + cell_w, y0 + cell_h))
        values.append(float(arr[i, j]))
        row_ids.append(int(i))
        col_ids.append(int(j))

    gdf = gpd.GeoDataFrame(
        {
            'value': values,
            'row': row_ids,
            'col': col_ids,
        },
        geometry=geoms,
        crs='EPSG:4326',
    )

    safe_label = ''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in (layer_label or 'heatmap'))
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f'{safe_label}_{stamp}'

    tmp_dir = tempfile.mkdtemp(prefix='heatmap_shp_')
    try:
        shp_path = os.path.join(tmp_dir, f'{base_name}.shp')
        gdf.to_file(shp_path, driver='ESRI Shapefile', encoding='utf-8')
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            for fname in os.listdir(tmp_dir):
                fpath = os.path.join(tmp_dir, fname)
                if os.path.isfile(fpath):
                    zf.write(fpath, arcname=fname)
        zip_buf.seek(0)
        return zip_buf, f'{base_name}.zip'
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _export_fmt_threshold_value(value):
    """与前端 formatThresholdValue 一致：最多保留两位小数。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return '—'
    if not np.isfinite(v):
        return '—'
    rounded = round(v * 100) / 100.0
    if abs(rounded) >= 100:
        if abs(rounded - round(rounded)) < 1e-9:
            return f'{int(round(rounded)):,}'
        return f'{rounded:,.2f}'
    if abs(rounded - round(rounded)) < 1e-9:
        return str(int(round(rounded)))
    return f'{rounded:.2f}'


def _export_fmt_threshold_range(lo, hi):
    a = _export_fmt_threshold_value(lo)
    b = _export_fmt_threshold_value(hi)
    return a if a == b else f'{a}–{b}'


def _build_export_legend_spec(layer, mode, data):
    """
    根据图层与热力图 API 返回数据，构建导出图例描述。
    返回 dict: {title, items:[{color,label}], gradient: optional}
    """
    layer = (layer or '').strip().lower()
    mode = (mode or '').strip().lower()

    if layer == 'accessibility':
        cfg = data.get('threshold_config') or {}
        colors = list(cfg.get('colors') or [
            '#313695', '#4575b4', '#74add1', '#abd9e9', '#ffffbf',
            '#fee090', '#fdae61', '#f46d43', '#d73027',
        ])
        items = []
        bounds = cfg.get('display_bounds') or []
        has_zero = bool(cfg.get('has_zero_bucket')) and len(bounds) >= len(colors) + 1
        if has_zero:
            for i, color in enumerate(colors):
                lo, hi = bounds[i], bounds[i + 1]
                label = '0' if i == 0 else _export_fmt_threshold_range(lo, hi)
                items.append({'color': color, 'label': label})
        else:
            thresholds = list(cfg.get('thresholds') or [])
            data_max = cfg.get('data_max')
            vmax = cfg.get('vmax')
            if not thresholds:
                thresholds = [0, 0.5, 1, 2, 5, 10, 20, 50, 100]
            for i, lo in enumerate(thresholds):
                if i < len(thresholds) - 1:
                    label = _export_fmt_threshold_range(lo, thresholds[i + 1])
                elif data_max is not None and float(data_max) > float(lo) + 1e-9:
                    label = _export_fmt_threshold_range(lo, data_max)
                elif vmax is not None and float(vmax) > float(lo) + 1e-9:
                    label = _export_fmt_threshold_range(lo, vmax)
                else:
                    label = f'{_export_fmt_threshold_value(lo)}+'
                color = colors[i] if i < len(colors) else colors[-1]
                items.append({'color': color, 'label': label})
        return {'title': 'Accessibility Score', 'items': items}

    if layer == 'carbon':
        if mode not in ('icb', 'acc', 'min_cb'):
            mode = 'icb'
        title_map = {
            'icb': 'ICB (g/person·day)',
            'acc': 'Multi-activity Accessibility',
            'min_cb': 'Minimum Carbon Budget',
        }
        title = title_map[mode]
        cfg = data.get('color_config') or {}
        colors = list(cfg.get('colors') or [])
        bounds = list(cfg.get('display_bounds') or [])
        items = []
        if colors and len(bounds) >= len(colors) + 1:
            for i, color in enumerate(colors):
                lo, hi = bounds[i], bounds[i + 1]
                if cfg.get('has_zero_bucket') and i == 0:
                    label = '0'
                elif mode == 'acc':
                    label = _export_fmt_threshold_range(lo, hi)
                elif mode == 'icb':
                    label = f'{_export_fmt_threshold_value(lo)}–{_export_fmt_threshold_value(hi)}'
                else:
                    label = f'{_export_fmt_threshold_value(lo)}–{_export_fmt_threshold_value(hi)} g'
                items.append({'color': color, 'label': label})
        stats = data.get('stats') or {}
        if mode == 'min_cb':
            rate = stats.get('min_cb_compliance_rate')
            total = stats.get('min_cb_demand_cells', stats.get('valid_cells'))
            ok = stats.get('min_cb_computable_cells')
            if rate is not None and total and ok is not None and ok < total:
                items.append({
                    'color': 'rgba(210, 210, 210, 0.65)',
                    'label': 'Time budget not met',
                })
        if not items:
            items = [{'color': '#cccccc', 'label': 'No classed data'}]
        return {'title': title, 'items': items}

    if layer == 'equity':
        if mode in ('moran',):
            mode = 'significance'
        if mode == 'cluster':
            return {
                'title': 'LISA Cluster',
                'items': [
                    {'color': 'rgb(33,113,181)', 'label': 'Low-Low (LL)'},
                    {'color': 'rgb(146,197,222)', 'label': 'Low-High (LH)'},
                    {'color': 'rgb(244,165,130)', 'label': 'High-Low (HL)'},
                    {'color': 'rgb(202,0,32)', 'label': 'High-High (HH)'},
                ],
            }
        if mode == 'significance':
            return {
                'title': 'LISA Significance (P-value)',
                'items': [],
                'gradient': {
                    'colors': ['rgb(165,0,38)', 'rgb(244,109,67)', 'rgb(254,224,144)', 'rgba(200,200,200,0.3)'],
                    'labels': ['P≈0 significant', 'P=0.05', 'P≥0.2 not significant'],
                },
            }
        title = 'GWR · GDP coefficient β₁' if mode == 'gdp' else 'GWR · Population density β₂'
        return {
            'title': title,
            'items': [],
            'gradient': {
                'colors': ['rgb(49,54,149)', 'rgb(255,255,255)', 'rgb(215,48,39)'],
                'labels': ['Negative', '0', 'Positive'],
            },
        }

    return {'title': 'Legend', 'items': []}


def _parse_css_color(color):
    """解析 #rgb / #rrggbb / rgb() / rgba() 为 (r,g,b,a) 0–255。"""
    if color is None:
        return (204, 204, 204, 255)
    s = str(color).strip()
    if s.startswith('#') and len(s) in (4, 7):
        if len(s) == 4:
            r, g, b = (int(s[i] * 2, 16) for i in range(1, 4))
        else:
            r, g, b = int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16)
        return (r, g, b, 255)
    if s.startswith('rgba'):
        nums = s[s.find('(') + 1:s.find(')')].split(',')
        r, g, b = [int(float(x)) for x in nums[:3]]
        a = float(nums[3]) if len(nums) > 3 else 1.0
        return (r, g, b, int(max(0, min(255, round(a * 255)))))
    if s.startswith('rgb'):
        nums = s[s.find('(') + 1:s.find(')')].split(',')
        r, g, b = [int(float(x)) for x in nums[:3]]
        return (r, g, b, 255)
    return (204, 204, 204, 255)


def _load_export_cjk_truetype(size=16):
    """加载系统中文字体，供 Pillow 绘制图例。"""
    from PIL import ImageFont
    candidates = [
        r'C:\Windows\Fonts\msyh.ttc',
        r'C:\Windows\Fonts\msyhbd.ttc',
        r'C:\Windows\Fonts\simhei.ttf',
        r'C:\Windows\Fonts\simsun.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/System/Library/Fonts/PingFang.ttc',
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _compose_heatmap_png_with_legend(png_bytes, legend_spec):
    """
    热力图与图例并排：图例在热力图外侧右下角，不与热力图像素重叠。
    布局示意：
        [ 热力图 ............ ]
        [ 热力图 ...... ][图例]
    """
    from PIL import Image, ImageDraw

    heatmap = Image.open(io.BytesIO(png_bytes)).convert('RGBA')
    hw, hh = heatmap.size

    legend = legend_spec or {}
    items = list(legend.get('items') or [])
    gradient = legend.get('gradient')
    title = str(legend.get('title') or '图例')

    scale = max(1.35, min(2.2, min(hw, hh) / 520.0))
    font_title = _load_export_cjk_truetype(max(22, int(26 * scale)))
    font_item = _load_export_cjk_truetype(max(18, int(20 * scale)))

    def _text_width(font, text):
        try:
            return font.getlength(text)
        except Exception:
            bbox = font.getbbox(text)
            return bbox[2] - bbox[0]

    max_label_w = _text_width(font_title, title)
    for it in items:
        max_label_w = max(max_label_w, _text_width(font_item, str(it.get('label') or '')))
    if gradient:
        for lab in (gradient.get('labels') or []):
            max_label_w = max(max_label_w, _text_width(font_item, str(lab)))

    swatch = max(28, int(32 * scale))
    row_h = max(36, int(40 * scale))
    pad_in = max(18, int(20 * scale))
    title_h = max(40, int(44 * scale))
    gap_swatch = max(12, int(14 * scale))

    if gradient:
        bar_h = max(200, int(220 * scale))
        content_h = title_h + bar_h + pad_in
    else:
        content_h = title_h + max(len(items), 1) * row_h + 8

    legend_w = int(pad_in * 2 + swatch + gap_swatch + max_label_w + 24)
    legend_w = max(int(280 * scale), legend_w)
    legend_h = int(content_h + pad_in * 2)
    # 图例过高时压缩行距，但始终放在热力图外侧
    max_legend_h = max(hh, legend_h)
    if legend_h > hh and not gradient and len(items) > 0:
        # 允许图例比热力图更高：画布会增高；若过高则略压缩
        if legend_h > hh * 1.35:
            avail = int(hh * 1.2) - pad_in * 2 - title_h
            row_h = max(30, int(avail / max(len(items), 1)))
            swatch = min(swatch, max(24, row_h - 8))
            legend_h = pad_in * 2 + title_h + len(items) * row_h + 8
    if gradient:
        bar_h = min(bar_h, max(160, hh - title_h - pad_in * 2))
        legend_h = pad_in * 2 + title_h + bar_h + 8

    outer_pad = max(16, int(18 * scale))
    gap = max(14, int(16 * scale))
    canvas_w = outer_pad + hw + gap + legend_w + outer_pad
    canvas_h = outer_pad + max(hh, legend_h) + outer_pad

    # 热力图贴左上（垂直居中于内容区偏上），图例贴右下外侧
    map_x = outer_pad
    map_y = outer_pad
    panel_x = outer_pad + hw + gap
    panel_y = canvas_h - outer_pad - legend_h

    canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
    canvas.paste(heatmap, (map_x, map_y), heatmap)

    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        [panel_x, panel_y, panel_x + legend_w - 1, panel_y + legend_h - 1],
        radius=max(8, int(10 * scale)),
        fill=(255, 255, 255),
        outline=(160, 170, 180),
        width=max(2, int(2 * scale)),
    )

    tx = panel_x + pad_in
    ty = panel_y + pad_in
    draw.text((tx, ty), title, fill=(31, 41, 55), font=font_title)
    ty += title_h

    if gradient:
        colors = [_parse_css_color(c) for c in (gradient.get('colors') or [])]
        labels = list(gradient.get('labels') or [])
        bar_w = max(32, int(36 * scale))
        bar_h = min(bar_h, legend_h - title_h - pad_in * 2)
        bar = Image.new('RGBA', (bar_w, max(bar_h, 1)), (0, 0, 0, 0))
        if len(colors) >= 2 and bar_h > 0:
            for y in range(bar_h):
                t = y / max(bar_h - 1, 1)
                seg = t * (len(colors) - 1)
                i0 = int(seg)
                i1 = min(i0 + 1, len(colors) - 1)
                f = seg - i0
                c0, c1 = colors[i0], colors[i1]
                rgba = tuple(int(c0[k] * (1 - f) + c1[k] * f) for k in range(4))
                for x in range(bar_w):
                    bar.putpixel((x, y), rgba)
        canvas.paste(bar, (tx, ty), bar)
        draw.rectangle(
            [tx, ty, tx + bar_w - 1, ty + bar_h - 1],
            outline=(152, 162, 179),
            width=max(1, int(scale)),
        )
        label_ys = [ty, ty + bar_h // 2, ty + bar_h - 1]
        for i, lab in enumerate(labels[:3]):
            draw.text(
                (tx + bar_w + gap_swatch, label_ys[i] - int(10 * scale)),
                str(lab), fill=(52, 64, 84), font=font_item,
            )
    else:
        for it in items:
            color = _parse_css_color(it.get('color'))
            label = str(it.get('label') or '')
            y0 = ty + max(0, (row_h - swatch) // 2)
            fill_rgb = color[:3]
            if len(color) > 3 and color[3] < 255:
                a = color[3] / 255.0
                fill_rgb = tuple(int(255 * (1 - a) + c * a) for c in fill_rgb)
            draw.rectangle(
                [tx, y0, tx + swatch, y0 + swatch],
                fill=fill_rgb,
                outline=(120, 130, 140),
                width=max(1, int(scale)),
            )
            draw.text(
                (tx + swatch + gap_swatch, ty + max(2, (row_h - int(20 * scale)) // 2)),
                label, fill=(52, 64, 84), font=font_item,
            )
            ty += row_h

    out = io.BytesIO()
    canvas.save(out, format='PNG')
    out.seek(0)
    return out.getvalue()


@app.route('/api/export_heatmap', methods=['GET'])
def export_heatmap():
    """导出当前热力图为 PNG 或 Shapefile（ZIP）。
    Query: layer=accessibility|equity|carbon, mode=..., format=png|shp
    """
    try:
        layer = (request.args.get('layer') or '').strip().lower()
        mode = (request.args.get('mode') or '').strip().lower()
        fmt = (request.args.get('format') or 'png').strip().lower()
        if fmt not in ('png', 'shp', 'shapefile', 'zip'):
            return jsonify({'success': False, 'message': 'format 仅支持 png 或 shp'}), 400
        if fmt in ('shapefile', 'zip'):
            fmt = 'shp'

        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if fmt == 'png':
            if layer == 'accessibility':
                data, status = _call_view_json(get_accessibility_map)
            elif layer == 'equity':
                eq_mode = mode or 'cluster'
                data, status = _call_view_json(
                    get_equity_heatmap_image, f'mode={eq_mode}'
                )
            elif layer == 'carbon':
                cb_mode = mode or 'icb'
                data, status = _call_view_json(
                    get_carbon_budget_heatmap_image, f'mode={cb_mode}'
                )
            else:
                return jsonify({'success': False, 'message': f'不支持的图层: {layer}'}), 400

            if status >= 400 or not data.get('success'):
                return jsonify({
                    'success': False,
                    'message': data.get('message') or data.get('error') or '无法生成热力图图片',
                }), status if status >= 400 else 400

            img_b64 = data.get('image_data')
            if not img_b64:
                return jsonify({'success': False, 'message': '热力图图片数据为空'}), 500
            png_bytes = base64.b64decode(img_b64)
            try:
                legend_spec = _build_export_legend_spec(layer, mode, data)
                png_bytes = _compose_heatmap_png_with_legend(png_bytes, legend_spec)
            except Exception as legend_err:
                # 图例合成失败时仍返回原热力图，并在日志中记录
                print(f'[export_heatmap] 图例合成失败，回退原图: {legend_err}')
            filename = f'heatmap_{layer}_{mode or "default"}_{stamp}.png'
            return send_file(
                io.BytesIO(png_bytes),
                mimetype='image/png',
                as_attachment=True,
                download_name=filename,
            )

        raster, bounds, label = _resolve_heatmap_export_raster(layer, mode)
        if raster is None:
            return jsonify({'success': False, 'message': bounds or label}), 400
        zip_buf, zip_name = _raster_to_shapefile_zip(raster, bounds, layer_label=label)
        return send_file(
            zip_buf,
            mimetype='application/zip',
            as_attachment=True,
            download_name=zip_name,
        )
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': f'导出热力图失败: {e}',
        }), 500


def _search_poi_by_keyword(q, limit=10, city='南京'):
    results = []
    try:
        from Data_loader import load_poi_from_geojson
        import pandas as pd

        backend_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(backend_dir)
        poi_data = load_poi_from_geojson(get_city_poi_dir(city))
        keyword = str(q).strip().lower()
        if not keyword:
            return results

        category_labels = {
            'work': '工作', 'education': '教育', 'medical': '医疗',
            'shopping': '购物', 'catering': '餐饮', 'recreation': '休闲娱乐',
            'transportation_station': '交通站点', 'transport': '交通站点',
        }
        seen = set()

        for category, gdf in poi_data.items():
            if gdf is None or gdf.empty:
                continue
            cat_label = category_labels.get(category, category)
            # 增加更多可能的搜索列（包括中英文）
            search_cols = [
                c for c in [
                    'name', '名称', 'address', '地址', 'type', '类型',
                    'category', '分类', 'adname', '城市', 'cityname', '类别',
                    'PoiID', '中类', '大类', '小类', 'fclass', 'subcategory', '子类别',
                ]
                if c in gdf.columns
            ]
            if not search_cols:
                continue
            mask = pd.Series([False] * len(gdf), index=gdf.index)
            for col in search_cols:
                ser = gdf[col].fillna('').astype(str).str.lower()
                # 搜索关键词中可能包含中文，转换编码后再搜索
                try:
                    mask = mask | ser.str.contains(keyword, na=False, regex=False)
                except:
                    pass
            matched = gdf[mask]
            for _, row in matched.iterrows():
                if len(results) >= limit:
                    return results
                try:
                    geom = row.geometry
                    if geom is None or geom.is_empty:
                        continue
                    lon = float(geom.x)
                    lat = float(geom.y)
                    name = ''
                    for col in ['name', '名称', '名称']:
                        if col in row.index and pd.notna(row[col]):
                            name = str(row[col]).strip()
                            break
                    addr = ''
                    for col in ['address', '地址', '地址']:
                        if col in row.index and pd.notna(row[col]):
                            addr = str(row[col]).strip()
                            break
                    display = name or addr or f'{cat_label} POI'
                    if addr and name and addr != name:
                        display = f'{name} - {addr}'
                    elif not display:
                        display = f'{cat_label} ({lon:.4f}, {lat:.4f})'
                    key = (round(lat, 6), round(lon, 6), display[:50])
                    if key in seen:
                        continue
                    seen.add(key)
                    poi_cat = ''
                    for col in ['中类', '小类', 'type', '类型', 'category', '分类']:
                        if col in row.index and pd.notna(row[col]):
                            poi_cat = str(row[col]).strip()
                            if poi_cat:
                                break
                    results.append({
                        'display_name': display,
                        'lat': lat,
                        'lon': lon,
                        'source': 'poi',
                        'category': cat_label,
                        'poi_category': category,
                        'poi_subtype': poi_cat,
                    })
                except Exception:
                    continue
        return results
    except Exception as e:
        print(f"[geocode] POI 搜索异常: {e}")
        return results

@app.route('/api/geocode', methods=['GET'])
def geocode():
    try:
        q = request.args.get('q', '').strip()
        if not q:
            return jsonify({'success': False, 'error': '请提供查询参数 q'}), 400

        city = request.args.get('city', '南京')
        poi_results = _search_poi_by_keyword(q, limit=15, city=city)
        results = [
            {
                'display_name': r['display_name'],
                'lat': r['lat'],
                'lon': r['lon'],
                'source': r.get('source', 'poi'),
                'category': r.get('category', ''),
                'poi_category': r.get('poi_category', ''),
                'poi_subtype': r.get('poi_subtype', ''),
            }
            for r in poi_results
        ]

        # 天地图搜索（行政区划 / 普通搜索 / 地理编码，与本地 POI 结果合并）
        tdt_key = os.environ.get('TIANDITU_API_KEY', '')
        tdt_error = None
        if tdt_key:
            from tianditu_poi import search_tianditu_geocode_candidates
            tdt_results, tdt_error = search_tianditu_geocode_candidates(q, city=city, limit=5)
            if tdt_results:
                results = tdt_results + results[:10]
        else:
            print("[geocode] 未设置 TIANDITU_API_KEY 环境变量，跳过在线地理编码补充")

        if not results:
            hint_parts = ['未在 POI 中找到匹配，可尝试更具体的关键词']
            if tdt_error:
                hint_parts.append(f'(天地图API: {tdt_error})')
            return jsonify({
                'success': True,
                'results': [],
                'message': ''.join(hint_parts),
            })

        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': '地理编码请求失败'
        }), 500

@app.route('/api/calculate_personal_accessibility', methods=['POST'])
def calculate_personal_accessibility():
    """基于个人的可达性计算API"""
    try:
        payload = request.json or {}
        from PersonalAccessibility import calculate_personal_accessibility as calc_fn
        result = calc_fn(payload)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e),
            'error': str(e)
        }), 500

@app.route('/api/community_equity_report', methods=['POST'])
def community_equity_report():
    """社区生活圈公平性报告：基于可达性栅格评估各品类配套水平。"""
    try:
        req_data = request.get_json() or {}
        lat = req_data.get('lat')
        lng = req_data.get('lng')
        if lat is None or lng is None:
            return jsonify({
                'success': False,
                'message': '请提供位置坐标 lat / lng',
            }), 400

        from PersonalEquity import compute_community_equity_report

        result = compute_community_equity_report(
            lat=float(lat),
            lng=float(lng),
            city=str(req_data.get('city') or '南京').strip(),
        )
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e),
        }), 500

@app.route('/api/personal_fairness_ranking', methods=['POST'])
def personal_fairness_ranking():
    """个人公平性排名：基于 POI 覆盖密度评估生活资源公平性。"""
    try:
        req_data = request.get_json() or {}
        lat = req_data.get('lat')
        lng = req_data.get('lng')
        if lat is None or lng is None:
            return jsonify({
                'success': False,
                'message': '请提供位置坐标 lat / lng',
            }), 400

        from PersonalEquity import compute_personal_fairness_ranking

        result = compute_personal_fairness_ranking(
            lat=float(lat),
            lng=float(lng),
            city=str(req_data.get('city') or '南京').strip(),
            travel_mode=str(req_data.get('travel_mode') or 'bus').strip(),
            time_budget_minutes=int(req_data.get('time_budget_minutes') or 30),
        )
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e),
        }), 500

@app.route('/api/location_comparison', methods=['POST'])
def location_comparison():
    """对比两处位置的个人公平性差异。"""
    try:
        req_data = request.get_json() or {}
        origin = req_data.get('origin') or {}
        target = req_data.get('target') or {}
        if origin.get('lat') is None or origin.get('lng') is None:
            return jsonify({
                'success': False,
                'message': '请提供出发地坐标 origin.lat / origin.lng',
            }), 400
        if target.get('lat') is None or target.get('lng') is None:
            return jsonify({
                'success': False,
                'message': '请提供目标位置坐标 target.lat / target.lng',
            }), 400

        from PersonalEquity import compute_location_comparison

        result = compute_location_comparison(
            origin={'lat': float(origin['lat']), 'lng': float(origin['lng'])},
            target={'lat': float(target['lat']), 'lng': float(target['lng'])},
            city=str(req_data.get('city') or '南京').strip(),
            travel_mode=str(req_data.get('travel_mode') or 'bus').strip(),
            time_budget_minutes=int(req_data.get('time_budget_minutes') or 30),
        )
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e),
        }), 500

@app.route('/api/improvement_suggestions', methods=['POST'])
def improvement_suggestions():
    """生成个人公平性改善建议（出行方式对比等）。"""
    try:
        req_data = request.get_json() or {}
        lat = req_data.get('lat')
        lng = req_data.get('lng')
        if lat is None or lng is None:
            return jsonify({
                'success': False,
                'message': '请提供位置坐标 lat / lng',
            }), 400

        from PersonalEquity import generate_improvement_suggestions

        result = generate_improvement_suggestions(
            lat=float(lat),
            lng=float(lng),
            city=str(req_data.get('city') or '南京').strip(),
            current_travel_mode=str(req_data.get('current_travel_mode') or req_data.get('travel_mode') or 'bus').strip(),
            time_budget_minutes=int(req_data.get('time_budget_minutes') or 30),
        )
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e),
        }), 500

@app.route('/api/travel_activity_poi_candidates', methods=['POST'])
def travel_activity_poi_candidates():
    """按活动类型关键词检索最近若干 POI，供用户选择后再规划路线。"""
    try:
        data = request.get_json(silent=True) or {}
        lat = data.get('lat', data.get('latitude'))
        lng = data.get('lng', data.get('longitude'))
        query = str(data.get('query') or data.get('activity_query') or '').strip()
        if lat is None or lng is None:
            return jsonify({
                'success': False,
                'message': '请提供当前位置 lat / lng',
            }), 400
        if not query:
            return jsonify({
                'success': False,
                'message': '请输入活动类型关键词',
            }), 400

        city = str(data.get('city') or '南京').strip()
        limit = max(1, min(int(data.get('limit') or 5), 10))
        max_distance_km = data.get('max_distance_km')
        if max_distance_km is not None:
            max_distance_km = float(max_distance_km)

        from travel_recommendation import (
            find_candidate_pois,
            _resolve_keywords,
            _poi_search_radius_m,
        )

        keywords, poi_cats = _resolve_keywords(query)
        radius_m = _poi_search_radius_m(max_distance_km)
        candidates = find_candidate_pois(
            float(lng),
            float(lat),
            keywords or [query],
            city=city,
            max_radius_m=radius_m,
            limit=limit,
            poi_categories=poi_cats,
        )
        places = []
        for item in candidates or []:
            places.append({
                'name': item.get('display_name') or item.get('name') or query,
                'lat': item.get('lat'),
                'lng': item.get('lon', item.get('lng')),
                'distance_m': item.get('distance_m'),
                'distance_km': item.get('distance_km'),
                'rating': item.get('rating'),
                'poi_category': item.get('poi_category') or '',
                'poi_subtype': item.get('poi_subtype') or '',
                'address': item.get('address') or '',
            })
        if not places:
            return jsonify({
                'success': False,
                'message': f'附近未找到「{query}」相关地点，请换关键词',
                'places': [],
            }), 400
        return jsonify({
            'success': True,
            'query': query,
            'count': len(places),
            'places': places,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e),
            'error': str(e),
        }), 500

@app.route('/api/travel_recommendation', methods=['POST'])
def travel_recommendation():
    """出行推荐：POI 检索 + 多模式路线 + 碳排放对比。"""
    try:
        data = request.get_json(silent=True) or {}
        lat = data.get('lat', data.get('latitude'))
        lng = data.get('lng', data.get('longitude'))
        if lat is None or lng is None:
            return jsonify({
                'success': False,
                'message': '请提供当前位置 lat / lng',
            }), 400

        city = str(data.get('city') or '南京').strip()
        max_distance_km = data.get('max_distance_km')
        if max_distance_km is not None:
            max_distance_km = float(max_distance_km)
        companions = data.get('companions')
        time_preference = data.get('time_preference')

        ready, auto_prepared = _ensure_carbon_budget_for_travel(city=city)
        icb_raster = carbon_budget_results.get('icb_raster') if ready else None
        transform = carbon_budget_results.get('transform') if ready else None

        from travel_recommendation import run_travel_recommendation

        result = run_travel_recommendation(
            origin_lat=float(lat),
            origin_lon=float(lng),
            activity_query=data.get('activity_query'),
            activity_queries=data.get('activity_queries'),
            preview_stops=data.get('preview_stops'),
            max_distance_km=max_distance_km,
            companions=companions,
            time_preference=time_preference,
            city=city,
            icb_raster=icb_raster,
            transform=transform,
            max_places=int(data.get('max_places') or 4),
            max_itineraries=int(data.get('max_itineraries') or 2),
            work_lat=data.get('work_lat'),
            work_lon=data.get('work_lng', data.get('work_lon')),
        )
        if auto_prepared and isinstance(result, dict):
            result['carbon_budget_auto_prepared'] = True
        status = 200 if result.get('success') else 400
        return jsonify(result), status
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e),
            'error': str(e),
        }), 500

@app.route('/api/travel_itinerary_scheme', methods=['POST'])
def travel_itinerary_scheme():
    """按站点顺序重算指定出行方案路线（绿色出行/最快到达）。"""
    try:
        data = request.get_json(silent=True) or {}
        stops = data.get('stops') or []
        scheme_id = str(data.get('scheme_id') or '').strip()
        city = str(data.get('city') or '南京').strip()
        from travel_recommendation import run_travel_itinerary_scheme_plan

        result = run_travel_itinerary_scheme_plan(
            stops=stops,
            scheme_id=scheme_id,
            city=city,
            carbon_budget_g=data.get('carbon_budget_g', data.get('remaining_budget_g')),
        )
        status = 200 if result.get('success') else 400
        return jsonify(result), status
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e),
            'error': str(e),
        }), 500

@app.route('/api/travel_chain_routes', methods=['POST'])
def travel_chain_routes():
    try:
        data = request.get_json(silent=True) or {}
        stops = data.get('stops') or []
        if not isinstance(stops, list) or len(stops) < 2:
            return jsonify({
                'success': False,
                'message': '请提供至少两个带坐标的站点',
            }), 400

        city = str(data.get('city') or '南京').strip()
        scheme_id = str(data.get('scheme_id') or 'low_carbon').strip()
        from travel_recommendation import run_travel_chain_routes

        result = run_travel_chain_routes(stops=stops, city=city, scheme_id=scheme_id)
        status = 200 if result.get('success') else 400
        return jsonify(result), status
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e),
            'error': str(e),
        }), 500

@app.route('/api/personal_activity_carbon', methods=['POST'])
def personal_activity_carbon():
    """个人活动出行链分段碳排放：居住位置→各选点，每段按所选交通方式计算并求和。"""
    try:
        data = request.get_json(silent=True) or {}
        home = data.get('home') or {}
        activities = data.get('activities') or []
        stops = data.get('stops') or []
        if not stops and (not isinstance(activities, list) or len(activities) < 1):
            return jsonify({
                'success': False,
                'message': '请提供至少两个已确认的活动地点',
            }), 400

        city = str(data.get('city') or '南京').strip()
        from ActivityTravelSimulation import run_personal_activity_carbon

        result = run_personal_activity_carbon(
            home=home if home else None,
            activities=activities,
            stops=stops if stops else None,
            city=city,
        )
        status = 200 if result.get('success') else 400
        return jsonify(result), status
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e),
            'error': str(e),
        }), 500

@app.route('/api/daily_activity_range', methods=['POST'])
def daily_activity_range():
    try:
        payload = request.get_json(silent=True) or {}
        activities = payload.get('activities', [])
        commute_modes = payload.get('commute_modes')
        if not isinstance(commute_modes, list):
            commute_modes = None
        travel_mode = str(payload.get('travel_mode', 'walk') or 'walk').strip().lower()
        selected_poi_types = payload.get('selected_poi_types', [])
        if not isinstance(selected_poi_types, list):
            selected_poi_types = []
        poi_activity_times = payload.get('poi_activity_times')
        if not isinstance(poi_activity_times, dict):
            poi_activity_times = None
        else:
            poi_activity_times = {
                str(k): float(v)
                for k, v in poi_activity_times.items()
                if v is not None and float(v) > 0
            }
        city = str(payload.get('city') or '南京').strip()

        if not isinstance(activities, list) or len(activities) < 2:
            return jsonify({
                'success': False,
                'message': '请至少提供2个活动（含开始时间和结束时间）',
            }), 400

        for i, act in enumerate(activities):
            if act.get('lat') is None or act.get('lng') is None:
                act_name = str(act.get('name') or '').strip() or f'活动{i + 1}'
                return jsonify({
                    'success': False,
                    'message': f"活动「{act_name}」缺少坐标，请完成地址搜索",
                }), 400
            if act.get('start_time') is None or act.get('end_time') is None:
                act_name = str(act.get('name') or '').strip() or f'活动{i + 1}'
                return jsonify({
                    'success': False,
                    'message': f"活动「{act_name}」缺少开始或结束时间",
                }), 400

        from daily_activity_range import compute_daily_activity_range
        result = compute_daily_activity_range(
            activities,
            selected_poi_types,
            city,
            commute_modes=commute_modes,
            travel_mode=travel_mode,
            poi_activity_times=poi_activity_times,
        )
        status = 200 if result.get('success') else 400
        return jsonify(result), status
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e),
            'error': str(e),
        }), 500

@app.route('/api/get_icb_at_point', methods=['GET'])
def get_icb_at_point():
    """查询任意经纬度坐标对应的人均日碳预算(ICB)值（单位：g/人·天）

    参数:
        lat: 纬度
        lng: 经度
    """
    try:
        lat = request.args.get('lat', type=float)
        lng = request.args.get('lng', type=float)

        if lat is None or lng is None:
            return jsonify({
                'success': False,
                'message': '请提供 lat 和 lng 参数'
            }), 400

        if carbon_budget_results['icb_raster'] is None or carbon_budget_results.get('transform') is None:
            return jsonify({
                'success': False,
                'message': '请先运行碳预算计算',
                'icb': None
            }), 400

        icb = carbon_budget_results['icb_raster']
        transform = carbon_budget_results['transform']

        # transform: (a, b, c, d, e, f) 其中
        # x = a * col + b * row + c
        # y = d * col + e * row + f
        # rasterio 原点在上方，但 ICB 栅格已用 origin='lower' 存储
        a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f

        # 反算行列号（适用于 origin='upper' 的情况，即 transform 描述的是左上角原点）
        # col = (x - c) / a
        # row = (y - f) / e  (e < 0)
        col = (lng - c) / a
        row = (lat - f) / e   # e 通常为负值（纬度方向向下）

        col_int = int(round(col))
        row_int = int(round(row))

        rows, cols = icb.shape
        if 0 <= row_int < rows and 0 <= col_int < cols:
            value = float(icb[row_int, col_int])
            if np.isnan(value):
                value = None
        else:
            value = None

        return jsonify({
            'success': True,
            'lat': lat,
            'lng': lng,
            'icb': value,
            'unit': 'g/人·天'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e),
            'icb': None
        }), 500

@app.route('/api/get_city_info', methods=['GET'])
def get_city_info():
    try:
        city = request.args.get('city', '南京')
        
        city_configs = {
            '南京': {
                'center': [32.06, 118.79],
                'zoom': 11,
                'bounds': {
                    'south': 31.14,
                    'north': 32.61,
                    'west': 118.22,
                    'east': 119.33
                }
            },
            '武汉': {
                'center': [30.59, 114.31],
                'zoom': 10,
                'bounds': {
                    'south': 29.97,
                    'north': 31.37,
                    'west': 113.69,
                    'east': 115.07
                }
            },
            '北京': {
                'center': [39.90, 116.41],
                'zoom': 10,
                'bounds': {
                    'south': 39.44,
                    'north': 41.05,
                    'west': 115.42,
                    'east': 117.50
                }
            },
            '上海': {
                'center': [31.23, 121.47],
                'zoom': 10,
                'bounds': {
                    'south': 30.68,
                    'north': 31.88,
                    'west': 120.85,
                    'east': 122.00
                }
            }
        }
        
        config = city_configs.get(city)
        if not config:
            return jsonify({
                'success': False,
                'message': f'不支持的城市: {city}'
            }), 400
        
        return jsonify({
            'success': True,
            'city': city,
            'center': config['center'],
            'zoom': config['zoom'],
            'bounds': config['bounds']
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e),
            'error': str(e)
        }), 500

if __name__ == '__main__':
    import time

    backend_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(backend_dir)
    print(f"当前工作目录: {os.getcwd()}")

    os.makedirs('results', exist_ok=True)
    print("Flask应用启动中...")
    print("高分辨率可达性分析服务已就绪")
    print("API端点:")
    print("  POST /api/calculate_accessibility - 启动可达性计算")
    print("  GET  /api/get_accessibility_map   - 获取可达性地图")
    print("  GET  /api/get_progress            - 获取计算进度")
    print("  POST /api/calculate_carbon_budget - 启动碳预算计算")
    print("  GET  /api/get_carbon_budget_progress - 获取碳预算计算进度")
    print("  POST /api/ensure_carbon_budget_for_travel - 出行推荐前确保碳预算栅格可用")
    print("  GET  /api/get_carbon_budget_results - 获取碳预算计算结果")
    print("  GET  /api/check_working_directory - 检查工作目录")
    print("  GET  /api/check_data_status       - 检查数据加载状态")
    print("  GET  /api/get_district_boundary   - 获取区域边界")
    print("  GET  /api/get_road_network        - 获取城市路网 GeoJSON（地图叠加）")
    print("  GET  /api/geocode                 - 地理编码")
    print("  GET  /api/health                  - 健康检查")
    print("  POST /api/calculate_personal_accessibility - 基于个人的可达性计算")
    print("  POST /api/travel_activity_poi_candidates - 活动类型附近 POI 候选")
    print("  POST /api/travel_recommendation    - 出行推荐（POI+多模式路线+碳对比）")
    print("  POST /api/travel_itinerary_scheme   - 重算出行方案路线")
    print("  GET  /api/get_icb_at_point         - 查询坐标对应ICB碳预算值")
    print("  GET  /api/export_heatmap           - 导出热力图 PNG / Shapefile")
    print("  GET  /api/get_city_info            - 获取城市地理信息")
    # Disable debug reloader in desktop/runtime mode to avoid duplicate backend processes.
    app.run(host='127.0.0.1', port=int(os.environ.get('IATP_BACKEND_PORT', '5000')), debug=False, use_reloader=False)
