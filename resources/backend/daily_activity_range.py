
from __future__ import annotations

import math
import threading
from datetime import datetime

from shapely.geometry import Point, mapping
from collections import defaultdict

from PersonalAccessibility import (
    _get_pois_in_polygon,
    _get_calculator,
    _calculate_activity_space_accessibility,
    _build_ppa_polygon,
    _build_ppa_bounds,
    _normalize_display_polygon,
    _route_linestring_from_coords,
    build_route_distance_context,
    _get_pois_in_radius,
    _distance_poi_to_network,
    _is_zero_activity_weight,
    ACTIVITY_LABELS,
    ACTIVITY_TO_POI,
    DECAY_BETA,
    DEFAULT_WEIGHTS,
    _DEG_LAT_TO_M,
    _commute_speed_ms,
    _lateral_expansion_speed_ms,
)
from tianditu_routing import _is_transit_travel_mode
from online_routing import plan_online_route
from travel_recommendation import (
    ROUTE_SCHEME_DEFS,
    _WALK_LEG_SCHEME,
    _build_leg_plan_for_scheme,
)

from transport_modes import (
    normalize_mode_key,
    get_route_geometry_coords,
    get_route_distance,
    path_length_from_coords,
    speed_ms,
    build_transit_path_geometry,
)

POI_TYPE_DEFS = {
    # 购物餐饮：关键词对齐 POI 文件「类别」字段（大类;中类;小类）
    'shopping_mall': ('shopping', ['购物相关场所', '商场', '购物中心', '奥特莱斯', '百货', '商业街']),
    'supermarket': ('shopping', ['超级市场', '综合市场', '农副产品市场', '超市', '生鲜', '菜市场']),
    'convenience': ('shopping', ['便民商店', '便利店']),
    'bookstore': ('shopping', ['书店', '书城']),
    'fast_food': ('catering', ['快餐厅']),
    'hotpot': ('catering', ['火锅店', '火锅']),
    'chinese_cuisine': ('catering', ['中餐厅', '川菜', '湘菜', '粤菜', '鲁菜', '江苏菜', '地方风味', '海鲜酒楼', '综合酒楼']),
    'bbq': ('catering', ['烧烤', '烤肉']),
    'hospital': ('medical', ['综合医院', '卫生院', '医院', '诊所']),
    'bank': ('work', ['银行', '金融保险服务', '金融保险', '自动提款机']),
    'pharmacy': ('medical', ['药店', '药房', '医药']),
    'florist': ('shopping', ['花店', '花卉', '鲜花']),
    'hotel': ('shopping', ['宾馆酒店', '住宿服务', '旅馆招待所', '经济型连锁酒店', '五星级宾馆', '酒店', '宾馆']),
    'park': ('recreation', ['公园', '公园广场']),
    'cinema': ('recreation', ['影剧院', '电影院', '影城']),
    'amusement_park': ('recreation', ['游乐场', '游乐园', '儿童乐园', '淘气堡']),
    'bar': ('recreation', ['酒吧']),
}

POI_TYPE_LABELS_ZH = {
    'shopping_mall': '商场',
    'supermarket': '超市',
    'convenience': '便利店',
    'bookstore': '书店',
    'fast_food': '快餐',
    'hotpot': '火锅',
    'chinese_cuisine': '中餐',
    'bbq': '烧烤',
    'hospital': '医院',
    'bank': '银行',
    'pharmacy': '药店',
    'florist': '花店',
    'hotel': '酒店',
    'park': '公园',
    'cinema': '电影院',
    'amusement_park': '游乐园',
    'bar': '酒吧',
}

POI_CATEGORY_COLORS = {
    'shopping': '#e6a23c',
    'catering': '#f56c6c',
    'recreation': '#67c23a',
    'medical': '#409eff',
    'education': '#9c27b0',
    'work': '#606266',
}

# 交通方式中文标签
MODE_LABELS_ZH = {
    'walk': '步行', 'bike': '骑行', 'metro': '地铁', 'bus': '公交', 'car': '驾车',
}

# 各 POI 子类默认活动时间（分钟），从空档弹性时间中扣除；多选时取最短
DEFAULT_POI_ACTIVITY_TIMES = {
    'shopping_mall': 30.0,
    'supermarket': 20.0,
    'convenience': 10.0,
    'bookstore': 10.0,
    'fast_food': 20.0,
    'hotpot': 10.0,
    'chinese_cuisine': 20.0,
    'bbq': 20.0,
    'hospital': 60.0,
    'bank': 15.0,
    'pharmacy': 10.0,
    'florist': 15.0,
    'hotel': 30.0,
    'park': 30.0,
    'cinema': 120.0,
    'amusement_park': 60.0,
    'bar': 30.0,
}

# 未知子类或未选 POI 时的兜底值
MIN_ACTIVITY_STAY_MINUTES = 5.0

def _get_poi_activity_time(type_id, poi_activity_times=None):
    """获取单个子类的活动时间（分钟）：优先用户输入，否则默认值。"""
    if poi_activity_times:
        raw = poi_activity_times.get(type_id)
        if raw is not None:
            val = float(raw)
            if val > 0:
                return val
    return float(DEFAULT_POI_ACTIVITY_TIMES.get(type_id, MIN_ACTIVITY_STAY_MINUTES))

def _resolve_stay_time_min(selected_poi_types, poi_activity_times=None):
    """多选 POI 子类时，活动空间按最短活动时间计算。"""
    type_ids = _normalize_selected_poi_types(selected_poi_types)
    if not type_ids:
        return MIN_ACTIVITY_STAY_MINUTES
    return min(_get_poi_activity_time(t, poi_activity_times) for t in type_ids)

def _normalize_commute_mode(travel_mode):
    """前端交通方式 → PersonalAccessibility 通勤模式键。"""
    mode_key = normalize_mode_key(travel_mode)
    commute_mode = mode_key if mode_key != 'drive' else 'car'
    if commute_mode == 'walk':
        commute_mode = 'foot'
    return commute_mode

def _scheme_for_travel_mode(travel_mode):
    key = normalize_mode_key(travel_mode)
    if key in ('walk', 'foot'):
        return _WALK_LEG_SCHEME
    scheme_id_map = {
        'bike': 'low_carbon',
        'metro': 'transit_metro',
        'subway': 'transit_metro',
        'bus': 'transit_bus',
        'car': 'regular',
        'drive': 'regular',
    }
    scheme_id = scheme_id_map.get(key, 'low_carbon')
    return next(s for s in ROUTE_SCHEME_DEFS if s.get('id') == scheme_id)

def _effective_commute_mode(travel_mode, route_plan):
    """路线回退为步行时，活动空间与可达性计算同步按步行处理。"""
    if not route_plan:
        return _normalize_commute_mode(travel_mode)
    if route_plan.get('walk_fallback') or route_plan.get('uses_walk_fallback'):
        return 'foot'
    if route_plan.get('profile') == 'walking':
        return 'foot'
    return _normalize_commute_mode(travel_mode)

def _estimate_transit_commute_duration_min(route_plan, travel_mode):
    """
    公交/地铁通勤：接驳步行段按步行速度，乘运段按公交/地铁速度分别计时。
    """
    parts = route_plan.get('route_parts') or []
    if not parts:
        return None

    mode_key = normalize_mode_key(travel_mode)
    default_ride_mode = 'metro' if mode_key in ('metro', 'subway') else 'bus'
    total_sec = 0.0
    has_ride = False

    for part in parts:
        coords = part.get('coords') or []
        if len(coords) < 2:
            continue
        dist_m = float(path_length_from_coords(coords) or 0)
        if dist_m <= 0:
            continue

        geometry = str(part.get('geometry') or '')
        part_type = str(part.get('type') or '')
        if geometry == 'transit_access' or part_type == 'walk':
            speed = speed_ms('walk')
        elif geometry == 'transit_ride':
            has_ride = True
            ride_mode = part_type if part_type in ('metro', 'bus') else default_ride_mode
            speed = speed_ms(ride_mode)
        else:
            speed = speed_ms(_normalize_commute_mode(travel_mode))
        total_sec += dist_m / max(speed, 0.1)

    if not has_ride:
        return None
    return total_sec / 60.0

def _format_route_plan_result(route_plan, travel_mode):
    duration_min = float(route_plan.get('duration_min') or 0)
    duration_sec = float(route_plan.get('duration_sec') or 0)
    distance_m = float(route_plan.get('distance_m') or 0)
    coords = route_plan.get('route_coords') or []

    # 从路线坐标反算距离；API 距离缺失或明显偏小时以坐标弧长为准
    if len(coords) >= 2:
        coord_dist_m = float(path_length_from_coords(coords) or 0)
        if coord_dist_m > 0 and (distance_m <= 0 or distance_m < coord_dist_m * 0.1):
            distance_m = coord_dist_m
    elif distance_m <= 0:
        distance_m = 0.0

    uses_transit = bool(
        route_plan.get('uses_transit_line')
        or any(
            p.get('geometry') == 'transit_ride'
            for p in (route_plan.get('route_parts') or [])
        )
    )
    if uses_transit:
        segmented_min = _estimate_transit_commute_duration_min(route_plan, travel_mode)
        if segmented_min is not None and segmented_min > 0:
            duration_min = segmented_min
        elif duration_sec > 0:
            duration_min = duration_sec / 60.0
        elif duration_min <= 0 and distance_m > 0:
            duration_min = distance_m / max(speed_ms('walk'), 0.1) / 60.0
    else:
        effective_mode = _effective_commute_mode(travel_mode, route_plan)
        speed_ms_val = _commute_speed_ms(effective_mode)
        if distance_m > 0:
            duration_min = distance_m / max(speed_ms_val, 0.1) / 60.0
        elif duration_min <= 0:
            duration_min = 0.0

    source = route_plan.get('source', 'unknown')
    if route_plan.get('walk_fallback') or route_plan.get('uses_walk_fallback'):
        source = 'tianditu_walk_fallback' if source == 'tianditu' else 'travel_recommendation_walk_fallback'

    return {
        'route_coords': route_plan.get('route_coords') or [],
        'distance_m': distance_m,
        'duration_min': round(duration_min, 2),
        'source': source,
        'profile': route_plan.get('profile'),
        'walk_fallback': bool(route_plan.get('walk_fallback') or route_plan.get('uses_walk_fallback')),
        'route_parts': route_plan.get('route_parts'),
        'uses_transit_line': route_plan.get('uses_transit_line'),
    }

def _transit_route_plan_usable(route_plan, travel_mode):
    """公交/地铁模式须含真实乘运段，不接受步行回退方案。"""
    if not route_plan:
        return False
    coords = route_plan.get('route_coords') or []
    if len(coords) < 2:
        return False
    if not _is_transit_travel_mode(travel_mode):
        return True
    if route_plan.get('walk_fallback') or route_plan.get('uses_walk_fallback'):
        return False
    if route_plan.get('uses_transit_line'):
        return True
    parts = route_plan.get('route_parts') or []
    return any(
        part.get('geometry') == 'transit_ride' and len(part.get('coords') or []) >= 2
        for part in parts
    )

def _plan_local_transit_route(calc, origin_pt, dest_pt, travel_mode):
    """天地图公交/地铁不可用时，回退本地公交/地铁线网。"""
    mode_key = normalize_mode_key(travel_mode)
    if mode_key not in ('bus', 'metro', 'subway'):
        return None
    if mode_key == 'subway':
        mode_key = 'metro'

    geom = build_transit_path_geometry(calc, origin_pt, dest_pt, mode_key)
    if not geom:
        return None

    parts = [
        dict(part) for part in (geom.get('parts') or [])
        if len(part.get('coords') or []) >= 2
    ]
    uses_transit_line = any(
        part.get('geometry') == 'transit_ride' and len(part.get('coords') or []) >= 2
        for part in parts
    )
    if not uses_transit_line:
        return None

    route_coords = geom.get('merged') or []
    if len(route_coords) < 2:
        return None

    distance_m = float(geom.get('total_distance_m') or 0)
    if distance_m <= 0:
        distance_m = float(path_length_from_coords(route_coords) or 0)
    if distance_m <= 0:
        return None

    speed_ms_val = speed_ms(mode_key)
    duration_min = distance_m / max(speed_ms_val, 0.1) / 60.0
    return _format_route_plan_result({
        'route_coords': route_coords,
        'route_parts': parts,
        'distance_m': distance_m,
        'duration_min': round(duration_min, 2),
        'source': 'local_network',
        'profile': mode_key,
        'uses_transit_line': True,
    }, travel_mode)

def _plan_local_network_route(calc, origin_pt, dest_pt, travel_mode):
    """天地图路径规划不可用时，回退本地路网/线网。"""
    mode_key = normalize_mode_key(travel_mode)
    if mode_key in ('bus', 'metro', 'subway'):
        local_transit = _plan_local_transit_route(calc, origin_pt, dest_pt, travel_mode)
        if local_transit:
            return local_transit

    commute_mode = _normalize_commute_mode(travel_mode)
    route_coords = get_route_geometry_coords(calc, origin_pt, dest_pt, commute_mode)
    if not route_coords or len(route_coords) < 2:
        return None

    distance_m = float(get_route_distance(calc, origin_pt, dest_pt, commute_mode) or 0)
    if distance_m <= 0:
        distance_m = float(path_length_from_coords(route_coords) or 0)
    if distance_m <= 0:
        return None

    speed_ms_val = speed_ms(commute_mode)
    duration_min = distance_m / max(speed_ms_val, 0.1) / 60.0
    return _format_route_plan_result({
        'route_coords': route_coords,
        'distance_m': distance_m,
        'duration_min': round(duration_min, 2),
        'source': 'local_network',
        'profile': commute_mode,
    }, travel_mode)

def _transit_walk_access_m(route_plan):
    """从公交/地铁路线方案中提取接驳步行总距离（米），用于比较方案优劣。"""
    if not route_plan:
        return -1.0
    parts = route_plan.get('route_parts') or []
    total_walk = 0.0
    has_walk = False
    for part in parts:
        if part.get('geometry') == 'transit_access':
            coords = part.get('coords') or []
            if len(coords) >= 2:
                from transport_modes import path_length_from_coords
                total_walk += path_length_from_coords(coords)
                has_walk = True
    return total_walk if has_walk else -1.0

def _plan_gap_commute_route(calc, origin_pt, dest_pt, travel_mode, city='南京'):
    """
    优先天地图，失败则高德；公交/地铁仍无乘运段时回退本地线网，最后才改步行。
    """
    local_box = {}
    local_thread = None
    if _is_transit_travel_mode(travel_mode):
        def _run_local_transit():
            try:
                local_box['plan'] = _plan_local_transit_route(
                    calc, origin_pt, dest_pt, travel_mode,
                )
            except Exception as exc:
                local_box['error'] = exc
                print(f'[daily_range] 本地公交/地铁规划失败: {exc}', flush=True)

        local_thread = threading.Thread(target=_run_local_transit, daemon=True)
        local_thread.start()

    online_plan = plan_online_route(
        float(origin_pt.x), float(origin_pt.y),
        float(dest_pt.x), float(dest_pt.y),
        travel_mode, city=city,
    )
    if _transit_route_plan_usable(online_plan, travel_mode):
        source = online_plan.get('source') or 'tianditu'
        print(f'[daily_range] 使用{source}通勤方案', flush=True)
        return _format_route_plan_result({**online_plan, 'source': source}, travel_mode)

    local_transit = None
    if local_thread is not None:
        local_thread.join(timeout=60.0)
        local_transit = local_box.get('plan')

    if local_transit:
        print(
            f'[daily_range] 在线{travel_mode}路线不可用，改用本地公交/地铁线网',
            flush=True,
        )
        return local_transit

    scheme = _scheme_for_travel_mode(travel_mode)
    leg = _build_leg_plan_for_scheme(
        {'lon': float(origin_pt.x), 'lat': float(origin_pt.y)},
        {'lon': float(dest_pt.x), 'lat': float(dest_pt.y)},
        scheme,
        city=city,
    )

    if leg and _transit_route_plan_usable({
        'route_coords': leg.get('route_coords') or [],
        'route_parts': leg.get('route_parts'),
        'uses_transit_line': leg.get('uses_transit_line'),
        'walk_fallback': leg.get('walk_fallback'),
    }, travel_mode):
        return _format_route_plan_result({
            'route_coords': leg.get('route_coords') or [],
            'distance_m': float(leg.get('distance_m') or 0),
            'duration_min': round(float(leg.get('time_min') or 0), 2),
            'source': 'travel_recommendation',
            'route_parts': leg.get('route_parts'),
            'uses_transit_line': leg.get('uses_transit_line'),
        }, travel_mode)

    if _is_transit_travel_mode(travel_mode):
        print(
            '[daily_range] 在线规划与本地线网均无地铁/公交乘运，改步行路线',
            flush=True,
        )
        leg = _build_leg_plan_for_scheme(
            {'lon': float(origin_pt.x), 'lat': float(origin_pt.y)},
            {'lon': float(dest_pt.x), 'lat': float(dest_pt.y)},
            _WALK_LEG_SCHEME,
            city=city,
        )
        if leg:
            return _format_route_plan_result({
                'route_coords': leg.get('route_coords') or [],
                'distance_m': float(leg.get('distance_m') or 0),
                'duration_min': round(float(leg.get('time_min') or 0), 2),
                'source': 'travel_recommendation',
                'profile': 'walking',
                'uses_walk_fallback': True,
                'route_parts': leg.get('route_parts'),
                'uses_transit_line': False,
            }, travel_mode)
    elif leg:
        return _format_route_plan_result({
            'route_coords': leg.get('route_coords') or [],
            'distance_m': float(leg.get('distance_m') or 0),
            'duration_min': round(float(leg.get('time_min') or 0), 2),
            'source': 'travel_recommendation',
            'route_parts': leg.get('route_parts'),
            'uses_transit_line': leg.get('uses_transit_line'),
        }, travel_mode)

    local_plan = _plan_local_network_route(calc, origin_pt, dest_pt, travel_mode)
    if local_plan:
        print(
            f'[daily_range] 在线路线不可用（{travel_mode}），改用本地路网',
            flush=True,
        )
        return local_plan
    return None

def _compute_single_gap_ppa(
    calc, gap, travel_mode, selected_poi_types, city='南京', poi_activity_times=None,
):
    """
    单个活动空档：先算上一活动→下一活动通勤时间，
    弹性时间 = 空档时间 - 通勤时间 - 所选 POI 子类最短活动时间，
    再按可达性分析 PPA 生成活动空间。
    """
    origin_pt = Point(gap['origin_lng'], gap['origin_lat'])
    dest_pt = Point(gap['dest_lng'], gap['dest_lat'])
    gap_minutes = float(gap['gap_minutes'])
    route_plan = _plan_gap_commute_route(calc, origin_pt, dest_pt, travel_mode, city=city)
    if not route_plan:
        return {
            'from_activity': gap['from_activity'],
            'to_activity': gap['to_activity'],
            'gap_minutes': gap_minutes,
            'origin_lat': gap['origin_lat'],
            'origin_lng': gap['origin_lng'],
            'dest_lat': gap['dest_lat'],
            'dest_lng': gap['dest_lng'],
            'valid': False,
            'method': 'route_plan_failed',
            'message': '无法规划通勤路线',
        }
    commute_mode = _effective_commute_mode(travel_mode, route_plan)
    speed_ms_val = _commute_speed_ms(commute_mode)

    route_coords = route_plan.get('route_coords') or []
    route_parts = route_plan.get('route_parts') or []
    commute_dist_m = float(route_plan.get('distance_m') or 0)
    commute_time_min = float(route_plan.get('duration_min') or 0)
    if commute_time_min <= 0 and commute_dist_m > 0:
        commute_time_min = commute_dist_m / max(speed_ms_val, 0.1) / 60.0
    route_source = route_plan.get('source', 'unknown')

    stay_time_min = _resolve_stay_time_min(selected_poi_types, poi_activity_times)
    elastic_time_min = gap_minutes - commute_time_min - stay_time_min
    total_travel_budget_min = gap_minutes - stay_time_min

    result = {
        'from_activity': gap['from_activity'],
        'to_activity': gap['to_activity'],
        'gap_minutes': gap_minutes,
        'commute_time_min': round(commute_time_min, 1),
        'commute_dist_m': round(commute_dist_m, 0),
        'stay_time_min': round(stay_time_min, 1),
        'poi_activity_times_used': {
            type_id: round(_get_poi_activity_time(type_id, poi_activity_times), 1)
            for type_id in _normalize_selected_poi_types(selected_poi_types)
        },
        'elastic_time_min': round(elastic_time_min, 1),
        'route_source': route_source,
        'walk_fallback': bool(route_plan.get('walk_fallback')),
        'origin_lat': gap['origin_lat'],
        'origin_lng': gap['origin_lng'],
        'dest_lat': gap['dest_lat'],
        'dest_lng': gap['dest_lng'],
        'valid': False,
    }

    if elastic_time_min <= 0:
        # 通勤+停留占满空档：不生成活动空间
        result['method'] = 'no_elastic_time'
        result['message'] = (
            f'空档 {gap_minutes:.0f} 分钟，通勤 {commute_time_min:.0f} 分钟，'
            f'停留 {stay_time_min:.0f} 分钟，无剩余弹性时间'
        )
        # 若通勤仍小于空档，保留通勤路线供地图展示
        if commute_time_min < gap_minutes and len(route_coords) >= 2:
            result['has_commute_route'] = True
            result['commute_route'] = route_coords
            if route_parts:
                result['route_parts'] = route_parts
            result['method'] = 'route_only_no_elastic'
            result['message'] = (
                f'空档 {gap_minutes:.0f} 分钟，通勤 {commute_time_min:.0f} 分钟，'
                f'停留 {stay_time_min:.0f} 分钟，无剩余弹性时间；已显示通勤路线'
            )
        return result

    poi_type_defs = _selected_poi_type_defs(selected_poi_types)

    ppa_travel_kwargs = {
        'total_travel_budget_min': total_travel_budget_min,
        'commute_time_min': commute_time_min,
    }
    ppa_result = _build_ppa_polygon(
        calc, origin_pt, elastic_time_min, commute_mode, work_pt=dest_pt,
        route_coords=route_coords if len(route_coords) >= 2 else None,
        route_parts=route_parts if route_parts else None,
        **ppa_travel_kwargs,
    )

    if not ppa_result or ppa_result.get('polygon') is None:
        result['method'] = 'ppa_failed'
        result['message'] = '潜在活动空间生成失败'
        return result

    route_line = _route_linestring_from_coords(ppa_result.get('route_coords'))
    max_one_way_m = float(
        ppa_result.get('max_one_way_m')
        or (elastic_time_min * 60.0 * _lateral_expansion_speed_ms(commute_mode) * 0.5)
    )
    route_dist_mode = commute_mode
    if normalize_mode_key(commute_mode) in ('metro', 'subway', 'bus'):
        route_dist_mode = 'foot'
    route_dist_ctx = build_route_distance_context(
        calc, ppa_result.get('route_coords'), route_dist_mode, max_one_way_m,
    )

    locations = [
        {
            'lat': gap['origin_lat'],
            'lng': gap['origin_lng'],
            'type': 'other',
            'name': str(gap['from_activity'].get('name', '活动')),
        },
        {
            'lat': gap['dest_lat'],
            'lng': gap['dest_lng'],
            'type': 'other',
            'name': str(gap['to_activity'].get('name', '活动')),
        },
    ]
    activity_bounds = _build_ppa_bounds(
        locations, origin_pt, ppa_result, elastic_time_min,
        _lateral_expansion_speed_ms(commute_mode),
    )
    from_name = gap['from_activity'].get('name', '活动')
    to_name = gap['to_activity'].get('name', '活动')
    activity_bounds['gap_label'] = f'{from_name} → {to_name}'
    activity_bounds['from_activity_name'] = from_name
    activity_bounds['to_activity_name'] = to_name
    activity_bounds['commute_time_min'] = round(commute_time_min, 1)
    activity_bounds['gap_minutes'] = round(gap_minutes, 1)
    activity_bounds['stay_time_min'] = round(stay_time_min, 1)

    category_scores, gap_overall, _ = _calculate_activity_space_accessibility(
        ppa_result['polygon'],
        origin_pt,
        calc,
        beta=DECAY_BETA,
        commute_mode=commute_mode,
        route_line=route_line,
        route_dist_ctx=route_dist_ctx,
        activity_time_budgets={'elastic': elastic_time_min},
        selected_poi_type_defs=poi_type_defs,
    )

    polygon = ppa_result['polygon']
    polygon = _normalize_display_polygon(polygon) if polygon is not None else None
    result.update({
        'valid': True,
        'evaluation_geometry': mapping(ppa_result['polygon']),
        'polygon': polygon,
        'polygon_coords': _polygon_to_coords(polygon),
        'method': ppa_result.get('method', 'space_time_prism_ppa'),
        'max_one_way_m': round(float(ppa_result.get('max_one_way_m', max_one_way_m)), 0),
        'activity_bounds': activity_bounds,
        'category_scores': category_scores,
        'overall_score': gap_overall,
    })
    return result

def _polygon_to_coords(polygon):
    """Shapely 多边形 → [[lon, lat], ...] 供前端 Leaflet 使用。"""
    if polygon is None or polygon.is_empty:
        return []
    try:
        from PersonalAccessibility import _simplify_activity_polygon
        polygon = _simplify_activity_polygon(polygon)
    except Exception:
        pass
    if polygon is None or getattr(polygon, 'is_empty', True):
        return []
    if polygon.geom_type == 'Polygon':
        return [[float(x), float(y)] for x, y in polygon.exterior.coords]
    if polygon.geom_type == 'MultiPolygon':
        largest = max(polygon.geoms, key=lambda p: p.area)
        return [[float(x), float(y)] for x, y in largest.exterior.coords]
    return []

def _pd_notna(val):
    try:
        import pandas as pd
        return pd.notna(val)
    except ImportError:
        return val is not None and str(val) not in ('', 'nan', 'NaN', 'None', 'null')

def _poi_category_blob(poi):
    parts = []
    cat = poi.get('category') or poi.get('类别') or ''
    if cat:
        text = str(cat).replace('/', ';')
        parts.extend(p.strip() for p in text.split(';') if p.strip())
    name = str(poi.get('name') or '').strip()
    if name:
        parts.append(name)
    return ';'.join(parts)

def _poi_matches_type_keywords(poi, keywords):
    """判断 POI 是否匹配所选类型的关键词（在类别各层级或名称中检索）。"""
    if not keywords:
        return True
    blob = _poi_category_blob(poi)
    if not blob:
        return False
    return any(kw in blob for kw in keywords)

def _enrich_poi_categories(pois, calc, poi_category):
    """为缺少类别字段的 POI 从 GeoJSON 属性表补全。"""
    if not pois or calc is None or not hasattr(calc, 'poi_trees'):
        return pois
    if poi_category not in calc.poi_trees:
        return pois
    _, _, gdf = calc.poi_trees[poi_category]
    if gdf is None or gdf.empty:
        return pois

    cat_field = next((f for f in ('类别', 'category', 'type', '大类') if f in gdf.columns), None)
    name_field = next((f for f in ('名称', 'name', 'NAME', 'Name') if f in gdf.columns), None)
    if not cat_field:
        return pois

    name_to_cat = {}
    if name_field:
        for _, row in gdf.iterrows():
            name = str(row[name_field]) if _pd_notna(row[name_field]) else ''
            if name and name not in name_to_cat:
                name_to_cat[name] = str(row[cat_field]) if _pd_notna(row[cat_field]) else ''

    for poi in pois:
        if poi.get('category'):
            continue
        poi_name = str(poi.get('name') or '')
        if poi_name and poi_name in name_to_cat:
            poi['category'] = name_to_cat[poi_name]
    return pois

def _filter_pois_by_subcategory(pois, keywords, calc, poi_category):
    """按 POI 文件「类别」字段（大类/中类/小类）及名称过滤。"""
    if not keywords or not pois:
        return pois
    pois = _enrich_poi_categories(pois, calc, poi_category)
    return [p for p in pois if _poi_matches_type_keywords(p, keywords)]

def _build_accessible_pois_from_type_results(type_results):
    """从所选子类 POI 构建可达 POI 展示列表（不含未勾选的大类 POI）。"""
    accessible_pois_list = []
    for type_id, pois in type_results.items():
        poi_category = POI_TYPE_DEFS[type_id][0]
        label = POI_TYPE_LABELS_ZH.get(type_id, type_id)
        color = POI_CATEGORY_COLORS.get(poi_category, '#909399')
        for poi in pois:
            coords = poi.get('coords', [0, 0])
            accessible_pois_list.append({
                'activity': type_id,
                'activity_label': label,
                'category': type_id,
                'category_label': label,
                'color': color,
                'coords': coords,
                'name': poi.get('name'),
                'activity_space_index': poi.get('activity_space_index'),
                'gap_label': poi.get('gap_name', ''),
            })
    return accessible_pois_list

def _parse_time_to_minutes(value):
    """将开始/结束时间解析为距 0 点的分钟数（支持 HH:MM 或分钟整数）。"""
    if value is None:
        return 0
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        if ':' in text:
            parts = text.split(':')
            hours = int(parts[0] or 0)
            minutes = int(parts[1] or 0) if len(parts) > 1 else 0
            return max(0, hours * 60 + minutes)
        value = float(text)
    number = float(value)
    if number <= 24 and number == int(number):
        return int(number) * 60
    return int(round(number))

def _format_time_minutes(minutes):
    """分钟数格式化为 HH:MM。"""
    total = int(minutes) % (24 * 60)
    hours, mins = divmod(total, 60)
    return f'{hours:02d}:{mins:02d}'

def _group_poi_types_by_category(poi_types):
    """将所选 POI 子类按大类分组，便于每个空档每大类只查一次多边形。"""
    groups = defaultdict(list)
    for type_id in poi_types:
        poi_category, keywords = POI_TYPE_DEFS[type_id]
        groups[poi_category].append((type_id, keywords))
    return groups

def _dedupe_pois_by_coords(pois):
    seen = set()
    unique = []
    for poi in pois:
        c = poi.get('coords', [0, 0])
        key = (round(float(c[0]), 5), round(float(c[1]), 5))
        if key not in seen:
            seen.add(key)
            unique.append(poi)
    return unique

def _search_pois_in_gaps_by_category(calc, valid_gap_results, category_groups):
    """每个活动空间 × 每个 POI 大类只检索一次，再按子类关键词拆分。"""
    type_results = {type_id: [] for cat_types in category_groups.values() for type_id, _ in cat_types}

    for poi_category, type_defs in category_groups.items():
        for gi, gap_res in enumerate(valid_gap_results):
            poly = gap_res.get('polygon')
            if poly is None or poly.is_empty:
                continue
            try:
                pois = _get_pois_in_polygon(calc, poi_category, poly)
            except Exception:
                continue
            if not pois:
                continue
            pois = _enrich_poi_categories(pois, calc, poi_category)

            gap_name = (
                f"{gap_res['from_activity'].get('name', '')} → "
                f"{gap_res['to_activity'].get('name', '')}"
            )
            activity_space_index = gap_res.get('activity_space_index')

            for type_id, keywords in type_defs:
                if keywords:
                    matched = [
                        p for p in pois
                        if _poi_matches_type_keywords(p, keywords)
                    ]
                else:
                    matched = pois
                for poi in matched:
                    item = dict(poi)
                    item['gap_index'] = gi
                    item['poi_type_id'] = type_id
                    item['interest_key'] = type_id
                    item['activity_space_index'] = activity_space_index
                    item['gap_name'] = gap_name
                    type_results[type_id].append(item)

    for type_id in type_results:
        type_results[type_id] = _dedupe_pois_by_coords(type_results[type_id])
    return type_results

def _normalize_selected_poi_types(selected_poi_types):
    if not selected_poi_types:
        return []
    seen = set()
    out = []
    for raw in selected_poi_types:
        type_id = str(raw or '').strip()
        if not type_id or type_id not in POI_TYPE_DEFS or type_id in seen:
            continue
        seen.add(type_id)
        out.append(type_id)
    return out

def _selected_poi_type_defs(selected_poi_types):
    """将勾选的 POI 类型 id 转为可达性计分用的 (type_id, 大类, 关键词) 列表。"""
    return [
        (type_id, POI_TYPE_DEFS[type_id][0], POI_TYPE_DEFS[type_id][1])
        for type_id in _normalize_selected_poi_types(selected_poi_types)
    ]

def _merge_gap_type_scores(per_gap_scores, gap_weights, selected_poi_types):
    """按空档时间权重合并各空档的 POI 子类可达性得分。"""
    type_ids = _normalize_selected_poi_types(selected_poi_types)
    total_w = sum(gap_weights)
    if total_w <= 0:
        total_w = 1.0
        gap_weights = [1.0] * len(per_gap_scores)

    merged = {}
    for type_id in type_ids:
        raw_sum = 0.0
        poi_count_sum = 0.0
        all_pois = []
        for scores, gw in zip(per_gap_scores, gap_weights):
            cs = scores.get(type_id, {})
            raw_sum += cs.get('raw_score', 0.0) * gw
            poi_count_sum += cs.get('poi_count', 0) * gw
            all_pois.extend(cs.get('pois', []))

        raw_avg = raw_sum / total_w
        merged[type_id] = {
            'raw_score': round(raw_avg, 4),
            'weight': 1.0,
            'weighted_score': round(raw_avg, 4),
            'poi_count': int(round(poi_count_sum / total_w)),
            'pois': all_pois,
            'time_budget': 0.0,
            'label': POI_TYPE_LABELS_ZH.get(type_id, type_id),
        }

    overall_score = sum(item['raw_score'] for item in merged.values())
    return merged, overall_score

def _poi_types_to_accessibility_weights(selected_poi_types):
    """将勾选的 POI 类型映射为可达性分析三类活动权重。"""
    counts = defaultdict(float)
    for type_id in _normalize_selected_poi_types(selected_poi_types):
        poi_category, _keywords = POI_TYPE_DEFS[type_id]
        counts[poi_category] += 1.0

    raw = {
        'shopping': counts.get('shopping', 0.0),
        'catering': counts.get('catering', 0.0),
        'recreation': counts.get('recreation', 0.0),
        'education': 0.0,
        'medical': 0.0,
    }
    total = sum(raw.values())
    if total <= 0:
        return {
            'shopping': 0.0,
            'catering': 0.0,
            'recreation': 0.0,
            'education': 0.0,
            'medical': 0.0,
        }
    return {k: v / total for k, v in raw.items()}

def _merge_gap_category_scores(per_gap_scores, gap_weights, final_weights):
    total_w = sum(gap_weights)
    if total_w <= 0:
        total_w = 1.0
        gap_weights = [1.0] * len(per_gap_scores)

    merged = {}
    for activity in DEFAULT_WEIGHTS:
        weight = final_weights.get(activity, 0.0)
        if _is_zero_activity_weight(weight):
            merged[activity] = {
                'raw_score': 0.0,
                'weight': weight,
                'weighted_score': 0.0,
                'poi_count': 0,
                'pois': [],
                'time_budget': 0.0,
            }
            continue

        raw_sum = 0.0
        poi_count_sum = 0.0
        all_pois = []
        for scores, gw in zip(per_gap_scores, gap_weights):
            cs = scores.get(activity, {})
            raw_sum += cs.get('raw_score', 0.0) * gw
            poi_count_sum += cs.get('poi_count', 0) * gw
            all_pois.extend(cs.get('pois', []))

        raw_avg = raw_sum / total_w
        merged[activity] = {
            'raw_score': round(raw_avg, 4),
            'weight': weight,
            'weighted_score': round(raw_avg * weight, 4),
            'poi_count': int(round(poi_count_sum / total_w)),
            'pois': all_pois,
            'time_budget': round(
                sum(
                    scores.get(activity, {}).get('time_budget', 0.0) * gw
                    for scores, gw in zip(per_gap_scores, gap_weights)
                ) / total_w,
                2,
            ),
        }

    overall = sum(s['weighted_score'] for s in merged.values())
    return merged, round(overall, 4)

def _build_route_only_space(gap_res):
    """无弹性活动空间时，构建仅含通勤路线的前端展示结构。"""
    from_name = gap_res['from_activity'].get('name', '活动')
    to_name = gap_res['to_activity'].get('name', '活动')
    origin_lat = float(gap_res.get('origin_lat') or 0)
    origin_lng = float(gap_res.get('origin_lng') or 0)
    dest_lat = float(gap_res.get('dest_lat') or 0)
    dest_lng = float(gap_res.get('dest_lng') or 0)
    space = {
        'cluster_id': -1,
        'route_only': True,
        'has_activity_space': False,
        'activity_space_index': None,
        'segment_label': gap_res.get('segment_label') or f'{from_name} → {to_name}',
        'gap_label': f'{from_name} → {to_name}',
        'from_activity_name': from_name,
        'to_activity_name': to_name,
        'commute_time_min': round(float(gap_res.get('commute_time_min') or 0), 1),
        'stay_time_min': round(float(gap_res.get('stay_time_min') or MIN_ACTIVITY_STAY_MINUTES), 1),
        'elastic_time_min': round(float(gap_res.get('elastic_time_min') or 0), 1),
        'gap_minutes': round(float(gap_res.get('gap_minutes') or 0), 1),
        'polygon': [],
        'method': gap_res.get('method', 'route_only_no_elastic'),
        'commute_route': gap_res.get('commute_route') or [],
        'input_points': [[origin_lat, origin_lng], [dest_lat, dest_lng]],
        'input_points_with_type': [
            {'coords': [origin_lat, origin_lng], 'type': 'other', 'name': from_name},
            {'coords': [dest_lat, dest_lng], 'type': 'other', 'name': to_name},
        ],
    }
    route_parts = gap_res.get('route_parts')
    if route_parts:
        space['route_parts'] = route_parts
    return space

def _build_text_summary(activities, commute_modes, total_by_type):
    act_strs = []
    for i, a in enumerate(activities):
        start_m = _parse_time_to_minutes(a.get('start_time', 0))
        end_m = _parse_time_to_minutes(a.get('end_time', 0))
        name = a.get('name', '活动')
        act_strs.append(f"{_format_time_minutes(start_m)}-{_format_time_minutes(end_m)} {name}")
        if i < len(activities) - 1:
            mode_key = normalize_mode_key(commute_modes[i] if i < len(commute_modes) else 'walk')
            mode_label = MODE_LABELS_ZH.get(mode_key, commute_modes[i] if i < len(commute_modes) else '步行')
            next_name = activities[i + 1].get('name', '活动')
            act_strs.append(f"  → {mode_label} → {next_name}")

    act_part = '\n'.join(act_strs) if act_strs else '无'

    type_parts = []
    for type_id, count in total_by_type.items():
        if count > 0:
            label = POI_TYPE_LABELS_ZH.get(type_id, type_id)
            type_parts.append(f"{count}个{label}")

    if type_parts:
        type_lines = [
            f"{part}、" if idx < len(type_parts) - 1 else part
            for idx, part in enumerate(type_parts)
        ]
        type_str = '\n'.join(type_lines)
        return (
            f"根据您今天的安排：\n"
            f"{act_part}。\n"
            f"按上述交通方式，可顺路到达：\n"
            f"{type_str}"
        )
    return (
        f"根据您今天的安排：\n"
        f"{act_part}。\n"
        f"按上述交通方式，暂未在可达范围内找到匹配您所选 POI 类型的场所。"
    )

def compute_daily_activity_range(
    activities, selected_poi_types, city='南京', commute_modes=None, travel_mode=None,
    poi_activity_times=None,
):
    """
    计算每日活动可达范围。

    参数
    ----
    activities : list[dict]
        每项含: name (str), lat (float), lng (float),
        start_time / end_time (int 分钟或 "HH:MM" 字符串，距 0 点)
    selected_poi_types : list[str]
        勾选的 POI 类型 id，见 POI_TYPE_DEFS
    poi_activity_times : dict[str, float], optional
        各 POI 子类用户设定的活动时间（分钟）；未提供则用 DEFAULT_POI_ACTIVITY_TIMES
    commute_modes : list[str], optional
        各活动空档的出行方式，长度应为 len(activities)-1，与按开始时间排序后的相邻活动对应
    travel_mode : str, optional
        兼容旧版：未提供 commute_modes 时，全部空档使用该方式
    city : str
        城市名（默认 "城市"）

    返回
    ----
    dict: {
        success, activity_markers, reachable_polygons, poi_markers,
        total_by_category, text_summary, gap_details
    }
    """
    poi_types = _normalize_selected_poi_types(selected_poi_types)
    if not poi_types:
        return {
            'success': False,
            'message': '请至少选择一种想去的 POI 类型',
        }

    if not activities or len(activities) < 2:
        return {
            'success': False,
            'message': '请至少添加2个活动以计算活动间隙的可达范围',
        }

    normalized_acts = []
    for act in activities:
        item = dict(act)
        item['_start_minutes'] = _parse_time_to_minutes(act.get('start_time', 0))
        item['_end_minutes'] = _parse_time_to_minutes(act.get('end_time', 0))
        normalized_acts.append(item)
    sorted_acts = sorted(normalized_acts, key=lambda a: a['_start_minutes'])

    gaps = []
    for i in range(len(sorted_acts) - 1):
        gap_start = sorted_acts[i]['_end_minutes']
        gap_end = sorted_acts[i + 1]['_start_minutes']
        if gap_end > gap_start:
            gap_minutes = float(gap_end - gap_start)
            gaps.append({
                'from_activity': sorted_acts[i],
                'to_activity': sorted_acts[i + 1],
                'gap_minutes': gap_minutes,
                'origin_lat': float(sorted_acts[i].get('lat', 0)),
                'origin_lng': float(sorted_acts[i].get('lng', 0)),
                'dest_lat': float(sorted_acts[i + 1].get('lat', 0)),
                'dest_lng': float(sorted_acts[i + 1].get('lng', 0)),
            })

    if not gaps:
        return {
            'success': False,
            'message': '当前活动安排无空闲时段（请确保活动之间有间隔）',
            'text_summary': '活动之间没有空闲时段，无法计算可达范围。',
        }

    calc = _get_calculator()
    default_mode = normalize_mode_key(travel_mode or 'walk')
    if commute_modes is None:
        commute_modes = []
    if not isinstance(commute_modes, list):
        commute_modes = []
    poi_type_defs = _selected_poi_type_defs(poi_types)
    active_type_ids = [item[0] for item in poi_type_defs]

    # 5. 各空档分别计算活动空间：活动1→活动2、活动2→活动3 …（N 个活动 → N-1 个空档）
    gap_results = []
    valid_gap_results = []
    activity_space_index = 0
    for gap_index, gap in enumerate(gaps):
        gap_travel_mode = commute_modes[gap_index] if gap_index < len(commute_modes) else default_mode
        print(
            f'[daily_range] 空档 {gap_index + 1}/{len(gaps)}：'
            f"{gap['from_activity'].get('name', '活动')} → "
            f"{gap['to_activity'].get('name', '活动')}（{gap_travel_mode}）",
            flush=True,
        )
        gap_res = _compute_single_gap_ppa(
            calc, gap, gap_travel_mode, poi_types, city=city,
            poi_activity_times=poi_activity_times,
        )
        print(
            f"[daily_range] 空档 {gap_index + 1} 完成 valid={gap_res.get('valid')} "
            f"method={gap_res.get('method')}",
            flush=True,
        )
        gap_res['travel_mode'] = normalize_mode_key(gap_travel_mode)
        gap_res['travel_mode_label'] = MODE_LABELS_ZH.get(
            normalize_mode_key(gap_travel_mode), gap_travel_mode
        )
        if gap_res.get('valid'):
            activity_space_index += 1
            gap_res['activity_space_index'] = activity_space_index
            gap_res['has_activity_space'] = True
            gap_res['segment_label'] = (
                f"活动空间 {activity_space_index}："
                f"{gap_res['from_activity'].get('name', '活动')} → "
                f"{gap_res['to_activity'].get('name', '活动')}"
            )
            valid_gap_results.append(gap_res)
        else:
            gap_res['activity_space_index'] = None
            gap_res['has_activity_space'] = False
            if gap_res.get('has_commute_route'):
                gap_res['segment_label'] = (
                    f"{gap_res['from_activity'].get('name', '活动')} → "
                    f"{gap_res['to_activity'].get('name', '活动')}（仅通勤路线）"
                )
            else:
                gap_res['segment_label'] = (
                    f"{gap_res['from_activity'].get('name', '活动')} → "
                    f"{gap_res['to_activity'].get('name', '活动')}（无弹性时间）"
                )
        gap_results.append(gap_res)

    route_only_results = [g for g in gap_results if g.get('has_commute_route')]

    if not valid_gap_results:
        if not route_only_results:
            reasons = [
                g.get('message') or f"{g['from_activity'].get('name', '')} → {g['to_activity'].get('name', '')}"
                for g in gap_results
            ]
            return {
                'success': False,
                'message': '空档时间不足以完成通勤，无法计算弹性活动空间',
                'text_summary': '空档时间不足以完成通勤，无法计算弹性活动空间。',
                'gap_details': [
                    {
                        'from': g['from_activity'].get('name', ''),
                        'to': g['to_activity'].get('name', ''),
                        'travel_mode': g.get('travel_mode', ''),
                        'travel_mode_label': g.get('travel_mode_label', ''),
                        'gap_minutes': round(g['gap_minutes'], 0),
                        'commute_time_min': round(g.get('commute_time_min', 0), 0),
                        'stay_time_min': round(g.get('stay_time_min', MIN_ACTIVITY_STAY_MINUTES), 0),
                        'elastic_time_min': round(g.get('elastic_time_min', 0), 0),
                        'method': g.get('method', ''),
                        'message': g.get('message', ''),
                        'route_source': g.get('route_source', ''),
                        'has_commute_route': bool(g.get('has_commute_route')),
                    }
                    for g in gap_results
                ],
                'gap_failure_reasons': reasons,
            }
        # 无活动空间，但有可展示的通勤路线
        route_only_spaces = [_build_route_only_space(g) for g in route_only_results]
        activity_markers = [
            {
                'lat': float(a.get('lat', 0)),
                'lng': float(a.get('lng', 0)),
                'name': str(a.get('name', '活动')),
                'start_time': a['_start_minutes'],
                'end_time': a['_end_minutes'],
                'type': 'activity',
            }
            for a in sorted_acts
        ]
        text_summary = (
            '空档时间不足以生成弹性活动空间（通勤+停留占用全部空档），'
            '已显示活动之间的通勤路线。'
        )
        return {
            'success': True,
            'overall_score': None,
            'timestamp': datetime.now().isoformat(),
            'activity_space_count': 0,
            'activity_space': {
                'cluster_count': 0,
                'spaces': route_only_spaces,
                'method': 'route_only_no_elastic',
                'input_points': [[m['lat'], m['lng']] for m in activity_markers],
                'input_points_with_type': [
                    {'coords': [m['lat'], m['lng']], 'type': 'other', 'name': m['name']}
                    for m in activity_markers
                ],
            },
            'activity_spaces': route_only_spaces,
            'radar_chart': {'labels': [], 'datasets': []},
            'accessibility_details': {},
            'accessible_pois': [],
            'activity_markers': activity_markers,
            'reachable_polygons': [],
            'poi_markers': [],
            'total_by_category': {},
            'text_summary': text_summary,
            'route_only': True,
            'gap_details': [
                {
                    'from': g['from_activity'].get('name', ''),
                    'to': g['to_activity'].get('name', ''),
                    'travel_mode': g.get('travel_mode', ''),
                    'travel_mode_label': g.get('travel_mode_label', ''),
                    'gap_minutes': round(g['gap_minutes'], 0),
                    'commute_time_min': round(g.get('commute_time_min', 0), 0),
                    'commute_dist_m': round(g.get('commute_dist_m', 0), 0),
                    'commute_dist_km': round(g.get('commute_dist_m', 0) / 1000.0, 2),
                    'stay_time_min': round(g.get('stay_time_min', MIN_ACTIVITY_STAY_MINUTES), 0),
                    'elastic_time_min': round(g.get('elastic_time_min', 0), 0),
                    'method': g.get('method', ''),
                    'message': g.get('message', ''),
                    'route_source': g.get('route_source', ''),
                    'has_activity_space': False,
                    'has_commute_route': bool(g.get('has_commute_route')),
                    'activity_space_index': None,
                    'segment_label': g.get('segment_label', ''),
                    'gap_score': None,
                    'accessible_poi_count': 0,
                }
                for g in gap_results
            ],
        }

    activity_spaces = []
    per_gap_scores = []
    gap_weight_list = []
    for gap_res in valid_gap_results:
        bounds = dict(gap_res['activity_bounds'])
        bounds['cluster_id'] = (gap_res.get('activity_space_index') or 1) - 1
        bounds['activity_space_index'] = gap_res.get('activity_space_index')
        bounds['segment_label'] = gap_res.get('segment_label')
        activity_spaces.append(bounds)
        gap_weight_list.append(gap_res['elastic_time_min'])
        per_gap_scores.append(gap_res.get('category_scores', {}))

    # 无弹性时间但仍有通勤路线的空档：一并放入 spaces 供地图展示
    for gap_res in route_only_results:
        activity_spaces.append(_build_route_only_space(gap_res))

    merged_scores, overall_score = _merge_gap_type_scores(
        per_gap_scores, gap_weight_list, poi_types
    )

    radar_labels = [POI_TYPE_LABELS_ZH.get(t, t) for t in active_type_ids]
    radar_data = [merged_scores.get(t, {}).get('raw_score', 0) for t in active_type_ids]

    all_activity_locations = [
        {
            'lat': float(a.get('lat', 0)),
            'lng': float(a.get('lng', 0)),
            'type': 'other',
            'name': str(a.get('name', '活动')),
        }
        for a in sorted_acts
    ]

    real_activity_spaces = [s for s in activity_spaces if not s.get('route_only')]
    ellipse_bounds = {
        'cluster_count': len(real_activity_spaces),
        'spaces': activity_spaces,
        'input_points': [[loc['lat'], loc['lng']] for loc in all_activity_locations],
        'input_points_with_type': [
            {'coords': [loc['lat'], loc['lng']], 'type': loc['type'], 'name': loc['name']}
            for loc in all_activity_locations
        ],
        'method': 'daily_gap_ppa_union',
    }
    if len(real_activity_spaces) == 1:
        ellipse_bounds.update(real_activity_spaces[0])
    elif real_activity_spaces:
        first = real_activity_spaces[0]
        ellipse_bounds.update({
            'center': first.get('center'),
            'center_lat': first.get('center_lat'),
            'center_lon': first.get('center_lon'),
            'polygon': first.get('polygon', []),
            'elastic_time_min': first.get('elastic_time_min'),
            'speed_ms': first.get('speed_ms'),
        })
        commute_route = first.get('commute_route')
        if commute_route:
            ellipse_bounds['commute_route'] = commute_route

    # 6. 在活动空间内按所选 POI 子类搜索（同大类合并查询）
    print('[daily_range] 开始活动空间内 POI 检索…', flush=True)
    category_groups = _group_poi_types_by_category(poi_types)
    type_results = _search_pois_in_gaps_by_category(calc, valid_gap_results, category_groups)
    total_by_category = {type_id: len(type_results.get(type_id, [])) for type_id in poi_types}
    print(f'[daily_range] POI 检索完成: {total_by_category}', flush=True)

    accessible_pois_list = _build_accessible_pois_from_type_results(type_results)

    # 7. 构建地图输出
    activity_markers = []
    for a in sorted_acts:
        activity_markers.append({
            'lat': float(a.get('lat', 0)),
            'lng': float(a.get('lng', 0)),
            'name': str(a.get('name', '活动')),
            'start_time': a['_start_minutes'],
            'end_time': a['_end_minutes'],
            'type': 'activity',
        })

    reachable_polygons = []
    for g in valid_gap_results:
        if g.get('polygon_coords'):
            reachable_polygons.append({
                'coords': g['polygon_coords'],
                'gap_minutes': round(g['gap_minutes'], 0),
                'commute_time_min': round(g.get('commute_time_min', 0), 0),
                'stay_time_min': round(g.get('stay_time_min', MIN_ACTIVITY_STAY_MINUTES), 0),
                'elastic_time_min': round(g.get('elastic_time_min', 0), 0),
                'from': g['from_activity'].get('name', ''),
                'to': g['to_activity'].get('name', ''),
                'method': g.get('method', ''),
                'max_one_way_m': g.get('max_one_way_m'),
            })

    poi_markers = []
    for type_id, pois in type_results.items():
        poi_category = POI_TYPE_DEFS[type_id][0]
        color = POI_CATEGORY_COLORS.get(poi_category, '#909399')
        label = POI_TYPE_LABELS_ZH.get(type_id, type_id)
        for poi in pois:
            c = poi.get('coords', [0, 0])
            poi_markers.append({
                'lat': float(c[0]),
                'lng': float(c[1]),
                'name': str(poi.get('name', '')),
                'category': type_id,
                'category_label': label,
                'color': color,
                'gap_info': str(poi.get('gap_name', '')),
                'activity_space_index': poi.get('activity_space_index'),
            })

    # 8. 构建文字摘要
    summary_commute_modes = [
        commute_modes[i] if i < len(commute_modes) else default_mode
        for i in range(len(sorted_acts) - 1)
    ]
    text_summary = _build_text_summary(sorted_acts, summary_commute_modes, total_by_category)

    return {
        'success': True,
        'overall_score': overall_score,
        'timestamp': datetime.now().isoformat(),
        'activity_space_count': len(real_activity_spaces),
        'activity_space': ellipse_bounds,
        'activity_spaces': activity_spaces,
        'radar_chart': {
            'labels': radar_labels,
            'datasets': [{
                'label': '可达性得分',
                'data': [round(d, 4) for d in radar_data],
            }],
        },
        'accessibility_details': {
            type_id: {
                'raw_score': merged_scores.get(type_id, {}).get('raw_score', 0),
                'poi_count': merged_scores.get(type_id, {}).get('poi_count', 0),
                'weight': merged_scores.get(type_id, {}).get('weight', 0),
                'label': POI_TYPE_LABELS_ZH.get(type_id, type_id),
            }
            for type_id in active_type_ids
        },
        'accessible_pois': accessible_pois_list,
        'evaluation_spaces': [
            {
                'geometry': gap['evaluation_geometry'],
                'scored_pois': [
                    {**poi, 'type': type_id}
                    for type_id, score in gap.get('category_scores', {}).items()
                    for poi in score.get('pois', [])
                ],
            }
            for gap in valid_gap_results
        ],
        'activity_markers': activity_markers,
        'reachable_polygons': reachable_polygons,
        'poi_markers': poi_markers,
        'total_by_category': total_by_category,
        'text_summary': text_summary,
        'gap_details': [
            {
                'from': g['from_activity'].get('name', ''),
                'to': g['to_activity'].get('name', ''),
                'travel_mode': g.get('travel_mode', ''),
                'travel_mode_label': g.get('travel_mode_label', ''),
                'gap_minutes': round(g['gap_minutes'], 0),
                'commute_time_min': round(g.get('commute_time_min', 0), 0),
                'commute_dist_m': round(g.get('commute_dist_m', 0), 0),
                'commute_dist_km': round(g.get('commute_dist_m', 0) / 1000.0, 2),
                'stay_time_min': round(g.get('stay_time_min', MIN_ACTIVITY_STAY_MINUTES), 0),
                'elastic_time_min': round(g.get('elastic_time_min', 0), 0),
                'method': g.get('method', ''),
                'max_one_way_m': g.get('max_one_way_m'),
                'message': g.get('message', ''),
                'route_source': g.get('route_source', ''),
                'walk_fallback': bool(g.get('walk_fallback')),
                'has_activity_space': bool(g.get('has_activity_space')),
                'has_commute_route': bool(g.get('has_commute_route')),
                'activity_space_index': g.get('activity_space_index'),
                'segment_label': g.get('segment_label', ''),
                'gap_score': round(g.get('overall_score', 0), 4) if g.get('valid') else None,
                'accessible_poi_count': sum(
                    1 for p in poi_markers
                    if p.get('activity_space_index') == g.get('activity_space_index')
                ) if g.get('has_activity_space') else 0,
            }
            for g in gap_results
        ],
    }
