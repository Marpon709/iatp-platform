import math
import os
from datetime import datetime

from shapely import concave_hull
from shapely.geometry import Point, LineString, Polygon, MultiPoint
from shapely.ops import unary_union
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from transport_modes import (
    get_route_geometry_coords,
    normalize_mode_key,
    normalize_route_mode_for_calc,
    speed_ms as _transport_speed_ms,
)

DEFAULT_TRAVEL_TIME_MIN = 30  # 分钟生活圈默认值，与基于位置可达性一致
DECAY_BETA = 0.003  # 距离衰减参数 β (距离单位：米)。若用公里则β=0.1
# 通勤路径采样：沿路线每 200m 取关键节点，在超级网络上做局部 STP 后网络缓冲/凹壳并集
PPA_ROUTE_SAMPLE_INTERVAL_M = 200.0
PPA_ROUTE_SAMPLE_MAX_K = 20
PPA_ROUTE_SAMPLE_MAX_K_CAP = 300
# 偏离路径再返回同一锚点时，局部探索时间按往返计（cost ≤ B_local/2）；单程偏离则 False
PPA_ROUND_TRIP_DEVIATION = True
# 必经路径走廊半宽（米）
PPA_PATH_CORRIDOR_HALF_WIDTH_M = 60.0
# 可达节点网络缓冲：默认半径与密度自适应上下限（米）
PPA_NETWORK_BUFFER_R_M = 80.0
PPA_NETWORK_BUFFER_R_MIN_M = 40.0
PPA_NETWORK_BUFFER_R_MAX_M = 150.0
# 缓冲并集点数上限（避免数百圆 unary_union 卡死）
PPA_BUFFER_POINT_MAX = 100
PPA_BUFFER_CIRCLE_POINTS = 16
PPA_CORRIDOR_POINT_MAX = 80
PPA_POLYGON_SIMPLIFY_DEG = 2.0e-5
# Alpha Shapes / 凹壳：ratio∈[0,1]，越小越贴合点集、越能保留空洞
PPA_ALPHA_RATIO = 0.25
PPA_ALPHA_ALLOW_HOLES = True
PPA_ALPHA_POINT_MAX = 400
PPA_MAX_POI_PER_CATEGORY = 0
# 路网距离场：相对通勤路径的最大搜索半径（米）
PPA_ROUTE_DIST_FIELD_BUFFER_M = 1500.0
_EARTH_RADIUS_M = 6371000.0
_DEG_LAT_TO_M = 111320.0

ACTIVITY_TO_POI = {
    'education': 'education',           # 教育
    'recreation': 'recreation',         # 娱乐
    'shopping': 'shopping',
    'catering': 'catering',
    'medical': 'medical',               # 医疗
}

ACTIVITY_LABELS = {
    'education': '教育',
    'recreation': '娱乐',
    'shopping': '购物',
    'catering': '餐饮',
    'medical': '医疗',
}

# 默认权重（等权重，5类活动）
DEFAULT_WEIGHTS = {
    'education': 1.0 / 5,
    'recreation': 1.0 / 5,
    'shopping': 1.0 / 5,
    'catering': 1.0 / 5,
    'medical': 1.0 / 5,
}

_WEIGHT_ZERO_EPS = 1e-9

def _is_zero_activity_weight(weight):
    """活动权重是否为 0%（不参与可达性计算与 POI 展示）。"""
    try:
        return float(weight) <= _WEIGHT_ZERO_EPS
    except (TypeError, ValueError):
        return False

def _parse_trip_counts(trip_counts):
    """解析前端 weekly trip counts，返回 {activity: int} 或 None。"""
    if not isinstance(trip_counts, dict):
        return None
    parsed = {}
    for activity in DEFAULT_WEIGHTS:
        if activity not in trip_counts:
            continue
        try:
            parsed[activity] = max(0, int(float(trip_counts[activity])))
        except (TypeError, ValueError):
            parsed[activity] = 0
    if not parsed:
        return None
    for activity in DEFAULT_WEIGHTS:
        parsed.setdefault(activity, 0)
    return parsed

def _resolve_final_weights(weights=None, trip_counts=None):
    final_weights = DEFAULT_WEIGHTS.copy()

    parsed_trips = _parse_trip_counts(trip_counts)
    if parsed_trips is not None:
        total_trips = sum(parsed_trips.values())
        if total_trips <= 0:
            return {activity: 0.0 for activity in DEFAULT_WEIGHTS}
        for activity in DEFAULT_WEIGHTS:
            count = parsed_trips.get(activity, 0)
            final_weights[activity] = 0.0 if count <= 0 else float(count) / float(total_trips)
        return final_weights

    if weights:
        try:
            final_weights.update({k: float(v) for k, v in weights.items()})
        except (TypeError, ValueError):
            pass
    return final_weights

def _active_activities(final_weights):
    return [
        activity for activity in DEFAULT_WEIGHTS
        if not _is_zero_activity_weight(final_weights.get(activity, 0))
    ]

def _build_accessible_pois_display_list(all_accessible_pois, category_scores, active_activity_keys):
    """
    构建前端「显示可达 POI」列表。
    优先使用时间预算内从出发地可达的 POI；若某类为空则回退到活动空间内 POI。
    """
    accessible_pois_list = []
    for activity in active_activity_keys:
        pois = list(all_accessible_pois.get(activity) or [])
        if not pois:
            for scored in category_scores.get(activity, {}).get("pois", []):
                pois.append({
                    "coords": scored.get("coords"),
                    "network_dist": scored.get("distance"),
                    "travel_time": scored.get("travel_time"),
                    "name": scored.get("name"),
                })

        for poi in pois:
            network_dist = poi.get("network_dist")
            if network_dist is None:
                network_dist = poi.get("distance")
            travel_time = poi.get("travel_time")
            decay = poi.get("decay")
            if decay is None and network_dist is not None:
                try:
                    decay = round(math.exp(-DECAY_BETA * float(network_dist)), 4)
                except (TypeError, ValueError):
                    decay = None

            accessible_pois_list.append({
                "activity": activity,
                "activity_label": ACTIVITY_LABELS.get(activity, activity),
                "coords": poi.get("coords"),
                "distance": round(float(network_dist), 1) if network_dist is not None else None,
                "travel_time": round(float(travel_time), 1) if travel_time is not None else None,
                "decay": decay,
                "contribution": decay if decay is not None else poi.get("contribution"),
                "name": poi.get("name"),
            })

    return accessible_pois_list

def _latlon_rad_for_haversine_tree(lon_arr, lat_arr):
    lon_arr = np.asarray(lon_arr, dtype=np.float64).reshape(-1)
    lat_arr = np.asarray(lat_arr, dtype=np.float64).reshape(-1)
    return np.column_stack([np.radians(lat_arr), np.radians(lon_arr)])

def _haversine_tree_radius_rad(radius_m):
    return float(radius_m) / _EARTH_RADIUS_M

def _commute_speed_ms(commute_mode):
    """前端交通方式 → 速度（m/s），与 transport_modes 一致。"""
    return _transport_speed_ms(commute_mode)

def _lateral_expansion_speed_ms(commute_mode):
    key = normalize_mode_key(commute_mode)
    if key in ("metro", "subway", "bus"):
        return max(_commute_speed_ms("walk"), 0.1)
    return max(_commute_speed_ms(commute_mode), 0.1)

def _haversine_m(lon1, lat1, lon2, lat2):
    """计算两点间的大圆距离（米）"""
    r = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return r * 2 * math.asin(math.sqrt(a))

def _get_calculator():
    from city_engine import get_shared_calculator
    return get_shared_calculator(city='南京')

def _get_pois_in_radius(calc, category, p_origin, radius_m, max_count=200):
    if calc is None or not hasattr(calc, 'poi_trees'):
        return []
    
    act_type = category.lower()
    
    if act_type not in calc.poi_trees:
        return []
    
    try:
        tree, coords, gdf = calc.poi_trees[act_type]
        if len(coords) == 0:
            return []
        
        name_field = None
        for field in ['名称', 'name', 'NAME', 'Name', 'title', 'Title']:
            if field in gdf.columns:
                name_field = field
                break
        
        # 使用haversine距离进行空间查询（返回米制距离）
        orig = np.array([[p_origin.x, p_origin.y]])
        k = min(max_count * 2, len(coords))
        distances, indices = tree.query(orig, k=k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            # BallTree返回的是欧氏距离（度），需要转换为米
            balltree_dist = distances[0][i]
            
            nc = coords[idx]
            poi_lon, poi_lat = float(nc[0]), float(nc[1])
            
            # 用haversine计算实际球面距离（米）
            haversine_dist = _haversine_m(p_origin.x, p_origin.y, poi_lon, poi_lat)
            
            # 使用haversine距离进行筛选
            if haversine_dist > radius_m * 1.5:
                continue
            
            poi_name = None
            if name_field is not None and idx < len(gdf):
                poi_name = str(gdf.iloc[idx][name_field]) if pd.notna(gdf.iloc[idx][name_field]) else None
            
            pt = Point(poi_lon, poi_lat)
            results.append({
                "point": pt,
                "distance": haversine_dist,  # 使用haversine距离
                "coords": [poi_lat, poi_lon],  # [lat, lon]
                "name": poi_name
            })
        
        return results
    except Exception as e:
        import traceback
        traceback.print_exc()
        return []

def _create_circle_polygon(center_lon, center_lat, radius_m, num_points=64):
    """创建圆形多边形（用于等时圈）"""
    points = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        dx = radius_m * math.cos(angle)
        dy = radius_m * math.sin(angle)
        lat = center_lat + (dy / 111320)
        lon = center_lon + (dx / (111320 * math.cos(math.radians(center_lat))))
        points.append([lon, lat])
    points.append(points[0])
    return Polygon(points)

def _densify_route_coords(coords, max_step_m=100.0):
    """沿路径折线加密顶点，使走廊贴合路网弯道而非起终点捷径。"""
    if not coords or len(coords) < 2:
        return list(coords or [])
    max_step_m = max(20.0, float(max_step_m))
    dense = [coords[0]]
    for i in range(1, len(coords)):
        prev, curr = coords[i - 1], coords[i]
        seg_m = _haversine_m(prev[1], prev[0], curr[1], curr[0])
        if seg_m <= max_step_m:
            if dense[-1] != curr:
                dense.append(curr)
            continue
        steps = max(1, int(math.ceil(seg_m / max_step_m)))
        for s in range(1, steps + 1):
            t = s / steps
            lat = prev[0] + (curr[0] - prev[0]) * t
            lng = prev[1] + (curr[1] - prev[1]) * t
            dense.append([lat, lng])
    return dense

def _buffer_route_polyline_m(route_coords, buffer_m):
    """沿规划路径折线（米制等宽）生成走廊多边形。"""
    if not route_coords or len(route_coords) < 2:
        return None
    buffer_m = max(float(buffer_m), 50.0)
    step_m = max(buffer_m * 0.9, 80.0)
    dense = _densify_route_coords(route_coords, max_step_m=step_m)
    if len(dense) > PPA_CORRIDOR_POINT_MAX:
        step = (len(dense) - 1) / (PPA_CORRIDOR_POINT_MAX - 1)
        dense = [dense[int(round(i * step))] for i in range(PPA_CORRIDOR_POINT_MAX)]
        if dense[-1] != route_coords[-1]:
            dense[-1] = route_coords[-1]
    circles = [
        _create_circle_polygon(
            float(lng), float(lat), buffer_m, num_points=PPA_BUFFER_CIRCLE_POINTS,
        )
        for lat, lng in dense
    ]
    if not circles:
        return None
    merged = unary_union(circles) if len(circles) > 1 else circles[0]
    return _simplify_activity_polygon(merged)

def _dest_aware_corridor_from_route(route_coords, total_budget_sec, commute_sec, speed_ms):
    """
    沿规划通勤路径（路网折线）按局部时间预算生成变宽走廊。
    B_local(k)=T_budget-t_actual(k)-t_remain_need(k)，remain 用弧长剩余。
    """
    if not route_coords or len(route_coords) < 2:
        return None, 0.0
    dense = _densify_route_coords(route_coords, max_step_m=100.0)
    cum_times_sec = _route_cumulative_times_by_arc(dense, commute_sec)
    circles = []
    max_lateral_m = 0.0
    for (lat, lng), t_accum in zip(dense, cum_times_sec):
        t_remain = max(0.0, float(commute_sec) - float(t_accum))
        b_local = _local_budget_sec(total_budget_sec, t_accum, t_remain)
        if b_local <= 0:
            continue
        radius_m = b_local * speed_ms * (0.5 if PPA_ROUND_TRIP_DEVIATION else 1.0)
        max_lateral_m = max(max_lateral_m, radius_m)
        circles.append(_create_circle_polygon(float(lng), float(lat), max(radius_m, 50.0)))
    if not circles:
        return None, max_lateral_m
    poly = unary_union(circles) if len(circles) > 1 else circles[0]
    return poly, max_lateral_m

def _mode_for_network_travel(commute_mode):
    """前端交通方式 → CarbonBudgetCalculator.mode_config 键。"""
    key = normalize_mode_key(commute_mode)
    return "car" if key == "drive" else key

def _extract_home_work(locations):
    home_loc = work_loc = None
    for loc in locations:
        t = loc.get("type")
        if t == "home" and home_loc is None:
            home_loc = loc
        elif t == "work" and work_loc is None:
            work_loc = loc
    return home_loc, work_loc

def _location_to_point(loc):
    return Point(float(loc["lng"]), float(loc["lat"]))

def _nearest_road_node(calc, point, mode_key):
    """定位路网上最近节点，返回 (node_id, node_lon, node_lat, snap_m)。"""
    if calc is None or mode_key not in getattr(calc, "network_trees", {}):
        return None, None, None, float("inf")
    tree, _coords, nodes = calc.network_trees[mode_key]
    q = np.array([[point.x, point.y]])
    _dist, idx = tree.query(q, k=1)
    idx = int(idx[0][0])
    node_id = nodes.index[idx]
    node_geom = nodes.iloc[idx].geometry
    snap_m = _haversine_m(point.x, point.y, node_geom.x, node_geom.y)
    return node_id, float(node_geom.x), float(node_geom.y), snap_m

def _batch_snap_road_nodes(calc, points, mode_key):
    """批量吸附到路网节点，返回 (node_ids, snap_ms)。"""
    n = len(points)
    if n == 0:
        return [], []
    if calc is None or mode_key not in getattr(calc, "network_trees", {}):
        return [None] * n, [float("inf")] * n

    tree, _coords, nodes = calc.network_trees[mode_key]
    q = np.array([[float(p.x), float(p.y)] for p in points], dtype=np.float64)
    _dist, idx = tree.query(q, k=1)
    node_ids = []
    snap_ms = []
    for i in range(n):
        j = int(idx[i][0])
        node_id = nodes.index[j]
        node_geom = nodes.iloc[j].geometry
        node_ids.append(node_id)
        snap_ms.append(
            _haversine_m(points[i].x, points[i].y, node_geom.x, node_geom.y)
        )
    return node_ids, snap_ms

def _collect_route_anchor_node_ids(calc, route_coords, mode_key):
    from transport_modes import _graph_node_id

    anchors = set()
    if not route_coords or calc is None:
        return anchors
    if mode_key not in getattr(calc, "network_trees", {}):
        return anchors

    network_key = calc.mode_config.get(mode_key, {}).get("network")
    G = calc.networks.get(network_key) if network_key else None

    for lat, lng in route_coords:
        pt = Point(float(lng), float(lat))
        nid, _, _, _ = _nearest_road_node(calc, pt, mode_key)
        if nid is None:
            continue
        if G is not None:
            anchors.add(_graph_node_id(G, nid))
        else:
            anchors.add(nid)
    return anchors

def _multi_source_dist_to_route_m(G, anchor_nodes, cutoff_m):
    """
    多源 Dijkstra：每个路网节点到最近路径锚点的路网距离（米）。
  步行网为有向 MultiDiGraph 时自动回退无向图。
    """
    import networkx as nx
    from transport_modes import _graph_node_id

    if G is None or not anchor_nodes:
        return {}

    sources = []
    seen = set()
    for nid in anchor_nodes:
        sid = _graph_node_id(G, nid)
        if sid in G and sid not in seen:
            sources.append(sid)
            seen.add(sid)
    if not sources:
        return {}

    cutoff_m = max(float(cutoff_m), 500.0)
    graphs = [G]
    if getattr(G, "is_directed", lambda: False)():
        try:
            graphs.append(G.to_undirected())
        except Exception:
            pass

    best = {}
    for g in graphs:
        try:
            dists = nx.multi_source_dijkstra_path_length(
                g, sources, weight="length", cutoff=cutoff_m
            )
        except Exception:
            try:
                dists = nx.multi_source_dijkstra_path_length(
                    g, sources, cutoff=cutoff_m
                )
            except Exception:
                continue
        for nid, d in dists.items():
            d = float(d)
            if not math.isfinite(d):
                continue
            if nid not in best or d < best[nid]:
                best[nid] = d
    return best

def build_route_distance_context(calc, route_coords, commute_mode, max_one_way_m):
    """
    预计算「路网节点 → 通勤路径」距离场，供活动空间内全部 POI 快速查表。

    返回 dict：mode_key, dist_to_route, cutoff_m, anchor_count；失败时 dist_to_route 为空。
    """
    mode_key = _mode_for_network_travel(commute_mode)
    ctx = {
        "mode_key": mode_key,
        "dist_to_route": {},
        "cutoff_m": 0.0,
        "anchor_count": 0,
    }
    if (
        calc is None
        or not route_coords
        or len(route_coords) < 2
        or mode_key not in getattr(calc, "network_trees", {})
        or mode_key not in getattr(calc, "mode_config", {})
    ):
        return ctx

    network_key = calc.mode_config[mode_key]["network"]
    G = calc.networks.get(network_key)
    if G is None or G.number_of_nodes() == 0:
        return ctx

    anchors = _collect_route_anchor_node_ids(calc, route_coords, mode_key)
    if not anchors:
        return ctx

    cutoff_m = max(
        float(max_one_way_m) + float(PPA_ROUTE_DIST_FIELD_BUFFER_M),
        800.0,
    )
    dist_field = _multi_source_dist_to_route_m(G, anchors, cutoff_m)
    print(
        f"[DEBUG] 通勤路径距离场: 锚点={len(anchors)}, "
        f"cutoff={cutoff_m:.0f}m, 覆盖节点={len(dist_field)}",
        flush=True,
    )
    ctx["dist_to_route"] = dist_field
    ctx["cutoff_m"] = cutoff_m
    ctx["anchor_count"] = len(anchors)
    return ctx

def _route_arc_length_m(coords):
    """路径折线 [[lat, lng], ...] 总弧长（米）。"""
    if not coords or len(coords) < 2:
        return 0.0
    total_m = 0.0
    for i in range(1, len(coords)):
        total_m += _haversine_m(
            coords[i - 1][1], coords[i - 1][0], coords[i][1], coords[i][0],
        )
    return total_m

def _sample_k_max_for_route(coords, interval_m=None):
    """按路径长度动态确定采样上限，避免长通勤路线采样过稀。"""
    interval_m = max(50.0, float(interval_m or PPA_ROUTE_SAMPLE_INTERVAL_M))
    route_m = _route_arc_length_m(coords)
    needed = int(route_m / interval_m) + 2
    return min(PPA_ROUTE_SAMPLE_MAX_K_CAP, max(PPA_ROUTE_SAMPLE_MAX_K, needed))

def _local_budget_sec(t_budget_sec, t_actual_sec, t_remain_need_sec):
    """
    路径关键节点 k 的局部时间预算：
    B_local(k) = T_budget - (t_actual(k) - t_0) - t_remain_need(k)
    （t_0=0，t_actual 为自起点累计实际到达时间）
    """
    return max(
        0.0,
        float(t_budget_sec) - float(t_actual_sec) - float(t_remain_need_sec),
    )

def _lateral_budget_sec_at_route_sample(remaining_sec, t_accum, commute_sec):
    """
    兼容旧调用：在 remaining_sec=T_budget-t_accum、t_remain=commute-t_accum 时
    等价于 B_local = T_budget - t_accum - (commute - t_accum) = 弹性时间。
    新逻辑请优先使用 _local_budget_sec + 反向搜索 remain_need。
    """
    t_remain_need = max(0.0, float(commute_sec) - float(t_accum))
    # remaining_sec 历史语义为「自当前点起剩余总预算」，换算回 T_budget
    t_budget = float(remaining_sec) + float(t_accum)
    return _local_budget_sec(t_budget, t_accum, t_remain_need)

def _sample_coords_along_path(coords, interval_m, k_max):
    """
    沿路径折线 [[lat, lng], ...] 按弧长间隔采样，首尾必含。
    """
    if not coords:
        return []
    if len(coords) == 1:
        return [coords[0]]

    interval_m = max(50.0, float(interval_m))
    k_max = max(2, int(k_max))
    samples = [coords[0]]
    acc = 0.0
    for i in range(1, len(coords)):
        d = _haversine_m(coords[i - 1][1], coords[i - 1][0], coords[i][1], coords[i][0])
        acc += d
        if acc >= interval_m:
            samples.append(coords[i])
            acc = 0.0
    if samples[-1] != coords[-1]:
        samples.append(coords[-1])

    # 去重（保持顺序）
    deduped = []
    seen = set()
    for c in samples:
        key = (round(c[0], 6), round(c[1], 6))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)

    if len(deduped) <= k_max:
        return deduped

    # 超过上限时均匀抽稀，始终保留首尾
    if k_max < 2:
        return [deduped[0], deduped[-1]]
    step = (len(deduped) - 1) / (k_max - 1)
    picked = [deduped[0]]
    for i in range(1, k_max - 1):
        picked.append(deduped[int(round(i * step))])
    if picked[-1] != deduped[-1]:
        picked.append(deduped[-1])
    return picked

def _route_cumulative_times_by_arc(sampled_coords, total_time_sec):
    """按路径弧长比例，为各采样点分配自起点沿通勤路径累计的通行时间（秒）。"""
    if not sampled_coords:
        return []
    total_time_sec = max(0.0, float(total_time_sec))
    if len(sampled_coords) == 1:
        return [0.0]

    cum_dist_m = [0.0]
    total_m = 0.0
    for i in range(1, len(sampled_coords)):
        prev, curr = sampled_coords[i - 1], sampled_coords[i]
        seg_m = _haversine_m(prev[1], prev[0], curr[1], curr[0])
        total_m += seg_m
        cum_dist_m.append(total_m)

    if total_m <= 1e-6:
        return [0.0] * len(sampled_coords)
    return [total_time_sec * (d / total_m) for d in cum_dist_m]

def _undirected_search_graph(G):
    """
    超级网络侧向搜索图：有向 MultiDiGraph 转无向，便于双向可达。
    按图对象缓存，避免每个采样点重复 to_undirected（城市路网极慢）。
    """
    if G is None:
        return None
    if not getattr(G, "is_directed", lambda: False)():
        return G
    cache = getattr(_undirected_search_graph, "_cache", None)
    if cache is None:
        cache = {}
        _undirected_search_graph._cache = cache
    key = id(G)
    hit = cache.get(key)
    if hit is not None:
        return hit
    n = G.number_of_nodes()
    print(f"[DEBUG] 构建无向搜索图缓存: nodes={n}, edges={G.number_of_edges()}", flush=True)
    try:
        ug = G.to_undirected()
    except Exception:
        ug = G
    cache[key] = ug
    # 防止无限增长：仅保留最近若干张图
    if len(cache) > 8:
        for old_key in list(cache.keys())[: len(cache) - 8]:
            cache.pop(old_key, None)
    return ug

def _subsample_lonlat_pts(pts, max_n):
    pts = list(pts or [])
    max_n = max(2, int(max_n))
    if len(pts) <= max_n:
        return pts
    step = (len(pts) - 1) / (max_n - 1)
    out = [pts[0]]
    for i in range(1, max_n - 1):
        out.append(pts[int(round(i * step))])
    out.append(pts[-1])
    return out

def _simplify_activity_polygon(polygon, tol=None):
    """简化活动空间边界，加速后续 POI contains 与前端传输。"""
    if polygon is None or getattr(polygon, "is_empty", True):
        return polygon
    tol = float(PPA_POLYGON_SIMPLIFY_DEG if tol is None else tol)
    try:
        simplified = polygon.simplify(tol, preserve_topology=True)
        if simplified is not None and not simplified.is_empty:
            return _normalize_display_polygon(simplified)
    except Exception:
        pass
    return _normalize_display_polygon(polygon)

def _network_dist_field_from_node(G, source_node_id, cutoff_m, undirected=True, search_graph=None):
    """路网节点到 source 的最短路径距离场（米）。"""
    import networkx as nx
    from transport_modes import _graph_node_id

    if G is None or source_node_id is None:
        return {}
    if search_graph is not None:
        g = search_graph
    else:
        g = _undirected_search_graph(G) if undirected else G
    sid = _graph_node_id(g, source_node_id)
    if sid not in g:
        return {}
    try:
        return nx.single_source_dijkstra_path_length(
            g, sid, weight="length", cutoff=max(float(cutoff_m), 500.0)
        )
    except Exception:
        try:
            return nx.single_source_dijkstra_path_length(
                g, sid, cutoff=max(float(cutoff_m), 500.0)
            )
        except Exception:
            return {}

def _reverse_remain_need_sec(
    dist_to_dest_field, node_id, snap_dest_m, speed_ms, arc_remain_sec=None,
):
    """
    由终点反向距离场得到 t_remain_need(k)（秒）。
    有路网距离时用 network；公交/地铁走廊上同时给弧长剩余时间时取二者较小值，
    避免步行网反向距离夸大必经剩余时间。
    """
    speed_ms = max(float(speed_ms), 0.1)
    net_sec = None
    if dist_to_dest_field and node_id is not None:
        dist_m = None
        for key in (node_id, str(node_id)):
            if key in dist_to_dest_field:
                dist_m = dist_to_dest_field[key]
                break
        if dist_m is not None and math.isfinite(float(dist_m)):
            net_sec = (float(dist_m) + float(snap_dest_m or 0.0)) / speed_ms

    if arc_remain_sec is not None:
        arc_sec = max(0.0, float(arc_remain_sec))
        if net_sec is None:
            return arc_sec
        return min(net_sec, arc_sec)
    if net_sec is not None:
        return max(0.0, net_sec)
    return 0.0

def _dijkstra_local_stp_reachable_lonlat(
    G, origin_node_id, snap_m, b_local_sec, speed_ms,
    round_trip=None, search_graph=None,
):
    """
    以节点 k 为中心，在超级网络上搜索满足
    cost(k→m) ≤ B_local(k)/2 （往返）或 ≤ B_local （单程偏离）的邻居。
    返回可达点 (lon, lat) 集合。
    """
    import networkx as nx
    from transport_modes import _graph_node_id

    if round_trip is None:
        round_trip = PPA_ROUND_TRIP_DEVIATION
    b_local_sec = max(0.0, float(b_local_sec))
    speed_ms = max(float(speed_ms), 0.1)
    if b_local_sec <= 0 or G is None or origin_node_id is None:
        return set()

    explore_sec = b_local_sec * 0.5 if round_trip else b_local_sec
    max_one_way_m = explore_sec * speed_ms
    g = search_graph if search_graph is not None else _undirected_search_graph(G)
    sid = _graph_node_id(g, origin_node_id)
    if sid not in g:
        return set()

    dijk_cutoff_m = max(max_one_way_m + float(snap_m or 0.0), 500.0)
    try:
        dists = nx.single_source_dijkstra_path_length(
            g, sid, weight="length", cutoff=dijk_cutoff_m
        )
    except Exception:
        try:
            dists = nx.single_source_dijkstra_path_length(
                g, sid, cutoff=dijk_cutoff_m
            )
        except Exception:
            return set()

    reachable = set()
    for node_id, network_dist_m in dists.items():
        # 吸附 + 路网偏离；往返时出/回各算一次网络距离
        out_m = float(snap_m or 0.0) + float(network_dist_m)
        if round_trip:
            total_m = out_m + float(network_dist_m) + float(snap_m or 0.0)
            if total_m / speed_ms > b_local_sec + 1e-6:
                continue
        else:
            if out_m / speed_ms > b_local_sec + 1e-6:
                continue
        nd = g.nodes[node_id] if node_id in g.nodes else G.nodes.get(node_id, {})
        lon, lat = nd.get("x"), nd.get("y")
        if lon is None or lat is None:
            continue
        reachable.add((float(lon), float(lat)))
    return reachable

def _dynamic_network_buffer_radius_m(reachable_pts, default_r=None):
    """按可达节点密度自适应缓冲半径 r（米）。"""
    default_r = float(default_r if default_r is not None else PPA_NETWORK_BUFFER_R_M)
    pts = list(reachable_pts or [])
    if len(pts) < 2:
        return max(PPA_NETWORK_BUFFER_R_MIN_M, default_r)

    # 子采样估计平均最近邻距离，避免 O(n²) 过大
    sample = pts
    if len(pts) > 80:
        step = max(1, len(pts) // 80)
        sample = pts[::step]
    nn_dists = []
    for i, (lon1, lat1) in enumerate(sample):
        best = None
        for j, (lon2, lat2) in enumerate(sample):
            if i == j:
                continue
            d = _haversine_m(lon1, lat1, lon2, lat2)
            if best is None or d < best:
                best = d
        if best is not None and best > 1.0:
            nn_dists.append(best)
    if not nn_dists:
        return max(PPA_NETWORK_BUFFER_R_MIN_M, default_r)
    mean_nn = float(sum(nn_dists) / len(nn_dists))
    r = 0.55 * mean_nn
    return float(min(PPA_NETWORK_BUFFER_R_MAX_M, max(PPA_NETWORK_BUFFER_R_MIN_M, r)))

def _buffer_lonlat_points_polygon(reachable_pts, radius_m):
    """节点投影后的欧氏缓冲并集：⋃ Buffer(n, r)；点数过多时抽稀。"""
    pts = _subsample_lonlat_pts(reachable_pts, PPA_BUFFER_POINT_MAX)
    if not pts:
        return None
    radius_m = max(float(radius_m), PPA_NETWORK_BUFFER_R_MIN_M)
    circles = [
        _create_circle_polygon(
            float(lon), float(lat), radius_m, num_points=PPA_BUFFER_CIRCLE_POINTS,
        )
        for lon, lat in pts
    ]
    if not circles:
        return None
    merged = unary_union(circles) if len(circles) > 1 else circles[0]
    return _simplify_activity_polygon(merged)

def _alpha_shape_polygon(reachable_pts, ratio=None, allow_holes=None):
    """
    Alpha Shapes / 凹壳边界：比凸包更能保留路网造成的空洞
    （河流、封闭小区等不可穿越区域）。
    """
    pts = _subsample_lonlat_pts(reachable_pts, PPA_ALPHA_POINT_MAX)
    if len(pts) < 3:
        return None
    ratio = float(PPA_ALPHA_RATIO if ratio is None else ratio)
    allow_holes = PPA_ALPHA_ALLOW_HOLES if allow_holes is None else bool(allow_holes)
    try:
        mp = MultiPoint([Point(float(lon), float(lat)) for lon, lat in pts])
        hull = concave_hull(mp, ratio=ratio, allow_holes=allow_holes)
    except Exception:
        return None
    if hull is None or getattr(hull, "is_empty", True):
        return None
    if hull.geom_type == "Polygon":
        return hull
    if hull.geom_type == "MultiPolygon":
        parts = [g for g in hull.geoms if g.geom_type == "Polygon" and not g.is_empty]
        return max(parts, key=lambda p: p.area) if parts else None
    if hull.geom_type == "LineString":
        try:
            return hull.buffer(1e-5)
        except Exception:
            return None
    return None

def _lonlat_set_to_activity_polygon(
    reachable_pts, fallback_pt, max_one_way_m, route_coords=None,
):
    """
    将超级网络可达节点集 V_STP 几何化：
    优先 Alpha Shapes 凹壳 + 必经路径走廊；节点缓冲仅对抽稀点集，避免卡顿。
    """
    import time as _time

    pts = list(reachable_pts or [])
    t0 = _time.perf_counter()
    radius_m = _dynamic_network_buffer_radius_m(
        pts, default_r=min(float(max_one_way_m or PPA_NETWORK_BUFFER_R_M), PPA_NETWORK_BUFFER_R_MAX_M),
    )
    parts = []

    # 凹壳为主边界（快）；节点缓冲为辅且严格限点数
    alpha = _alpha_shape_polygon(pts)
    if alpha is not None and not alpha.is_empty:
        try:
            alpha_buf = alpha.buffer(0)
            if alpha_buf is not None and not alpha_buf.is_empty:
                parts.append(alpha_buf)
            else:
                parts.append(alpha)
        except Exception:
            parts.append(alpha)

    buf = _buffer_lonlat_points_polygon(pts, radius_m)
    if buf is not None and not buf.is_empty:
        parts.append(buf)

    if route_coords and len(route_coords) >= 2:
        corridor = _buffer_route_polyline_m(
            route_coords, max(PPA_PATH_CORRIDOR_HALF_WIDTH_M, radius_m * 0.5),
        )
        if corridor is not None and not getattr(corridor, "is_empty", True):
            parts.append(corridor)

    if parts:
        merged = unary_union(parts) if len(parts) > 1 else parts[0]
        poly = _simplify_activity_polygon(merged)
        print(
            f"[DEBUG] 活动空间几何化: V_STP={len(pts)}, parts={len(parts)}, "
            f"{(_time.perf_counter() - t0) * 1000:.0f}ms",
            flush=True,
        )
        return poly

    if fallback_pt is not None:
        return _create_rect_polygon(
            fallback_pt.x, fallback_pt.y, max(float(max_one_way_m or 0.0), 200.0),
        )
    return None

def _dijkstra_reachable_to_dest_lonlat(
    G, sample_node_id, snap_sample_m, dist_to_dest_field, snap_dest_m,
    budget_sec, speed_ms, max_search_m,
):
    """
    从通勤路径采样点侧向扩张：经 detour 点仍能在剩余时间内到达目的地。
    约束：(snap_s + dist(S,X) + dist(X,D) + snap_d) / speed <= budget_sec
    """
    import networkx as nx
    from transport_modes import _graph_node_id

    budget_sec = max(0.0, float(budget_sec))
    speed_ms = max(float(speed_ms), 0.1)
    if budget_sec <= 0 or G is None or sample_node_id is None:
        return set()

    g = _undirected_search_graph(G)
    sid = _graph_node_id(g, sample_node_id)
    if sid not in g:
        return set()

    dijk_cutoff_m = max(float(max_search_m) + float(snap_sample_m), 500.0)
    try:
        dists_from_sample = nx.single_source_dijkstra_path_length(
            g, sid, weight="length", cutoff=dijk_cutoff_m
        )
    except Exception:
        return set()

    reachable = set()
    for node_id, dist_sm in dists_from_sample.items():
        dist_to_dest = dist_to_dest_field.get(node_id)
        if dist_to_dest is None:
            continue
        total_m = (
            float(snap_sample_m) + float(dist_sm)
            + float(dist_to_dest) + float(snap_dest_m)
        )
        if total_m / speed_ms > budget_sec + 1e-6:
            continue
        nd = g.nodes[node_id] if node_id in g.nodes else {}
        lon, lat = nd.get("x"), nd.get("y")
        if lon is None or lat is None:
            continue
        reachable.add((float(lon), float(lat)))
    return reachable

def _expand_mode_key_for_lateral(commute_mode):
    """侧向扩张吸附的路网模式（公交/地铁段仍按步行路网扩张）。"""
    mode_key = _mode_for_network_travel(commute_mode)
    if normalize_mode_key(commute_mode) in ('metro', 'subway', 'bus'):
        return 'walk'
    return mode_key

def _build_ppa_polygon_route_dest_aware(
    calc, route_coords, commute_mode, work_pt,
    total_travel_budget_min, commute_time_min,
    fallback_pt=None, route_parts=None,
):
    """
    STP = 必经路径缓冲 + 自由探索区域。
    对路径关键节点计算 B_local，在超级网络上局部蔓延后合并，
    几何输出为网络缓冲 ⋃ Alpha Shapes（不用凸包）。
    """
    lateral_speed_ms = _lateral_expansion_speed_ms(commute_mode)
    total_budget_sec = max(0.0, float(total_travel_budget_min)) * 60.0
    commute_sec = max(0.0, float(commute_time_min)) * 60.0
    if total_budget_sec <= 0 or not route_coords or len(route_coords) < 2:
        return None

    if fallback_pt is None:
        fallback_pt = Point(route_coords[0][1], route_coords[0][0])

    sampled_coords = _sample_coords_along_path(
        _merge_route_sample_coords(route_coords, PPA_ROUTE_SAMPLE_INTERVAL_M),
        PPA_ROUTE_SAMPLE_INTERVAL_M,
        _sample_k_max_for_route(route_coords),
    )
    cum_times_sec = _route_cumulative_times_by_arc(sampled_coords, commute_sec)

    # 用路径弧长剩余作初值；有反向距离场后会再校正
    origin_b_local = _local_budget_sec(total_budget_sec, 0.0, commute_sec)
    dest_b_local = _local_budget_sec(total_budget_sec, commute_sec, 0.0)
    max_lateral_m = max(origin_b_local * lateral_speed_ms * 0.5, 80.0)
    expand_mode_key = _expand_mode_key_for_lateral(commute_mode)
    method = "space_time_prism_supernode_stp_buffer_alpha"
    G = None
    activity_locations = _activity_locations_for_endpoints(
        fallback_pt, work_pt, origin_b_local, dest_b_local,
    )

    dist_to_dest = {}
    snap_dest_m = 0.0
    if (
        calc is not None
        and expand_mode_key in getattr(calc, "network_trees", {})
        and expand_mode_key in getattr(calc, "mode_config", {})
    ):
        network_key = calc.mode_config[expand_mode_key]["network"]
        G = calc.networks.get(network_key)
        if G is not None and G.number_of_nodes() > 0 and work_pt is not None:
            dest_node_id, _, _, snap_dest_m = _nearest_road_node(
                calc, work_pt, expand_mode_key,
            )
            if dest_node_id is not None:
                # 终点反向搜索：覆盖总预算对应的路网半径
                reverse_cutoff_m = max(
                    total_budget_sec * lateral_speed_ms + 500.0,
                    1500.0,
                )
                dist_to_dest = _network_dist_field_from_node(
                    G, dest_node_id, reverse_cutoff_m, undirected=True,
                )

    all_reachable, valid_samples, max_lateral_m, poly = (
        _build_route_activity_space_by_sample_hull_union(
            calc,
            G,
            expand_mode_key,
            sampled_coords,
            lateral_speed_ms,
            fallback_pt,
            cum_times_sec=cum_times_sec,
            total_budget_sec=total_budget_sec,
            commute_sec=commute_sec,
            activity_locations=activity_locations,
            dist_to_dest_field=dist_to_dest,
            snap_dest_m=snap_dest_m,
            route_coords=route_coords,
        )
    )

    if poly is None or poly.is_empty:
        poly = _finalize_route_activity_space_polygon(
            set(), route_coords, max_lateral_m, fallback_pt,
            activity_locations=activity_locations,
        )
        method = "space_time_prism_planned_route_buffer"
        if poly is None or poly.is_empty:
            return None

    print(
        f"[DEBUG] 超级网络 STP: 采样间隔={PPA_ROUTE_SAMPLE_INTERVAL_M:.0f}m, "
        f"采样K={valid_samples}, V_STP节点={len(all_reachable)}, "
        f"反向场覆盖={len(dist_to_dest)}",
        flush=True,
    )
    return {
        "polygon": poly,
        "reachable_node_count": len(all_reachable),
        "method": method,
        "max_one_way_m": max_lateral_m,
        "sample_count": valid_samples or len(sampled_coords),
        "route_coords": route_coords,
        "route_parts": route_parts,
    }

def _dijkstra_reachable_lonlat(G, origin_node_id, snap_m, elastic_sec, speed_ms, max_one_way_m):
    """从路网节点出发，按往返弹性时间约束收集可达点 (lon, lat)。"""
    del max_one_way_m
    return _dijkstra_local_stp_reachable_lonlat(
        G, origin_node_id, snap_m, elastic_sec, speed_ms,
        round_trip=PPA_ROUND_TRIP_DEVIATION,
    )

def _merge_route_sample_coords(route_coords, interval_m):
    """200m 间隔加密路线折线（保留全部拐点），避免弯道处漏采。"""
    return _densify_route_coords(route_coords, max_step_m=max(50.0, float(interval_m)))

def _create_rect_polygon(center_lon, center_lat, half_side_m):
    half_side_m = max(float(half_side_m), 50.0)
    lat_deg = half_side_m / _DEG_LAT_TO_M
    lon_deg = half_side_m / (_DEG_LAT_TO_M * max(math.cos(math.radians(center_lat)), 0.2))
    return Polygon([
        (center_lon - lon_deg, center_lat - lat_deg),
        (center_lon + lon_deg, center_lat - lat_deg),
        (center_lon + lon_deg, center_lat + lat_deg),
        (center_lon - lon_deg, center_lat + lat_deg),
        (center_lon - lon_deg, center_lat - lat_deg),
    ])

def _straight_buffer_segment(lon1, lat1, lon2, lat2, half_width_m):
    half_width_m = max(float(half_width_m), 50.0)
    seg_m = _haversine_m(lon1, lat1, lon2, lat2)
    if seg_m < 1e-3:
        return _create_rect_polygon(lon1, lat1, half_width_m)

    cos_lat = max(math.cos(math.radians((lat1 + lat2) / 2.0)), 0.2)
    dlon = (lon2 - lon1) * _DEG_LAT_TO_M * cos_lat
    dlat = (lat2 - lat1) * _DEG_LAT_TO_M
    norm = math.hypot(dlon, dlat) or 1.0
    nx = -dlat / norm * half_width_m
    ny = dlon / norm * half_width_m
    ox = nx / (_DEG_LAT_TO_M * cos_lat)
    oy = ny / _DEG_LAT_TO_M

    return Polygon([
        (lon1 - ox, lat1 - oy),
        (lon2 - ox, lat2 - oy),
        (lon2 + ox, lat2 + oy),
        (lon1 + ox, lat1 + oy),
        (lon1 - ox, lat1 - oy),
    ])

def _sanitize_route_coords_for_corridor(route_coords):
    if not route_coords or len(route_coords) < 2:
        return list(route_coords or [])
    try:
        from Get_Network import (
            dedupe_consecutive_coords,
            remove_path_loops,
            remove_spur_backtracks,
        )

        lonlat = [
            (float(c[1]), float(c[0]))
            for c in route_coords
            if c and len(c) >= 2
        ]
        working = dedupe_consecutive_coords(lonlat)
        working = remove_spur_backtracks(working)
        working = remove_path_loops(working)
        working = dedupe_consecutive_coords(working)
        if len(working) >= 2:
            return [[lat, lon] for lon, lat in working]
    except Exception:
        pass
    return list(route_coords)

def _route_corridor_hull_points(route_coords, half_width_m):
    half_width_m = max(float(half_width_m), 80.0)
    if not route_coords or len(route_coords) < 2:
        return set()
    coords = _sanitize_route_coords_for_corridor(route_coords)
    if len(coords) < 2:
        coords = list(route_coords)
    densified = _merge_route_sample_coords(
        coords, min(100.0, max(50.0, half_width_m * 0.75)),
    )
    pts = set()
    for i in range(len(densified) - 1):
        lat1, lng1 = densified[i]
        lat2, lng2 = densified[i + 1]
        if _haversine_m(lng1, lat1, lng2, lat2) < 1e-3:
            continue
        seg_poly = _straight_buffer_segment(
            float(lng1), float(lat1), float(lng2), float(lat2), half_width_m,
        )
        for lon, lat in seg_poly.exterior.coords[:-1]:
            pts.add((float(lon), float(lat)))
    return pts

def _buffer_route_coords_polygon(route_coords, half_width_m):
    """兼容旧调用：路线走廊缓冲（网络约束风格，不用凸包）。"""
    half_width_m = max(float(half_width_m), 80.0)
    poly = _buffer_route_polyline_m(route_coords, half_width_m)
    if poly is None or getattr(poly, "is_empty", True):
        pts = _route_corridor_hull_points(route_coords, half_width_m)
        if len(pts) < 3:
            return None
        return _lonlat_set_to_activity_polygon(pts, None, half_width_m, route_coords)
    return _normalize_display_polygon(poly)

def _union_activity_space_polygons(polygons, fallback_pt=None, max_lateral_m=200.0):
    """对多个采样点活动空间多边形做并集，得到最终活动空间。"""
    polys = [
        p for p in (polygons or [])
        if p is not None and not getattr(p, "is_empty", True)
    ]
    if not polys:
        if fallback_pt is not None:
            return _create_rect_polygon(
                fallback_pt.x, fallback_pt.y, max(float(max_lateral_m), 200.0),
            )
        return None
    if len(polys) == 1:
        return polys[0]
    merged = unary_union(polys)
    if merged is None or merged.is_empty:
        return polys[0]
    if merged.geom_type == "Polygon":
        return merged
    if merged.geom_type == "MultiPolygon":
        parts = [g for g in merged.geoms if g.geom_type == "Polygon" and not g.is_empty]
        return max(parts, key=lambda p: p.area) if parts else merged
    return merged

def _convex_hull_activity_space_at_sample(reachable_pts, center_pt, max_one_way_m):
    """兼容旧名：单点侧向蔓延后的活动空间（网络缓冲+凹壳，非凸包）。"""
    return _lonlat_set_to_activity_polygon(reachable_pts, center_pt, max_one_way_m)

def _geometric_activity_space_at_sample(lng, lat, radius_m):
    return _create_circle_polygon(float(lng), float(lat), max(float(radius_m), 50.0))

def _activity_locations_for_endpoints(
    fallback_pt, work_pt, origin_lateral_sec, dest_lateral_sec,
):
    """起终点活动地点各对应一次侧向蔓延（坐标用真实活动地，非路线折线端点）。"""
    locations = []
    origin_sec = max(0.0, float(origin_lateral_sec or 0.0))
    dest_sec = max(0.0, float(dest_lateral_sec or 0.0))
    if fallback_pt is not None and origin_sec > 0:
        locations.append({
            'lat': float(fallback_pt.y),
            'lng': float(fallback_pt.x),
            'lateral_budget_sec': origin_sec,
        })
    if work_pt is not None and dest_sec > 0:
        locations.append({
            'lat': float(work_pt.y),
            'lng': float(work_pt.x),
            'lateral_budget_sec': dest_sec,
        })
    return locations

def _spread_sample_polygon_at_point(
    calc, G, expand_mode_key, lat, lng, lateral_budget_sec, speed_ms,
    search_graph=None, build_polygon=True,
):
    """单点：超级网络侧向蔓延 → 网络缓冲/凹壳（无路网时用几何圆）。"""
    lateral_budget_sec = max(0.0, float(lateral_budget_sec or 0.0))
    if lateral_budget_sec <= 0 or speed_ms <= 0:
        return set(), None, 0.0

    sample_max_one_way_m = lateral_budget_sec * speed_ms * (
        0.5 if PPA_ROUND_TRIP_DEVIATION else 1.0
    )
    pt = Point(float(lng), float(lat))
    sample_pts = set()
    network_ready = (
        calc is not None
        and G is not None
        and G.number_of_nodes() > 0
        and expand_mode_key
        and expand_mode_key in getattr(calc, "network_trees", {})
        and expand_mode_key in getattr(calc, "mode_config", {})
    )
    if network_ready:
        sample_node_id, _, _, snap_m = _nearest_road_node(calc, pt, expand_mode_key)
        if sample_node_id is not None:
            sample_pts = _dijkstra_local_stp_reachable_lonlat(
                G,
                sample_node_id,
                snap_m,
                lateral_budget_sec,
                speed_ms,
                round_trip=PPA_ROUND_TRIP_DEVIATION,
                search_graph=search_graph,
            )
    if not build_polygon:
        return sample_pts, None, sample_max_one_way_m
    if sample_pts:
        sample_poly = _lonlat_set_to_activity_polygon(
            sample_pts, pt, sample_max_one_way_m,
        )
    else:
        sample_poly = _geometric_activity_space_at_sample(
            lng, lat, sample_max_one_way_m,
        )
    return sample_pts, sample_poly, sample_max_one_way_m

def _build_route_activity_space_by_sample_hull_union(
    calc,
    G,
    expand_mode_key,
    sampled_coords,
    speed_ms,
    fallback_pt,
    cum_times_sec=None,
    total_budget_sec=None,
    commute_sec=None,
    elastic_sec=None,
    activity_locations=None,
    dist_to_dest_field=None,
    snap_dest_m=0.0,
    route_coords=None,
):
    """
    沿路线采样关键节点 k：
      B_local(k)=T_budget-t_actual(k)-t_remain_need(k)
      超级网络搜索 cost(k→m)≤B_local/2 的局部可达集并合并；
    最终几何：必经路径缓冲 ⋃ 网络节点缓冲 ⋃ Alpha Shapes。
    """
    from transport_modes import _graph_node_id

    all_reachable = set()
    valid_samples = 0
    max_lateral_m = 0.0
    use_dest_aware = (
        cum_times_sec is not None
        and total_budget_sec is not None
        and commute_sec is not None
    )
    fixed_elastic_sec = max(0.0, float(elastic_sec or 0.0))
    fixed_max_one_way_m = fixed_elastic_sec * speed_ms * 0.5 if fixed_elastic_sec > 0 else 0.0
    dist_to_dest_field = dist_to_dest_field or {}
    snap_dest_m = float(snap_dest_m or 0.0)

    if fallback_pt is None and sampled_coords:
        lat0, lng0 = sampled_coords[0]
        fallback_pt = Point(float(lng0), float(lat0))

    network_ready = (
        calc is not None
        and G is not None
        and G.number_of_nodes() > 0
        and expand_mode_key
        and expand_mode_key in getattr(calc, "network_trees", {})
        and expand_mode_key in getattr(calc, "mode_config", {})
    )
    g_search = _undirected_search_graph(G) if network_ready else None

    time_iter = (
        cum_times_sec
        if use_dest_aware
        else [0.0] * len(sampled_coords or [])
    )

    for (lat, lng), t_accum in zip(sampled_coords or [], time_iter):
        pt = Point(float(lng), float(lat))
        sample_node_id = None
        snap_m = 0.0
        mapped_node = None

        if network_ready:
            sample_node_id, _nlon, _nlat, snap_m = _nearest_road_node(
                calc, pt, expand_mode_key,
            )
            if sample_node_id is not None and g_search is not None:
                mapped_node = _graph_node_id(g_search, sample_node_id)

        if use_dest_aware:
            t_actual = float(t_accum)
            arc_remain = max(0.0, float(commute_sec) - t_actual)
            t_remain_need = _reverse_remain_need_sec(
                dist_to_dest_field,
                mapped_node if mapped_node is not None else sample_node_id,
                snap_dest_m,
                speed_ms,
                arc_remain_sec=arc_remain,
            )
            b_local_sec = _local_budget_sec(
                float(total_budget_sec), t_actual, t_remain_need,
            )
            if b_local_sec <= 0:
                # 仍将路径锚点纳入 V_STP（必经走廊）
                if mapped_node is not None and g_search is not None:
                    nd = g_search.nodes.get(mapped_node) or {}
                    lon, lat_n = nd.get("x"), nd.get("y")
                    if lon is not None and lat_n is not None:
                        all_reachable.add((float(lon), float(lat_n)))
                else:
                    all_reachable.add((float(lng), float(lat)))
                continue
            sample_max_one_way_m = b_local_sec * speed_ms * (
                0.5 if PPA_ROUND_TRIP_DEVIATION else 1.0
            )
        else:
            b_local_sec = fixed_elastic_sec
            sample_max_one_way_m = fixed_max_one_way_m
            if b_local_sec <= 0 or sample_max_one_way_m <= 0:
                continue

        max_lateral_m = max(max_lateral_m, sample_max_one_way_m)
        sample_pts = set()

        if network_ready and sample_node_id is not None:
            valid_samples += 1
            sample_pts = _dijkstra_local_stp_reachable_lonlat(
                G,
                sample_node_id,
                snap_m,
                b_local_sec,
                speed_ms,
                round_trip=PPA_ROUND_TRIP_DEVIATION,
                search_graph=g_search,
            )
            all_reachable.update(sample_pts)
            # 锚点本身始终纳入
            all_reachable.add((float(lng), float(lat)))
        else:
            all_reachable.add((float(lng), float(lat)))

    activity_spreads = 0
    for act_loc in activity_locations or []:
        lateral_budget_sec = float(act_loc.get('lateral_budget_sec', 0) or 0.0)
        if lateral_budget_sec <= 0:
            continue
        act_pts, _act_poly, act_max_m = _spread_sample_polygon_at_point(
            calc,
            G,
            expand_mode_key,
            act_loc['lat'],
            act_loc['lng'],
            lateral_budget_sec,
            speed_ms,
            search_graph=g_search,
            build_polygon=False,
        )
        if act_pts:
            all_reachable.update(act_pts)
        all_reachable.add((float(act_loc['lng']), float(act_loc['lat'])))
        max_lateral_m = max(max_lateral_m, act_max_m)
        activity_spreads += 1

    # 必经路径折线顶点并入 V_STP
    for lat, lng in (route_coords or sampled_coords or []):
        all_reachable.add((float(lng), float(lat)))

    union_poly = _lonlat_set_to_activity_polygon(
        all_reachable,
        fallback_pt,
        max(max_lateral_m, 80.0),
        route_coords=route_coords or sampled_coords,
    )
    if union_poly is None or getattr(union_poly, "is_empty", True):
        union_poly = _union_activity_space_polygons(
            [], fallback_pt, max(max_lateral_m, 80.0),
        )
    return all_reachable, valid_samples + activity_spreads, max_lateral_m, union_poly

def _reachable_spans_route(reachable_pts, route_coords, margin_m=400.0):
    if not reachable_pts or not route_coords or len(route_coords) < 2:
        return False
    lons = [float(p[0]) for p in reachable_pts]
    lats = [float(p[1]) for p in reachable_pts]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    margin_deg = float(margin_m) / _DEG_LAT_TO_M
    for lat, lng in (route_coords[0], route_coords[-1]):
        lat, lng = float(lat), float(lng)
        cos_lat = max(math.cos(math.radians(lat)), 0.2)
        if not (
            min_lon - margin_deg / cos_lat <= lng <= max_lon + margin_deg / cos_lat
            and min_lat - margin_deg <= lat <= max_lat + margin_deg
        ):
            return False
    return True

def _collect_hull_points_for_activity_space(all_reachable, route_coords=None, max_lateral_m=None):
    """路网侧向蔓延最远点；未覆盖全程时用路线外廓顶点补足。"""
    pts = set(all_reachable or [])
    if route_coords and len(route_coords) >= 2 and max_lateral_m is not None:
        if len(pts) < 3 or not _reachable_spans_route(pts, route_coords):
            pts.update(_route_corridor_hull_points(route_coords, max_lateral_m))
    return pts

def _finalize_route_activity_space_polygon(
    all_reachable, route_coords, max_lateral_m, fallback_pt,
    activity_locations=None,
):
    """
    回退路径：沿路线采样后做网络风格缓冲并集（无路网时使用）。
    """
    del all_reachable
    half_w = max(float(max_lateral_m), 80.0)
    sample_pts = set()
    if route_coords and len(route_coords) >= 2:
        sampled_coords = _sample_coords_along_path(
            _merge_route_sample_coords(route_coords, PPA_ROUTE_SAMPLE_INTERVAL_M),
            PPA_ROUTE_SAMPLE_INTERVAL_M,
            _sample_k_max_for_route(route_coords),
        )
        for lat, lng in sampled_coords:
            sample_pts.add((float(lng), float(lat)))

    for act_loc in activity_locations or []:
        lateral_budget_sec = float(act_loc.get('lateral_budget_sec', 0) or 0.0)
        if lateral_budget_sec <= 0:
            continue
        sample_pts.add((float(act_loc['lng']), float(act_loc['lat'])))

    if sample_pts:
        poly = _lonlat_set_to_activity_polygon(
            sample_pts, fallback_pt, half_w, route_coords=route_coords,
        )
        if poly is not None and not getattr(poly, "is_empty", True):
            return poly

    if fallback_pt is not None:
        return _create_rect_polygon(fallback_pt.x, fallback_pt.y, half_w)
    return None

def _lonlat_set_to_polygon(reachable_pts, fallback_pt, max_one_way_m):
    """将路网可达点集转为活动空间多边形（网络缓冲 + Alpha Shapes，非凸包）。"""
    return _lonlat_set_to_activity_polygon(reachable_pts, fallback_pt, max_one_way_m)

def _expand_route_samples_on_network(
    calc, G, expand_mode_key, sampled_coords, cum_times_sec,
    speed_ms, total_budget_sec, commute_sec,
    dist_to_dest=None, snap_dest_m=0.0,
    activity_locations=None,
):
    """
    沿路线采样点吸附超级网络并做局部 STP；合并后网络缓冲+凹壳。
    """
    fallback_pt = None
    if sampled_coords:
        lat0, lng0 = sampled_coords[0]
        fallback_pt = Point(float(lng0), float(lat0))
    return _build_route_activity_space_by_sample_hull_union(
        calc,
        G,
        expand_mode_key,
        sampled_coords,
        speed_ms,
        fallback_pt,
        cum_times_sec=cum_times_sec,
        total_budget_sec=total_budget_sec,
        commute_sec=commute_sec,
        activity_locations=activity_locations,
        dist_to_dest_field=dist_to_dest,
        snap_dest_m=snap_dest_m,
        route_coords=sampled_coords,
    )

def _build_ppa_polygon_single(calc, home_pt, elastic_time_min, commute_mode):
    lateral_speed_ms = _lateral_expansion_speed_ms(commute_mode)
    elastic_sec = max(0.0, float(elastic_time_min)) * 60.0
    if elastic_sec <= 0 or lateral_speed_ms <= 0:
        return None

    expand_mode_key = _expand_mode_key_for_lateral(commute_mode)
    max_one_way_m = elastic_sec * lateral_speed_ms * 0.5

    if (
        calc is None
        or expand_mode_key not in getattr(calc, "network_trees", {})
        or expand_mode_key not in getattr(calc, "mode_config", {})
    ):
        rect = _create_rect_polygon(home_pt.x, home_pt.y, max_one_way_m)
        return {
            "polygon": rect,
            "reachable_node_count": 0,
            "method": "space_time_prism_fallback_rect",
            "max_one_way_m": max_one_way_m,
            "sample_count": 1,
        }

    network_key = calc.mode_config[expand_mode_key]["network"]
    G = calc.networks.get(network_key)
    if G is None or G.number_of_nodes() == 0:
        rect = _create_rect_polygon(home_pt.x, home_pt.y, max_one_way_m)
        return {
            "polygon": rect,
            "reachable_node_count": 0,
            "method": "space_time_prism_fallback_rect",
            "max_one_way_m": max_one_way_m,
            "sample_count": 1,
        }

    origin_node_id, _nlon, _nlat, snap_m = _nearest_road_node(calc, home_pt, expand_mode_key)
    if origin_node_id is None:
        rect = _create_rect_polygon(home_pt.x, home_pt.y, max_one_way_m)
        return {
            "polygon": rect,
            "reachable_node_count": 0,
            "method": "space_time_prism_fallback_rect",
            "max_one_way_m": max_one_way_m,
            "sample_count": 1,
        }

    reachable_pts = _dijkstra_reachable_lonlat(
        G, origin_node_id, snap_m, elastic_sec, lateral_speed_ms, max_one_way_m
    )
    hull = _lonlat_set_to_polygon(reachable_pts, home_pt, max_one_way_m)
    return {
        "polygon": hull,
        "reachable_node_count": len(reachable_pts),
        "method": "space_time_prism_ppa",
        "max_one_way_m": max_one_way_m,
        "sample_count": 1,
    }

def _build_ppa_polygon_route_union(calc, home_pt, work_pt, elastic_time_min, commute_mode):
    """
    通勤路径 K 点采样 + 多次 Dijkstra 并集，构建沿走廊展开的潜在活动空间。
    """
    mode_cb = normalize_route_mode_for_calc(commute_mode)
    route_coords = get_route_geometry_coords(calc, home_pt, work_pt, mode_cb)
    if not route_coords or len(route_coords) < 2:
        return _build_ppa_polygon_single(calc, home_pt, elastic_time_min, commute_mode)
    return _build_ppa_polygon_from_route_coords(
        calc, route_coords, elastic_time_min, commute_mode, fallback_pt=home_pt,
    )

def _route_part_network_and_speed(part_type):
    """通勤分段侧向扩张：均在步行路网上按步行速度蔓延（与分段交通工具无关）。"""
    _ = part_type
    walk_speed = max(_commute_speed_ms("walk"), 0.1)
    return "walk", walk_speed

def _merged_coords_from_route_parts(route_parts):
    """合并 route_parts 各段坐标为一条折线。"""
    merged = []
    for part in route_parts or []:
        coords = part.get('coords') or []
        if not coords:
            continue
        if not merged:
            merged = list(coords)
            continue
        for coord in coords:
            prev = merged[-1]
            if abs(prev[0] - coord[0]) > 1e-6 or abs(prev[1] - coord[1]) > 1e-6:
                merged.append(coord)
    return merged

def _build_ppa_polygon_from_route_parts(
    calc, route_parts, elastic_time_min, commute_mode, fallback_pt=None, route_coords=None,
    work_pt=None,
):
    """
    沿步行+地铁/公交分段分别采样，各点在超级网络上局部 STP 后合并。
    """
    elastic_sec = max(0.0, float(elastic_time_min)) * 60.0
    if elastic_sec <= 0 or not route_parts:
        return None

    if route_coords is None:
        route_coords = _merged_coords_from_route_parts(route_parts)
    if not route_coords or len(route_coords) < 2:
        return None

    if fallback_pt is None:
        fallback_pt = Point(route_coords[0][1], route_coords[0][0])

    dest_pt = work_pt
    if dest_pt is None:
        dest_pt = Point(route_coords[-1][1], route_coords[-1][0])
    activity_locations = _activity_locations_for_endpoints(
        fallback_pt, dest_pt, elastic_sec, elastic_sec,
    )

    all_reachable = set()
    all_sample_polys = []
    valid_samples = 0
    max_one_way_overall = 0.0

    for part in route_parts:
        part_coords = part.get('coords') or []
        if len(part_coords) < 2:
            continue
        network_mode, speed_ms = _route_part_network_and_speed(part.get('type'))
        max_one_way_m = elastic_sec * speed_ms * 0.5
        max_one_way_overall = max(max_one_way_overall, max_one_way_m)
        sampled_coords = _sample_coords_along_path(
            part_coords, PPA_ROUTE_SAMPLE_INTERVAL_M,
            _sample_k_max_for_route(part_coords),
        )

        expand_mode_key = _expand_mode_key_for_lateral(network_mode)
        G = None
        if (
            calc is not None
            and expand_mode_key in getattr(calc, "network_trees", {})
            and expand_mode_key in getattr(calc, "mode_config", {})
        ):
            network_key = calc.mode_config[expand_mode_key]["network"]
            G = calc.networks.get(network_key)

        part_reachable, part_valid, part_max_m, part_poly = (
            _build_route_activity_space_by_sample_hull_union(
                calc,
                G,
                expand_mode_key,
                sampled_coords,
                speed_ms,
                fallback_pt,
                elastic_sec=elastic_sec,
                route_coords=part_coords,
            )
        )
        all_reachable.update(part_reachable)
        valid_samples += part_valid
        max_one_way_overall = max(max_one_way_overall, part_max_m)
        if part_poly is not None and not part_poly.is_empty:
            all_sample_polys.append(part_poly)

    expand_mode_key = _expand_mode_key_for_lateral(commute_mode)
    G = None
    if (
        calc is not None
        and expand_mode_key in getattr(calc, "network_trees", {})
        and expand_mode_key in getattr(calc, "mode_config", {})
    ):
        network_key = calc.mode_config[expand_mode_key]["network"]
        G = calc.networks.get(network_key)
    _, speed_ms = _route_part_network_and_speed(
        (route_parts[0] or {}).get('type') if route_parts else None,
    )
    for act_loc in activity_locations:
        act_pts, act_poly, act_max_m = _spread_sample_polygon_at_point(
            calc,
            G,
            expand_mode_key,
            act_loc['lat'],
            act_loc['lng'],
            act_loc['lateral_budget_sec'],
            speed_ms,
            build_polygon=False,
        )
        all_reachable.update(act_pts)
        valid_samples += 1
        max_one_way_overall = max(max_one_way_overall, act_max_m)
        del act_poly

    # 优先对合并后的 V_STP 做一次网络缓冲+凹壳
    poly = _lonlat_set_to_activity_polygon(
        all_reachable, fallback_pt, max(max_one_way_overall, 200.0),
        route_coords=route_coords,
    )
    if poly is None or poly.is_empty:
        poly = _union_activity_space_polygons(
            all_sample_polys, fallback_pt, max(max_one_way_overall, 200.0),
        )
    if poly is None or poly.is_empty:
        poly = _finalize_route_activity_space_polygon(
            set(), route_coords, max(max_one_way_overall, 200.0), fallback_pt,
            activity_locations=activity_locations,
        )
    if poly is None or poly.is_empty:
        return None

    print(
        f"[DEBUG] 分段通勤超级网络 STP: 段数={len(route_parts)}, 采样K={valid_samples}, "
        f"V_STP节点={len(all_reachable)}",
        flush=True,
    )
    return {
        "polygon": poly,
        "reachable_node_count": len(all_reachable),
        "method": "space_time_prism_transit_route_stp_buffer_alpha",
        "max_one_way_m": max_one_way_overall,
        "sample_count": valid_samples,
        "route_coords": route_coords,
        "route_parts": route_parts,
    }

def _build_ppa_polygon_from_route_coords(
    calc, route_coords, elastic_time_min, commute_mode, fallback_pt=None, route_parts=None,
    work_pt=None, total_travel_budget_min=None, commute_time_min=None,
):
    """沿已规划路径（高德/出行推荐等）采样构建潜在活动空间。"""
    if (
        work_pt is not None
        and total_travel_budget_min is not None
        and commute_time_min is not None
    ):
        dest_aware = _build_ppa_polygon_route_dest_aware(
            calc, route_coords, commute_mode, work_pt,
            total_travel_budget_min, commute_time_min,
            fallback_pt=fallback_pt, route_parts=route_parts,
        )
        if dest_aware is not None:
            return dest_aware

    if route_parts and len(route_parts) >= 1:
        composite = _build_ppa_polygon_from_route_parts(
            calc, route_parts, elastic_time_min, commute_mode,
            fallback_pt=fallback_pt, route_coords=route_coords, work_pt=work_pt,
        )
        if composite is not None:
            return composite

    speed_ms = _lateral_expansion_speed_ms(commute_mode)
    elastic_sec = max(0.0, float(elastic_time_min)) * 60.0
    if elastic_sec <= 0 or speed_ms <= 0 or not route_coords or len(route_coords) < 2:
        return None

    expand_mode_key = _expand_mode_key_for_lateral(commute_mode)
    max_one_way_m = elastic_sec * speed_ms * 0.5
    sampled_coords = _sample_coords_along_path(
        _merge_route_sample_coords(route_coords, PPA_ROUTE_SAMPLE_INTERVAL_M),
        PPA_ROUTE_SAMPLE_INTERVAL_M,
        _sample_k_max_for_route(route_coords),
    )

    if fallback_pt is None:
        mid_lat = (route_coords[0][0] + route_coords[-1][0]) / 2.0
        mid_lng = (route_coords[0][1] + route_coords[-1][1]) / 2.0
        fallback_pt = Point(mid_lng, mid_lat)

    dest_pt = work_pt
    if dest_pt is None:
        dest_pt = Point(route_coords[-1][1], route_coords[-1][0])
    activity_locations = _activity_locations_for_endpoints(
        fallback_pt, dest_pt, elastic_sec, elastic_sec,
    )

    G = None
    if (
        calc is not None
        and expand_mode_key in getattr(calc, "network_trees", {})
        and expand_mode_key in getattr(calc, "mode_config", {})
    ):
        network_key = calc.mode_config[expand_mode_key]["network"]
        G = calc.networks.get(network_key)

    all_reachable, valid_samples, max_one_way_m, poly = (
        _build_route_activity_space_by_sample_hull_union(
            calc,
            G,
            expand_mode_key,
            sampled_coords,
            speed_ms,
            fallback_pt,
            elastic_sec=elastic_sec,
            activity_locations=activity_locations,
            route_coords=route_coords,
        )
    )

    if poly is None or poly.is_empty:
        poly = _finalize_route_activity_space_polygon(
            set(), route_coords, max_one_way_m, fallback_pt,
            activity_locations=activity_locations,
        )
    if poly is None or poly.is_empty:
        return None

    method = (
        "space_time_prism_planned_route_stp_buffer_alpha"
        if all_reachable
        else "space_time_prism_planned_route_buffer"
    )
    print(
        f"[DEBUG] 规划路径超级网络 STP: 路径点={len(route_coords)}, 采样K={valid_samples}, "
        f"V_STP节点={len(all_reachable)}",
        flush=True,
    )
    return {
        "polygon": poly,
        "reachable_node_count": len(all_reachable),
        "method": method,
        "max_one_way_m": max_one_way_m,
        "sample_count": valid_samples,
        "route_coords": route_coords,
    }

def _build_ppa_polygon(calc, home_pt, elastic_time_min, commute_mode, work_pt=None, route_coords=None, route_parts=None, total_travel_budget_min=None, commute_time_min=None):
    """构建潜在活动空间：有下一活动地时沿规划路径 K 点采样并集，否则单起点 PPA。"""
    if work_pt is not None:
        if route_parts and len(route_parts) >= 1:
            merged = route_coords if route_coords and len(route_coords) >= 2 else _merged_coords_from_route_parts(route_parts)
            if merged and len(merged) >= 2:
                return _build_ppa_polygon_from_route_coords(
                    calc, merged, elastic_time_min, commute_mode,
                    fallback_pt=home_pt, route_parts=route_parts,
                    work_pt=work_pt,
                    total_travel_budget_min=total_travel_budget_min,
                    commute_time_min=commute_time_min,
                )
        if route_coords and len(route_coords) >= 2:
            return _build_ppa_polygon_from_route_coords(
                calc, route_coords, elastic_time_min, commute_mode, fallback_pt=home_pt,
                work_pt=work_pt,
                total_travel_budget_min=total_travel_budget_min,
                commute_time_min=commute_time_min,
            )
        return _build_ppa_polygon_route_union(
            calc, home_pt, work_pt, elastic_time_min, commute_mode
        )
    return _build_ppa_polygon_single(calc, home_pt, elastic_time_min, commute_mode)

def _normalize_display_polygon(polygon):
    """将并集结果规范为单一 Polygon，避免再次凸包拉直弯道。"""
    if polygon is None or getattr(polygon, "is_empty", True):
        return polygon
    if polygon.geom_type == "Polygon":
        return polygon
    if polygon.geom_type in ("MultiPolygon", "GeometryCollection"):
        polys = [g for g in polygon.geoms if g.geom_type == "Polygon" and not g.is_empty]
        if not polys:
            return polygon
        merged = unary_union(polys)
        if merged.geom_type == "Polygon":
            return merged
        if merged.geom_type == "MultiPolygon":
            polys = [g for g in merged.geoms if g.geom_type == "Polygon" and not g.is_empty]
            return max(polys, key=lambda p: p.area) if polys else merged
        return merged
    if polygon.geom_type == "GeometryCollection":
        polys = [g for g in polygon.geoms if g.geom_type == "Polygon" and not g.is_empty]
        return max(polys, key=lambda p: p.area) if polys else polygon
    return polygon

def _build_ppa_bounds(
    locations,
    home_pt,
    ppa_result,
    elastic_time_min,
    speed_ms,
):
    """构建前端展示用的活动空间边界（潜在活动空间多边形）。"""
    polygon = _simplify_activity_polygon(
        _normalize_display_polygon(ppa_result["polygon"])
    )

    center = polygon.centroid
    center_lon, center_lat = float(center.x), float(center.y)
    bounds = polygon.bounds
    width_m = (bounds[2] - bounds[0]) * _DEG_LAT_TO_M * max(
        math.cos(math.radians(center_lat)), 0.2
    )
    height_m = (bounds[3] - bounds[1]) * _DEG_LAT_TO_M

    bounds = {
        "cluster_id": 0,
        "point_count": len(locations),
        "center": [center_lat, center_lon],
        "center_lon": center_lon,
        "center_lat": center_lat,
        "semi_major": width_m / _DEG_LAT_TO_M / 2.0,
        "semi_minor": height_m / _DEG_LAT_TO_M / 2.0,
        "theta_deg": 0.0,
        "polygon": list(polygon.exterior.coords) if polygon else [],
        "method": ppa_result.get("method", "space_time_prism_ppa"),
        "elastic_time_min": round(float(elastic_time_min), 2),
        "speed_ms": round(float(speed_ms), 3),
        "reachable_node_count": int(ppa_result.get("reachable_node_count", 0)),
        "sample_count": int(ppa_result.get("sample_count", 1)),
        "input_points": [[loc.get("lat"), loc.get("lng")] for loc in locations],
        "input_points_with_type": [
            {
                "coords": [loc.get("lat"), loc.get("lng")],
                "type": loc.get("type"),
                "name": loc.get("name"),
            }
            for loc in locations
        ],
    }
    route_coords = ppa_result.get("route_coords")
    if route_coords:
        bounds["commute_route"] = route_coords
    route_parts = ppa_result.get("route_parts")
    if route_parts:
        bounds["route_parts"] = route_parts
    return bounds

def _route_linestring_from_coords(route_coords):
    """[[lat, lng], ...] → Shapely LineString(lon, lat)。"""
    if not route_coords or len(route_coords) < 2:
        return None
    return LineString([(float(c[1]), float(c[0])) for c in route_coords])

def _haversine_distance_to_route_line(poi_pt, route_line, fallback_pt=None):
    """POI 到通勤折线的几何距离（米），无路网时使用。"""
    if route_line is not None and not route_line.is_empty:
        nearest_on_route = route_line.interpolate(route_line.project(poi_pt))
        return _haversine_m(
            float(nearest_on_route.x), float(nearest_on_route.y),
            float(poi_pt.x), float(poi_pt.y),
        )
    if fallback_pt is not None:
        return _haversine_m(
            float(fallback_pt.x), float(fallback_pt.y),
            float(poi_pt.x), float(poi_pt.y),
        )
    return 0.0

def _lookup_network_dist_to_route(dist_field, node_id):
    if not dist_field or node_id is None:
        return None
    for key in (node_id, str(node_id)):
        if key in dist_field:
            v = float(dist_field[key])
            if math.isfinite(v):
                return v
    return None

def _network_dist_poi_to_route(calc, poi_pt, route_dist_ctx, route_line, fallback_pt):
    """基于预计算距离场：d_mean ≈ 吸附距离 + 路网节点到路径锚点的最短距离。"""
    dist_field = route_dist_ctx.get("dist_to_route")
    mode_key = route_dist_ctx.get("mode_key")
    if not dist_field or not mode_key:
        return _haversine_distance_to_route_line(poi_pt, route_line, fallback_pt)

    _nid, _, _, snap_m = _nearest_road_node(calc, poi_pt, mode_key)
    d_route = _lookup_network_dist_to_route(dist_field, _nid)
    if d_route is not None:
        return float(snap_m) + float(d_route)
    return _haversine_distance_to_route_line(poi_pt, route_line, fallback_pt)

def _batch_network_dist_poi_to_route(calc, poi_points, route_dist_ctx, route_line, fallback_pt):
    """批量计算 POI 到通勤路径的路网距离（米）。"""
    n = len(poi_points)
    if n == 0:
        return []
    dist_field = route_dist_ctx.get("dist_to_route")
    mode_key = route_dist_ctx.get("mode_key")
    if not dist_field or not mode_key:
        return [
            _haversine_distance_to_route_line(pt, route_line, fallback_pt)
            for pt in poi_points
        ]

    node_ids, snap_ms = _batch_snap_road_nodes(calc, poi_points, mode_key)
    out = []
    for pt, nid, snap_m in zip(poi_points, node_ids, snap_ms):
        d_route = _lookup_network_dist_to_route(dist_field, nid)
        if d_route is not None:
            out.append(float(snap_m) + float(d_route))
        else:
            out.append(_haversine_distance_to_route_line(pt, route_line, fallback_pt))
    return out

def _distance_poi_to_network(calc, origin_pt, poi_pt, commute_mode):
    """
    计算从出发地到POI的路网距离（米）。
    
    使用Dijkstra算法计算实际路网距离，而非直线距离。
    """
    import networkx as nx

    try:
        mode_key = _mode_for_network_travel(commute_mode)
        
        G = calc.road_networks.get(mode_key)
        if G is None:
            # 如果没有路网，使用Haversine距离
            return _haversine_m(origin_pt.x, origin_pt.y, poi_pt.x, poi_pt.y)
        
        # 吸附到最近路网节点
        origin_node = _nearest_road_node(calc, origin_pt, mode_key)
        poi_node = _nearest_road_node(calc, poi_pt, mode_key)
        
        if origin_node[0] is None or poi_node[0] is None:
            return _haversine_m(origin_pt.x, origin_pt.y, poi_pt.x, poi_pt.y)
        
        # 计算最短路径长度
        try:
            length = nx.dijkstra_path_length(G, origin_node[0], poi_node[0], weight='length')
            return length
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # 如果没有路径，使用Haversine距离
            return _haversine_m(origin_pt.x, origin_pt.y, poi_pt.x, poi_pt.y)
    except Exception as e:
        # 失败时使用Haversine距离
        return _haversine_m(origin_pt.x, origin_pt.y, poi_pt.x, poi_pt.y)

def _build_ppa_polygon_with_pois(
    calc, home_pt, elastic_time_min, commute_mode, work_pt=None, reachable_pois=None,
    route_coords=None, route_parts=None, total_travel_budget_min=None, commute_time_min=None,
):
    """
    基于可达 POI 辅助构建 PPA 范围。

    有下一活动地点时优先沿通勤路径构建时空棱镜（POI 凹壳无法覆盖沿途走廊）；
    仅单地点场景才用 POI 分布限定范围。
    """
    try:
        if work_pt is not None:
            route_result = _build_ppa_polygon(
                calc, home_pt, elastic_time_min, commute_mode, work_pt=work_pt,
                route_coords=route_coords, route_parts=route_parts,
                total_travel_budget_min=total_travel_budget_min,
                commute_time_min=commute_time_min,
            )
            if route_result and route_result.get("polygon") is not None:
                if reachable_pois:
                    route_result["reachable_poi_count"] = len(reachable_pois)
                return route_result

        lateral_speed_ms = _lateral_expansion_speed_ms(commute_mode)
        elastic_sec = elastic_time_min * 60.0
        max_one_way_m = elastic_sec * lateral_speed_ms * 0.5

        # 单起点：有可达 POI 时用凹壳/网络缓冲限定范围（不用凸包）
        if reachable_pois and len(reachable_pois) > 0:
            print(f"[DEBUG] 基于{len(reachable_pois)}个可达POI构建PPA范围", flush=True)
            
            # 收集所有可达POI的坐标
            poi_coords = []
            for poi in reachable_pois:
                if 'coords' in poi:
                    # coords格式是 [lat, lon]
                    lat, lon = poi['coords'][0], poi['coords'][1]
                    poi_coords.append((lon, lat))
                elif 'point' in poi:
                    pt = poi['point']
                    poi_coords.append((pt.x, pt.y))
            
            if len(poi_coords) >= 3:
                hull = _lonlat_set_to_activity_polygon(
                    poi_coords, home_pt, max_one_way_m,
                )
                if hull is None or getattr(hull, "is_empty", True):
                    hull = _alpha_shape_polygon(poi_coords)
                if hull is None or getattr(hull, "is_empty", True):
                    hull = _buffer_lonlat_points_polygon(
                        poi_coords, min(max_one_way_m, PPA_NETWORK_BUFFER_R_MAX_M),
                    )
                
                result = {
                    "polygon": hull,
                    "method": "poi_based_alpha_buffer",
                    "reachable_poi_count": len(reachable_pois)
                }

                if route_coords and len(route_coords) >= 2:
                    result["route_coords"] = route_coords
                    result["commute_route"] = route_coords
                if route_parts:
                    result["route_parts"] = route_parts
                elif work_pt:
                    try:
                        mode_cb = normalize_route_mode_for_calc(commute_mode)
                        planned = get_route_geometry_coords(calc, home_pt, work_pt, mode_cb)
                        if planned:
                            result["route_coords"] = planned
                            result["commute_route"] = planned
                    except Exception as e:
                        print(f"[DEBUG] 获取通勤路径失败: {e}", flush=True)

                return result
            
            elif len(poi_coords) == 1:
                lon, lat = poi_coords[0]
                rect = _create_rect_polygon(lon, lat, max_one_way_m)
                return {
                    "polygon": rect,
                    "method": "poi_based_rect",
                    "reachable_poi_count": 1
                }
            
            elif len(poi_coords) == 2:
                # 有两个POI，创建线段缓冲区
                from shapely.geometry import LineString
                line = LineString(poi_coords)
                buffer_poly = line.buffer(max_one_way_m * 0.5)
                return {
                    "polygon": buffer_poly,
                    "method": "poi_based_buffer",
                    "reachable_poi_count": len(reachable_pois)
                }
        
        # 如果没有可达POI或POI数量不足，回退到原始方法
        print("[DEBUG] 回退到原始时空棱镜方法", flush=True)
        return _build_ppa_polygon(
            calc, home_pt, elastic_time_min, commute_mode, work_pt=work_pt,
            route_coords=route_coords, route_parts=route_parts,
            total_travel_budget_min=total_travel_budget_min,
            commute_time_min=commute_time_min,
        )
        
    except Exception as e:
        import traceback
        print(f"[ERROR] _build_ppa_polygon_with_pois 失败: {e}", flush=True)
        traceback.print_exc()
        return _build_ppa_polygon(
            calc, home_pt, elastic_time_min, commute_mode, work_pt=work_pt,
            route_coords=route_coords, route_parts=route_parts,
            total_travel_budget_min=total_travel_budget_min,
            commute_time_min=commute_time_min,
        )

def _distance_poi_to_route(
    calc, poi_pt, route_line, commute_mode, fallback_pt=None, route_dist_ctx=None,
):
    """POI 到通勤路径的距离 d_mean（米）。"""
    if route_dist_ctx and route_dist_ctx.get("dist_to_route"):
        return _network_dist_poi_to_route(
            calc, poi_pt, route_dist_ctx, route_line, fallback_pt
        )
    return _haversine_distance_to_route_line(poi_pt, route_line, fallback_pt)

def _poi_category_blob(poi):
    parts = []
    cat = poi.get("category") or poi.get("类别") or ""
    if cat:
        text = str(cat).replace("/", ";")
        parts.extend(p.strip() for p in text.split(";") if p.strip())
    name = str(poi.get("name") or "").strip()
    if name:
        parts.append(name)
    return ";".join(parts)

def _poi_matches_subcategory_keywords(poi, keywords):
    if not keywords:
        return True
    blob = _poi_category_blob(poi)
    if not blob:
        return False
    return any(kw in blob for kw in keywords)

def _enrich_poi_categories(pois, calc, poi_category):
    """为缺少类别字段的 POI 从 GeoJSON 属性表补全。"""
    if not pois or calc is None or not hasattr(calc, "poi_trees"):
        return pois
    if poi_category not in calc.poi_trees:
        return pois
    _, _, gdf = calc.poi_trees[poi_category]
    if gdf is None or gdf.empty:
        return pois

    cat_field = next((f for f in ("类别", "category", "type", "大类") if f in gdf.columns), None)
    name_field = next((f for f in ("名称", "name", "NAME", "Name") if f in gdf.columns), None)
    if not cat_field:
        return pois

    name_to_cat = {}
    if name_field:
        for _, row in gdf.iterrows():
            name = str(row[name_field]) if row[name_field] is not None else ""
            if name and name not in name_to_cat:
                name_to_cat[name] = str(row[cat_field]) if row[cat_field] is not None else ""

    for poi in pois:
        if poi.get("category"):
            continue
        poi_name = str(poi.get("name") or "")
        if poi_name and poi_name in name_to_cat:
            poi["category"] = name_to_cat[poi_name]
    return pois

def _filter_pois_by_subcategory(pois, keywords, calc, poi_category):
    """按 POI 文件「类别」字段（大类/中类/小类）及名称过滤。"""
    if not keywords or not pois:
        return pois
    pois = _enrich_poi_categories(pois, calc, poi_category)
    return [p for p in pois if _poi_matches_subcategory_keywords(p, keywords)]

def _get_pois_in_polygon(calc, poi_category, polygon, max_count=None):
    if calc is None or polygon is None or polygon.is_empty:
        return []
    if not hasattr(calc, "poi_trees"):
        return []

    # 复杂边界先简化，加速 prepared.contains
    polygon = _simplify_activity_polygon(polygon)
    if polygon is None or getattr(polygon, "is_empty", True):
        return []

    cap = PPA_MAX_POI_PER_CATEGORY if max_count is None else int(max_count)
    act_type = poi_category.lower()
    if act_type not in calc.poi_trees:
        return []

    try:
        from shapely.prepared import prep

        minx, miny, maxx, maxy = polygon.bounds
        center_lon = (minx + maxx) / 2.0
        center_lat = (miny + maxy) / 2.0
        search_radius_m = max(
            _haversine_m(minx, miny, maxx, maxy),
            _haversine_m(center_lon, center_lat, maxx, maxy),
            800.0,
        )

        tree, coords, gdf = calc.poi_trees[act_type]
        if len(coords) == 0:
            return []

        name_field = None
        cat_field = None
        for field in ["名称", "name", "NAME", "Name", "title", "Title"]:
            if field in gdf.columns:
                name_field = field
                break
        for field in ["类别", "category", "type", "大类"]:
            if field in gdf.columns:
                cat_field = field
                break

        poi_lon_arr = coords[:, 0].astype(np.float64)
        poi_lat_arr = coords[:, 1].astype(np.float64)
        center_rad = np.array([[math.radians(center_lat), math.radians(center_lon)]])
        candidate_idx = tree.query_radius(
            center_rad, r=_haversine_tree_radius_rad(search_radius_m)
        )[0]

        prepared_poly = prep(polygon)
        results = []
        for idx in candidate_idx:
            if cap > 0 and len(results) >= cap:
                break
            plon = float(poi_lon_arr[idx])
            plat = float(poi_lat_arr[idx])
            if not prepared_poly.contains(Point(plon, plat)):
                continue
            poi_name = None
            poi_category_text = None
            if idx < len(gdf):
                row = gdf.iloc[idx]
                if name_field is not None:
                    val = row[name_field]
                    poi_name = str(val) if pd.notna(val) else None
                if cat_field is not None:
                    val = row[cat_field]
                    poi_category_text = str(val) if pd.notna(val) else None
            results.append({
                "point": Point(plon, plat),
                "coords": [plat, plon],
                "name": poi_name,
                "category": poi_category_text,
            })
        return results
    except Exception:
        import traceback
        traceback.print_exc()
        return []

def _score_pois_in_activity_space(
    pois_in_space,
    center_pt,
    calc,
    beta,
    commute_mode,
    route_line,
    route_dist_ctx,
    speed_ms,
):
    pois_scored = []
    for poi in pois_in_space:
        poi_pt = poi["point"]
        if route_dist_ctx and route_dist_ctx.get("dist_to_route"):
            dist_list = _batch_network_dist_poi_to_route(
                calc, [poi_pt], route_dist_ctx, route_line, center_pt
            )
            network_dist = dist_list[0] if dist_list else float("inf")
        else:
            network_dist = _distance_poi_to_route(
                calc, poi_pt, route_line, commute_mode,
                fallback_pt=center_pt, route_dist_ctx=route_dist_ctx,
            )
        travel_time = network_dist / speed_ms / 60.0
        pois_scored.append((poi, network_dist, travel_time))

    total_score = 0.0
    scored_pois = []
    for poi, network_dist, travel_time in pois_scored:
        decay = math.exp(-beta * network_dist)
        contribution = decay
        total_score += contribution
        scored_pois.append({
            "coords": poi["coords"],
            "distance": round(network_dist, 1),
            "travel_time": round(travel_time, 1),
            "decay": round(decay, 4),
            "contribution": round(contribution, 4),
            "name": poi.get("name"),
        })
    return total_score, scored_pois, len(pois_scored)

def _calculate_activity_space_accessibility(
    space_polygon, center_pt, calc, weights=None, beta=DECAY_BETA,
    commute_mode="foot", route_line=None, route_dist_ctx=None,
    activity_time_budgets=None, selected_poi_type_defs=None,
):
    """
    第四步：计算基于潜在活动空间（PPA 多边形）的综合可达性得分

    公式:
    A_space = Σ_{k∈K} W_k · (Σ_{j∈Ω_space(k)} O_j · e^(-β·d_mean(j)))

    若提供 selected_poi_type_defs，则仅对勾选的 POI 子类计分，各子类得分直接相加；
    否则按活动大类权重，对多边形内该大类全部 POI 计分。
    """
    category_scores = {}
    all_pois = []
    speed_ms = max(_commute_speed_ms(commute_mode), 0.1)

    if selected_poi_type_defs:
        type_defs = [
            item for item in (selected_poi_type_defs or [])
            if item and item[0] and item[1]
        ]
        type_count = len(type_defs)
        if type_count <= 0:
            return {}, 0.0, []

        default_time_budget = DEFAULT_TRAVEL_TIME_MIN
        if activity_time_budgets:
            positive_budgets = [
                float(v) for v in activity_time_budgets.values() if float(v) > 0
            ]
            if positive_budgets:
                default_time_budget = sum(positive_budgets) / len(positive_budgets)

        for type_id, poi_category, keywords in type_defs:
            pois_in_space = _get_pois_in_polygon(calc, poi_category, space_polygon)
            pois_in_space = _filter_pois_by_subcategory(
                pois_in_space, keywords, calc, poi_category,
            )
            total_score, activity_pois, poi_count = _score_pois_in_activity_space(
                pois_in_space,
                center_pt,
                calc,
                beta,
                commute_mode,
                route_line,
                route_dist_ctx,
                speed_ms,
            )
            print(
                f"[DEBUG] 活动空间可达性: {type_id} POI={poi_count} (仅所选子类计入)",
                flush=True,
            )
            category_scores[type_id] = {
                "raw_score": round(total_score, 4),
                "weight": 1.0,
                "weighted_score": round(total_score, 4),
                "poi_count": poi_count,
                "pois": activity_pois,
                "time_budget": default_time_budget,
                "poi_category": poi_category,
            }
            for poi in activity_pois:
                poi["activity"] = type_id
                all_pois.append(poi)

        total_score = sum(s["raw_score"] for s in category_scores.values())
        return category_scores, total_score, all_pois

    final_weights = DEFAULT_WEIGHTS.copy()
    if weights:
        final_weights.update(weights)

    if activity_time_budgets:
        time_budgets = activity_time_budgets.copy()
    else:
        elastic_time = DEFAULT_TRAVEL_TIME_MIN
        time_budgets = {
            activity: elastic_time * weight
            for activity, weight in final_weights.items()
        }

    for activity, weight in final_weights.items():
        if _is_zero_activity_weight(weight):
            category_scores[activity] = {
                "raw_score": 0.0,
                "weight": weight,
                "weighted_score": 0.0,
                "poi_count": 0,
                "pois": [],
                "time_budget": 0.0,
            }
            continue

        poi_category = ACTIVITY_TO_POI.get(activity, activity)
        pois_in_space = _get_pois_in_polygon(calc, poi_category, space_polygon)
        activity_time = time_budgets.get(activity, DEFAULT_TRAVEL_TIME_MIN)
        total_score, activity_pois, poi_count = _score_pois_in_activity_space(
            pois_in_space,
            center_pt,
            calc,
            beta,
            commute_mode,
            route_line,
            route_dist_ctx,
            speed_ms,
        )

        print(
            f"[DEBUG] 活动空间可达性: {activity} POI={poi_count} (多边形内全部计入)",
            flush=True,
        )

        category_scores[activity] = {
            "raw_score": round(total_score, 4),
            "weight": weight,
            "weighted_score": round(total_score * weight, 4),
            "poi_count": poi_count,
            "pois": activity_pois,
            "time_budget": activity_time,
        }

        for poi in activity_pois:
            poi["activity"] = activity
            all_pois.append(poi)

    total_score = sum(s["weighted_score"] for s in category_scores.values())
    return category_scores, total_score, all_pois

def calculate_personal_accessibility(payload):
    """
    计算个人可达性 — 时空棱镜 / 潜在活动路径（PPA）方法（按权重分配时间预算）

    第一步：构建时空锚点（出发地 + 目的地）
    第二步：按活动偏好权重分配可自由支配时间
    第三步：分别计算每类POI在各自时间预算内的可达范围
    第四步：以所有可到达的POI为限定生成PPA范围
    第五步：A_space = Σ W_k · Σ O_j · exp(-β·d_mean(j))（多边形内各类 POI 全部计入）
    """
    try:
        locations = payload.get("locations", [])
        raw_elastic = payload.get("elastic_time", payload.get("travel_time", DEFAULT_TRAVEL_TIME_MIN))
        elastic_time_min = max(5, float(raw_elastic))
        commute_mode = str(payload.get("commute_mode", "foot")).strip().lower()
        raw_weights = payload.get("weights")
        trip_counts = payload.get("trip_counts")

        print(f"[DEBUG] 接收到的参数: elastic_time={elastic_time_min}min, commute_mode={commute_mode}", flush=True)
        print(f"[DEBUG] 接收到的权重: {raw_weights}", flush=True)
        print(f"[DEBUG] 接收到的出行次数: {trip_counts}", flush=True)

        final_weights = _resolve_final_weights(raw_weights, trip_counts)
        active_activity_keys = _active_activities(final_weights)
        if not active_activity_keys:
            return {"success": False, "message": "请至少为一项活动设置大于 0 的每周出行次数"}

        if not locations:
            return {"success": False, "message": "请至少选择一个位置（出发地）"}

        home_loc, work_loc = _extract_home_work(locations)
        if home_loc is None:
            return {"success": False, "message": "请至少选择一个出发地"}

        calc = _get_calculator()
        if calc is None:
            return {
                "success": False,
                "message": "可达性计算引擎初始化失败，请确认路网与 POI 数据已加载",
            }

        home_pt = _location_to_point(home_loc)
        work_pt = _location_to_point(work_loc) if work_loc else None
        speed_ms = _commute_speed_ms(commute_mode)

        print(
            f"[DEBUG] 时空棱镜: 弹性时间={elastic_time_min:.1f}min, 速度={speed_ms:.2f}m/s, "
            f"模式={'路径K点并集' if work_pt else '单起点'}"
        )

        # 按权重分配时间预算，计算每类POI的可达范围
        # 计算每类POI的时间预算
        activity_time_budgets = {}
        all_accessible_pois = {}  # {activity: [poi, ...]}
        
        print(f"[DEBUG] 按权重分配时间预算 (总时间={elastic_time_min}min):", flush=True)
        for activity, weight in final_weights.items():
            if _is_zero_activity_weight(weight):
                activity_time_budgets[activity] = 0.0
                all_accessible_pois[activity] = []
                print(f"  - {activity}: 权重 0%，跳过可达 POI 计算", flush=True)
                continue

            # 该活动的时间预算 = 权重 × 可自由支配时间
            activity_time = elastic_time_min * weight
            activity_time_budgets[activity] = activity_time
            print(f"  - {activity}: {weight:.2%} × {elastic_time_min}min = {activity_time:.1f}min", flush=True)
            
            # 计算该活动在时间预算内的可达POI
            poi_category = ACTIVITY_TO_POI.get(activity, activity)
            max_network_distance = activity_time * 60 * speed_ms  # 转换为米
            search_radius = max_network_distance * 2.0  # 留2倍余量搜索
            
            pois = _get_pois_in_radius(calc, poi_category, home_pt, search_radius, max_count=500)
            
            # 计算每个POI的实际路网距离
            accessible_pois_for_activity = []
            for poi in pois:
                # 计算从出发地到POI的路网距离
                network_dist = _distance_poi_to_network(
                    calc, home_pt, poi["point"], commute_mode
                )
                
                travel_time = network_dist / speed_ms / 60.0  # 转换为分钟
                if travel_time <= activity_time:
                    accessible_pois_for_activity.append({
                        **poi,
                        "network_dist": network_dist,
                        "travel_time": travel_time,
                        "activity": activity
                    })
            
            all_accessible_pois[activity] = accessible_pois_for_activity
            print(f"  - {activity}: 在{activity_time:.1f}min内可到达 {len(accessible_pois_for_activity)} 个POI", flush=True)
        
        # 收集所有可达的POI，构建POI集合用于限定PPA范围
        all_reachable_poi_points = []
        for activity, pois in all_accessible_pois.items():
            all_reachable_poi_points.extend(pois)
        
        print(f"[DEBUG] 总共可到达 {len(all_reachable_poi_points)} 个POI", flush=True)
        
        # 如果有可达POI，用它们来限定PPA范围
        if all_reachable_poi_points:
            # 基于可达 POI 构建网络缓冲/凹壳活动空间
            ppa_result = _build_ppa_polygon_with_pois(
                calc, home_pt, elastic_time_min, commute_mode, 
                work_pt=work_pt, reachable_pois=all_reachable_poi_points
            )
        else:
            # 如果没有可达POI，回退到原始的PPA生成方式
            ppa_result = _build_ppa_polygon(
                calc, home_pt, elastic_time_min, commute_mode, work_pt=work_pt
            )
        
        if not ppa_result or ppa_result.get("polygon") is None:
            return {"success": False, "message": "潜在活动空间生成失败"}

        print("[DEBUG] 潜在活动空间多边形已生成，开始计算空间内可达性…", flush=True)

        route_line = _route_linestring_from_coords(ppa_result.get("route_coords"))
        max_one_way_m = elastic_time_min * 60.0 * _lateral_expansion_speed_ms(commute_mode) * 0.5
        route_dist_ctx = build_route_distance_context(
            calc,
            ppa_result.get("route_coords"),
            commute_mode,
            max_one_way_m,
        )

        activity_bounds = _build_ppa_bounds(
            locations,
            home_pt,
            ppa_result,
            elastic_time_min,
            _lateral_expansion_speed_ms(commute_mode),
        )

        # 计算可达性得分，传递每类POI的时间预算
        category_scores, overall_score, all_pois = _calculate_activity_space_accessibility(
            ppa_result["polygon"],
            home_pt,
            calc,
            final_weights,
            DECAY_BETA,
            commute_mode,
            route_line=route_line,
            route_dist_ctx=route_dist_ctx,
            activity_time_budgets=activity_time_budgets,
        )

        print(
            f"[DEBUG] 空间可达性完成: 总分={overall_score:.4f}, POI数={len(all_pois)}",
            flush=True,
        )

        activity_spaces = [activity_bounds]
        ellipse_bounds = {
            **activity_bounds,
            "cluster_count": 1,
            "r_max_m": round(
                elastic_time_min * 60.0 * _lateral_expansion_speed_ms(commute_mode) * 0.5, 1
            ),
            "spaces": activity_spaces,
        }
        if activity_bounds.get("commute_route"):
            ellipse_bounds["commute_route"] = activity_bounds["commute_route"]

        commute_route = activity_bounds.get("commute_route")

        radar_labels = [ACTIVITY_LABELS.get(a, a) for a in active_activity_keys]
        radar_data = [category_scores.get(a, {}).get("raw_score", 0) for a in active_activity_keys]

        # 可达 POI 展示：仅含出行次数>0 的活动
        accessible_pois_list = _build_accessible_pois_display_list(
            all_accessible_pois, category_scores, active_activity_keys
        )
        print(f"[DEBUG] 可达 POI 展示列表: {len(accessible_pois_list)} 个", flush=True)

        return {
            "success": True,
            "overall_score": round(overall_score, 4),
            "timestamp": datetime.now().isoformat(),
            "activity_space_count": 1,
            "activity_space": ellipse_bounds,
            "activity_spaces": activity_spaces,
            "space_time_prism": {
                "elastic_time_min": round(elastic_time_min, 2),
                "speed_ms": round(_lateral_expansion_speed_ms(commute_mode), 3),
                "method": ppa_result.get("method"),
                "sample_count": int(ppa_result.get("sample_count", 1)),
                "sample_interval_m": PPA_ROUTE_SAMPLE_INTERVAL_M,
            },
            "radar_chart": {
                "labels": radar_labels,
                "datasets": [{
                    "label": "可达性得分",
                    "data": [round(d, 4) for d in radar_data],
                }],
            },
            "accessibility_details": {
                activity: {
                    "raw_score": category_scores.get(activity, {}).get("raw_score", 0),
                    "poi_count": category_scores.get(activity, {}).get("poi_count", 0),
                    "weight": final_weights.get(activity, 0),
                }
                for activity in active_activity_keys
            },
            "accessible_pois": accessible_pois_list,
            "commute": {
                "route_coords": commute_route,
                "mode": commute_mode,
            } if commute_route else None,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"计算失败: {str(e)}",
            "error": str(e),
        }

