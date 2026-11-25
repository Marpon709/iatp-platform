"""
全项目统一的交通方式速度、路网/公共交通线网加载，以及公交地铁可达性线网工具。

各模块应从此处读取速度；路网 GraphML、地铁/公交 GeoJSON 的加载与接驳最短路亦集中在此。
"""
from __future__ import annotations

import math
import os
import threading
from collections import defaultdict
from typing import Callable

import geopandas as gpd
import numpy as np
import rasterio

# 速度 / 方式别名

CARBON_BUDGET_MODE_SPEEDS_KMH: dict[str, float] = {
    "walk": 4.8,
    "bike": 12.0,
    "bus": 25.0,
    "metro": 30.0,
    "drive": 50.0,
}

_MODE_ALIASES: dict[str, str] = {
    "foot": "walk",
    "walk": "walk",
    "bike": "bike",
    "ebike": "bike",
    "bus": "bus",
    "subway": "metro",
    "metro": "metro",
    "drive": "drive",
    "car": "drive",
    "private_car": "drive",
    "pt": "bus",
    "bnr": "bus",
    "pnr": "metro",
    "slow_vehicle": "bus",
}

DEFAULT_MODE_KEY = "walk"
DEFAULT_SPEED_KMH = CARBON_BUDGET_MODE_SPEEDS_KMH[DEFAULT_MODE_KEY]

def normalize_mode_key(mode: str | None) -> str:
    """将任意交通方式字符串规范为核心键（walk/bike/bus/metro/drive）。"""
    m = str(mode or DEFAULT_MODE_KEY).strip().lower()
    return _MODE_ALIASES.get(m, m if m in CARBON_BUDGET_MODE_SPEEDS_KMH else DEFAULT_MODE_KEY)

def speed_kmh(mode: str | None) -> float:
    """返回速度（km/h）。"""
    key = normalize_mode_key(mode)
    return float(CARBON_BUDGET_MODE_SPEEDS_KMH.get(key, DEFAULT_SPEED_KMH))

def speed_ms(mode: str | None) -> float:
    return speed_kmh(mode) / 3.6

def carbon_budget_mode_speeds() -> dict[str, float]:
    """CarbonBudgetParams 默认 mode_speeds 副本。"""
    return CARBON_BUDGET_MODE_SPEEDS_KMH.copy()

def carbon_calculator_mode_config() -> dict[str, dict]:
    """
    CalculateDistance.CarbonBudgetCalculator.mode_config 的速度字段（km/h）。
    排放因子仍由各模块自行维护。
    """
    return {
        "walk": {"speed": speed_kmh("walk"), "network": "foot", "emission_factor": 0.0},
        "car": {"speed": speed_kmh("drive"), "network": "drive", "emission_factor": 0.184},
        "bike": {"speed": speed_kmh("bike"), "network": "bike", "emission_factor": 0.0},
        "bus": {"speed": speed_kmh("bus"), "network": "bus", "emission_factor": 0.015},
        "metro": {"speed": speed_kmh("metro"), "network": "metro", "emission_factor": 0.024},
    }

SPEED_WALK_MS = speed_ms("walk")
SPEED_BIKE_MS = speed_ms("bike")
SPEED_BUS_MS = speed_ms("bus")
SPEED_METRO_MS = speed_ms("metro")
SPEED_DRIVE_MS = speed_ms("drive")

MODE_TO_SPEED_MS: dict[str, float] = {
    "foot": SPEED_WALK_MS,
    "walk": SPEED_WALK_MS,
    "bike": SPEED_BIKE_MS,
    "ebike": SPEED_BIKE_MS,
    "bus": SPEED_BUS_MS,
    "metro": SPEED_METRO_MS,
    "subway": SPEED_METRO_MS,
    "car": SPEED_DRIVE_MS,
    "drive": SPEED_DRIVE_MS,
    "private_car": SPEED_DRIVE_MS,
}

# 路网路径几何（绘制坐标 / 路网距离）

EARTH_RADIUS_M = 6371000.0
ROAD_PATH_MODES = frozenset({"walk", "bike", "car"})
MAX_ROUTE_POINTS = 400
MAX_TRANSIT_RIDE_POINTS = 2500
_TRANSIT_PATH_MIN_STEP_M = 8.0
_STATION_SNAP_M = 25.0
_TRANSIT_STATION_CANDIDATES_K = 8

class _LRUCache:
    """线程安全的小型 LRU 缓存。"""

    __slots__ = ("_maxsize", "_data", "_lock", "_order")

    def __init__(self, maxsize):
        self._maxsize = max(1, int(maxsize))
        self._data = {}
        self._lock = threading.Lock()
        self._order = []

    def get(self, key):
        with self._lock:
            if key not in self._data:
                return None
            try:
                self._order.remove(key)
            except ValueError:
                pass
            self._order.append(key)
            return self._data[key]

    def set(self, key, value):
        with self._lock:
            if key in self._data:
                try:
                    self._order.remove(key)
                except ValueError:
                    pass
            self._data[key] = value
            self._order.append(key)
            while len(self._order) > self._maxsize:
                old = self._order.pop(0)
                self._data.pop(old, None)

# 公交/地铁缓存版本号：修改 _search_transit_od / transit_access 逻辑后递增，
# 避免旧缓存导致站点选择/步行距离计算沿用修复前的结果。
_TRANSIT_CACHE_VERSION = 4

_ACCESS_WALK_M_CACHE = _LRUCache(4096)
_ACCESS_WALK_COORDS_CACHE = _LRUCache(2048)
_TRANSIT_OD_GEOM_CACHE = _LRUCache(1024)

def _point_cache_key(p) -> tuple[float, float]:
    return (round(float(p.x), 5), round(float(p.y), 5))

def _od_cache_key(p_from, p_to, suffix: str = "") -> tuple:
    return (_point_cache_key(p_from), _point_cache_key(p_to), f"v{_TRANSIT_CACHE_VERSION}:{suffix}")

def _clean_transit_ride_topology_latlng(coords_latlng):
    """对 [lat, lng] 乘运折线做拓扑清洗（委托 Get_Network 线网清洗）。"""
    if not coords_latlng or len(coords_latlng) < 2:
        return coords_latlng
    try:
        from Get_Network import clean_transit_line_topology

        lonlat = [(float(c[1]), float(c[0])) for c in coords_latlng if c and len(c) >= 2]
        cleaned = clean_transit_line_topology(lonlat)
        if len(cleaned) < 2:
            return coords_latlng
        return [[lat, lon] for lon, lat in cleaned]
    except Exception:
        return _clean_transit_path_coords(coords_latlng)

def haversine_m(lon1, lat1, lon2, lat2):
    """两点大圆距离（米）。"""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return EARTH_RADIUS_M * 2.0 * math.asin(min(1.0, math.sqrt(max(0.0, a))))

def normalize_route_mode_for_calc(commute_mode):
    """前端交通方式 → CarbonBudgetCalculator 路网模式（walk / car / bike / bus / metro）。"""
    m = (commute_mode or "foot").strip().lower()
    if m in ("drive", "car", "private_car"):
        return "car"
    if m in ("ebike", "bike"):
        return "bike"
    if m in ("subway", "metro"):
        return "metro"
    if m in ("foot", "walk"):
        return "walk"
    if m == "bus":
        return "bus"
    return "walk"

def mode_for_path_geometry(mode):
    m = (mode or "bus").strip().lower()
    if m in ("drive", "car", "private_car"):
        return "car"
    if m in ("ebike", "bike"):
        return "bike"
    if m in ("subway", "metro"):
        return "metro"
    if m == "bus":
        return "bus"
    if m in ("foot", "walk", "slow"):
        return "walk"
    return "walk"

def straight_line_coords(p_from, p_to):
    return [
        [float(p_from.y), float(p_from.x)],
        [float(p_to.y), float(p_to.x)],
    ]

def simplify_route_coords(coords, max_points=MAX_ROUTE_POINTS, min_keep=2):
    if not coords:
        return []
    out = [[float(c[0]), float(c[1])] for c in coords if c and len(c) >= 2]
    if len(out) <= max(max_points, min_keep):
        return out
    step = max(1, len(out) // max_points)
    simplified = out[::step]
    if simplified[-1] != out[-1]:
        simplified.append(out[-1])
    return simplified

def merge_path_segments(*parts):
    path = []
    for part in parts or []:
        if not part:
            continue
        seg = [[float(c[0]), float(c[1])] for c in part if c and len(c) >= 2]
        if not seg:
            continue
        if not path:
            path.extend(seg)
            continue
        if path[-1] == seg[0]:
            path.extend(seg[1:])
        else:
            path.extend(seg)
    return path if len(path) >= 2 else None

def path_length_from_coords(path):
    """[[lat, lng], ...] 折线弧长（米）。"""
    if not path or len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        total += haversine_m(path[i][1], path[i][0], path[i + 1][1], path[i + 1][0])
    return total

def is_straight_line_route(path, p_from, p_to, tol_ratio=1.03):
    """路径是否实质为起终点直线（路网求解失败时的回退形态）。"""
    if not path or len(path) < 2:
        return True
    straight_m = haversine_m(p_from.x, p_from.y, p_to.x, p_to.y)
    if straight_m < 30.0:
        return len(path) <= 2
    path_m = path_length_from_coords(path)
    return path_m <= straight_m * tol_ratio

def _accept_road_path_for_od(path, p_from, p_to):
    """路网折线是否可用于 OD 绘制（多于 2 个点即视为沿道路，避免误判为直线）。"""
    if not path or len(path) < 2:
        return False
    if len(path) > 2:
        return True
    return not is_straight_line_route(path, p_from, p_to)

def _graph_node_id(G, node_id):
    """GraphML 节点 id 多为 str，与 GeoDataFrame index 对齐。"""
    if node_id in G:
        return node_id
    s = str(node_id)
    if s in G:
        return s
    try:
        i = int(node_id)
        if i in G:
            return i
        if str(i) in G:
            return str(i)
    except (TypeError, ValueError):
        pass
    return node_id

def _shortest_path_on_graph(G, origin_node_id, dest_node_id, allow_undirected_fallback=True):
    """
    在路网上求最短路。
    步行/骑行：有向失败时可回退无向图（双向通行）。
    驾车（allow_undirected_fallback=False）：严格有向，避免单行道逆行。
    """
    import networkx as nx

    o = _graph_node_id(G, origin_node_id)
    d = _graph_node_id(G, dest_node_id)
    if o not in G or d not in G:
        raise nx.NodeNotFound(f"node not in graph: {o!r} or {d!r}")

    graphs = [G]
    if allow_undirected_fallback and getattr(G, "is_directed", lambda: False)():
        try:
            graphs.append(G.to_undirected())
        except Exception:
            pass

    last_err = None
    for g in graphs:
        for weight in ("length", "weight", None):
            try:
                if weight is None:
                    return nx.shortest_path(g, o, d)
                return nx.shortest_path(g, o, d, weight=weight)
            except Exception as e:
                last_err = e
    if last_err is not None:
        raise last_err
    raise nx.NetworkXNoPath(f"no path between {o} and {d}")

def _network_od_reference_m(calc, p_from, p_to, path_mode, straight_m):
    """路网参考距离（米），用于校验绘制路径是否异常绕远。"""
    if calc is None or not hasattr(calc, "_calculate_travel_time_and_distance"):
        return float(straight_m)
    mode_key = path_mode if path_mode in ("walk", "bike", "car") else "walk"
    try:
        _, dist = calc._calculate_travel_time_and_distance(p_from, p_to, mode_key)
        if dist is not None and math.isfinite(dist) and dist > 0:
            return float(dist)
    except Exception:
        pass
    return float(straight_m)

def _node_lonlat(nodes_gdf, node_id):
    nid = node_id if node_id in nodes_gdf.index else str(node_id)
    geom = nodes_gdf.loc[nid].geometry
    return float(geom.x), float(geom.y)

def _find_best_od_route(
    G,
    tree,
    nodes,
    p_from,
    p_to,
    *,
    k=8,
    allow_undirected=False,
    include_snap_cost=True,
):
    """
    在起终点各自 k 近邻路网节点中搜索最优路径，缓解单点吸附到弱连通分量。
    include_snap_cost=True 时总代价含起终点到吸附节点的直线距离，
    确保找到的是「起终点之间的真实最短路径」而非仅路网段最短（避免因忽略
    接驳距离而选中绕远节点对，导致步行/骑行等出现绕路）。
    返回 (route_node_ids, used_undirected, route_weight_m) 或 (None, False, None)。
    """
    import networkx as nx

    if G is None or tree is None or nodes is None or len(nodes) == 0:
        return None, False, None

    k = max(1, min(int(k), len(nodes)))
    oc = np.array([[float(p_from.x), float(p_from.y)]])
    dc = np.array([[float(p_to.x), float(p_to.y)]])
    _, oi = tree.query(oc, k=k)
    _, di = tree.query(dc, k=k)

    def _unique_node_ids(indices):
        out = []
        seen = set()
        for idx in indices:
            nid = _graph_node_id(G, nodes.index[int(idx)])
            if nid in seen:
                continue
            seen.add(nid)
            out.append(nid)
        return out

    origin_ids = _unique_node_ids(oi[0])
    dest_ids = _unique_node_ids(di[0])

    def _snap_o_m(node_id):
        lon, lat = _node_lonlat(nodes, node_id)
        return haversine_m(float(p_from.x), float(p_from.y), lon, lat)

    def _snap_d_m(node_id):
        lon, lat = _node_lonlat(nodes, node_id)
        return haversine_m(lon, lat, float(p_to.x), float(p_to.y))

    def _search(on_graph):
        best_w = None
        best_route = None
        for o in origin_ids:
            snap_o = _snap_o_m(o)
            for d in dest_ids:
                try:
                    if o == d:
                        # 起终点吸附到同一节点 → 路径仅含接驳段
                        w = snap_o + _snap_d_m(d)
                        route_nodes = [o]
                    else:
                        w = float(nx.shortest_path_length(on_graph, o, d, weight="length"))
                        route_nodes = nx.shortest_path(on_graph, o, d, weight="length")
                        if include_snap_cost:
                            w += snap_o + _snap_d_m(d)
                except Exception:
                    continue
                if best_w is None or w < best_w:
                    best_w = w
                    best_route = route_nodes
        return best_route, best_w

    # 优先在有向图上搜索（驾车遵守单行道）
    route, route_w = _search(G)
    if route is not None:
        return route, False, route_w

    # 步行/骑行：有向图无路径时回退无向图（人行道、自行车道双向通行）
    if allow_undirected and getattr(G, "is_directed", lambda: False)():
        try:
            route, route_w = _search(G.to_undirected())
            if route is not None:
                return route, True, route_w
        except Exception:
            pass
    return None, False, None

def _resolve_road_network_context(calc, path_mode):
    if calc is None or path_mode not in ROAD_PATH_MODES:
        return None, None, None

    cfg = getattr(calc, "mode_config", {}).get(path_mode)
    if not cfg:
        return None, None, None
    network_key = cfg.get("network")
    G = getattr(calc, "networks", {}).get(network_key)
    if G is None:
        return None, None, None

    network_trees = getattr(calc, "network_trees", {})
    if path_mode in network_trees:
        tree, _coords, nodes = network_trees[path_mode]
        return G, tree, nodes

    try:
        import osmnx as ox  # 第三方开源库，用于路网节点坐标提取与索引构建
        from sklearn.neighbors import BallTree

        nodes = ox.graph_to_gdfs(G, edges=False)
        if nodes is None or len(nodes) == 0:
            return None, None, None
        coords = np.array([[float(n.x), float(n.y)] for n in nodes.geometry])
        tree = BallTree(coords, leaf_size=40)
        print(f"[路网] 临时构建 {path_mode} 索引（network={network_key}，节点 {len(nodes)}）")
        return G, tree, nodes
    except Exception as e:
        print(f"[路网] 无法构建 {path_mode} 索引: {e}")
        return None, None, None

def fetch_road_od_path_fast(calc, p_from, p_to, mode="walk"):
    """
    快速路网 OD，用于批量出行推荐。
    所有路网模式（步行/骑行/驾车）均使用多候选吸附搜索总路径最短的节点对，
    避免单点吸附到弱连通分量导致绕路（尤其是步行网络较稀疏的场景）。
    """
    if calc is None:
        return straight_line_coords(p_from, p_to)

    path_mode = mode_for_path_geometry(mode)
    if path_mode not in ROAD_PATH_MODES:
        return straight_line_coords(p_from, p_to)

    try:
        from CalculateDistance import attach_od_endpoints_to_path, route_nodes_to_path_coords

        G, tree, nodes = _resolve_road_network_context(calc, path_mode)
        if G is None or tree is None or nodes is None:
            return straight_line_coords(p_from, p_to)

        straight_m = haversine_m(p_from.x, p_from.y, p_to.x, p_to.y)

        # 所有模式统一使用多候选吸附，按总路径（含接驳）最短选优
        if path_mode == "car":
            snap_k = 10
            allow_undirected = True
        elif path_mode == "walk":
            # 步行网络稀疏，增大候选数以覆盖更广的接入点
            snap_k = 6
            allow_undirected = True
        else:
            # bike
            snap_k = 5
            allow_undirected = True

        route, _, _ = _find_best_od_route(
            G,
            tree,
            nodes,
            p_from,
            p_to,
            k=snap_k,
            allow_undirected=allow_undirected,
            include_snap_cost=True,
        )

        if not route:
            return straight_line_coords(p_from, p_to)

        path = route_nodes_to_path_coords(G, route, nodes)
        path = attach_od_endpoints_to_path(path, p_from, p_to)
        if _accept_road_path_for_od(path, p_from, p_to):
            return simplify_route_coords(path, max_points=MAX_ROUTE_POINTS)
    except Exception as e:
        print(f"[DEBUG] fetch_road_od_path_fast({path_mode}) 失败: {e}")

    return straight_line_coords(p_from, p_to)

def fetch_road_od_path(calc, p_from, p_to, mode="walk", *, for_transit_access=False):
    """步行/骑行/驾车沿对应道路网的 OD 路径 [[lat, lng], ...]。
    所有模式均通过多候选吸附搜索「起终点间总路径最短」的节点对，
    避免因仅优化路网段而忽略接驳距离导致的绕路。"""
    if calc is None:
        return straight_line_coords(p_from, p_to)

    path_mode = mode_for_path_geometry(mode)
    if path_mode not in ROAD_PATH_MODES:
        return straight_line_coords(p_from, p_to)

    try:
        import numpy as np
        from CalculateDistance import attach_od_endpoints_to_path, route_nodes_to_path_coords

        G, tree, nodes = _resolve_road_network_context(calc, path_mode)
        if G is None or tree is None or nodes is None:
            print(f"[DEBUG] 道路网不可用: mode={path_mode}")
            return straight_line_coords(p_from, p_to)

        straight_m = haversine_m(p_from.x, p_from.y, p_to.x, p_to.y)
        if for_transit_access and path_mode == "walk":
            snap_k = 4
        elif path_mode == "car":
            snap_k = 10
        elif path_mode == "walk":
            # 步行网络较稀疏，增大候选数以找到不绕路的入口
            snap_k = 6 if straight_m > 3000.0 else 5
        else:
            # bike
            snap_k = 5 if straight_m > 3000.0 else 4

        def _build_path(snap_k_val):
            route, _used_undirected, route_weight_m = _find_best_od_route(
                G,
                tree,
                nodes,
                p_from,
                p_to,
                k=snap_k_val,
                allow_undirected=True,
                include_snap_cost=True,
            )
            if not route:
                return None, route_weight_m
            geom = route_nodes_to_path_coords(G, route, nodes)
            if for_transit_access:
                geom = _snap_path_endpoints(geom, p_from, p_to)
            else:
                geom = attach_od_endpoints_to_path(geom, p_from, p_to)
            return geom, route_weight_m

        path, route_weight_m = _build_path(snap_k)
        if not path:
            print(f"[DEBUG] fetch_road_od_path({path_mode}) 无连通路径，回退直线")
            return straight_line_coords(p_from, p_to)

        if for_transit_access and path_mode == "walk":
            path_m = path_length_from_coords(path)
            if path_m > max(straight_m * 1.55, straight_m + 80.0):
                alt_path, _ = _build_path(1)
                if alt_path:
                    alt_m = path_length_from_coords(alt_path)
                    if alt_m < path_m:
                        path = alt_path

        if len(path) >= 2 and (
            for_transit_access or _accept_road_path_for_od(path, p_from, p_to)
        ):
            if for_transit_access:
                return simplify_route_coords(path)
            return simplify_route_coords(path)
    except Exception as e:
        print(f"[DEBUG] fetch_road_od_path({path_mode}) 失败: {e}")

    return straight_line_coords(p_from, p_to)

def _clean_transit_path_coords(coords, min_step_m=_TRANSIT_PATH_MIN_STEP_M):
    if not coords or len(coords) < 2:
        return coords
    out = [[float(coords[0][0]), float(coords[0][1])]]
    for i in range(1, len(coords)):
        lat, lon = float(coords[i][0]), float(coords[i][1])
        prev = out[-1]
        if haversine_m(prev[1], prev[0], lon, lat) < min_step_m:
            continue
        out.append([lat, lon])
    if len(out) < 2:
        return [
            [float(coords[0][0]), float(coords[0][1])],
            [float(coords[-1][0]), float(coords[-1][1])],
        ]
    return out

def _ensure_path_endpoints(path, p_from, p_to):
    if not path:
        return straight_line_coords(p_from, p_to)
    out = [[float(c[0]), float(c[1])] for c in path if c and len(c) >= 2]
    if not out:
        return straight_line_coords(p_from, p_to)
    orig_lat, orig_lon = float(p_from.y), float(p_from.x)
    dest_lat, dest_lon = float(p_to.y), float(p_to.x)
    if abs(out[0][0] - orig_lat) > 1e-6 or abs(out[0][1] - orig_lon) > 1e-6:
        out.insert(0, [orig_lat, orig_lon])
    if abs(out[-1][0] - dest_lat) > 1e-6 or abs(out[-1][1] - dest_lon) > 1e-6:
        out.append([dest_lat, dest_lon])
    return out

def _snap_path_endpoints(path, p_from, p_to):
    """将折线首尾对齐到起终点（替换而非插入），避免接驳步行出现 V 形毛刺。"""
    if not path:
        return straight_line_coords(p_from, p_to)
    out = [[float(c[0]), float(c[1])] for c in path if c and len(c) >= 2]
    if not out:
        return straight_line_coords(p_from, p_to)
    out[0] = [float(p_from.y), float(p_from.x)]
    out[-1] = [float(p_to.y), float(p_to.x)]
    return out

def _clean_access_walk_path(path):
    """接驳步行仅去重过密点，不做公交线网拓扑清洗（避免把道路折线洗成绕远环）。"""
    return _clean_transit_path_coords(path)

def _transit_access_walk_coords_uncached(calc, p_from, p_to):
    """
    公交/地铁接驳步行几何：沿步行路网规划；仅路网不可达或路径明显绕远时回退直线。
    """
    road = fetch_road_od_path(calc, p_from, p_to, "walk", for_transit_access=True)
    road = _snap_path_endpoints(road, p_from, p_to)
    if road and len(road) >= 2:
        # 安全检查：若路网步行路径超过直线距离 2 倍，回退直线（避免绕远）
        road_m = path_length_from_coords(road)
        straight_m = haversine_m(p_from.x, p_from.y, p_to.x, p_to.y)
        if road_m > max(straight_m * 2.0, straight_m + 200.0):
            return _clean_access_walk_path(straight_line_coords(p_from, p_to))
        return _clean_access_walk_path(road)
    return _clean_access_walk_path(straight_line_coords(p_from, p_to))

def transit_access_walk_coords(calc, p_from, p_to):
    key = _od_cache_key(p_from, p_to, "coords_v3")
    cached = _ACCESS_WALK_COORDS_CACHE.get(key)
    if cached is not None:
        return cached
    path = _transit_access_walk_coords_uncached(calc, p_from, p_to)
    _ACCESS_WALK_COORDS_CACHE.set(key, path)
    return path

def transit_access_walk_m(calc, p_from, p_to) -> float:
    key = _od_cache_key(p_from, p_to, "m_v3")
    cached = _ACCESS_WALK_M_CACHE.get(key)
    if cached is not None:
        return float(cached)
    coords = transit_access_walk_coords(calc, p_from, p_to)
    dist = path_length_from_coords(coords)
    if dist > 0:
        val = float(dist)
    else:
        val = float(haversine_m(p_from.x, p_from.y, p_to.x, p_to.y))
    _ACCESS_WALK_M_CACHE.set(key, val)
    return val

def _ensure_ride_connects_stations(ride_coords, lat_o, lon_o, lat_d, lon_d):
    if not ride_coords:
        return [[float(lat_o), float(lon_o)], [float(lat_d), float(lon_d)]]
    out = [[float(c[0]), float(c[1])] for c in ride_coords if c and len(c) >= 2]
    if not out:
        return [[float(lat_o), float(lon_o)], [float(lat_d), float(lon_d)]]
    try:
        from Get_Network import TOPO_SNAP_STOP_M, trim_polyline_to_stop

        lonlat = [(float(c[1]), float(c[0])) for c in out]
        if haversine_m(lon_o, lat_o, out[0][1], out[0][0]) > 1.0:
            lonlat = trim_polyline_to_stop(lonlat, lon_o, lat_o, "start", TOPO_SNAP_STOP_M)
        if len(lonlat) >= 2 and haversine_m(lon_d, lat_d, lonlat[-1][1], lonlat[-1][0]) > 1.0:
            lonlat = trim_polyline_to_stop(lonlat, lon_d, lat_d, "end", TOPO_SNAP_STOP_M)
        if len(lonlat) >= 2:
            out = [[lat, lon] for lon, lat in lonlat]
    except Exception:
        if haversine_m(lon_o, lat_o, out[0][1], out[0][0]) > 1.0:
            out[0] = [float(lat_o), float(lon_o)]
        if haversine_m(lon_d, lat_d, out[-1][1], out[-1][0]) > 1.0:
            out[-1] = [float(lat_d), float(lon_d)]
    if haversine_m(lon_o, lat_o, out[0][1], out[0][0]) > 1.0:
        out.insert(0, [float(lat_o), float(lon_o)])
    if haversine_m(lon_d, lat_d, out[-1][1], out[-1][0]) > 1.0:
        out.append([float(lat_d), float(lon_d)])
    out = _clean_transit_ride_topology_latlng(out)
    return _clean_transit_path_coords(out)

def fetch_transit_access_coords(calc, p_from, p_to):
    return transit_access_walk_coords(calc, p_from, p_to)

def _build_transit_ride_stops(tn, stop_a, stop_b):
    if tn is None or not stop_a or not stop_b:
        return []
    stops = []
    seen = set()
    for sid in tn.stop_pair_path_ids(stop_a, stop_b):
        if sid in seen:
            continue
        seen.add(sid)
        ll = tn.stop_lonlat.get(sid)
        if not ll:
            continue
        lon, lat = ll
        name = tn.stop_name(sid) or sid
        stops.append({"name": name, "lat": lat, "lon": lon})
    return stops

def _build_transit_walk_only_geometry(calc, p_from, p_to, total_distance_m):
    seg_walk = fetch_transit_access_coords(calc, p_from, p_to)
    walk_m = path_length_from_coords(seg_walk)
    dist_m = float(walk_m if walk_m > 0 else total_distance_m)
    return {
        "merged": seg_walk,
        "total_distance_m": dist_m,
        "parts": [
            {"type": "walk", "geometry": "transit_access", "coords": seg_walk},
        ],
        "ride_point_count": 0,
    }

def _metro_origin_beyond_snap(calc, origin, tn):
    if calc is None or tn is None:
        return False
    if not hasattr(calc, "_origin_beyond_metro_snap"):
        return False
    try:
        return bool(calc._origin_beyond_metro_snap(origin, tn))
    except Exception:
        return False

def _estimate_access_walk_m(p_from, p_to) -> float:
    """站点组合搜索阶段用直线距离近似接驳步行，避免大量路网最短路。"""
    return float(haversine_m(p_from.x, p_from.y, p_to.x, p_to.y))

def _search_transit_od(calc, p_from, p_to, mode="metro"):
    """
    在候选站点中搜索最优公交/地铁 OD。
    返回 (best_ride, best_walk_only)：
      best_ride = (total_m, s_o, s_d, lon_o, lat_o, lon_d, lat_d) 或 None
      best_walk_only = float 或 None
    """
    from shapely.geometry import Point as ShapelyPoint

    networks = getattr(calc, "transit_networks", None) or {}
    tn = networks.get(mode)
    if tn is None:
        return None, None

    if mode == "metro" and _metro_origin_beyond_snap(calc, p_from, tn):
        walk_m = transit_access_walk_m(calc, p_from, p_to)
        return None, float(walk_m)

    orig_stops = tn.nearest_stops(
        p_from.x, p_from.y, k=_TRANSIT_STATION_CANDIDATES_K, max_dist_m=2500.0
    )
    dest_stops = tn.nearest_stops(
        p_to.x, p_to.y, k=_TRANSIT_STATION_CANDIDATES_K, max_dist_m=2500.0
    )
    if not orig_stops or not dest_stops:
        return None, None

    # 预计算所有候选起/终点站的道路网步行距离（最多 K 个），避免仅凭直线估计排序
    # 而漏掉"直线略远但路网步行更近"的站点（如受河流/高架阻隔）。
    orig_walk = {}   # s_o → walk_m
    for s_o, lon_o, lat_o, _ in orig_stops:
        station_o = ShapelyPoint(lon_o, lat_o)
        orig_walk[s_o] = (transit_access_walk_m(calc, p_from, station_o), lon_o, lat_o)

    dest_walk = {}   # s_d → walk_m
    for s_d, lon_d, lat_d, _ in dest_stops:
        station_d = ShapelyPoint(lon_d, lat_d)
        dest_walk[s_d] = (transit_access_walk_m(calc, station_d, p_to), lon_d, lat_d)

    # 步行惩罚系数：1m 接驳步行等价于 N 米乘车，使算法优先选择步行更短的站点
    _WALK_PENALTY = 3.0

    best_walk_only = None
    best_ride = None
    best_total = None

    for s_o, (walk_o, lon_o, lat_o) in orig_walk.items():
        for s_d, (walk_d, lon_d, lat_d) in dest_walk.items():
            if s_o == s_d:
                total = walk_o + walk_d
                if best_walk_only is None or total < best_walk_only:
                    best_walk_only = total
                continue
            ride = tn.stop_pair_distance_m(s_o, s_d)
            if not math.isfinite(ride) or ride <= 0:
                total = walk_o + walk_d
                if best_walk_only is None or total < best_walk_only:
                    best_walk_only = total
                continue
            # 加权总分 = 步行×惩罚系数 + 乘车距离，优先选择接驳步行更短的站点
            total = walk_o * _WALK_PENALTY + ride + walk_d * _WALK_PENALTY
            if best_total is None or total < best_total:
                best_total = total
                best_ride = (walk_o + ride + walk_d, s_o, s_d, lon_o, lat_o, lon_d, lat_d)

    return best_ride, best_walk_only

def compute_transit_od_distance_m(calc, p_from, p_to, mode="metro"):
    """公交/地铁 OD 总距离（米），不含几何构建。"""
    if calc is None:
        return haversine_m(p_from.x, p_from.y, p_to.x, p_to.y)
    best_ride, best_walk_only = _search_transit_od(calc, p_from, p_to, mode)
    if best_ride is None:
        if best_walk_only is None:
            return None
        return float(best_walk_only)
    if best_walk_only is not None and best_walk_only < best_ride[0]:
        return float(best_walk_only)
    return float(best_ride[0])

def _geometry_from_transit_od(calc, p_from, p_to, mode, best_ride, best_walk_only):
    from shapely.geometry import Point as ShapelyPoint

    if best_ride is None:
        if best_walk_only is None:
            return None
        return _build_transit_walk_only_geometry(calc, p_from, p_to, best_walk_only)

    if best_walk_only is not None and best_walk_only < best_ride[0]:
        return _build_transit_walk_only_geometry(calc, p_from, p_to, best_walk_only)

    networks = getattr(calc, "transit_networks", None) or {}
    tn = networks.get(mode)
    if tn is None:
        return None

    _, s_o, s_d, lon_o, lat_o, lon_d, lat_d = best_ride
    station_o = ShapelyPoint(lon_o, lat_o)
    station_d = ShapelyPoint(lon_d, lat_d)
    seg_walk_o = fetch_transit_access_coords(calc, p_from, station_o)
    seg_walk_d = fetch_transit_access_coords(calc, station_d, p_to)
    seg_ride = _ensure_ride_connects_stations(
        [[lat, lon] for lon, lat in tn.stop_pair_path_coords(s_o, s_d)],
        lat_o,
        lon_o,
        lat_d,
        lon_d,
    )
    ride_stops = _build_transit_ride_stops(tn, s_o, s_d)
    merged = merge_path_segments(seg_walk_o, seg_ride, seg_walk_d)
    return {
        "merged": merged,
        "total_distance_m": float(best_ride[0]),
        "parts": [
            {"type": "walk", "geometry": "transit_access", "coords": seg_walk_o},
            {"type": mode, "geometry": "transit_ride", "coords": seg_ride, "stops": ride_stops},
            {"type": "walk", "geometry": "transit_access", "coords": seg_walk_d},
        ],
        "ride_point_count": len(seg_ride),
    }

def build_transit_path_geometry(calc, p_from, p_to, mode="metro"):
    if calc is None:
        return None
    mode = normalize_mode_key(mode)
    if mode not in ("bus", "metro"):
        return None

    cache_key = _od_cache_key(p_from, p_to, f"geom:{mode}")
    cached = _TRANSIT_OD_GEOM_CACHE.get(cache_key)
    if cached is not None:
        return cached

    best_ride, best_walk_only = _search_transit_od(calc, p_from, p_to, mode)
    result = _geometry_from_transit_od(calc, p_from, p_to, mode, best_ride, best_walk_only)
    if result is not None:
        _TRANSIT_OD_GEOM_CACHE.set(cache_key, result)
    return result

def get_route_geometry_coords(calc, p_from, p_to, mode="bus"):
    if calc is None:
        return straight_line_coords(p_from, p_to)

    mode_key = normalize_mode_key(mode)

    def _accept_road_path(path):
        return (
            path
            and len(path) >= 2
            and not is_straight_line_route(path, p_from, p_to)
        )

    try:
        if mode_key in ("bus", "metro"):
            transit = build_transit_path_geometry(calc, p_from, p_to, mode_key)
            if transit and isinstance(transit, dict):
                merged = transit.get("merged") or []
                if _accept_road_path(merged):
                    return merged
            for walk_mode in ("walk", "foot"):
                walk_path = fetch_road_od_path(calc, p_from, p_to, walk_mode)
                if _accept_road_path(walk_path):
                    return walk_path
        else:
            road_try = []
            pm = mode_for_path_geometry(mode)
            if pm:
                road_try.append(pm)
            if mode_key == "walk" and "walk" not in road_try:
                road_try.append("walk")
            if mode_key == "drive" and "car" not in road_try:
                road_try.append("car")
            for rm in road_try:
                road_path = fetch_road_od_path(calc, p_from, p_to, rm)
                if _accept_road_path(road_path):
                    return road_path
    except Exception as e:
        print(f"[DEBUG] 路网路径几何失败，尝试计算器回退: {e}")

    try:
        mode_cb = normalize_route_mode_for_calc(mode)
        path = calc._get_route_path_geometry(p_from, p_to, mode_cb)
        if _accept_road_path(path):
            return path
    except Exception as e:
        print(f"[DEBUG] 计算器路径几何失败: {e}")

    for walk_mode in ("walk", "foot"):
        try:
            walk_path = fetch_road_od_path(calc, p_from, p_to, walk_mode)
            if _accept_road_path(walk_path):
                return walk_path
        except Exception:
            pass

    return straight_line_coords(p_from, p_to)

def get_route_distance(calc, p_from, p_to, mode="bus"):
    """获取两点间路网/线网距离（米）。"""
    if calc is None:
        return haversine_m(p_from.x, p_from.y, p_to.x, p_to.y)
    path_mode = mode_for_path_geometry(mode)
    if path_mode in ("metro", "bus"):
        dist = compute_transit_od_distance_m(calc, p_from, p_to, path_mode)
        if dist is not None and math.isfinite(dist) and dist > 0:
            return float(dist)
    if path_mode in ROAD_PATH_MODES:
        path = fetch_road_od_path(calc, p_from, p_to, mode)
        if path and len(path) >= 2 and not is_straight_line_route(path, p_from, p_to):
            total_dist = path_length_from_coords(path)
            straight_dist = haversine_m(p_from.x, p_from.y, p_to.x, p_to.y)
            if total_dist > straight_dist * 3:
                return straight_dist
            return total_dist
    try:
        mode_cb = normalize_route_mode_for_calc(mode)
        path = calc._get_route_path_geometry(p_from, p_to, mode_cb)
        if path and len(path) >= 2:
            total_dist = path_length_from_coords(path)
            straight_dist = haversine_m(p_from.x, p_from.y, p_to.x, p_to.y)
            if total_dist > straight_dist * 3:
                return straight_dist
            return total_dist
    except Exception:
        pass
    return haversine_m(p_from.x, p_from.y, p_to.x, p_to.y)

def fetch_route_coords(calc, p_from, p_to, mode):
    """出行推荐等模块用：按方式返回路径坐标（公交/地铁不回落道路网时为直线）。"""
    if calc is None:
        return straight_line_coords(p_from, p_to)
    path_mode = mode_for_path_geometry(mode)
    if path_mode in ("metro", "bus"):
        transit_path = build_transit_path_geometry(calc, p_from, p_to, path_mode)
        if transit_path:
            if isinstance(transit_path, dict):
                return simplify_route_coords(
                    transit_path.get("merged") or [],
                    max_points=MAX_TRANSIT_RIDE_POINTS,
                )
            return simplify_route_coords(transit_path, max_points=MAX_TRANSIT_RIDE_POINTS)
        return straight_line_coords(p_from, p_to)
    return fetch_road_od_path(calc, p_from, p_to, mode)

# 路网 / 公交·地铁线网加载

def webgis_base_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))

def get_city_data_config(city: str) -> dict:
    city_name = str(city or "南京").strip()
    mapping = {
        "南京": {
            "folder": "Network",
            "poi_folder": os.path.join("POI", "Nanjing"),
            "prefix": "nanjing",
            "road_files": {
                "foot": os.path.join("Network", "Nanjing", "NanjingRoad_foot.graphml"),
                "bike": os.path.join("Network", "Nanjing", "NanjingRoad_bike.graphml"),
                "drive": os.path.join("Network", "Nanjing", "NanjingRoad_car.graphml"),
            },
            "boundary_file": os.path.join("Network", "Nanjing", "Nanjing_NoRiver.shp"),
            "population_file": os.path.join("Network", "Nanjing", "Nanjing_population.tif"),
            "gdp_file": os.path.join("Network", "Nanjing", "Nan_GDP_2020.tif"),
        },
        "武汉": {
            "folder": "Network",
            "poi_folder": os.path.join("POI", "Wuhan"),
            "prefix": "wuhan",
            "road_file": os.path.join("Network", "Wuhan", "WuhanRoad.graphml"),
            "boundary_file": os.path.join("Network", "Wuhan", "Wuhan_NoRiver.shp"),
            "population_file": os.path.join("Network", "Wuhan", "Wuhan_Population.tif"),
        },
        "北京": {
            "folder": "Network",
            "poi_folder": os.path.join("POI", "Beijing"),
            "prefix": "beijing",
            "road_file": os.path.join("Network", "Beijing", "BeijingRoad.graphml"),
            "boundary_file": os.path.join("Network", "Beijing", "Beijing_NoRiverr.shp"),
            "population_file": os.path.join("Network", "Beijing", "Beijing_Poplation.tif"),
        },
        "上海": {
            "folder": "Network",
            "poi_folder": os.path.join("POI", "Shanghai"),
            "prefix": "shanghai",
            "road_file": os.path.join("Network", "Shanghai", "ShanghaiRoad.graphml"),
            "boundary_file": os.path.join("Network", "Shanghai", "Shanghai_NoRiverr.shp"),
            "population_file": os.path.join("Network", "Shanghai", "shanghai_population.tif"),
        },
    }
    return mapping.get(city_name, mapping["南京"])

def get_city_gdp_path(city: str, base_dir: str | None = None) -> str | None:
    cfg = get_city_data_config(city)
    rel = cfg.get("gdp_file")
    if not rel:
        return None
    path = os.path.join(base_dir or webgis_base_dir(), rel)
    return path if os.path.isfile(path) else None

def _read_graphml_road_network(path: str):
    """读取 GraphML 路网并规范化节点数值属性。"""
    import networkx as nx

    road_network = nx.read_graphml(path)
    road_network.graph["crs"] = "EPSG:4326"
    for node in road_network.nodes():
        for attr in list(road_network.nodes[node].keys()):
            value = road_network.nodes[node][attr]
            if isinstance(value, str):
                try:
                    road_network.nodes[node][attr] = float(value)
                except ValueError:
                    pass
    return road_network

def load_road_networks(city: str = "南京", base_dir: str | None = None) -> dict:
    """从 GraphML 加载步行/骑行/驾车道路网，返回 networks 字典（键：foot、bike、drive）。"""
    city_name = str(city or "南京").strip()
    base_dir = base_dir or webgis_base_dir()
    cfg = get_city_data_config(city_name)
    data_dir = os.path.join(base_dir, cfg["folder"])
    prefix = cfg["prefix"]

    networks: dict = {}
    road_files = cfg.get("road_files")

    if road_files:
        for net_key, rel_path in road_files.items():
            road_path = os.path.join(base_dir, rel_path)
            print(f"[路网] 城市: {city_name}, {net_key}: {road_path}")
            print(f"[路网] 文件存在: {os.path.exists(road_path)}")
            if not os.path.exists(road_path):
                print(f"[路网] 未找到路网文件: {road_path}")
                continue
            try:
                G = _read_graphml_road_network(road_path)
                networks[net_key] = G
                print(
                    f"  [OK] {net_key}: {len(G.nodes)} 个节点, {len(G.edges)} 条边"
                )
            except Exception as e:
                print(f"  [FAIL] 读取 {net_key} 路网失败: {e}")
        return networks

    road_file = cfg.get("road_file")
    road_path = (
        os.path.join(base_dir, road_file)
        if road_file
        else os.path.join(data_dir, f"{prefix}_drive.graphml")
    )

    print(f"[路网] 城市: {city_name}, 尝试路径: {road_path}")
    print(f"[路网] 文件存在: {os.path.exists(road_path)}")

    if not os.path.exists(road_path):
        print(f"[路网] 未找到路网文件: {road_path}")
        return networks

    try:
        road_network = _read_graphml_road_network(road_path)
        networks["drive"] = road_network
        networks["foot"] = road_network
        networks["bike"] = road_network
        print(f"已读取路网: {len(road_network.nodes)} 个节点, {len(road_network.edges)} 条边")
    except Exception as e:
        print(f"读取路网失败: {e}")

    return networks

def load_transit_networks(city: str = "南京", base_dir: str | None = None) -> dict:
    """加载城市地铁/公交线网（当前仅城市提供 GeoJSON）。"""
    city_name = str(city or "南京").strip()
    if city_name != "南京":
        return {}
    from Get_Network import load_nanjing_transit_networks

    return load_nanjing_transit_networks(base_dir)

# 城市数据集（路网 + POI + 人口 + 边界）

_CITY_DATA_CACHE: dict = {}
_CITY_DATA_LOCK = threading.Lock()
# 路网分文件（foot/bike/car）后 bump，避免沿用旧版合并路网缓存
_CITY_DATA_CACHE_VERSION = "foot-bike-drive-v1"

def _city_data_cache_key(city_name: str) -> str:
    return f"{city_name}:{_CITY_DATA_CACHE_VERSION}"

def read_all_city_data(city: str = "南京", use_cache: bool = True):
    """
    读取城市全套数据。use_cache=True 时首次加载后缓存。
    返回 (networks, poi_data, population_data, pop_transform,
          nodes_with_population, admin_boundary, pop_raster_crs)
    """
    city_name = str(city or "南京").strip()
    cache_key = _city_data_cache_key(city_name)
    if use_cache:
        with _CITY_DATA_LOCK:
            if cache_key in _CITY_DATA_CACHE:
                print(f"使用已缓存的城市数据（{cache_key}，跳过重复加载）")
                return _CITY_DATA_CACHE[cache_key]

    print(f"开始读取城市数据: {city_name} ...")
    base_dir = webgis_base_dir()
    cfg = get_city_data_config(city_name)
    print(f"[数据加载] {city_name} 配置: {cfg}")

    data_dir = os.path.join(base_dir, cfg["folder"])
    poi_dir = os.path.join(base_dir, cfg.get("poi_folder", cfg["folder"]))
    prefix = cfg["prefix"]

    networks = load_road_networks(city_name, base_dir)

    from Data_loader import load_poi_from_geojson

    print(f"[POI] 加载目录: {poi_dir}")
    poi_data = load_poi_from_geojson(poi_dir, use_cache=True)
    print(
        f"[POI] 加载结果: "
        f"{[(k, len(v)) if v is not None and not v.empty else (k, 0) for k, v in poi_data.items()]}"
    )

    pop_file = cfg.get("population_file")
    pop_path = (
        os.path.join(base_dir, pop_file)
        if pop_file
        else os.path.join(data_dir, f"{prefix}_population.tif")
    )
    pop_raster_crs = None
    pop_ext = os.path.splitext(pop_path)[1].lower()

    if pop_ext == ".tif" and os.path.exists(pop_path):
        with rasterio.open(pop_path) as src:
            population_data = src.read(1)
            pop_transform = src.transform
            raster_crs = src.crs
        print(f"已读取人口栅格数据: {os.path.basename(pop_path)}, CRS: {raster_crs}")
        pop_raster_crs = raster_crs
        nodes_with_population = gpd.GeoDataFrame(
            columns=["geometry"],
            crs=raster_crs.to_string() if raster_crs else "EPSG:4326",
        )
    elif pop_ext == ".geojson" and os.path.exists(pop_path):
        nodes_with_population = gpd.read_file(pop_path)
        if nodes_with_population.crs is None:
            nodes_with_population = nodes_with_population.set_crs("EPSG:4326", allow_override=True)
        elif nodes_with_population.crs.to_string() != "EPSG:4326":
            nodes_with_population = nodes_with_population.to_crs(epsg=4326)
        print(f"已读取人口矢量数据: {os.path.basename(pop_path)}, {len(nodes_with_population)} 个点")
        population_data = None
        pop_transform = None
    elif pop_ext == ".shp" and os.path.exists(pop_path):
        pop_gdf = gpd.read_file(pop_path)
        pop_raster_crs = pop_gdf.crs
        pop_col = "Z" if "Z" in pop_gdf.columns else pop_gdf.columns[0]
        print(
            f"已读取人口网格数据: {os.path.basename(pop_path)}, "
            f"{len(pop_gdf)} 个网格, CRS: {pop_raster_crs}, 人口列: {pop_col}"
        )
        if pop_gdf.crs is not None and pop_gdf.crs.to_string() != "EPSG:4326":
            pop_gdf = pop_gdf.to_crs(epsg=4326)
        nodes_with_population = pop_gdf
        population_data = None
        pop_transform = None
    else:
        alt_geojson_path = os.path.join(data_dir, "net_population.geojson")
        alt_tif_path = os.path.join(data_dir, f"{prefix}_population.tif")
        if os.path.exists(alt_geojson_path):
            nodes_with_population = gpd.read_file(alt_geojson_path)
            print(f"已读取人口矢量数据: net_population.geojson, {len(nodes_with_population)} 个点")
            population_data = None
            pop_transform = None
            pop_raster_crs = None
        elif os.path.exists(alt_tif_path):
            with rasterio.open(alt_tif_path) as src:
                population_data = src.read(1)
                pop_transform = src.transform
                pop_raster_crs = src.crs
            print(f"已读取人口栅格数据: {os.path.basename(alt_tif_path)}")
            nodes_with_population = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
        else:
            print("警告: 未找到人口数据文件")
            population_data = None
            pop_transform = None
            pop_raster_crs = None
            nodes_with_population = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")

    boundary_path = None
    boundary_file = cfg.get("boundary_file")
    if boundary_file:
        if city_name == "上海":
            boundary_path = os.path.join(base_dir, "Network", "Shanghai", os.path.basename(boundary_file))
        elif city_name == "南京":
            boundary_path = os.path.join(base_dir, boundary_file)
        elif city_name == "武汉":
            boundary_path = os.path.join(base_dir, "Network", "Wuhan", os.path.basename(boundary_file))
        elif city_name == "北京":
            boundary_path = os.path.join(base_dir, "Network", "Beijing", os.path.basename(boundary_file))

    if not boundary_path or not os.path.exists(boundary_path):
        for bf in cfg.get("boundary_files", []):
            candidate = os.path.join(data_dir, bf)
            if os.path.exists(candidate):
                boundary_path = candidate
                break

    if boundary_path and os.path.exists(boundary_path):
        try:
            admin_boundary = gpd.read_file(boundary_path, encoding="utf-8")
        except Exception:
            try:
                admin_boundary = gpd.read_file(boundary_path, encoding="gbk")
            except Exception:
                admin_boundary = gpd.read_file(boundary_path)
        if admin_boundary.crs is None:
            admin_boundary = admin_boundary.set_crs("EPSG:4326", allow_override=True)
        elif admin_boundary.crs.to_string() != "EPSG:4326":
            admin_boundary = admin_boundary.to_crs(epsg=4326)
        print(f"已读取行政边界: {len(admin_boundary)} 个多边形 (来自 {os.path.basename(boundary_path)})")
    else:
        print(f"警告: 未找到行政边界文件 {boundary_path}")
        admin_boundary = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")

    result = (
        networks,
        poi_data,
        population_data,
        pop_transform,
        nodes_with_population,
        admin_boundary,
        pop_raster_crs,
    )
    if use_cache:
        with _CITY_DATA_LOCK:
            _CITY_DATA_CACHE[cache_key] = result
    print("数据读取完成\n")
    return result

def read_all_nanjing_data(use_cache: bool = True):
    """兼容旧调用：默认读取城市数据。"""
    return read_all_city_data(city="南京", use_cache=use_cache)

# 公交/地铁 G2SFCA 线网接驳与距离场（从 AccessibilityAnalysis 抽离）

AccessWalkFn = Callable[[float, float, float, float], float]

class TransitAccessibilityRouter:
    """
    公共交通可达性线网工具：站点接驳、多源 Dijkstra 距离场、OD 查表。
    access_walk_fn(lon1, lat1, lon2, lat2) -> 米（由计算器注入，区分公交直线/地铁路网）。
    """

    def __init__(self, access_walk_fn: AccessWalkFn):
        self._access_walk = access_walk_fn

    def snap_access_rows(self, lon, lat, tn, k_stops=3, snap_m=2500.0):
        rows = []
        for sid, slon, slat, _ in tn.nearest_stops(lon, lat, k=k_stops, max_dist_m=snap_m):
            if sid not in tn.stop_lonlat:
                continue
            w = self._access_walk(lon, lat, slon, slat)
            rows.append((sid, float(w)))
        return rows

    @staticmethod
    def sources_from_access(access_rows, cutoff_m):
        sources = {}
        for sid, w in access_rows:
            if w <= cutoff_m:
                sources[sid] = w
        return sources

    def dist_from_point_to_stops(self, lon, lat, tn, cutoff_m, k_stops=3, snap_m=2500.0):
        sources = self.sources_from_access(
            self.snap_access_rows(lon, lat, tn, k_stops=k_stops, snap_m=snap_m),
            cutoff_m,
        )
        if not sources:
            return {}
        return tn.multi_source_distances_m(sources, cutoff_m)

    def precompute_grid_access(
        self, coords, tn, k_stops=3, snap_m=2500.0, progress_every=0, progress_fn=None
    ):
        n = len(coords)
        out = []
        for i in range(n):
            if progress_fn and progress_every > 0 and i % progress_every == 0:
                try:
                    progress_fn(i, n)
                except Exception:
                    pass
            out.append(
                self.snap_access_rows(coords[i, 0], coords[i, 1], tn, k_stops=k_stops, snap_m=snap_m)
            )
        return out

    @staticmethod
    def access_sources_key(access_rows, cutoff_m):
        parts = [
            (sid, int(round(float(w))))
            for sid, w in access_rows
            if float(w) <= float(cutoff_m)
        ]
        return tuple(sorted(parts)) if parts else None

    @staticmethod
    def demand_geo_cluster_key(i, demand_coords, access_rows, geo_round=2):
        lon, lat = float(demand_coords[i, 0]), float(demand_coords[i, 1])
        scale = 10 ** int(geo_round)
        if access_rows:
            return (str(access_rows[0][0]), int(round(lon * scale)), int(round(lat * scale)))
        return ("__none__", int(round(lon * scale)), int(round(lat * scale)))

    def build_demand_fields_clustered(
        self, demand_coords, demand_access, tn, cutoff_m, geo_round=2
    ):
        n = len(demand_access)
        fields = [{} for _ in range(n)]
        src_cache = {}
        geo_buckets = defaultdict(list)

        for i, rows in enumerate(demand_access):
            sk = self.access_sources_key(rows, cutoff_m)
            if sk is not None and sk in src_cache:
                fields[i] = src_cache[sk]
                continue
            gk = self.demand_geo_cluster_key(i, demand_coords, rows, geo_round=geo_round)
            geo_buckets[gk].append((i, sk))

        n_exact = sum(1 for f in fields if f)
        n_geo = len(geo_buckets)
        print(
            f"  [线网] Step2 聚类: {n} 格 → 精确源复用 {n_exact} 格, "
            f"地理簇 {n_geo} 个（主站+{geo_round} 位坐标）"
        )

        for gi, (_gk, members) in enumerate(geo_buckets.items()):
            if gi % 200 == 0:
                print(f"    需求侧线网扩散(簇): {gi}/{n_geo}")
            pending = []
            for i, sk in members:
                if sk is not None and sk in src_cache:
                    fields[i] = src_cache[sk]
                else:
                    pending.append((i, sk))
            if not pending:
                continue

            primary_sid = _gk[0] if _gk[0] not in ("__none__",) else None
            if primary_sid:
                rep_i = min(
                    pending,
                    key=lambda t: next(
                        (float(w) for sid, w in demand_access[t[0]] if sid == primary_sid),
                        float("inf"),
                    ),
                )[0]
            else:
                rep_i = min(
                    pending,
                    key=lambda t: sum(float(w) for _, w in demand_access[t[0]]),
                )[0]

            sources = self.sources_from_access(demand_access[rep_i], cutoff_m)
            rep_sk = self.access_sources_key(demand_access[rep_i], cutoff_m)
            if not sources:
                cluster_field = {}
            else:
                cluster_field = tn.multi_source_distances_m(sources, float(cutoff_m))
            if rep_sk is not None:
                src_cache[rep_sk] = cluster_field
            for i, sk in pending:
                fields[i] = src_cache[sk] if sk is not None and sk in src_cache else cluster_field

        return fields

    @staticmethod
    def build_stop_to_facility_index(facility_access, rj=None):
        """站点 → [(设施索引, 接驳步行米)]，供 Step2 汇总可达性倒排。"""
        index: dict[str, list[tuple[int, float]]] = defaultdict(list)
        rj_arr = None if rj is None else np.asarray(rj, dtype=float)
        for j, rows in enumerate(facility_access):
            if rj_arr is not None and rj_arr[j] <= 0:
                continue
            for sid, w in rows:
                index[sid].append((j, float(w)))
        return index

    @staticmethod
    def summarize_accessibility_ai(
        demand_fields,
        stop_to_fac,
        rj,
        d0,
        beta=1.5,
        progress_every=2000,
    ):
        """
        Step2 汇总 A_i：沿需求侧距离场的可达站点倒排挂靠设施，避免每格 Haversine 查 POI。
        """
        n = len(demand_fields)
        ai = np.zeros(n, dtype=float)
        rj_arr = np.asarray(rj, dtype=float)
        d0 = float(d0)
        if d0 <= 0:
            return ai
        # 高斯衰减：G = [exp(-1/2·(d/d0)^2) - exp(-1/2)] / [1 - exp(-1/2)]
        e_half = np.exp(-0.5)
        denom_g = 1.0 - e_half
        if denom_g <= 0:
            return ai

        for i in range(n):
            if progress_every and i % progress_every == 0:
                print(f"    汇总可达性: {i}/{n}")
            field = demand_fields[i]
            if not field:
                continue
            acc = 0.0
            for sid, net_d in field.items():
                net_d = float(net_d)
                if net_d > d0:
                    continue
                for j, walk_f in stop_to_fac.get(sid, ()):
                    rj_v = float(rj_arr[j])
                    if rj_v <= 0:
                        continue
                    dist = net_d + walk_f
                    if dist > d0:
                        continue
                    numer = np.exp(-0.5 * (dist / d0) ** 2) - e_half
                    acc += rj_v * (numer / denom_g)
            ai[i] = acc
        return ai

    @staticmethod
    def build_stop_to_demand_index(demand_access):
        """站点 → [(需求格索引, 接驳步行米)]，供设施侧 Step1 倒排剪枝。"""
        index: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for di, rows in enumerate(demand_access):
            for sid, w in rows:
                index[sid].append((di, float(w)))
        return index

    @staticmethod
    def service_ratio_denom_from_facility_field(
        facility_field,
        stop_index,
        demand_pop,
        demand_pop_valid,
        d0,
        beta=1.5,
    ):
        """
        单一设施线网距离场下，用站点倒排累计 Step1 分母 Σ P_i·G(d_ij)。
        仅遍历线网可达站点及其挂靠需求格，避免每设施 Haversine 大球查询。
        """
        if not facility_field:
            return 0.0
        d0 = float(d0)
        if d0 <= 0:
            return 0.0

        best_dist: dict[int, float] = {}
        for sid, net_d in facility_field.items():
            net_d = float(net_d)
            if net_d > d0:
                continue
            for di, walk_m in stop_index.get(sid, ()):
                if not demand_pop_valid[di]:
                    continue
                total = net_d + walk_m
                if total > d0:
                    continue
                prev = best_dist.get(di)
                if prev is None or total < prev:
                    best_dist[di] = total

        if not best_dist:
            return 0.0

        # 高斯衰减：G = [exp(-1/2·(d/d0)^2) - exp(-1/2)] / [1 - exp(-1/2)]
        e_half = np.exp(-0.5)
        denom_g = 1.0 - e_half
        if denom_g <= 0:
            return 0.0

        pop = np.asarray(demand_pop, dtype=float)
        total = 0.0
        for di, dist in best_dist.items():
            numer = np.exp(-0.5 * (dist / d0) ** 2) - e_half
            g = numer / denom_g
            total += pop[di] * g
        return float(total)

    def facility_fields_by_access(
        self, facility_access, tn, cutoff_m, progress_every=500
    ):
        """按接驳源去重：相同 boarding 源只跑一次设施侧 Dijkstra。返回 list[dict]。"""
        n = len(facility_access)
        fields: list[dict] = [{} for _ in range(n)]
        cache: dict[tuple, dict] = {}
        n_unique = 0
        for j in range(n):
            if progress_every and j % progress_every == 0:
                print(f"    设施线网扩散: {j}/{n}")
            sk = self.access_sources_key(facility_access[j], cutoff_m)
            if sk is not None and sk in cache:
                fields[j] = cache[sk]
                continue
            sources = self.sources_from_access(facility_access[j], cutoff_m)
            if not sources:
                field = {}
            else:
                field = tn.multi_source_distances_m(sources, float(cutoff_m))
                n_unique += 1
            if sk is not None:
                cache[sk] = field
            fields[j] = field
        print(f"  [线网] 设施侧 Dijkstra: {n} 个 POI → {n_unique} 次（接驳源去重）")
        return fields

    @staticmethod
    def od_from_facility_field(facility_field, demand_access_rows, cutoff_m):
        best = float("inf")
        for sid, walk_to_demand in demand_access_rows:
            base = facility_field.get(sid)
            if base is None:
                continue
            total = float(base) + float(walk_to_demand)
            if total < best:
                best = total
        if not np.isfinite(best) or best > cutoff_m:
            return float("inf")
        return best

    @staticmethod
    def od_from_demand_field(demand_field, facility_access_rows, cutoff_m):
        best = float("inf")
        for sid, walk_to_facility in facility_access_rows:
            base = demand_field.get(sid)
            if base is None:
                continue
            total = float(base) + float(walk_to_facility)
            if total < best:
                best = total
        if not np.isfinite(best) or best > cutoff_m:
            return float("inf")
        return best

    def od_distance_m(self, stop_dist_from_origin, fx, fy, tn, cutoff_m, k_stops=3, snap_m=2500.0):
        fac_access = self.snap_access_rows(fx, fy, tn, k_stops=k_stops, snap_m=snap_m)
        return self.od_from_demand_field(stop_dist_from_origin, fac_access, cutoff_m)
