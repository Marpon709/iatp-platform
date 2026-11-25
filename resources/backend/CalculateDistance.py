import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.neighbors import BallTree
import warnings
import os
import time
import threading
from numba import njit
from rasterio.transform import from_origin

warnings.filterwarnings('ignore')

_DEGREE_TO_METER = 111000.0
# 与 Haversine 实现一致；sklearn BallTree(metric='haversine') 的距离为弧度，× 本常数 → 米
_EARTH_RADIUS_M = 6371000.0

def _latlon_rad_for_haversine_tree(lon_arr, lat_arr):
    """sklearn BallTree(metric='haversine') 要求 shape (n,2)，列为 (lat_rad, lon_rad)。"""
    lon_arr = np.asarray(lon_arr, dtype=np.float64).reshape(-1)
    lat_arr = np.asarray(lat_arr, dtype=np.float64).reshape(-1)
    return np.column_stack([np.radians(lat_arr), np.radians(lon_arr)])

def _poi_haversine_balltree(lon_arr, lat_arr, leaf_size=40):
    return BallTree(_latlon_rad_for_haversine_tree(lon_arr, lat_arr), metric='haversine', leaf_size=leaf_size)

def _haversine_tree_radius_rad(radius_m):
    """直线距离阈值（米）→ Haversine BallTree 的 query_radius 弧度半径。"""
    return float(radius_m) / _EARTH_RADIUS_M

def _coords_near(a, b, eps=1e-7):
    return abs(float(a[0]) - float(b[0])) < eps and abs(float(a[1]) - float(b[1])) < eps

def _graph_node_latlng(nodes_gdf, node_id):
    geom = nodes_gdf.loc[node_id].geometry
    return [float(geom.y), float(geom.x)]

def _pick_min_length_edge_key(G, u, v):
    if G.has_edge(u, v):
        edges = G[u][v]
    elif G.has_edge(v, u):
        edges = G[v][u]
        u, v = v, u
    else:
        return None
    if not getattr(G, "is_multigraph", lambda: False)():
        return None
    best_k = None
    best_len = float("inf")
    for k, data in edges.items():
        try:
            length = float(data.get("length", float("inf")))
        except (TypeError, ValueError):
            length = float("inf")
        if length < best_len:
            best_len = length
            best_k = k
    return best_k

def _edge_geometry_latlng(edge_data, nodes_gdf, u, v):
    """从边属性解析折线坐标 [[lat, lng], ...]；无几何时返回 None。"""
    if not isinstance(edge_data, dict):
        return None
    geom = edge_data.get("geometry")
    if geom is None:
        return None
    try:
        from shapely import wkt
        from shapely.geometry import LineString

        if isinstance(geom, str):
            parsed = wkt.loads(geom)
        elif isinstance(geom, LineString):
            parsed = geom
        else:
            parsed = None
        if parsed is not None and not parsed.is_empty:
            if parsed.geom_type == "LineString":
                return [[float(y), float(x)] for x, y in parsed.coords]
            if parsed.geom_type == "MultiLineString":
                out = []
                for part in parsed.geoms:
                    out.extend([[float(y), float(x)] for x, y in part.coords])
                return out or None
    except Exception:
        pass
    if isinstance(geom, (list, tuple)) and geom:
        first = geom[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            out = []
            for pt in geom:
                if len(pt) < 2:
                    continue
                x, y = float(pt[0]), float(pt[1])
                if abs(x) <= 180 and abs(y) <= 90:
                    out.append([y, x])
                else:
                    out.append([x, y])
            return out or None
    return None

def route_nodes_to_path_coords(G, route, nodes_gdf):
    """将最短路节点序列转为沿有向边几何的 [[lat, lng], ...]。"""
    if not route:
        return []
    if len(route) == 1:
        return [_graph_node_latlng(nodes_gdf, route[0])]

    path = []
    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        edge_key = _pick_min_length_edge_key(G, u, v)
        if edge_key is None and not G.has_edge(u, v) and not G.has_edge(v, u):
            seg = [_graph_node_latlng(nodes_gdf, u), _graph_node_latlng(nodes_gdf, v)]
        else:
            if G.has_edge(u, v):
                eu, ev = u, v
            else:
                eu, ev = v, u
            if edge_key is not None:
                edge_data = G[eu][ev][edge_key]
            else:
                edge_data = G[eu][ev]
            seg = _edge_geometry_latlng(edge_data, nodes_gdf, eu, ev)
            if seg is None:
                seg = [_graph_node_latlng(nodes_gdf, u), _graph_node_latlng(nodes_gdf, v)]
            elif eu == v and ev == u:
                seg = list(reversed(seg))

        if not path:
            path.extend(seg)
            continue
        if _coords_near(path[-1], seg[0]):
            path.extend(seg[1:])
        elif _coords_near(path[-1], seg[-1]):
            path.extend(reversed(seg[:-1]))
        else:
            # 边几何与上一段未衔接时，按节点序列推进，避免乱序拼接成折返环
            v_pt = _graph_node_latlng(nodes_gdf, v)
            if not _coords_near(path[-1], v_pt):
                path.append(v_pt)

    return path

def attach_od_endpoints_to_path(path, origin, destination):
    if not path:
        return [
            [float(origin.y), float(origin.x)],
            [float(destination.y), float(destination.x)],
        ]
    out = [[float(c[0]), float(c[1])] for c in path]
    o_pt = [float(origin.y), float(origin.x)]
    d_pt = [float(destination.y), float(destination.x)]
    if not _coords_near(out[0], o_pt):
        out.insert(0, o_pt)
    if not _coords_near(out[-1], d_pt):
        out.append(d_pt)
    return out

def merge_path_segments(*segments):
    path = []
    for part in segments:
        if not part:
            continue
        seg = [[float(c[0]), float(c[1])] for c in part if c and len(c) >= 2]
        if not seg:
            continue
        if not path:
            path.extend(seg)
            continue
        if _coords_near(path[-1], seg[0]):
            path.extend(seg[1:])
        else:
            path.extend(seg)
    return path

# 个人碳预算（多活动可达性 + RCB 反比分配）常量
# 最低碳预算标准：分钟生活圈时间内四类 POI 各可达 1 个（不含娱乐/餐饮）
CARBON_MOBILITY_STANDARD_ACTIVITIES = (
    'work', 'shopping', 'education', 'medical',
)
DEFAULT_MIN_MOBILITY_POI_COUNTS = {k: 1 for k in CARBON_MOBILITY_STANDARD_ACTIVITIES}
# 最低碳预算：每类活动直线最近 POI，分别计算出发地→POI 驾车路网 OD 距离并求和判时
MIN_CB_ACTIVITY_CACHE_LETTERS = {
    'work': 'W',
    'shopping': 'S',
    'education': 'E',
    'medical': 'M',
}
MIN_CB_Dijk_CAP_M = 0.0  # 0=仅按出行时间×驾车速度截断；>0 时与预算距离取较大值（上限）
MIN_CB_CACHE_VER = '4cat_ud'  # 缓存版本：四类 POI（无娱乐/餐饮）+ 无向驾车 Dijkstra
# 多活动可达性公式中的 w_k（总和为 1.0）
CARBON_ACTIVITY_WEIGHTS = {
    'medical': 0.18,
    'work': 0.16,
    'education': 0.15,
    'transportation_station': 0.14,
    'shopping': 0.14,
    'catering': 0.13,
    'recreation': 0.10,
}
# 多活动可达性：先按交通方式加权（再按活动类型加权），权重和为 1
CARBON_BUDGET_MODE_WEIGHTS = {
    'walk': 0.25,
    'bus': 0.25,
    'metro': 0.25,
    'drive': 0.25,
}
# G2SFCA 分区可达性缓存标签默认 d0（米）
CARBON_ACCESSIBILITY_RADIUS_M = 10000.0
# 参与机动碳池分配的年排放 RCB_total（吨/年）= 前端「区域碳排放总目标」（吨/年）× 本系数
RCB_TOTAL_ALLOCATION_FACTOR = 0.1
# 与前端「全部区域」默认 3000 万吨/年 一致（CarbonBudget.vue: rcb×10000 → 吨/年）
CARBON_DEFAULT_RCB_TONS_YEAR = 30_000_000.0

def _snap_lonlat_to_network_nodes_hav(graph_index, lon_arr, lat_arr):
    """需求/POI 坐标 → 最近路网节点（Haversine）。"""
    node_ids = graph_index['node_ids']
    node_lon = graph_index['node_lon']
    node_lat = graph_index['node_lat']
    node_tree_hav = graph_index['node_tree_hav']
    lon_arr = np.asarray(lon_arr, dtype=np.float64).reshape(-1)
    lat_arr = np.asarray(lat_arr, dtype=np.float64).reshape(-1)
    q = np.column_stack([np.radians(lat_arr), np.radians(lon_arr)])
    _, nn_idx = node_tree_hav.query(q, k=1)
    nn_idx = nn_idx.ravel()
    nearest_ids = node_ids[nn_idx]
    snap_dists = _haversine_m_vec(lon_arr, lat_arr, node_lon[nn_idx], node_lat[nn_idx])
    return nearest_ids, snap_dists

def _haversine_m_vec(lon1, lat1, lon2, lat2):
    """批量 Haversine（米），lon1/lat1/lon2/lat2 为可广播的 ndarray。"""
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi * 0.5) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam * 0.5) ** 2
    a = np.clip(a, 0.0, 1.0)
    return R * 2.0 * np.arcsin(np.sqrt(a))

class CarbonBudgetCalculator:
    def __init__(self, networks, poi_data, admin_boundary, selected_district=None, city=None):

        self.networks = networks
        self.poi_data = poi_data
        self.admin_boundary = admin_boundary
        self.selected_district = selected_district
        self.city = city or '南京'

        if selected_district and selected_district != '全部区域':
            self.district_boundary = self._filter_district(selected_district)
        else:
            self.district_boundary = admin_boundary

        self.activity_types = ['work', 'medical', 'shopping', 'catering', 'education', 'recreation', 'transportation_station']

        # speed: km/h；碳预算出行时间 t=D/V（速度与 transport_modes 统一）
        from transport_modes import carbon_calculator_mode_config
        self.mode_config = carbon_calculator_mode_config()

        self.transit_networks = {}
        if str(self.city or '南京').strip() == '南京':
            try:
                from Get_Network import load_nanjing_transit_networks
                self.transit_networks = load_nanjing_transit_networks()
            except Exception as ex:
                print(f"[Transit] 南京公交/地铁线网加载失败，将回退道路网: {ex}")

        self.network_trees = {}
        self.network_nodes = {}
        self.poi_trees = {}
        # 方案 H：按 network_key 缓存路网节点 BallTree，避免每种活动×方式重复构建
        self._graph_spatial_index = {}
        self._network_csr = {}
        self._min_cb_dijkstra_array_cache = {}
        self._min_cb_cache_lock = threading.Lock()
        # 热力图可达性 d0（米）：由出行时间×交通方式速度换算，或兼容旧版 search_radius_m
        self._carbon_acc_search_radius_m = None
        # 最近一次碳预算热力图中，多活动可达性是否使用路网距离 d_ij（否则为无路网时的直线回退）
        self._carbon_acc_uses_road_distance = False

        print(f"碳预算计算器初始化完成 (区域: {selected_district or '全部区域'})")

    def _filter_district(self, district_name):
        from shapely.geometry import Point as ShapelyPoint
        from shapely.ops import unary_union
        import shapely

        possible_fields = ['县', 'name', 'NAME', 'Name', '名称', 'district', 'DISTRICT', 'District',
                          'QHMC', 'qhmc', 'xzqmc', 'XZQMC', '区县', '区', '市县']

        for field in possible_fields:
            if field in self.admin_boundary.columns:
                filtered = self.admin_boundary[self.admin_boundary[field] == district_name]
                if not filtered.empty:
                    print(f"成功过滤区域 '{district_name}' (使用字段: {field}, 精确匹配)")
                    return filtered
                try:
                    filtered = self.admin_boundary[self.admin_boundary[field].astype(str).str.contains(district_name, na=False)]
                    if not filtered.empty:
                        print(f"成功过滤区域 '{district_name}' (使用字段: {field}, 包含匹配)")
                        return filtered
                except Exception as e:
                    print(f"字段 {field} 包含匹配出错: {e}")

        print(f"警告：未找到区域 '{district_name}'，使用全部区域")
        print(f"可用字段: {list(self.admin_boundary.columns)}")
        return self.admin_boundary

    def precompute_indices(self):
        print("\n" + "="*60)
        print("开始预计算索引...")
        print(f"POI数据键: {list(self.poi_data.keys())}")
        print(f"需要的活动类型: {self.activity_types}")
        print("="*60)

        import osmnx as ox  # 第三方开源库，用于路网节点坐标提取与索引构建
        print("\n[路网索引]")
        for mode, config in self.mode_config.items():
            if mode in self.transit_networks:
                tn = self.transit_networks[mode]
                print(f"  [OK] {mode}: 公交/地铁线网 {len(tn.stop_lonlat)} 个站点")
                continue
            network_key = config['network']
            if network_key in self.networks and self.networks[network_key] is not None:
                G = self.networks[network_key]
                nodes = ox.graph_to_gdfs(G, edges=False)
                coords = np.array([[n.x, n.y] for n in nodes.geometry])
                tree = BallTree(coords, leaf_size=40)
                self.network_trees[mode] = (tree, coords, nodes)
                self.network_nodes[mode] = nodes
                print(f"  [OK] {mode}: {len(nodes)} 个节点")
            else:
                print(f"  [FAIL] {mode}: 路网数据不存在")

        # 公交/地铁接驳步行使用道路网索引
        walk_ref = self.network_trees.get('walk') or self.network_trees.get('car')
        if walk_ref is not None:
            for tm in ('bus', 'metro'):
                if tm in self.transit_networks and tm not in self.network_trees:
                    self.network_trees[tm] = walk_ref

        print("\n[POI索引]")
        poi_count = 0
        for activity, poi_gdf in self.poi_data.items():
            is_empty = poi_gdf.empty if hasattr(poi_gdf, 'empty') else True
            poi_len = 0 if is_empty else len(poi_gdf)
            print(f"  处理 {activity}: 类型={type(poi_gdf)}, 是否为空={is_empty}, POI数量={poi_len}")
            if not is_empty and poi_len > 0:
                try:
                    coords = np.array([[p.x, p.y] for p in poi_gdf.geometry])
                    tree = _poi_haversine_balltree(coords[:, 0], coords[:, 1])
                    self.poi_trees[activity] = (tree, coords, poi_gdf)
                    poi_count += poi_len
                    print(f"    [OK] 成功建立索引: {poi_len} 个POI")
                except Exception as e:
                    print(f"    [FAIL] 建立索引失败: {e}")
            else:
                print(f"    [FAIL] 数据为空，跳过")

        print(f"\n[总结] POI索引建立完成，已建立索引的POI类型: {list(self.poi_trees.keys())}，总POI数量: {poi_count}")
        missing = set(self.activity_types) - set(self.poi_trees.keys())
        if missing:
            print(f"  [WARNING] 以下活动类型没有POI数据: {missing}")
        else:
            print(f"  [OK] 所有需要的活动类型都有POI数据")
        print("="*60 + "\n")
        if len(self.poi_trees) == 0:
            raise ValueError("无法建立任何POI索引！请检查POI数据。")

    @staticmethod
    @njit
    def _haversine_distance(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = np.radians(lon1), np.radians(lat1), np.radians(lon2), np.radians(lat2)
        dlon, dlat = lon2 - lon1, lat2 - lat1
        a = np.sin(dlat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon * 0.5) ** 2
        return 6371000.0 * 2.0 * np.arcsin(np.sqrt(a))

    def _road_walk_distance_m(self, origin, destination):
        """道路网步行接驳距离（米）。"""
        _, dist = self._calculate_travel_time_and_distance(origin, destination, 'walk')
        if dist is None or not np.isfinite(dist):
            return self._haversine_distance(origin.x, origin.y, destination.x, destination.y)
        return float(dist)

    def _straight_walk_distance_m(self, origin, destination):
        """公交接驳：步行速度下按直线几何距离（米）。"""
        return float(self._haversine_distance(origin.x, origin.y, destination.x, destination.y))

    def _metro_snap_threshold_m(self, travel_minutes=None, metro_search_radius_m=None):
        """地铁接驳容许距离（米）= 出行时间 × 步行 5km/h。"""
        from transport_modes import speed_ms

        if travel_minutes is not None:
            minutes = float(travel_minutes)
        elif metro_search_radius_m is not None:
            minutes = float(metro_search_radius_m) / (60.0 * speed_ms('metro'))
        else:
            minutes = 20.0
        return minutes * 60.0 * speed_ms('walk')

    def _origin_beyond_metro_snap(self, origin, tn, snap_m=None):
        from shapely.geometry import Point as ShapelyPoint

        stops = tn.nearest_stops(origin.x, origin.y, k=1, max_dist_m=50000.0)
        if not stops:
            return True
        _, slon, slat, _ = stops[0]
        access = self._road_walk_distance_m(origin, ShapelyPoint(slon, slat))
        thresh = float(snap_m if snap_m is not None else self._metro_snap_threshold_m())
        return access > thresh

    def _access_walk_distance_m(self, origin, destination, transit_mode):
        """接驳步行：路网与直线取更合理者（与地图绘制一致）。"""
        _ = transit_mode
        try:
            from transport_modes import transit_access_walk_m
            return float(transit_access_walk_m(self, origin, destination))
        except Exception:
            return self._road_walk_distance_m(origin, destination)

    def _calculate_transit_distance_m(self, origin, destination, mode, metro_snap_m=None):
        """
        公交/地铁：接驳步行（道路网）+ 线网最短路 + 接驳步行（道路网）。
        地铁：若起点到最近站超过 snap（默认 出行时间×步行速度），整段 OD 改道路网步行。
        """
        tn = self.transit_networks.get(mode)
        if tn is None:
            return None
        try:
            from transport_modes import compute_transit_od_distance_m

            if mode == 'metro' and self._origin_beyond_metro_snap(origin, tn, snap_m=metro_snap_m):
                _, dist = self._calculate_travel_time_and_distance(origin, destination, 'walk')
                if dist is not None and np.isfinite(dist) and dist > 0:
                    return float(dist)
                return float(self._haversine_distance(origin.x, origin.y, destination.x, destination.y))
            dist = compute_transit_od_distance_m(self, origin, destination, mode)
            if dist is not None and np.isfinite(dist) and dist > 0:
                return float(dist)
        except Exception:
            pass
        return None

    def _calculate_travel_time_and_distance(self, origin, destination, mode, origin_node_info=None):
        inf = float('inf')
        if mode not in self.mode_config:
            return (inf, inf)

        if mode in self.transit_networks:
            dist_m = self._calculate_transit_distance_m(origin, destination, mode)
            if dist_m is not None and np.isfinite(dist_m) and dist_m > 0:
                speed_ms = self.mode_config[mode]['speed'] * 1000.0 / 3600.0
                return (dist_m / speed_ms / 60.0, dist_m)

        network_key = self.mode_config[mode]['network']
        if network_key not in self.networks or self.networks[network_key] is None:
            distance = self._haversine_distance(origin.x, origin.y, destination.x, destination.y)
            speed_ms = self.mode_config[mode]['speed'] * 1000.0 / 3600.0
            return (distance / speed_ms / 60.0, distance)
        if mode not in self.network_trees:
            distance = self._haversine_distance(origin.x, origin.y, destination.x, destination.y)
            speed_ms = self.mode_config[mode]['speed'] * 1000.0 / 3600.0
            return (distance / speed_ms / 60.0, distance)
        try:
            import networkx as nx
            tree, coords, nodes = self.network_trees[mode]
            G = self.networks[network_key]

            # 多候选吸附：避免单点吸附到弱连通分量导致绕路
            snap_k = 5 if mode in ('walk', 'bike') else 8
            snap_k = min(snap_k, len(nodes))

            dest_coord = np.array([[destination.x, destination.y]])
            _, dest_indices = tree.query(dest_coord, k=snap_k)

            if origin_node_info is not None:
                origin_node_id, origin_node_geom = origin_node_info
                origin_dist_m = self._haversine_distance(origin.x, origin.y, origin_node_geom.x, origin_node_geom.y)
                origin_candidates = [(origin_node_id, origin_dist_m)]
            else:
                origin_coord = np.array([[origin.x, origin.y]])
                _, origin_indices = tree.query(origin_coord, k=snap_k)
                origin_candidates = []
                seen_o = set()
                for oi in origin_indices[0]:
                    nid = nodes.index[int(oi)]
                    if nid in seen_o:
                        continue
                    seen_o.add(nid)
                    ng = nodes.iloc[int(oi)].geometry
                    d = self._haversine_distance(origin.x, origin.y, ng.x, ng.y)
                    origin_candidates.append((nid, d))

            # 搜索总路径（接驳+路网+接驳）最短的节点对
            best_distance = inf
            dest_candidates = []
            seen_d = set()
            for di in dest_indices[0]:
                nid = nodes.index[int(di)]
                if nid in seen_d:
                    continue
                seen_d.add(nid)
                ng = nodes.iloc[int(di)].geometry
                d = self._haversine_distance(destination.x, destination.y, ng.x, ng.y)
                dest_candidates.append((nid, d))

            for o_nid, o_snap in origin_candidates:
                for d_nid, d_snap in dest_candidates:
                    if o_nid == d_nid:
                        # 同节点：直接用直线距离
                        candidate = self._haversine_distance(origin.x, origin.y, destination.x, destination.y)
                    else:
                        try:
                            network_dist = nx.shortest_path_length(G, o_nid, d_nid, weight='length')
                            candidate = o_snap + network_dist + d_snap
                        except Exception:
                            continue
                    if candidate < best_distance:
                        best_distance = candidate

            if not np.isfinite(best_distance):
                distance = self._haversine_distance(origin.x, origin.y, destination.x, destination.y)
            else:
                distance = best_distance

            speed_ms = self.mode_config[mode]['speed'] * 1000.0 / 3600.0
            return (distance / speed_ms / 60.0, distance)
        except Exception:
            distance = self._haversine_distance(origin.x, origin.y, destination.x, destination.y)
            speed_ms = self.mode_config[mode]['speed'] * 1000.0 / 3600.0
            return (distance / speed_ms / 60.0, distance)

    def _calculate_travel_time(self, origin, destination, mode, origin_node_info=None):
        time_min, _ = self._calculate_travel_time_and_distance(origin, destination, mode, origin_node_info)
        return time_min

    def _get_road_path_geometry(self, origin, destination, mode="walk", for_transit_access=False):
        """沿道路路网几何的 OD 路径 [[lat, lng], ...]（步行网无向回退，见 transport_modes）。"""
        from transport_modes import fetch_road_od_path, fetch_transit_access_coords

        if for_transit_access:
            path = fetch_transit_access_coords(self, origin, destination)
        else:
            path = fetch_road_od_path(self, origin, destination, mode)
        if path and len(path) >= 2:
            return path
        return [[origin.y, origin.x], [destination.y, destination.x]]

    def _get_route_path_geometry(self, origin, destination, mode):
        if mode in self.transit_networks:
            try:
                from shapely.geometry import Point as ShapelyPoint
                tn = self.transit_networks[mode]
                if mode == 'metro' and self._origin_beyond_metro_snap(origin, tn):
                    return [[origin.y, origin.x], [destination.y, destination.x]]
                orig_stops = tn.nearest_stops(origin.x, origin.y, k=3, max_dist_m=2500.0)
                dest_stops = tn.nearest_stops(destination.x, destination.y, k=3, max_dist_m=2500.0)
                best = None
                best_path = None
                for s_o, lon_o, lat_o, _ in orig_stops:
                    p_stop_o = ShapelyPoint(lon_o, lat_o)
                    for s_d, lon_d, lat_d, _ in dest_stops:
                        p_stop_d = ShapelyPoint(lon_d, lat_d)
                        ride = tn.stop_pair_distance_m(s_o, s_d)
                        walk_o = self._access_walk_distance_m(origin, p_stop_o, mode)
                        walk_d = self._access_walk_distance_m(p_stop_d, destination, mode)
                        total = walk_o + ride + walk_d
                        if best is None or total < best:
                            best = total
                            seg_walk_o = self._get_road_path_geometry(
                                origin, p_stop_o, "walk", for_transit_access=True
                            )
                            seg_walk_d = self._get_road_path_geometry(
                                p_stop_d, destination, "walk", for_transit_access=True
                            )
                            seg_ride = [
                                [lat, lon] for lon, lat in tn.stop_pair_path_coords(s_o, s_d)
                            ]
                            best_path = (seg_walk_o, seg_ride, seg_walk_d)
                if best_path:
                    path = merge_path_segments(*best_path)
                    if len(path) >= 2:
                        return path
            except Exception:
                pass
            # 地铁/公交未连通时不退回道路网，避免路径画成车行路线
            if mode in getattr(self, 'transit_networks', {}):
                return [[origin.y, origin.x], [destination.y, destination.x]]

        return self._get_road_path_geometry(origin, destination, mode)

    def _acc_radius_cache_tag(self):
        r = getattr(self, '_carbon_acc_search_radius_m', None)
        if r is None:
            r = CARBON_ACCESSIBILITY_RADIUS_M
        return int(round(float(r)))

    def _get_acc_cache_dir(self, resolution_m, district_override=None):
        base = os.path.dirname(os.path.abspath(__file__))
        if district_override is not None:
            district_suffix = district_override
        else:
            district_suffix = (
                self.selected_district
                if (self.selected_district and self.selected_district != '全部区域')
                else 'all'
            )
        cache_dir = os.path.join(base, 'POI', 'Nanjing', 'CarbonBudget', f'resol_{resolution_m}', district_suffix)
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def _get_district_g2sfca_cache_path(self, resolution_m):
        tag = self._acc_radius_cache_tag()
        tt = int(getattr(self, '_carbon_acc_travel_time_min', 30) or 30)
        mode = str(getattr(self, '_carbon_acc_transport_mode', 'bike') or 'bike')
        fname = f'g2sfca_total_acc_r{tag}_t{tt}_{mode}.npz'
        return os.path.join(self._get_acc_cache_dir(resolution_m), fname)

    def _load_district_g2sfca_cache(self, demand_coords, resolution_m):
        """读取分区 G2SFCA 综合可达性缓存（仅当 demand_coords 与缓存一致时命中）。"""
        path = self._get_district_g2sfca_cache_path(resolution_m)
        if not os.path.exists(path):
            return None
        try:
            data = np.load(path, allow_pickle=True)
            cached_coords = data['coords']
            if cached_coords.shape != demand_coords.shape:
                return None
            if not np.allclose(cached_coords, demand_coords, rtol=1e-5, atol=1e-8):
                return None
            total_acc = data['total_acc'].astype(np.float64)
            if len(total_acc) != len(demand_coords):
                return None
            print(f"  [分区缓存命中] {self.selected_district} → {os.path.basename(path)}")
            return {'total_acc': total_acc, 'coords': cached_coords}
        except Exception as e:
            print(f"  分区 G2SFCA 缓存加载失败: {e}")
            return None

    def _save_district_g2sfca_cache(self, demand_coords, total_acc, resolution_m):
        """保存分区 G2SFCA 综合可达性缓存。"""
        path = self._get_district_g2sfca_cache_path(resolution_m)
        try:
            np.savez_compressed(
                path,
                coords=demand_coords.astype(np.float64),
                total_acc=np.asarray(total_acc, dtype=np.float32),
                district=str(self.selected_district or ''),
                resolution_m=int(resolution_m),
                travel_time_min=int(getattr(self, '_carbon_acc_travel_time_min', 30) or 30),
                transport_mode=str(getattr(self, '_carbon_acc_transport_mode', 'bike') or 'bike'),
            )
            print(f"  [分区缓存已保存] {self.selected_district} → {os.path.basename(path)}")
        except Exception as e:
            print(f"  分区 G2SFCA 缓存保存失败: {e}")

    def _min_cb_counts_cache_tag(self, min_poi_counts):
        """缓存文件名片段，如 W1_S1_E1_M1（字母=活动类型，数字=界面 POI 数量）。"""
        min_counts = self._normalize_min_mobility_poi_counts(min_poi_counts)
        parts = []
        for key in CARBON_MOBILITY_STANDARD_ACTIVITIES:
            letter = MIN_CB_ACTIVITY_CACHE_LETTERS.get(key, key[:1].upper())
            n = int(min_counts.get(key, 1) or 1)
            parts.append(f'{letter}{n}')
        return '_'.join(parts)

    def _min_cb_cache_path(self, resolution_m, min_poi_counts, mobility_tt):
        counts_tag = self._min_cb_counts_cache_tag(min_poi_counts)
        fname = f'Min_{counts_tag}_t{int(mobility_tt)}_{MIN_CB_CACHE_VER}.npz'
        return os.path.join(self._get_acc_cache_dir(resolution_m), fname)

    def _load_min_cb_cache(self, demand_coords, resolution_m, min_poi_counts, mobility_tt):
        path = self._min_cb_cache_path(resolution_m, min_poi_counts, mobility_tt)
        if not os.path.exists(path):
            return None
        try:
            data = np.load(path, allow_pickle=True)
            cached_coords = data['coords']
            if cached_coords.shape != demand_coords.shape:
                return None
            if not np.allclose(cached_coords, demand_coords, rtol=1e-5, atol=1e-8):
                return None
            values = data['values'].astype(np.float64)
            if len(values) != len(demand_coords):
                return None
            print(f'  [最低碳预算-缓存命中] {os.path.basename(path)}')
            return values
        except Exception as ex:
            print(f'  [最低碳预算-缓存加载失败] {ex}')
            return None

    def _save_min_cb_cache(self, demand_coords, values, resolution_m, min_poi_counts, mobility_tt):
        path = self._min_cb_cache_path(resolution_m, min_poi_counts, mobility_tt)
        try:
            np.savez_compressed(
                path,
                values=values.astype(np.float64),
                coords=demand_coords.astype(np.float64),
            )
            print(f'  [最低碳预算-缓存已保存] {os.path.basename(path)}')
        except Exception as ex:
            print(f'  [最低碳预算-缓存保存失败] {ex}')

    @staticmethod
    def _min_cb_skip_transit():
        # 默认计算公交/地铁线网碳排；设 MIN_CB_SKIP_TRANSIT=1 可跳过（加速但结果不完整）
        return os.environ.get('MIN_CB_SKIP_TRANSIT', '0').strip().lower() in ('1', 'true', 'yes')

    @staticmethod
    def _build_demand_grid(boundary, resolution_degree, grid_bbox=None):
        """在指定边界内按分辨率生成需求格网。

        grid_bbox: 格网 x/y 序列的起算范围 (minx, miny, maxx, maxy)。
                   分区与全市须共用同一 bbox，格点才能对齐并从全市缓存提取子集。
        """
        bbox = grid_bbox if grid_bbox is not None else boundary.total_bounds
        x_coords = np.arange(bbox[0], bbox[2], resolution_degree)
        y_coords = np.arange(bbox[1], bbox[3], resolution_degree)
        rows, cols = len(y_coords), len(x_coords)
        demand_cells = []
        for j, y in enumerate(y_coords):
            for i, x in enumerate(x_coords):
                pt = Point(x, y)
                if boundary.contains(pt).any():
                    demand_cells.append((i, j, x, y))
        demand_coords = np.array(
            [[x, y] for _, _, x, y in demand_cells], dtype=np.float64
        ) if demand_cells else np.empty((0, 2), dtype=np.float64)
        return demand_cells, demand_coords, x_coords, y_coords, bbox, rows, cols

    @staticmethod
    def _map_subset_indices_from_full(
        subset_coords, full_coords, resolution_degree=None
    ):
        full_map = {}
        for idx, (x, y) in enumerate(full_coords):
            key = (round(float(x), 8), round(float(y), 8))
            full_map[key] = idx
        tol = (
            max(float(resolution_degree) * 0.51, 1e-9)
            if resolution_degree is not None
            else 1e-8
        )
        indices = []
        for x, y in subset_coords:
            key = (round(float(x), 8), round(float(y), 8))
            if key in full_map:
                indices.append(full_map[key])
                continue
            # 浮点累积误差时，在半格距内吸附到最近全市格点
            dx = full_coords[:, 0] - float(x)
            dy = full_coords[:, 1] - float(y)
            dists = np.hypot(dx, dy)
            j = int(np.argmin(dists))
            if dists[j] <= tol:
                indices.append(j)
                continue
            raise ValueError(
                f"分区格点 ({float(x):.6f}, {float(y):.6f}) 不在全市网格中，无法从全市缓存提取"
            )
        return np.asarray(indices, dtype=np.int64)

    def _compute_g2sfca_total_accessibility(
        self,
        g2sfca_calc,
        demand_coords,
        demand_pop,
        demand_pop_valid,
        resolution_degree,
        travel_time_min,
        mode_weights_norm,
        valid_activities,
        progress_callback=None,
        progress_start=8,
        progress_end=46,
    ):
        """多活动 G2SFCA 加权求和，返回 total_acc 向量。"""
        n_cells = len(demand_coords)
        total_acc = np.zeros(n_cells, dtype=np.float64)
        total_stages = max(len(valid_activities), 1)
        span = max(progress_end - progress_start, 1)

        for stage, activity in enumerate(valid_activities, start=1):
            pct = progress_start + int(span * stage / total_stages)
            if callable(progress_callback):
                try:
                    progress_callback(
                        pct,
                        f"可达性(2SFCA) {activity} ({stage}/{total_stages})",
                    )
                except Exception:
                    pass

            a_ik = self._compute_activity_g2sfca_accessibility(
                g2sfca_calc,
                activity,
                demand_coords,
                demand_pop,
                demand_pop_valid,
                resolution_degree,
                travel_time_min=travel_time_min,
                mode_weights=mode_weights_norm,
            )
            w_k = float(CARBON_ACTIVITY_WEIGHTS.get(activity, 1.0))
            total_acc += w_k * a_ik

        return total_acc

    def _ensure_graph_spatial_index(self, network_key):
        """方案 H：对同一路网只构建一次节点坐标与 BallTree。"""
        if network_key not in self.networks or self.networks[network_key] is None:
            raise ValueError(f"路网不存在: {network_key}")
        if network_key not in self._graph_spatial_index:
            G = self.networks[network_key]
            node_ids = np.array(list(G.nodes()))
            node_coords = np.array(
                [[G.nodes[n].get('x', 0.0), G.nodes[n].get('y', 0.0)] for n in node_ids],
                dtype=np.float64,
            )
            node_lon = node_coords[:, 0]
            node_lat = node_coords[:, 1]
            node_tree_hav = BallTree(
                np.column_stack([np.radians(node_lat), np.radians(node_lon)]),
                metric='haversine',
                leaf_size=40,
            )
            self._graph_spatial_index[network_key] = {
                'node_ids': node_ids,
                'node_coords': node_coords,
                'node_lon': node_lon,
                'node_lat': node_lat,
                'node_tree_hav': node_tree_hav,
            }
            print(f"  [路网索引-H] 已缓存 network_key={network_key}，节点数={len(node_ids)}（Haversine 吸附）")
        return self._graph_spatial_index[network_key]

    def _get_location_based_g2sfca_calculator(
        self, population_data=None, pop_transform=None, nodes_with_population=None
    ):
        """与「基于位置」模块共用的 NanjingAccessibilityCalculator（2SFCA + Location-based 缓存）。"""
        from AccessibilityAnalysis import NanjingAccessibilityCalculator, CarbonBudgetParams

        self._g2sfca_calc = NanjingAccessibilityCalculator(
            self.networks,
            self.poi_data,
            self.district_boundary,
            population_data=population_data,
            pop_transform=pop_transform,
            city=self.city or '南京',
            nodes_with_population=nodes_with_population,
            carbon_params=CarbonBudgetParams(),
        )
        return self._g2sfca_calc

    def _compute_activity_g2sfca_accessibility(
        self,
        g2sfca_calc,
        activity,
        demand_coords,
        demand_pop,
        demand_pop_valid,
        resolution_degree,
        travel_time_min,
        mode_weights=None,
    ):
        """
        单类活动可达性 A_(i,k) = Σ_m w_m·A_(i,k,m)。
        各交通方式优先读 Location-based 缓存（POI+子类+t{分钟}+mode），与基于位置模块一致。
        """
        poi_gdf = self.poi_data.get(activity)
        if poi_gdf is None or getattr(poi_gdf, 'empty', True):
            return np.zeros(len(demand_coords), dtype=np.float64)

        weights = mode_weights if mode_weights else CARBON_BUDGET_MODE_WEIGHTS
        beta = float(g2sfca_calc.carbon_params.distance_decay_beta)
        return g2sfca_calc.compute_activity_accessibility_mode_weighted(
            activity,
            poi_gdf,
            demand_coords,
            demand_pop,
            demand_pop_valid,
            resolution=resolution_degree,
            mode_weights=weights,
            travel_time_min=travel_time_min,
            subcategory_keyword=None,
            beta=beta,
            progress_callback=None,
        )

    @staticmethod
    def _normalize_min_mobility_poi_counts(raw):
        """解析最低碳预算标准 POI 数量（固定四类各 1 个；兼容旧接口传入的 dict）。"""
        out = dict(DEFAULT_MIN_MOBILITY_POI_COUNTS)
        if not isinstance(raw, dict):
            return out
        for key in CARBON_MOBILITY_STANDARD_ACTIVITIES:
            if key not in raw:
                continue
            try:
                n = int(float(raw[key]))
            except (TypeError, ValueError):
                continue
            out[key] = max(1, min(n, 100))
        return out

    def _snap_origin_to_network(self, origin, mode):
        """将起点吸附到指定方式的路网节点（供多次 OD 复用）。"""
        if mode not in self.network_trees:
            return None
        try:
            tree, _coords, nodes = self.network_trees[mode]
            origin_coord = np.array([[origin.x, origin.y]])
            _, origin_idx = tree.query(origin_coord, k=1)
            origin_idx = int(origin_idx[0][0])
            origin_node_id = nodes.index[origin_idx]
            origin_node_geom = nodes.iloc[origin_idx].geometry
            return (origin_node_id, origin_node_geom)
        except Exception:
            return None

    def _ensure_network_csr(self, network_key):
        """驾车/步行等路网 → SciPy CSR（供 csgraph.dijkstra 批量加速）。"""
        if network_key in self._network_csr:
            return self._network_csr[network_key]
        from scipy.sparse import csr_matrix

        G = self.networks[network_key]
        gi = self._ensure_graph_spatial_index(network_key)
        node_ids = gi['node_ids']
        n = len(node_ids)
        id2i = {}
        for i in range(n):
            nid = node_ids[i]
            id2i[nid] = i
            try:
                id2i[int(nid)] = i
            except (TypeError, ValueError):
                pass
        rows, cols, data = [], [], []
        for u, v, edata in G.edges(data=True):
            if u not in id2i or v not in id2i:
                continue
            w = float(edata.get('length', 1.0))
            if w <= 0 or not np.isfinite(w):
                continue
            i, j = id2i[u], id2i[v]
            rows.append(i)
            cols.append(j)
            data.append(w)
        mat = csr_matrix((data, (rows, cols)), shape=(n, n))
        import networkx as nx
        Gu = nx.Graph()
        for u, v, edata in G.edges(data=True):
            w = float(edata.get('length', 1.0))
            if w <= 0 or not np.isfinite(w):
                continue
            if Gu.has_edge(u, v):
                Gu[u][v]['length'] = min(float(Gu[u][v]['length']), w)
            else:
                Gu.add_edge(u, v, length=w)
        rows_u, cols_u, data_u = [], [], []
        for u, v, edata in Gu.edges(data=True):
            if u not in id2i or v not in id2i:
                continue
            w = float(edata.get('length', 1.0))
            if w <= 0 or not np.isfinite(w):
                continue
            i, j = id2i[u], id2i[v]
            rows_u.append(i)
            cols_u.append(j)
            data_u.append(w)
        mat_ud = csr_matrix((data_u, (rows_u, cols_u)), shape=(n, n))
        pack = {
            'csr': mat, 'csr_undirected': mat_ud,
            'node_ids': node_ids, 'id2i': id2i, 'n': n,
        }
        self._network_csr[network_key] = pack
        print(f'  [路网CSR] network_key={network_key}，节点={n}，边={len(data)}')
        return pack

    def _get_dijkstra_distances_array_cached(
        self, network_key, origin_node, cutoff_m, cache, undirected=False,
    ):
        """单源 Dijkstra → 稠密距离向量（SciPy csgraph，比 NetworkX dict 快）。"""
        from scipy.sparse.csgraph import dijkstra

        cutoff_m = float(cutoff_m)
        cache_key = (str(network_key), origin_node, int(round(cutoff_m)), bool(undirected))
        with self._min_cb_cache_lock:
            dist_arr = cache.get(cache_key)
            if dist_arr is not None:
                return dist_arr
        csr_pack = self._ensure_network_csr(network_key)
        id2i = csr_pack['id2i']
        csr = csr_pack['csr_undirected'] if undirected else csr_pack['csr']
        if origin_node not in id2i:
            dist_arr = np.full(csr_pack['n'], np.inf, dtype=np.float64)
        else:
            oi = id2i[origin_node]
            raw = dijkstra(
                csr,
                directed=not undirected,
                indices=oi,
                limit=cutoff_m,
            )
            dist_arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        with self._min_cb_cache_lock:
            cache[cache_key] = dist_arr
        return dist_arr

    def _get_dijkstra_distances_cached(self, G, network_key, origin_node, cutoff_m, cache):
        """单源 Dijkstra（带 cutoff），按 (network_key, 锚点, cutoff) 缓存。"""
        import networkx as nx

        cutoff_m = float(cutoff_m)
        cache_key = (str(network_key), origin_node, int(round(cutoff_m)))
        dists = cache.get(cache_key)
        if dists is None:
            try:
                dists = nx.single_source_dijkstra_path_length(
                    G, origin_node, cutoff=cutoff_m, weight='length'
                )
            except Exception:
                dists = {}
            cache[cache_key] = dists
        return dists

    @staticmethod
    def _network_od_distance_m(d_to_net, dists_from_nn, dest_nn, dest_snap):
        if dest_nn not in dists_from_nn:
            return float('inf')
        return float(d_to_net) + float(dists_from_nn[dest_nn]) + float(dest_snap)

    def _prepare_min_cb_poi_packs(self, gi_drive):
        """各活动 POI 吸附到驾车路网，供批量最短路查表（所有出行方式共用）。"""
        packs = {}
        for activity in CARBON_MOBILITY_STANDARD_ACTIVITIES:
            if activity not in self.poi_trees:
                continue
            tree, coords, _gdf = self.poi_trees[activity]
            if coords is None or len(coords) == 0:
                continue
            poi_lon = coords[:, 0].astype(np.float64)
            poi_lat = coords[:, 1].astype(np.float64)
            poi_nn, poi_snap = _snap_lonlat_to_network_nodes_hav(gi_drive, poi_lon, poi_lat)
            packs[activity] = {
                'tree': tree,
                'coords': coords,
                'lon': poi_lon,
                'lat': poi_lat,
                'poi_nn_drive': poi_nn,
                'poi_snap_drive': poi_snap,
            }
        return packs

    @staticmethod
    def _haversine_k_nearest_indices(pack, demand_lon, demand_lat, k):
        """直线距离最近的 k 个 POI 索引（k 不超过该类 POI 总数）。"""
        n_poi = len(pack['lon'])
        if n_poi == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        k = min(max(1, int(k)), n_poi)
        q = _latlon_rad_for_haversine_tree(
            np.array([demand_lon], dtype=np.float64),
            np.array([demand_lat], dtype=np.float64),
        )
        dist_rad, indices = pack['tree'].query(q, k=k)
        if k == 1:
            idxs = np.array([int(indices[0][0])], dtype=np.int64)
            dists_m = np.array([float(dist_rad[0][0]) * _EARTH_RADIUS_M], dtype=np.float64)
        else:
            idxs = np.asarray(indices[0], dtype=np.int64)
            dists_m = np.asarray(dist_rad[0], dtype=np.float64) * _EARTH_RADIUS_M
        return idxs, dists_m

    def _road_select_top_n_poi_indices(
        self, cand_idxs, n_need, d_to_net, dists_foot, pack, origin_pt=None, road_mode='walk',
    ):
        """
        在直线候选 POI 中按路网距离排序，返回 (前 n 个索引列表, 链式出行用的最近 1 个索引)。
        候选不足 n 或路网不可达时返回 (None, None)。
        """
        n_need = max(1, int(n_need or 1))
        if cand_idxs is None or len(cand_idxs) < n_need:
            return None, None
        poi_nn = pack['poi_nn']
        poi_snap = pack['poi_snap']
        candidates = []
        origin_node = None
        if origin_pt is not None and dists_foot is None:
            snap_mode = road_mode if road_mode in self.network_trees else 'walk'
            origin_node = self._snap_origin_to_network(origin_pt, snap_mode)
        for idx in cand_idxs:
            idx = int(idx)
            if dists_foot is not None:
                d_net = self._network_od_distance_m(
                    d_to_net, dists_foot, poi_nn[idx], poi_snap[idx]
                )
            elif origin_pt is not None:
                dest = Point(float(pack['lon'][idx]), float(pack['lat'][idx]))
                _t, d_net = self._calculate_travel_time_and_distance(
                    origin_pt, dest, road_mode, origin_node_info=origin_node
                )
                if d_net is None or not np.isfinite(d_net):
                    d_net = float('inf')
            else:
                d_net = float('inf')
            if np.isfinite(d_net):
                candidates.append((float(d_net), idx))
        if len(candidates) < n_need:
            return None, None
        candidates.sort(key=lambda x: x[0])
        top_n = [int(c[1]) for c in candidates[:n_need]]
        return top_n, top_n[0]

    def _batch_min_cb_haversine_candidates(self, demand_lon, demand_lat, poi_packs, min_poi_counts=None):
        """
        Step1：每类活动取直线距离最近 1 个 POI。
        返回 poi_hits[activity][i] = (idxs, dists_m_hav)，cell_ok[i] 表示四类均有 POI。
        """
        n_cells = len(demand_lon)
        min_counts = self._normalize_min_mobility_poi_counts(min_poi_counts)
        poi_hits = {act: [None] * n_cells for act in CARBON_MOBILITY_STANDARD_ACTIVITIES}
        cell_ok = np.ones(n_cells, dtype=bool)

        q_all = _latlon_rad_for_haversine_tree(
            demand_lon.astype(np.float64), demand_lat.astype(np.float64)
        )

        for activity in CARBON_MOBILITY_STANDARD_ACTIVITIES:
            pack = poi_packs.get(activity)
            if pack is None:
                cell_ok[:] = False
                return poi_hits, min_counts, cell_ok
            n_poi = len(pack['lon'])
            if n_poi == 0:
                cell_ok[:] = False
                return poi_hits, min_counts, cell_ok
            dist_rad, indices = pack['tree'].query(q_all, k=1)
            hits = poi_hits[activity]
            for i in range(n_cells):
                idxs = np.array([int(indices[i][0])], dtype=np.int64)
                dists_m = np.array([float(dist_rad[i][0]) * _EARTH_RADIUS_M], dtype=np.float64)
                hits[i] = (idxs, dists_m)
        return poi_hits, min_counts, cell_ok

    def _pick_dest_packs_for_cell(self, cell_i, poi_packs, poi_hits):
        dest_packs = []
        for activity in CARBON_MOBILITY_STANDARD_ACTIVITIES:
            pack = poi_packs.get(activity)
            hit = poi_hits.get(activity)
            if pack is None or hit is None or hit[cell_i] is None:
                return None
            cand_idxs, _ = hit[cell_i]
            if cand_idxs is None or len(cand_idxs) == 0:
                return None
            dest_packs.append((pack, int(cand_idxs[0])))
        return dest_packs

    @staticmethod
    def _min_cb_drive_dijk_cutoff_m(time_budget_min):
        """驾车路网 Dijkstra 截止距离（米）：默认仅 出行时间×驾车速度，不再强制 30km。"""
        from transport_modes import speed_ms
        budget_cap = float(time_budget_min) * 60.0 * speed_ms('drive')
        try:
            extra_cap = float(os.environ.get('MIN_CB_Dijk_CAP_M', str(MIN_CB_Dijk_CAP_M)))
        except ValueError:
            extra_cap = MIN_CB_Dijk_CAP_M
        # 四段独立 OD：单段搜索半径至少覆盖一次驾车预算，并留出吸附余量
        snap_margin = 2000.0
        cap = budget_cap + snap_margin
        if extra_cap > 0:
            return max(cap, extra_cap)
        return cap

    @staticmethod
    def _min_cb_mode_plan(time_budget_min, mode_config, skip_transit):
        """各方式下四段独立 OD 总路程在预算内的最大距离（米）及碳排因子。"""
        from carbon_budget import CARBON_ALPHA_G_PER_KM
        from transport_modes import speed_ms

        budget_s = float(time_budget_min) * 60.0
        plan = []
        for mode, alpha_key in CarbonBudgetCalculator._min_cb_mode_trials():
            if skip_transit and mode in ('bus', 'metro'):
                continue
            if mode not in mode_config:
                continue
            spd = speed_ms('drive') if mode == 'car' else speed_ms(mode)
            max_d = budget_s * float(spd)
            if alpha_key is None:
                alpha = 0.0
            else:
                alpha = float(CARBON_ALPHA_G_PER_KM.get(alpha_key, 0.0))
            plan.append((max_d, alpha))
        return plan

    def _origin_drive_snap_m(self, origin_pt, origin_node_drive=None):
        """起点到驾车路网节点的吸附距离（米）。"""
        if origin_node_drive is None:
            origin_node_drive = self._snap_origin_to_network(origin_pt, 'car')
        if origin_node_drive is None:
            return 0.0
        geom = origin_node_drive[1]
        return float(self._haversine_distance(origin_pt.x, origin_pt.y, geom.x, geom.y))

    @staticmethod
    def _assign_min_cb_from_total_d(total_d, valid_mask, mode_plan):
        """总路程÷速度≤预算时，按出行方式顺序赋值最低碳预算。"""
        out = np.full(total_d.shape[0], np.nan, dtype=np.float64)
        remaining = valid_mask & np.isfinite(total_d)
        for max_d, alpha in mode_plan:
            ok = remaining & (total_d <= max_d + 1e-6)
            if alpha <= 0.0:
                out[ok] = 0.0
            else:
                out[ok] = (total_d[ok] / 1000.0) * alpha
            remaining &= ~ok
        return out

    def _build_min_cb_poi_di_arrays(self, poi_packs, poi_hits, n_cells, id2i):
        """预计算每格点四类 POI 在驾车路网上的节点索引与吸附距离。"""
        n_act = len(CARBON_MOBILITY_STANDARD_ACTIVITIES)
        poi_di = np.full((n_cells, n_act), -1, dtype=np.int32)
        poi_snap = np.full((n_cells, n_act), np.nan, dtype=np.float64)
        cell_ok = np.ones(n_cells, dtype=bool)
        id_get = id2i.get
        for k, activity in enumerate(CARBON_MOBILITY_STANDARD_ACTIVITIES):
            pack = poi_packs.get(activity)
            hits = poi_hits.get(activity)
            if pack is None or hits is None:
                cell_ok[:] = False
                return poi_di, poi_snap, cell_ok
            poi_nn = pack['poi_nn_drive']
            poi_sn = pack['poi_snap_drive']
            idxs = np.empty(n_cells, dtype=np.int64)
            for i in range(n_cells):
                hit = hits[i]
                if hit is None or hit[0] is None or len(hit[0]) == 0:
                    cell_ok[i] = False
                    idxs[i] = 0
                    continue
                idxs[i] = int(hit[0][0])
            nn = poi_nn[idxs]
            snaps = poi_sn[idxs]
            di_col = np.fromiter(
                (id_get(x, id_get(int(x), -1)) for x in nn), dtype=np.int32, count=n_cells,
            )
            poi_di[:, k] = di_col
            poi_snap[:, k] = snaps
        return poi_di, poi_snap, cell_ok

    def _build_min_cb_poi_hav_arrays(self, poi_hits, n_cells):
        """每格点四类 POI 的直线距离（米），用于路网不可达时回退。"""
        n_act = len(CARBON_MOBILITY_STANDARD_ACTIVITIES)
        poi_hav = np.full((n_cells, n_act), np.nan, dtype=np.float64)
        for k, activity in enumerate(CARBON_MOBILITY_STANDARD_ACTIVITIES):
            hits = poi_hits.get(activity)
            if hits is None:
                continue
            for i in range(n_cells):
                hit = hits[i]
                if hit is None or hit[1] is None or len(hit[1]) == 0:
                    continue
                poi_hav[i, k] = float(hit[1][0])
        return poi_hav

    def _eval_min_cb_batch_for_anchor(
        self, idx_arr, d_snap_arr, dist_arr, poi_di, poi_snap, poi_hav, cell_ok, mode_plan,
    ):
        """同一驾车锚点下批量格点：四条独立 OD 驾车路网距离 + 向量化方式判定。"""
        idx_arr = np.asarray(idx_arr, dtype=np.int64)
        di = poi_di[idx_arr]
        sn = poi_snap[idx_arr]
        hav = poi_hav[idx_arr]
        d_snap = d_snap_arr[idx_arr]
        di_safe = np.where(di >= 0, di, 0)
        d_legs_net = dist_arr[di_safe].astype(np.float64, copy=True)
        d_legs_net[di < 0] = np.inf
        leg_od_net = d_snap[:, None] + d_legs_net + sn
        leg_od = np.where(np.isfinite(d_legs_net), leg_od_net, hav)
        valid = cell_ok[idx_arr] & np.isfinite(leg_od).all(axis=1)
        total_d = np.sum(leg_od, axis=1)
        total_d[~valid] = np.nan
        return idx_arr, self._assign_min_cb_from_total_d(total_d, valid, mode_plan)

    @staticmethod
    def _min_cb_leg_time_min(d_m, speed_kmh):
        spd_ms = float(speed_kmh) * 1000.0 / 3600.0
        if spd_ms <= 0 or not np.isfinite(d_m):
            return float('inf')
        return float(d_m) / spd_ms / 60.0

    @staticmethod
    def _min_cb_mode_trials():
        return (
            ('walk', None),
            ('bike', None),
            ('bus', 'bus'),
            ('metro', 'subway'),
            ('car', 'drive'),
        )

    def _eval_min_cb_cell(
        self, cell_i, d_to_net_drive, dists_drive,
        poi_packs, poi_hits, time_budget_min, skip_transit=False, cell_ok=True,
    ):
        """单格点：四条独立 OD 驾车路网距离之和，总路程÷速度≤预算。"""
        from carbon_budget import CARBON_ALPHA_G_PER_KM

        if not cell_ok or dists_drive is None or d_to_net_drive is None:
            return np.nan

        budget = float(time_budget_min)
        dest_packs = self._pick_dest_packs_for_cell(cell_i, poi_packs, poi_hits)
        if dest_packs is None:
            return np.nan

        leg_dists = []
        for pack, idx in dest_packs:
            d_m = self._network_od_distance_m(
                d_to_net_drive, dists_drive,
                pack['poi_nn_drive'][idx], pack['poi_snap_drive'][idx],
            )
            if not np.isfinite(d_m):
                return np.nan
            leg_dists.append(float(d_m))

        d_origin = float(d_to_net_drive)
        n_legs = len(leg_dists)
        total_d = sum(leg_dists) - d_origin * max(0, n_legs - 1)

        for mode, alpha_key in self._min_cb_mode_trials():
            if skip_transit and mode in ('bus', 'metro'):
                continue
            if mode not in self.mode_config:
                continue
            speed = float(self.mode_config[mode]['speed'])
            t_min = self._min_cb_leg_time_min(total_d, speed)
            if not np.isfinite(t_min) or t_min > budget + 1e-6:
                continue
            if alpha_key is None:
                return 0.0
            alpha = float(CARBON_ALPHA_G_PER_KM.get(alpha_key, 0.0))
            return float((total_d / 1000.0) * alpha)
        return np.nan

    def _find_nearest_poi_by_haversine(self, origin_pt, activity):
        if activity not in self.poi_trees:
            return None
        tree, coords, _gdf = self.poi_trees[activity]
        if coords is None or len(coords) == 0:
            return None
        pack = {
            'tree': tree,
            'lon': coords[:, 0].astype(np.float64),
            'lat': coords[:, 1].astype(np.float64),
        }
        cand_idxs, _ = self._haversine_k_nearest_indices(pack, origin_pt.x, origin_pt.y, 1)
        if len(cand_idxs) == 0:
            return None
        idx = int(cand_idxs[0])
        return Point(float(coords[idx][0]), float(coords[idx][1]))

    def _total_od_chain_time_distance(self, origin, destinations, mode, origin_node_info=None):
        """对多个目的地累加出行时间与路网距离（分钟、米）。"""
        inf = float('inf')
        total_t = 0.0
        total_d = 0.0
        node_info = origin_node_info
        for dest in destinations:
            t_min, d_m = self._calculate_travel_time_and_distance(
                origin, dest, mode, origin_node_info=node_info
            )
            if t_min is None or d_m is None or not np.isfinite(t_min) or not np.isfinite(d_m):
                return inf, inf
            total_t += float(t_min)
            total_d += float(d_m)
        return total_t, total_d

    def _min_cb_total_d_drive_from_destinations(self, origin_pt, destinations, origin_node_drive=None):
        """四条独立 OD 总路程：一次起点吸附 + 四段路网+终点吸附。"""
        legs = []
        for dest in destinations:
            _t_car, d_m = self._calculate_travel_time_and_distance(
                origin_pt, dest, 'car', origin_node_info=origin_node_drive,
            )
            if d_m is None or not np.isfinite(d_m):
                return None
            legs.append(float(d_m))
        if not legs:
            return None
        d_origin = self._origin_drive_snap_m(origin_pt, origin_node_drive)
        return sum(legs) - d_origin * max(0, len(legs) - 1)

    def _compute_min_carbon_budget_for_origin(self, origin, min_poi_counts, time_budget_min):
        """
        四类活动各取直线最近 POI；分别计算出发地→各 POI 的驾车路网 OD 距离并求和，
        总路程÷速度≤预算时，按步行→骑行→公交→地铁→私家车顺序判定碳排。
        """
        from carbon_budget import CARBON_ALPHA_G_PER_KM

        origin_pt = origin if hasattr(origin, 'x') else Point(float(origin[0]), float(origin[1]))
        destinations = []
        for activity in CARBON_MOBILITY_STANDARD_ACTIVITIES:
            dest = self._find_nearest_poi_by_haversine(origin_pt, activity)
            if dest is None:
                return np.nan, None
            destinations.append(dest)

        budget = float(time_budget_min)
        origin_node_drive = (
            self._snap_origin_to_network(origin_pt, 'car')
            if 'car' in self.network_trees else None
        )
        leg_dists = self._min_cb_total_d_drive_from_destinations(
            origin_pt, destinations, origin_node_drive=origin_node_drive,
        )
        if leg_dists is None:
            return np.nan, None
        total_d = leg_dists

        for mode, alpha_key in self._min_cb_mode_trials():
            if mode not in self.mode_config:
                continue
            speed = float(self.mode_config[mode]['speed'])
            t_min = self._min_cb_leg_time_min(total_d, speed)
            if not np.isfinite(t_min) or t_min > budget + 1e-6:
                continue
            if alpha_key is None:
                alpha = 0.0
            else:
                alpha = float(CARBON_ALPHA_G_PER_KM.get(alpha_key, CARBON_ALPHA_G_PER_KM.get(mode, 0.0)))
            return float((total_d / 1000.0) * alpha), mode
        return np.nan, None

    def _compute_min_carbon_budget_raster(
        self, demand_coords, min_poi_counts, time_budget_min,
        progress_callback=None, resolution_m=1000,
    ):
        """
        为全部需求格点计算最低碳预算（克/次，四条独立 OD 之和）。
        加速：磁盘缓存、直线最近 POI、驾车路网 Dijkstra 查表（所有方式共用距离）。
        """
        from collections import defaultdict

        n = len(demand_coords)
        values = np.full(n, np.nan, dtype=np.float64)
        if n == 0:
            return values

        cached = self._load_min_cb_cache(
            demand_coords, resolution_m, min_poi_counts, time_budget_min
        )
        if cached is not None:
            if callable(progress_callback):
                try:
                    progress_callback(57, f'最低碳预算缓存命中 ({n} 格点)')
                except Exception:
                    pass
            return cached

        drive_key = self.mode_config.get('car', {}).get('network', 'drive')
        G_drive = self.networks.get(drive_key) if drive_key in self.networks else None
        if G_drive is None:
            print('[最低碳预算] 无驾车路网，回退逐格点计算（较慢）')
            return self._compute_min_carbon_budget_raster_legacy(
                demand_coords, min_poi_counts, time_budget_min, progress_callback
            )

        try:
            gi_drive = self._ensure_graph_spatial_index(drive_key)
        except Exception as ex:
            print(f'[最低碳预算] 驾车路网索引失败，回退逐格点计算: {ex}')
            return self._compute_min_carbon_budget_raster_legacy(
                demand_coords, min_poi_counts, time_budget_min, progress_callback
            )

        demand_lon = demand_coords[:, 0].astype(np.float64)
        demand_lat = demand_coords[:, 1].astype(np.float64)
        d_nn_drive, d_snap_drive = _snap_lonlat_to_network_nodes_hav(gi_drive, demand_lon, demand_lat)
        poi_packs = self._prepare_min_cb_poi_packs(gi_drive)
        if len(poi_packs) < len(CARBON_MOBILITY_STANDARD_ACTIVITIES):
            print('[最低碳预算] POI 数据不完整，回退逐格点计算')
            return self._compute_min_carbon_budget_raster_legacy(
                demand_coords, min_poi_counts, time_budget_min, progress_callback
            )

        if not hasattr(self, '_carbon_dijkstra_cache') or self._carbon_dijkstra_cache is None:
            self._carbon_dijkstra_cache = {}
        cache = self._carbon_dijkstra_cache
        skip_transit = self._min_cb_skip_transit()
        dijk_cutoff = self._min_cb_drive_dijk_cutoff_m(time_budget_min)
        mode_plan = self._min_cb_mode_plan(time_budget_min, self.mode_config, skip_transit)

        if callable(progress_callback):
            try:
                progress_callback(48, '正在批量检索直线最近 POI…')
            except Exception:
                pass
        poi_hits, _min_counts, cell_ok_hav = self._batch_min_cb_haversine_candidates(
            demand_lon, demand_lat, poi_packs, min_poi_counts
        )

        csr_pack = self._ensure_network_csr(drive_key)
        id2i = csr_pack['id2i']
        poi_di, poi_snap, cell_ok_net = self._build_min_cb_poi_di_arrays(
            poi_packs, poi_hits, n, id2i,
        )
        poi_hav = self._build_min_cb_poi_hav_arrays(poi_hits, n)
        cell_ok = cell_ok_hav & cell_ok_net

        by_anchor = defaultdict(list)
        for i in range(n):
            by_anchor[d_nn_drive[i]].append(i)
        n_unique = len(by_anchor)
        try:
            n_workers = max(1, int(os.environ.get('MIN_CB_WORKERS', '0')))
        except ValueError:
            n_workers = 0
        if n_workers <= 0:
            import multiprocessing
            n_workers = min(4, max(1, multiprocessing.cpu_count() - 1))
        print(
            f'  [最低碳预算-加速] {n} 格点 → {n_unique} 个驾车锚点，'
            f'dijk≤{dijk_cutoff:.0f}m（SciPy CSR），workers={n_workers}，'
            f'距离=四条独立OD(驾车路网)，'
        )

        cells_done = 0
        report_every = max(1, n_unique // 40)
        last_report_ts = 0.0
        anchor_items = list(by_anchor.items())

        def _maybe_report(force=False):
            nonlocal last_report_ts
            if not callable(progress_callback):
                return
            now = time.monotonic()
            if not force and cells_done > 0 and cells_done % max(50, n // 100) != 0:
                if now - last_report_ts < 2.0:
                    return
            pct = 50 + int(7 * cells_done / max(n, 1))
            try:
                progress_callback(pct, f'最低碳预算 {cells_done}/{n}')
                last_report_ts = now
            except Exception:
                pass

        def _run_anchor(item):
            anchor, idx_list = item
            dist_arr = self._get_dijkstra_distances_array_cached(
                drive_key, anchor, dijk_cutoff, cache, undirected=True,
            )
            idx_arr, batch_vals = self._eval_min_cb_batch_for_anchor(
                idx_list, d_snap_drive, dist_arr, poi_di, poi_snap, poi_hav, cell_ok, mode_plan,
            )
            return idx_arr, batch_vals

        if n_workers <= 1 or n_unique < 8:
            for g_idx, item in enumerate(anchor_items):
                idx_arr, batch_vals = _run_anchor(item)
                values[idx_arr] = batch_vals
                cells_done += len(idx_arr)
                if g_idx == 0 or g_idx == n_unique - 1 or (g_idx + 1) % report_every == 0:
                    _maybe_report(force=True)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_run_anchor, item): item for item in anchor_items}
                done_anchors = 0
                for fut in as_completed(futures):
                    idx_arr, batch_vals = fut.result()
                    values[idx_arr] = batch_vals
                    cells_done += len(idx_arr)
                    done_anchors += 1
                    if done_anchors == 1 or done_anchors == n_unique or done_anchors % report_every == 0:
                        _maybe_report(force=True)

        _maybe_report(force=True)

        self._save_min_cb_cache(
            demand_coords, values, resolution_m, min_poi_counts, time_budget_min
        )
        return values

    def _compute_min_carbon_budget_raster_legacy(
        self, demand_coords, min_poi_counts, time_budget_min, progress_callback=None
    ):
        """逐格点最短路（无路网批量加速时的回退路径）。"""
        n = len(demand_coords)
        values = np.full(n, np.nan, dtype=np.float64)
        report_every = max(1, min(50, n // 100))
        last_report_ts = 0.0
        for idx in range(n):
            lon, lat = demand_coords[idx]
            val, _mode = self._compute_min_carbon_budget_for_origin(
                Point(float(lon), float(lat)),
                min_poi_counts,
                time_budget_min,
            )
            values[idx] = val
            if callable(progress_callback):
                now = time.monotonic()
                done = idx + 1
                if done == 1 or done == n or done % report_every == 0 or now - last_report_ts >= 2.0:
                    pct = 50 + int(7 * idx / max(n - 1, 1))
                    try:
                        progress_callback(pct, f'最低碳预算 {done}/{n}')
                        last_report_ts = now
                    except Exception:
                        pass
        return values

    def calculate_carbon_budget_heatmap(
        self,
        rcb_tons,
        population_data=None,
        pop_transform=None,
        resolution_m=1000,
        progress_callback=None,
        search_radius_m=None,
        travel_time_min=None,
        transport_mode=None,
        min_mobility_poi_counts=None,
        mobility_standard_minutes=None,
    ):
        """
        个人碳预算公式（当前实现）：

        ①② A_(i,k,m)：各交通方式 G2SFCA（d0=出行时间×该方式速度），读 Location-based 缓存；
            A_(i,k)=Σ_m w_m·A_(i,k,m)；A_i = Σ_k W_k·A_(i,k)（W_k=CARBON_ACTIVITY_WEIGHTS）。

        ③ 最低碳预算标准：分钟生活圈时间内，工作/购物/教育/医疗四类 POI 各取直线最近 1 个，
            分别算出发地→POI 驾车路网 OD 并求和；总路程÷速度≤分钟生活圈时间时，
            按步行→骑行→公交→地铁→私家车顺序判定，碳排=四段距离之和×因子。

        ④ 将全部 RCB 按可达性反比分配：ICB_i = λ / A_i，
            λ = RCB_日 / Σ(P_i / A_i)，保证 Σ(P_i · ICB_i) = RCB_日。

        travel_time_min + transport_mode：与基于位置一致，d0 = 出行时间(分) × 速度(m/s) × 60；
            同时作为最低碳预算标准的时间预算（分钟生活圈）。
        search_radius_m：兼容旧接口，直接指定 d0（米）。
        min_mobility_poi_counts / mobility_standard_minutes：已废弃，保留参数仅为兼容旧调用。
        """
        def _report(pct, status):
            if callable(progress_callback):
                try:
                    progress_callback(int(pct), status)
                except Exception:
                    pass

        resolution_degree = resolution_m / _DEGREE_TO_METER
        # 前端区域碳排放总目标（吨/年）→ RCB_total（吨/年）→ 克/天参与 pool
        rcb_tons_input = float(rcb_tons)
        rcb_total_tons = max(0.0, rcb_tons_input * RCB_TOTAL_ALLOCATION_FACTOR)
        rcb_annual_g = rcb_total_tons * 1e6
        rcb_daily_g = rcb_annual_g / 365.0

        from transport_modes import speed_ms, normalize_mode_key
        acc_transport_mode = normalize_mode_key(
            transport_mode or getattr(self, '_carbon_acc_transport_mode', 'bike')
        )
        if travel_time_min is not None:
            tt = max(5, int(float(travel_time_min)))
            acc_radius_m = tt * 60.0 * speed_ms(acc_transport_mode)
        elif search_radius_m is not None:
            acc_radius_m = float(search_radius_m)
            tt = max(5, int(round(acc_radius_m / (60.0 * speed_ms(acc_transport_mode)))))
        else:
            tt = 30
            acc_transport_mode = 'bike'
            acc_radius_m = tt * 60.0 * speed_ms(acc_transport_mode)
        acc_radius_m = max(500.0, min(float(acc_radius_m), 100000.0))
        self._carbon_acc_search_radius_m = acc_radius_m
        self._carbon_acc_travel_time_min = tt
        self._carbon_acc_transport_mode = acc_transport_mode

        # 最低碳预算标准：分钟生活圈时间内，四类 POI 各可达 1 个
        min_poi_counts = dict(DEFAULT_MIN_MOBILITY_POI_COUNTS)
        mobility_tt = tt

        is_district_subset = bool(
            self.selected_district and self.selected_district != '全部区域'
        )

        _report(3, f"正在生成网格...")
        # 分区与全市共用行政边界 bbox 起算格网，避免分区单独 bbox 导致格点 lattice 错位
        city_grid_bbox = self.admin_boundary.total_bounds
        demand_cells, demand_coords, x_coords, y_coords, bbox, rows, cols = (
            self._build_demand_grid(
                self.district_boundary,
                resolution_degree,
                grid_bbox=city_grid_bbox,
            )
        )

        acc_raster = np.full((rows, cols), np.nan, dtype=np.float64)
        icb_raster = np.full((rows, cols), np.nan, dtype=np.float64)
        min_cb_raster = np.full((rows, cols), np.nan, dtype=np.float64)
        grid_inside_mask = np.zeros((rows, cols), dtype=bool)
        for i, j, _, _ in demand_cells:
            grid_inside_mask[j, i] = True

        if not demand_cells:
            return {
                'icb_raster': icb_raster,
                'acc_raster': acc_raster,
                'min_cb_raster': min_cb_raster,
                'grid_inside_mask': grid_inside_mask,
                'transform': None,
                'valid_cells': 0, 'stats': {}
            }

        n_cells = len(demand_cells)
        print(
            f"[碳预算热力图] 有效网格数: {n_cells}，分辨率: {resolution_m}m，"
            f"区域: {self.selected_district or '全部区域'}，"
            f"出行时间={tt}min，交通方式={acc_transport_mode}，d0={acc_radius_m:.0f}m，"
            f"RCB日配额={rcb_daily_g:.2f}g/天（反比分配）"
        )

        _report(5, "正在提取人口数据...")
        has_pop_raster = population_data is not None and pop_transform is not None
        if has_pop_raster:
            pop_values = np.zeros(n_cells, dtype=np.float64)
        else:
            # 无栅格：每格权重 1，P_total = n_cells
            pop_values = np.ones(n_cells, dtype=np.float64)

        if has_pop_raster:
            import rasterio
            height, width = population_data.shape
            pop_valid_count = 0
            for idx, (x, y) in enumerate(demand_coords):
                try:
                    r, c = rasterio.transform.rowcol(pop_transform, x, y)
                    if 0 <= r < height and 0 <= c < width:
                        val = float(population_data[r, c])
                        if np.isfinite(val) and val > 0 and val < 1e12:
                            pop_values[idx] = val
                            pop_valid_count += 1
                except Exception:
                    pass
            P_pop = float(np.sum(pop_values))
            print(f"[碳预算-人口] 人口栅格尺寸: {population_data.shape}, 需求格点: {n_cells}, "
                  f"栅格命中(>0): {pop_valid_count}, P_total={P_pop:,.2f}, "
                  f"单格人口范围: [{pop_values.min():.2f}, {pop_values.max():.2f}]")
        else:
            P_pop = float(np.sum(pop_values))
            print(f"[碳预算-人口] 警告: 未提供人口栅格，P_total 退化为需求格点数 n_cells={n_cells} "
                  f"（每格权重 1）")

        P_pop = max(P_pop, 1.0)

        valid_activities = [
            a for a in self.activity_types
            if a in self.poi_data
            and self.poi_data[a] is not None
            and not getattr(self.poi_data[a], 'empty', True)
        ]

        self._g2sfca_calc = None
        g2sfca_calc = self._get_location_based_g2sfca_calculator(
            population_data=population_data,
            pop_transform=pop_transform,
        )
        mode_weights_norm = g2sfca_calc._normalize_mode_weights(CARBON_BUDGET_MODE_WEIGHTS)
        g2sfca_calc.travel_time_minutes = tt
        g2sfca_beta = float(g2sfca_calc.carbon_params.distance_decay_beta)
        demand_pop_g2, demand_pop_valid_g2 = g2sfca_calc._extract_population_for_points(demand_coords)
        if not np.any(demand_pop_valid_g2):
            demand_pop_g2 = np.maximum(np.asarray(pop_values, dtype=np.float64), 0.1)
            demand_pop_valid_g2 = np.ones(n_cells, dtype=bool)

        drive_ok = (
            self.networks.get('drive') is not None
            and len(self.networks['drive'].nodes) > 0
        )
        self._carbon_acc_uses_road_distance = bool(drive_ok)
        print(
            f"[碳预算-G2SFCA] 活动数={len(valid_activities)}，出行时间={tt}min，"
            f"交通方式加权={mode_weights_norm}，β={g2sfca_beta}，"
            f"路网={'是' if drive_ok else '否(回退直线)'}，"
            f"缓存=Location-based/resol_{int(resolution_m)}/ + CarbonBudget/resol_{int(resolution_m)}/{{分区}}"
        )

        accessibility_from_district_cache = False
        accessibility_from_full_city = False
        cached_district = (
            self._load_district_g2sfca_cache(demand_coords, resolution_m)
            if is_district_subset else None
        )

        if cached_district is not None:
            total_acc = cached_district['total_acc']
            accessibility_from_district_cache = True
            _report(46, f"已使用 {self.selected_district} 分区可达性缓存")
        elif is_district_subset:
            _report(
                8,
                f"未找到 {self.selected_district} 分区缓存，正在计算全市可达性后提取…",
            )
            print(
                f"[碳预算-分区] 未命中 {self.selected_district} 分区缓存，"
                f"将先计算全市 G2SFCA 再提取对应区域"
            )
            full_cells, full_coords, *_ = self._build_demand_grid(
                self.admin_boundary,
                resolution_degree,
                grid_bbox=city_grid_bbox,
            )
            if len(full_coords) == 0:
                raise ValueError("全市需求格网为空，无法为分区提取可达性")
            print(
                f"[碳预算-分区] 全市格点 {len(full_coords)} 个，"
                f"目标分区 {self.selected_district} 格点 {n_cells} 个"
            )

            full_pop_g2, full_pop_valid_g2 = g2sfca_calc._extract_population_for_points(full_coords)
            if not np.any(full_pop_valid_g2):
                full_pop_g2 = np.ones(len(full_coords), dtype=np.float64) * 0.1
                full_pop_valid_g2 = np.ones(len(full_coords), dtype=bool)

            total_acc_full = self._compute_g2sfca_total_accessibility(
                g2sfca_calc,
                full_coords,
                full_pop_g2,
                full_pop_valid_g2,
                resolution_degree,
                tt,
                mode_weights_norm,
                valid_activities,
                progress_callback=_report,
                progress_start=8,
                progress_end=46,
            )
            subset_idx = self._map_subset_indices_from_full(
                demand_coords, full_coords, resolution_degree
            )
            total_acc = total_acc_full[subset_idx]
            self._save_district_g2sfca_cache(demand_coords, total_acc, resolution_m)
            accessibility_from_full_city = True
            _report(46, f"已从全市结果提取 {self.selected_district} 分区可达性")
            print(
                f"[碳预算-分区] 已从全市结果提取 {self.selected_district} "
                f"({n_cells}/{len(full_coords)} 格点)，并写入分区缓存"
            )
        else:
            total_acc = self._compute_g2sfca_total_accessibility(
                g2sfca_calc,
                demand_coords,
                demand_pop_g2,
                demand_pop_valid_g2,
                resolution_degree,
                tt,
                mode_weights_norm,
                valid_activities,
                progress_callback=_report,
                progress_start=8,
                progress_end=46,
            )

        self._g2sfca_calc = None

        _report(47, "正在计算最低碳预算标准…")
        min_cb_values = self._compute_min_carbon_budget_raster(
            demand_coords, min_poi_counts, mobility_tt,
            progress_callback=progress_callback,
            resolution_m=resolution_m,
        )
        n_min_cb_finite = int(np.isfinite(min_cb_values).sum())
        print(
            f"[碳预算-最低碳预算] 出行时间预算={mobility_tt}min，"
            f"POI数量={min_poi_counts}，有效格点={n_min_cb_finite}/{n_cells}"
        )

        _report(55, "正在计算碳预算分配（λ / A_i）...")
        valid_mask = total_acc > 0
        n_valid = int(np.sum(valid_mask))
        if n_valid == 0:
            return {
                'icb_raster': icb_raster,
                'acc_raster': acc_raster,
                'min_cb_raster': min_cb_raster,
                'transform': from_origin(bbox[0], bbox[3], resolution_degree, resolution_degree),
                'valid_cells': 0, 'stats': {'error': '无可达性数据'}
            }

        allocation_pool_g = float(rcb_daily_g)
        eps_a = 1e-15
        den = float(np.sum(pop_values[valid_mask] / (total_acc[valid_mask] + eps_a)))

        if allocation_pool_g <= 0 or den <= 0:
            lambda_val = 0.0
            print(
                f"[碳预算-分配] 警告: RCB日配额={allocation_pool_g:.2f} g/天, "
                f"Σ(P_i/A_i)={den:.6e}；"
                f"无法反比分配（请检查区域碳排放总目标或可达性数据）。"
            )
        else:
            lambda_val = allocation_pool_g / den

        ICB_i = np.full(n_cells, np.nan, dtype=np.float64)
        ICB_i[valid_mask] = lambda_val / (total_acc[valid_mask] + eps_a)

        _report(60, "正在回填栅格...")
        for idx, (i, j, _, _) in enumerate(demand_cells):
            acc_raster[j, i] = total_acc[idx]
            icb_raster[j, i] = ICB_i[idx]
            min_cb_raster[j, i] = min_cb_values[idx]

        valid_icb = ICB_i[np.isfinite(ICB_i)]
        sum_alloc_check = float(np.sum(pop_values[valid_mask] * ICB_i[valid_mask]))
        print(f"[碳预算-调试] 区域目标输入={rcb_tons_input:.2f}t/年, RCB_total={rcb_total_tons:.2f}t/年(×{RCB_TOTAL_ALLOCATION_FACTOR}), "
              f"RCB日={rcb_daily_g:.2f}g, λ={lambda_val:.6e}, "
              f"Σ(P_i·ICB)={sum_alloc_check:.2f}g, "
              f"ICB均值={np.mean(valid_icb):.2f}, P_total={P_pop:,.2f}")

        _report(65, "正在计算统计信息...")

        def _safe_stat(arr):
            v = arr[~np.isnan(arr)]
            if len(v) == 0:
                return {'mean': None, 'median': None, 'min': None, 'max': None,
                        'vmin': None, 'vmax': None, 'std': None, 'count': 0}
            return {
                'mean': float(np.mean(v)), 'median': float(np.median(v)),
                'min': float(np.min(v)), 'max': float(np.max(v)),
                'vmin': float(np.percentile(v, 5)),
                'vmax': float(np.percentile(v, 95)),
                'std': float(np.std(v)), 'count': int(len(v))
            }

        stats = {
            'icb': _safe_stat(ICB_i),
            'acc': _safe_stat(total_acc),
            'rcb_tons': float(rcb_tons_input),
            'rcb_tons_year': float(rcb_tons_input),
            'rcb_total_tons': float(rcb_total_tons),
            'rcb_total_allocation_factor': float(RCB_TOTAL_ALLOCATION_FACTOR),
            'rcb_g_per_day': float(rcb_daily_g),
            'rcb_g_annual': float(rcb_annual_g),
            'lambda_inverse_alloc': float(lambda_val),
            'allocation_pool_g_per_day': float(allocation_pool_g),
            'allocation_model': 'rcb_inverse_accessibility',
            'min_mobility_poi_counts': dict(min_poi_counts),
            'mobility_standard_minutes': int(mobility_tt),
            'min_cb': _safe_stat(min_cb_values),
            'min_cb_demand_cells': int(n_cells),
            'min_cb_computable_cells': n_min_cb_finite,
            'min_cb_compliance_rate': float(n_min_cb_finite / n_cells) if n_cells else 0.0,
            'search_radius_m': float(acc_radius_m),
            'travel_time_min': int(tt),
            'transport_mode': str(acc_transport_mode),
            'accessibility_mode_weights': dict(mode_weights_norm or CARBON_BUDGET_MODE_WEIGHTS),
            'accessibility_model': 'g2sfca_location_based_multimode',
            'accessibility_g2sfca_d0_m': float(acc_radius_m),
            'accessibility_g2sfca_beta': float(g2sfca_beta),
            'accessibility_uses_road_distance': bool(getattr(self, '_carbon_acc_uses_road_distance', False)),
            'accessibility_cache': 'Location-based/resol_{m}/{poi}_{subcategory}_t{travel_time}_{mode}.npz',
            'accessibility_from_district_cache': bool(accessibility_from_district_cache),
            'accessibility_from_full_city': bool(accessibility_from_full_city),
            'selected_district': str(self.selected_district or '全部区域'),
            'total_accessibility': float(np.sum(total_acc[valid_mask])),
            'valid_cells': n_valid,
            'resolution_m': resolution_m,
            'population_P': float(P_pop),
            'population_source': 'raster_sum' if has_pop_raster else 'grid_cell_count',
            'allocated_total_g_per_day': float(sum_alloc_check),
            'allocation_denominator': float(den),
        }

        transform = from_origin(bbox[0], bbox[3], resolution_degree, resolution_degree)
        _report(100, "碳预算热力图计算完成")

        return {
            'icb_raster': icb_raster,
            'acc_raster': acc_raster,
            'min_cb_raster': min_cb_raster,
            'grid_inside_mask': grid_inside_mask,
            'transform': transform,
            'valid_cells': n_valid,
            'stats': stats,
        }
