"""
路网与公共交通线网工具。

- TransitNetwork / load_nanjing_transit_networks：城市地铁、公交 GeoJSON 线网
- 脚本模式（python Get_Network.py）：从 OSM 下载并保存步行/骑行/驾车 GraphML
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from collections import defaultdict

import networkx as nx
import numpy as np
from sklearn.neighbors import BallTree

EARTH_RADIUS_M = 6371000.0
TRANSFER_PENALTY_M = 120.0
COLLOCATE_ROUND = 4

def _networkx_shortest_path_errors():
    """兼容不同 NetworkX 版本的最短路异常类型。"""
    errors = [nx.NetworkXNoPath, KeyError]
    node_not_found = getattr(nx, "NetworkXNodeNotFound", None)
    if node_not_found is not None:
        errors.append(node_not_found)
    return tuple(errors)

_NX_SHORTEST_PATH_ERRORS = _networkx_shortest_path_errors()

def straight_walk_distance_m(lon1, lat1, lon2, lat2) -> float:
    """步行接驳直线距离（米），用于公交到站/出站（不按道路网绕路）。"""
    return haversine_m(lon1, lat1, lon2, lat2)

def haversine_m(lon1, lat1, lon2, lat2):
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi * 0.5) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam * 0.5) ** 2
    return float(EARTH_RADIUS_M * 2.0 * np.arcsin(np.sqrt(min(1.0, a))))

def linestring_length_m(coords):
    if not coords or len(coords) < 2:
        return 0.0
    total = 0.0
    for i in range(len(coords) - 1):
        lon1, lat1 = float(coords[i][0]), float(coords[i][1])
        lon2, lat2 = float(coords[i + 1][0]), float(coords[i + 1][1])
        total += haversine_m(lon1, lat1, lon2, lat2)
    return total

# 线网几何拼接 / 拓扑清洗阈值（米或角度）
TOPO_JOIN_M = 15.0
TOPO_DEDUP_M = 4.0
TOPO_LOOP_CLOSE_M = 20.0
TOPO_SNAP_STOP_M = 150.0
TOPO_COLLINEAR_DEG = 12.0
TOPO_MIN_DETOUR_M = 22.0
TOPO_MIN_SPUR_LEG_M = 8.0
TOPO_SKIP_TRANSFER_SEG_M = 25.0

def coords_near_m(
    lonlat_a: tuple[float, float],
    lonlat_b: tuple[float, float],
    tol_m: float = TOPO_JOIN_M,
) -> bool:
    return haversine_m(lonlat_a[0], lonlat_a[1], lonlat_b[0], lonlat_b[1]) <= float(tol_m)

def _turn_angle_deg(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    import math

    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 < 1e-15 or n2 < 1e-15:
        return 180.0
    cos_val = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cos_val = max(-1.0, min(1.0, cos_val))
    return float(math.degrees(math.acos(cos_val)))

def _project_point_on_segment(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> tuple[float, float]:
    """将点 P 投影到线段 A-B 上（局部平面近似，适用于短线段）。"""
    dx = bx - ax
    dy = by - ay
    if abs(dx) < 1e-15 and abs(dy) < 1e-15:
        return (ax, ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return (ax + t * dx, ay + t * dy)

def nearest_point_on_polyline(
    lon: float,
    lat: float,
    coords: list[tuple[float, float]],
) -> tuple[tuple[float, float], float, int]:
    """
    求 (lon, lat) 到折线的最近点。
    返回 (投影点 lon/lat, 距离米, 所在线段的起点索引)。
    """
    if not coords:
        return ((lon, lat), 0.0, 0)
    if len(coords) == 1:
        pt = coords[0]
        return (pt, haversine_m(lon, lat, pt[0], pt[1]), 0)

    best_pt = coords[0]
    best_d = float("inf")
    best_i = 0
    for i in range(len(coords) - 1):
        ax, ay = coords[i]
        bx, by = coords[i + 1]
        px, py = _project_point_on_segment(lon, lat, ax, ay, bx, by)
        d = haversine_m(lon, lat, px, py)
        if d < best_d:
            best_d = d
            best_pt = (px, py)
            best_i = i
    return (best_pt, best_d, best_i)

def trim_polyline_to_stop(
    coords: list[tuple[float, float]],
    stop_lon: float,
    stop_lat: float,
    end: str,
    max_snap_m: float = TOPO_SNAP_STOP_M,
) -> list[tuple[float, float]]:
    """
    将折线起点/终点对齐到站点在折线上的投影点。
    避免把端点直接替换为站外坐标而产生 V 形绕路。
    """
    if not coords or len(coords) < 2:
        return coords
    pt, d, seg_i = nearest_point_on_polyline(stop_lon, stop_lat, coords)
    if d > max_snap_m:
        return coords
    if end == "start":
        trimmed = [pt] + coords[seg_i + 1 :]
    elif end == "end":
        trimmed = coords[: seg_i + 1] + [pt]
    else:
        return coords
    trimmed = dedupe_consecutive_coords(trimmed)
    return trimmed if len(trimmed) >= 2 else coords

def dedupe_consecutive_coords(
    coords: list[tuple[float, float]],
    min_step_m: float = TOPO_DEDUP_M,
) -> list[tuple[float, float]]:
    if not coords:
        return []
    out = [(float(coords[0][0]), float(coords[0][1]))]
    for pt in coords[1:]:
        cur = (float(pt[0]), float(pt[1]))
        if haversine_m(out[-1][0], out[-1][1], cur[0], cur[1]) >= min_step_m:
            out.append(cur)
    if len(out) < 2 and len(coords) >= 2:
        return [(float(coords[0][0]), float(coords[0][1])), (float(coords[-1][0]), float(coords[-1][1]))]
    return out

def remove_path_loops(
    coords: list[tuple[float, float]],
    close_m: float = TOPO_LOOP_CLOSE_M,
) -> list[tuple[float, float]]:
    """剔除折返环：路径回到较早邻近点时，删除中间折返段。"""
    if len(coords) < 4:
        return coords
    changed = True
    while changed and len(coords) >= 4:
        changed = False
        n = len(coords)
        for i in range(n - 1, 2, -1):
            for j in range(i - 2, -1, -1):
                if coords_near_m(coords[i], coords[j], close_m) and (i - j) >= 2:
                    del coords[j + 1:i]
                    changed = True
                    break
            if changed:
                break
    return coords

def remove_spur_backtracks(
    coords: list[tuple[float, float]],
    min_detour_m: float = TOPO_MIN_DETOUR_M,
    min_leg_m: float = TOPO_MIN_SPUR_LEG_M,
    sharp_turn_deg: float = 155.0,
) -> list[tuple[float, float]]:
    if len(coords) < 3:
        return coords
    changed = True
    while changed and len(coords) >= 3:
        changed = False
        for i in range(1, len(coords) - 1):
            a, b, c = coords[i - 1], coords[i], coords[i + 1]
            d_ac = haversine_m(a[0], a[1], c[0], c[1])
            d_ab = haversine_m(a[0], a[1], b[0], b[1])
            d_bc = haversine_m(b[0], b[1], c[0], c[1])
            if d_ab < TOPO_DEDUP_M or d_bc < TOPO_DEDUP_M:
                del coords[i]
                changed = True
                break
            detour = d_ab + d_bc - d_ac
            turn = _turn_angle_deg(a, b, c)
            if (
                detour >= min_detour_m
                and d_ab >= min_leg_m
                and d_bc >= min_leg_m
                and turn >= sharp_turn_deg
            ):
                del coords[i]
                changed = True
                break
    return coords

def merge_collinear_coords(
    coords: list[tuple[float, float]],
    angle_tol_deg: float = TOPO_COLLINEAR_DEG,
) -> list[tuple[float, float]]:
    if len(coords) < 3:
        return coords
    straight_min = 180.0 - float(angle_tol_deg)
    out = [coords[0]]
    for i in range(1, len(coords) - 1):
        if _turn_angle_deg(out[-1], coords[i], coords[i + 1]) >= straight_min:
            continue
        out.append(coords[i])
    out.append(coords[-1])
    return out

def clean_transit_line_topology(
    coords: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """公交/地铁线网折线拓扑清洗：去重、去环、去折返（保留弯道几何，不做共线合并）。"""
    if not coords:
        return []
    working = dedupe_consecutive_coords([(float(c[0]), float(c[1])) for c in coords])
    if len(working) < 2:
        return working
    working = remove_path_loops(working)
    working = remove_spur_backtracks(working)
    working = dedupe_consecutive_coords(working)
    return working if len(working) >= 2 else working[:2]

class TransitNetwork:
    """单种公共交通线网（地铁或公交）；区段按 s_stopid→e_stopid 有向连通。"""

    def __init__(self, geojson_path: str, name: str = "transit"):
        self.geojson_path = os.path.abspath(geojson_path)
        self.name = name
        self.graph = nx.DiGraph()
        self.stop_lonlat: dict[str, tuple[float, float]] = {}
        self.stop_names: dict[str, str] = {}
        # 区段 LineString 坐标 (lon, lat)，用于绘制沿线路走向的路径
        self.edge_line_coords: dict[tuple[str, str], list[tuple[float, float]]] = {}
        self._stop_ids: list[str] = []
        self._coord_arr: np.ndarray | None = None
        self._tree: BallTree | None = None
        self._adj_ready = False
        self._adj: list[list[tuple[int, float]]] | None = None
        self._stop_to_idx: dict[str, int] = {}
        self._load()

    def _cache_path(self) -> str:
        base = os.path.dirname(os.path.abspath(__file__))
        mtime = os.path.getmtime(self.geojson_path)
        key = f"{self.geojson_path}:{mtime}:{self.name}:linegeom_v7_projected"
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
        cache_dir = os.path.join(base, "cache", "transit")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{self.name}_{digest}.pkl")

    def _load(self):
        cache = self._cache_path()
        if os.path.exists(cache):
            try:
                with open(cache, "rb") as f:
                    data = pickle.load(f)
                self.graph = data["graph"]
                self.stop_lonlat = data["stop_lonlat"]
                self.stop_names = data.get("stop_names") or {}
                self.edge_line_coords = data.get("edge_line_coords") or {}
                self._adj_ready = False
                self._build_spatial_index()
                if not self.edge_line_coords:
                    self._load_segment_line_coords_from_geojson()
                if not self.stop_names:
                    self._load_stop_names_from_geojson()
                print(
                    f"[Transit] 缓存加载 {self.name}: "
                    f"{len(self.stop_lonlat)} 站, {self.graph.number_of_edges()} 边"
                )
                return
            except Exception as ex:
                print(f"[Transit] 缓存读取失败，重新构建: {ex}")

        self._build_from_geojson()
        try:
            with open(cache, "wb") as f:
                pickle.dump(
                    {
                        "graph": self.graph,
                        "stop_lonlat": self.stop_lonlat,
                        "stop_names": self.stop_names,
                        "edge_line_coords": self.edge_line_coords,
                    },
                    f,
                    protocol=4,
                )
        except Exception as ex:
            print(f"[Transit] 缓存写入失败: {ex}")
        self._build_spatial_index()

    @staticmethod
    def _stop_label_from_props(props) -> str:
        """从站点属性提取展示名（兼容 name_cn / stop_cn 等字段）。"""
        if not props:
            return ""
        for key in ("name_cn", "stop_cn", "name", "stop_en", "name_en"):
            val = props.get(key)
            if val is None:
                continue
            label = str(val).strip()
            if label:
                return label
        return ""

    def _load_stop_names_from_geojson(self):
        """从 GeoJSON 补载站点名称（兼容旧缓存）。"""
        self.stop_names = {}
        try:
            with open(self.geojson_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            stop_features = (data.get("stops") or {}).get("features") or []
            for feat in stop_features:
                props = feat.get("properties") or {}
                sid = props.get("stop_id") or props.get("s_stopid") or props.get("e_stopid")
                if not sid or sid in self.stop_names:
                    continue
                name = self._stop_label_from_props(props)
                if name:
                    self.stop_names[str(sid)] = name
        except Exception as ex:
            print(f"[Transit] 站点名称加载失败 {self.name}: {ex}")

    def _load_segment_line_coords_from_geojson(self):
        """从 GeoJSON 区段补载 LineString 几何（兼容旧缓存）。"""
        self.edge_line_coords = {}
        try:
            with open(self.geojson_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            segment_features = (data.get("segments") or {}).get("features") or []
            self._ingest_segment_features(segment_features)
        except Exception as ex:
            print(f"[Transit] 区段几何加载失败 {self.name}: {ex}")

    @staticmethod
    def _extract_linestring_coords(geom) -> list[tuple[float, float]]:
        if not geom:
            return []
        gtype = geom.get("type")
        lines: list[list] = []
        if gtype == "LineString":
            lines = [geom.get("coordinates") or []]
        elif gtype == "MultiLineString":
            lines = geom.get("coordinates") or []
        out: list[tuple[float, float]] = []
        for raw in lines:
            for c in raw or []:
                if isinstance(c, (list, tuple)) and len(c) >= 2:
                    out.append((float(c[0]), float(c[1])))
        return out

    def _register_edge_geometry(self, s_id: str, e_id: str, line: list[tuple[float, float]]):
        if len(line) < 2:
            return
        self.edge_line_coords[(s_id, e_id)] = line
        self.edge_line_coords[(e_id, s_id)] = list(reversed(line))

    def _ingest_segment_features(self, segment_features):
        for feat in segment_features:
            props = feat.get("properties") or {}
            s_id = props.get("s_stopid")
            e_id = props.get("e_stopid")
            if not s_id or not e_id:
                continue
            s_id, e_id = str(s_id), str(e_id)
            line = self._extract_linestring_coords(feat.get("geometry") or {})
            if len(line) < 2:
                continue
            self._register_edge_geometry(s_id, e_id, line)

    def _build_from_geojson(self):
        with open(self.geojson_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        stops_fc = data.get("stops") or {}
        segments_fc = data.get("segments") or {}
        stop_features = stops_fc.get("features") or []
        segment_features = segments_fc.get("features") or []
        self.edge_line_coords = {}

        for feat in stop_features:
            props = feat.get("properties") or {}
            sid = props.get("stop_id") or props.get("s_stopid") or props.get("e_stopid")
            geom = feat.get("geometry") or {}
            if not sid or geom.get("type") != "Point":
                continue
            coords = geom.get("coordinates") or []
            if len(coords) < 2:
                continue
            lon, lat = float(coords[0]), float(coords[1])
            sid = str(sid)
            self.stop_lonlat[sid] = (lon, lat)
            name = self._stop_label_from_props(props)
            if name and sid not in self.stop_names:
                self.stop_names[sid] = name

        self._ingest_segment_features(segment_features)

        for feat in segment_features:
            props = feat.get("properties") or {}
            s_id = props.get("s_stopid")
            e_id = props.get("e_stopid")
            if not s_id or not e_id:
                continue
            s_id, e_id = str(s_id), str(e_id)
            geom = feat.get("geometry") or {}
            weight = None
            if geom.get("type") == "LineString":
                weight = linestring_length_m(geom.get("coordinates") or [])
            if weight is None or weight <= 0:
                dist_km = props.get("distance")
                if dist_km is not None:
                    try:
                        weight = float(dist_km) * 1000.0
                    except (TypeError, ValueError):
                        weight = None
            line = self.edge_line_coords.get((s_id, e_id))
            if weight is None or weight <= 0:
                if line and len(line) >= 2:
                    weight = linestring_length_m([[lon, lat] for lon, lat in line])
                elif s_id in self.stop_lonlat and e_id in self.stop_lonlat:
                    lon1, lat1 = self.stop_lonlat[s_id]
                    lon2, lat2 = self.stop_lonlat[e_id]
                    weight = haversine_m(lon1, lat1, lon2, lat2)
                else:
                    continue
            self._add_edge(s_id, e_id, weight, line_coords=line)

        self._add_transfer_edges()
        self._adj_ready = False
        print(
            f"[Transit] 构建 {self.name}: {len(self.stop_lonlat)} 站, "
            f"{len(segment_features)} 线段, 图边 {self.graph.number_of_edges()}"
        )

    def _add_edge(self, a: str, b: str, weight: float, line_coords=None):
        if a == b or weight <= 0:
            return
        line = line_coords
        if line is None:
            line = self.edge_line_coords.get((a, b))
        if self.graph.has_edge(a, b):
            w_old = self.graph[a][b].get("weight", float("inf"))
            if weight < w_old:
                self.graph[a][b]["weight"] = weight
                if line and len(line) >= 2:
                    self.graph[a][b]["line_coords"] = line
        else:
            attrs = {"weight": float(weight)}
            if line and len(line) >= 2:
                attrs["line_coords"] = line
            self.graph.add_edge(a, b, **attrs)

    def _add_transfer_edges(self):
        groups: dict[tuple[float, float], list[str]] = defaultdict(list)
        for sid, (lon, lat) in self.stop_lonlat.items():
            key = (round(lon, COLLOCATE_ROUND), round(lat, COLLOCATE_ROUND))
            groups[key].append(sid)
        for sids in groups.values():
            if len(sids) < 2:
                continue
            for i in range(len(sids)):
                for j in range(i + 1, len(sids)):
                    self._add_edge(sids[i], sids[j], TRANSFER_PENALTY_M)
                    self._add_edge(sids[j], sids[i], TRANSFER_PENALTY_M)

    def _build_spatial_index(self):
        self._invalidate_adjacency()
        self._stop_ids = list(self.stop_lonlat.keys())
        if not self._stop_ids:
            self._coord_arr = np.zeros((0, 2), dtype=np.float64)
            self._tree = None
            return
        self._coord_arr = np.array(
            [[self.stop_lonlat[s][0], self.stop_lonlat[s][1]] for s in self._stop_ids],
            dtype=np.float64,
        )
        lat_rad = np.radians(self._coord_arr[:, 1])
        lon_rad = np.radians(self._coord_arr[:, 0])
        self._tree = BallTree(
            np.column_stack([lat_rad, lon_rad]),
            metric="haversine",
            leaf_size=40,
        )

    def _invalidate_adjacency(self):
        self._adj_ready = False
        self._adj = None
        self._stop_to_idx = {}

    def _ensure_adjacency(self):
        """将 NetworkX 图转为邻接表，供快速截断多源 Dijkstra 使用。"""
        if self._adj_ready and self._adj is not None:
            return
        self._stop_to_idx = {s: i for i, s in enumerate(self._stop_ids)}
        n = len(self._stop_ids)
        adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        for u, v, data in self.graph.edges(data=True):
            ui = self._stop_to_idx.get(u)
            vi = self._stop_to_idx.get(v)
            if ui is None or vi is None:
                continue
            w = float(data.get("weight", 0.0) or 0.0)
            if w <= 0:
                continue
            adj[ui].append((vi, w))
        self._adj = adj
        self._adj_ready = True

    def multi_source_distances_m(
        self, sources: dict[str, float], cutoff_m: float
    ) -> dict[str, float]:
        """
        多源 Dijkstra（截断 cutoff_m），返回 {stop_id: 距离(米)}。
        sources: 各起点站点及其初始代价（通常为接驳步行距离）。
        """
        import heapq

        self._ensure_adjacency()
        if not self._adj or not self._stop_ids:
            return {}

        n = len(self._stop_ids)
        cutoff = float(cutoff_m)
        best = np.full(n, np.inf, dtype=np.float64)
        heap: list[tuple[float, int]] = []

        for sid, w0 in sources.items():
            idx = self._stop_to_idx.get(sid)
            if idx is None:
                continue
            w0 = float(w0)
            if w0 <= cutoff and w0 < best[idx]:
                best[idx] = w0
                heapq.heappush(heap, (w0, idx))

        if not heap:
            return {}

        adj = self._adj
        stop_ids = self._stop_ids
        while heap:
            d, u = heapq.heappop(heap)
            if d > best[u] or d > cutoff:
                continue
            for v, w in adj[u]:
                nd = d + w
                if nd <= cutoff and nd < best[v]:
                    best[v] = nd
                    heapq.heappush(heap, (nd, v))

        out: dict[str, float] = {}
        for i in range(n):
            if best[i] <= cutoff:
                out[stop_ids[i]] = float(best[i])
        return out

    def nearest_stops(self, lon: float, lat: float, k: int = 5, max_dist_m: float = 2500.0):
        if self._tree is None or len(self._stop_ids) == 0:
            return []
        k = min(int(k), len(self._stop_ids))
        q = np.array([[np.radians(lat), np.radians(lon)]], dtype=np.float64)
        dist_rad, indices = self._tree.query(q, k=k)
        out = []
        max_rad = float(max_dist_m) / EARTH_RADIUS_M
        for dist_r, idx in zip(dist_rad[0], indices[0]):
            if dist_r > max_rad:
                continue
            sid = self._stop_ids[int(idx)]
            slon, slat = self.stop_lonlat[sid]
            out.append((sid, slon, slat, float(dist_r) * EARTH_RADIUS_M))
        return out

    def stop_name(self, stop_id: str) -> str:
        return self.stop_names.get(str(stop_id), "")

    def stop_pair_path_ids(self, stop_a: str, stop_b: str) -> list[str]:
        if stop_a == stop_b:
            return [stop_a] if stop_a in self.stop_lonlat else []
        try:
            return list(nx.shortest_path(self.graph, stop_a, stop_b, weight="weight"))
        except _NX_SHORTEST_PATH_ERRORS:
            return []

    def stop_pair_distance_m(self, stop_a: str, stop_b: str) -> float:
        if stop_a == stop_b:
            return 0.0
        if stop_a not in self.graph or stop_b not in self.graph:
            if stop_a in self.stop_lonlat and stop_b in self.stop_lonlat:
                lon1, lat1 = self.stop_lonlat[stop_a]
                lon2, lat2 = self.stop_lonlat[stop_b]
                return haversine_m(lon1, lat1, lon2, lat2)
            return float("inf")
        try:
            return float(
                nx.shortest_path_length(self.graph, stop_a, stop_b, weight="weight")
            )
        except _NX_SHORTEST_PATH_ERRORS:
            return float("inf")

    def _append_line_coords(self, path: list[tuple[float, float]], segment: list[tuple[float, float]]):
        if not segment:
            return
        seg = [(float(c[0]), float(c[1])) for c in segment if c and len(c) >= 2]
        if len(seg) < 2:
            return
        if not path:
            path.extend(seg)
            return

        last = path[-1]
        first, end = seg[0], seg[-1]

        if coords_near_m(last, first):
            path.extend(seg[1:])
            return
        if coords_near_m(last, end):
            path.extend(list(reversed(seg[:-1])))
            return

        # 与已有路径中某点相接时，从最近点继续向前拼接，避免重复折线
        best_i = None
        best_d = TOPO_JOIN_M
        for i, pt in enumerate(seg):
            d = haversine_m(last[0], last[1], pt[0], pt[1])
            if d < best_d:
                best_d = d
                best_i = i
        if best_i is not None and best_i < len(seg) - 1:
            path.extend(seg[best_i + 1:])
            return

        seg_len = linestring_length_m(seg)
        if seg_len <= TOPO_SKIP_TRANSFER_SEG_M and (
            coords_near_m(last, first) or coords_near_m(last, end)
        ):
            return

        path.extend(seg)

    def _snap_segment_to_stops(
        self,
        segment: list[tuple[float, float]],
        stop_from: str,
        stop_to: str,
        max_snap_m: float = TOPO_SNAP_STOP_M,
    ) -> list[tuple[float, float]]:
        """将区段端点对齐到站点在折线上的投影点，避免站外坐标造成 V 形绕路。"""
        if not segment or len(segment) < 2:
            return segment
        out = [(float(c[0]), float(c[1])) for c in segment]
        if stop_from in self.stop_lonlat:
            lon_a, lat_a = self.stop_lonlat[stop_from]
            out = trim_polyline_to_stop(out, lon_a, lat_a, "start", max_snap_m)
        if len(out) >= 2 and stop_to in self.stop_lonlat:
            lon_b, lat_b = self.stop_lonlat[stop_to]
            out = trim_polyline_to_stop(out, lon_b, lat_b, "end", max_snap_m)
        return out if len(out) >= 2 else segment

    def _orient_segment_for_stops(
        self,
        segment: list[tuple[float, float]],
        stop_from: str,
        stop_to: str,
    ) -> list[tuple[float, float]]:
        """将区段折线朝向 stop_from -> stop_to（兼容旧缓存/缺失几何）。"""
        if not segment or len(segment) < 2:
            return segment
        if stop_from not in self.stop_lonlat or stop_to not in self.stop_lonlat:
            return segment
        lon_a, lat_a = self.stop_lonlat[stop_from]
        lon_b, lat_b = self.stop_lonlat[stop_to]
        lon_f, lat_f = segment[0]
        lon_l, lat_l = segment[-1]
        d_start_a = haversine_m(lon_a, lat_a, lon_f, lat_f)
        d_end_a = haversine_m(lon_a, lat_a, lon_l, lat_l)
        if d_end_a + 1.0 < d_start_a:
            return list(reversed(segment))
        return segment

    def _segment_coords_for_edge(self, stop_from: str, stop_to: str) -> list[tuple[float, float]] | None:
        """取 stop_from -> stop_to 的有向区段几何（优先 edge_line_coords）。"""
        seg = self.edge_line_coords.get((stop_from, stop_to))
        if seg and len(seg) >= 2:
            oriented = self._orient_segment_for_stops(list(seg), stop_from, stop_to)
            return self._snap_segment_to_stops(oriented, stop_from, stop_to)
        if not self.graph.has_edge(stop_from, stop_to):
            return None
        seg = self.graph[stop_from][stop_to].get("line_coords")
        if not seg or len(seg) < 2:
            return None
        oriented = self._orient_segment_for_stops(list(seg), stop_from, stop_to)
        return self._snap_segment_to_stops(oriented, stop_from, stop_to)

    def stop_pair_path_coords(self, stop_a: str, stop_b: str) -> list[tuple[float, float]]:
        """沿线路区段几何的坐标序列 [(lon, lat), ...]。"""
        if stop_a == stop_b:
            if stop_a in self.stop_lonlat:
                return [self.stop_lonlat[stop_a]]
            return []
        try:
            path_ids = nx.shortest_path(self.graph, stop_a, stop_b, weight="weight")
        except _NX_SHORTEST_PATH_ERRORS:
            return []

        if not getattr(self, "edge_line_coords", None):
            self.edge_line_coords = {}

        out: list[tuple[float, float]] = []
        for i in range(len(path_ids) - 1):
            u, v = path_ids[i], path_ids[i + 1]
            is_transfer = (
                u in self.stop_lonlat
                and v in self.stop_lonlat
                and haversine_m(
                    self.stop_lonlat[u][0],
                    self.stop_lonlat[u][1],
                    self.stop_lonlat[v][0],
                    self.stop_lonlat[v][1],
                )
                < TOPO_SKIP_TRANSFER_SEG_M
            )
            seg = self._segment_coords_for_edge(u, v)
            if seg:
                if is_transfer and out and coords_near_m(out[-1], seg[-1]):
                    continue
                self._append_line_coords(out, seg)
            elif u in self.stop_lonlat and v in self.stop_lonlat:
                lon1, lat1 = self.stop_lonlat[u]
                lon2, lat2 = self.stop_lonlat[v]
                if haversine_m(lon1, lat1, lon2, lat2) >= TOPO_SKIP_TRANSFER_SEG_M:
                    self._append_line_coords(
                        out,
                        [self.stop_lonlat[u], self.stop_lonlat[v]],
                    )

        if len(out) < 2:
            fallback = [self.stop_lonlat[s] for s in path_ids if s in self.stop_lonlat]
            return clean_transit_line_topology(fallback) if len(fallback) >= 2 else fallback
        return clean_transit_line_topology(out)

def multimodal_transit_distance_m(
    origin_lon: float,
    origin_lat: float,
    dest_lon: float,
    dest_lat: float,
    mode: str,
    transit_net: TransitNetwork,
    road_walk_distance_fn=None,
) -> float | None:
    """
    公交/地铁出行距离（米）：接驳步行 + 线网最短路 + 接驳步行。
    road_walk_distance_fn(lon1, lat1, lon2, lat2) -> 米；公交宜传 straight_walk_distance_m，
    地铁可传道路网步行函数；未提供时用 Haversine×1.25 近似。
    """
    if transit_net is None:
        return None

    def _walk_m(lon1, lat1, lon2, lat2):
        if callable(road_walk_distance_fn):
            d = road_walk_distance_fn(lon1, lat1, lon2, lat2)
            if d is not None and d > 0 and d < float("inf"):
                return float(d)
        return haversine_m(lon1, lat1, lon2, lat2) * 1.25

    best = float("inf")
    orig_stops = transit_net.nearest_stops(origin_lon, origin_lat, k=5, max_dist_m=2500.0)
    dest_stops = transit_net.nearest_stops(dest_lon, dest_lat, k=5, max_dist_m=2500.0)
    if not orig_stops or not dest_stops:
        return None

    for s_o, lon_o, lat_o, _ in orig_stops:
        walk_o = _walk_m(origin_lon, origin_lat, lon_o, lat_o)
        for s_d, lon_d, lat_d, _ in dest_stops:
            walk_d = _walk_m(lon_d, lat_d, dest_lon, dest_lat)
            ride = transit_net.stop_pair_distance_m(s_o, s_d)
            if not np.isfinite(ride):
                continue
            total = walk_o + ride + walk_d
            if total < best:
                best = total
    if not np.isfinite(best):
        return None
    return float(best)

def load_nanjing_transit_networks(base_dir: str | None = None) -> dict[str, TransitNetwork]:
    """加载城市地铁、公交线网（Network/Nanjing/*.geojson）。"""
    base = base_dir or os.path.dirname(os.path.abspath(__file__))
    net_dir = os.path.join(base, "Network", "Nanjing")
    out: dict[str, TransitNetwork] = {}
    metro_path = os.path.join(net_dir, "nanjing_metro.geojson")
    bus_path = os.path.join(net_dir, "nanjing_bus.geojson")
    if os.path.exists(metro_path):
        out["metro"] = TransitNetwork(metro_path, "metro")
    if os.path.exists(bus_path):
        out["bus"] = TransitNetwork(bus_path, "bus")
    return out

def _download_osm_road_networks():
    """从 OSM 下载城市局部道路网并保存 GraphML（仅脚本入口调用）。"""
    import matplotlib.pyplot as plt
    import osmnx as ox  # 第三方开源库，用于从 OpenStreetMap 下载并保存道路网 GraphML

    center_point = (32.06, 118.78)
    dist = 5000

    print("正在下载所有道路数据...")
    all_net = ox.graph_from_point(center_point, dist=dist, network_type="all", simplify=True)
    print(f"所有道路: {len(all_net.nodes)}节点, {len(all_net.edges)}边")

    def filter_network(G, allowed_highways):
        filtered_edges = []
        for u, v, k, data in G.edges(keys=True, data=True):
            highway = data.get("highway", "")
            if isinstance(highway, list):
                highway = highway[0] if highway else ""
            if highway in allowed_highways:
                filtered_edges.append((u, v, k))
        return G.edge_subgraph(filtered_edges)

    walk_highways = [
        "footway", "path", "pedestrian", "steps", "corridor",
        "living_street", "residential", "service", "unclassified",
    ]
    bike_highways = [
        "cycleway", "path", "footway", "living_street",
        "residential", "service", "unclassified", "tertiary",
        "secondary", "primary", "trunk",
    ]
    drive_highways = [
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "unclassified", "residential", "service", "living_street",
    ]

    print("正在过滤步行网络...")
    walk_net = filter_network(all_net, walk_highways)
    print(f"步行网络: {len(walk_net.nodes)}节点, {len(walk_net.edges)}边")

    print("正在过滤自行车网络...")
    bike_net = filter_network(all_net, bike_highways)
    print(f"自行车网络: {len(bike_net.nodes)}节点, {len(bike_net.edges)}边")

    print("正在过滤驾车网络...")
    drive_net = filter_network(all_net, drive_highways)
    print(f"驾车网络: {len(drive_net.nodes)}节点, {len(drive_net.edges)}边")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    def get_all_bounds():
        bounds_list = []
        for net in (drive_net, walk_net, bike_net):
            if len(net.edges) > 0:
                nodes = ox.graph_to_gdfs(net, edges=False)
                bounds_list.append(nodes.total_bounds)
        if bounds_list:
            all_bounds = np.vstack(bounds_list)
            return (
                np.min(all_bounds[:, 0]), np.min(all_bounds[:, 1]),
                np.max(all_bounds[:, 2]), np.max(all_bounds[:, 3]),
            )
        return None

    if len(drive_net.edges) > 0:
        ox.plot_graph(drive_net, ax=ax1, node_size=0, edge_color="red", edge_linewidth=1.5, show=False)
        ax1.set_title(f"驾车网络 ({len(drive_net.edges)}条道路)", fontsize=12)
    else:
        ax1.text(0.5, 0.5, "无驾车道路数据", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("驾车网络 (无数据)")

    if len(walk_net.edges) > 0:
        ox.plot_graph(walk_net, ax=ax2, node_size=0, edge_color="green", edge_linewidth=1.5, show=False)
        ax2.set_title(f"步行网络 ({len(walk_net.edges)}条道路)", fontsize=12)
    else:
        ax2.text(0.5, 0.5, "无步行道路数据", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("步行网络 (无数据)")

    if len(bike_net.edges) > 0:
        ox.plot_graph(bike_net, ax=ax3, node_size=0, edge_color="blue", edge_linewidth=1.5, show=False)
        ax3.set_title(f"自行车网络 ({len(bike_net.edges)}条道路)", fontsize=12)
    else:
        ax3.text(0.5, 0.5, "无自行车道路数据", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("自行车网络 (无数据)")

    bounds = get_all_bounds()
    if bounds:
        xmin, ymin, xmax, ymax = bounds
        margin_x = (xmax - xmin) * 0.05
        margin_y = (ymax - ymin) * 0.05
        for ax in (ax1, ax2, ax3):
            ax.set_xlim(xmin - margin_x, xmax + margin_x)
            ax.set_ylim(ymin - margin_y, ymax + margin_y)

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.tight_layout()
    plt.show()

    if len(drive_net.edges) > 0:
        ox.save_graphml(drive_net, "nanjing_drive_net.graphml")
    if len(walk_net.edges) > 0:
        ox.save_graphml(walk_net, "nanjing_walk_net.graphml")
    if len(bike_net.edges) > 0:
        ox.save_graphml(bike_net, "nanjing_bike_net.graphml")
    print("数据保存完成")

if __name__ == "__main__":
    _download_osm_road_networks()
