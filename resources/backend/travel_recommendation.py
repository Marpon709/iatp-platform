"""
出行推荐：天地图 POI 检索 + 天地图路径规划 + 碳排放对比。
"""
from __future__ import annotations

import itertools
import math
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from shapely.geometry import Point

from transport_modes import (
    haversine_m as _haversine_m,
    simplify_route_coords as _simplify_route_coords,
    speed_ms,
)
from online_routing import has_online_routing_key, plan_online_route
from tianditu_poi import require_tianditu_api_key, search_tianditu_pois

ROUTE_SCHEME_DEFS = [
    {
        "id": "low_carbon",
        "label": "极致低碳",
        "icon": "🚴",
        "modes": ["foot", "bike"],
        "ui_mode": "bike",
    },
    {
        "id": "transit_metro",
        "label": "地铁",
        "icon": "🚇",
        "modes": ["metro"],
        "ui_mode": "metro",
    },
    {
        "id": "transit_bus",
        "label": "公交",
        "icon": "🚌",
        "modes": ["bus"],
        "ui_mode": "bus",
    },
    {
        "id": "regular",
        "label": "常规出行",
        "icon": "🚗",
        "modes": ["drive"],
        "ui_mode": "drive",
    },
]

# 优化权重含义：(碳排, 时间, 舒适性)；舒适性越高越好（评分时取反）
OPTIMIZED_SCHEME_DEFS = [
    {
        "id": "model_low_carbon",
        "label": "绿色出行",
        "icon": "♻",
        "weights": (0.5, 0.20, 0.3),
        "strategy": "low_carbon",
    },
    {
        "id": "model_time_shortest",
        "label": "最快到达",
        "icon": "⏱",
        "weights": (0.2, 0.50, 0.3),
        "strategy": "time_shortest",
    },
]

OPTIMIZED_SCHEME_IDS = frozenset(s["id"] for s in OPTIMIZED_SCHEME_DEFS)

MODE_LABELS_ZH = {
    "foot": "步行",
    "walk": "步行",
    "bike": "骑行",
    "metro": "地铁",
    "subway": "地铁",
    "bus": "公交",
    "drive": "开车",
}

# 交通方式舒适性：
# - 步行/骑行：C(d) = C0 · exp(-β · d)，d 为出行距离(km)，β>0 时随距离衰减
# - 公交/地铁/驾车：舒适性近似恒定（β=0）
MODE_COMFORT_PARAMS = {
    "foot": {"C0": 0.60, "beta": 0.60},
    "walk": {"C0": 0.60, "beta": 0.60},
    "bike": {"C0": 0.70, "beta": 0.30},
    "bus": {"C0": 0.55, "beta": 0.0},
    "metro": {"C0": 0.85, "beta": 0.0},
    "subway": {"C0": 0.85, "beta": 0.0},
    "drive": {"C0": 0.90, "beta": 0.0},
    "car": {"C0": 0.90, "beta": 0.0},
    "private_car": {"C0": 0.90, "beta": 0.0},
}
# 共享单车按单段取还计费；若接入城市/运营商实时价格，可在此替换为对应费率。
SHARED_BIKE_BASE_FARE_YUAN = 1.8
SHARED_BIKE_INCLUDED_MINUTES = 15.0
SHARED_BIKE_EXTRA_BLOCK_MINUTES = 20.0
SHARED_BIKE_EXTRA_BLOCK_FARE_YUAN = 1.5

# 公交/地铁不满足单段条件时的接驳步行方案（仍纳入原方案路径）
_WALK_LEG_SCHEME = {
    "id": "walk_leg",
    "label": "步行",
    "icon": "🚶",
    "modes": ["foot", "walk"],
    "ui_mode": "foot",
}

def _scheme_display_label(scheme_def, used_mode):
    """推荐结果展示标签：统一为实际交通方式，不使用方案营销名。"""
    mode_label = MODE_LABELS_ZH.get(used_mode, used_mode)
    if str(scheme_def.get("id", "")) in ("low_carbon", "regular"):
        return mode_label
    return str(scheme_def.get("label") or mode_label)

EAGER_SCHEME_IDS = frozenset({"low_carbon", "regular"})
TRANSIT_SCHEME_IDS = frozenset({"transit_metro", "transit_bus"})

MAX_MULTI_ITINERARIES = 2
MAX_CANDIDATES_PER_ACTIVITY_MULTI = 2
ROUGH_ITINERARY_CANDIDATES = 2
MAX_SINGLE_PLACES = 2
MIN_DISTANCE_FOR_TRANSIT_BUS_M = 2000.0
MIN_DISTANCE_FOR_TRANSIT_METRO_M = 3000.0
MIN_DISTANCE_BETWEEN_RECOMMENDED_POIS_M = 200.0
METRO_STATION_NEAR_M = 1200.0
_EARTH_RADIUS_M = 6371000.0

_LEG_PLAN_CACHE_MAX = 5000
# 天地图周边搜索半径上限（米）；出行推荐不再做 5 km 业务截断
DEFAULT_POI_SEARCH_RADIUS_M = 50_000

def _poi_search_radius_m(max_distance_km=None):
    """POI 检索半径（米）。未指定时使用天地图周边搜索上限。"""
    if max_distance_km is None:
        return float(DEFAULT_POI_SEARCH_RADIUS_M)
    return max(500.0, float(max_distance_km) * 1000.0)

class _LRULegPlanCache:
    """跨请求保留的段级路线 LRU 缓存（线程安全，供并行算路使用）。"""

    __slots__ = ("_maxsize", "_data", "_lock")

    def __init__(self, maxsize):
        self._maxsize = max(1, int(maxsize))
        self._data = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def set(self, key, value):
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)

_LEG_PLAN_CACHE = _LRULegPlanCache(_LEG_PLAN_CACHE_MAX)

# 关键词 → POI 大类，用于缩小检索范围（未命中时仍搜全部类别）
_KEYWORD_CATEGORY_HINTS = (
    ("咖啡", ("catering",)),
    ("奶茶", ("catering",)),
    ("餐", ("catering",)),
    ("饭", ("catering",)),
    ("饭店", ("catering",)),
    ("餐厅", ("catering",)),
    ("超市", ("shopping",)),
    ("商场", ("shopping",)),
    ("购物", ("shopping",)),
    ("便利店", ("shopping",)),
    ("公园", ("recreation",)),
    ("电影", ("recreation",)),
    ("健身", ("recreation",)),
    ("医院", ("medical",)),
    ("诊所", ("medical",)),
    ("药店", ("medical",)),
    ("学校", ("education",)),
    ("图书馆", ("education",)),
    ("地铁", ("transportation_station",)),
    ("公交", ("transportation_station",)),
    ("车站", ("transportation_station",)),
)

# 路线规划缓存版本：修改 _build_route_plan 逻辑后递增，避免旧缓存返回距离为 0 的结果。
_LEG_PLAN_CACHE_VERSION = 3

def _leg_plan_cache_key(scheme_id, origin_pt, dest_pt, city='南京'):
    return (
        str(scheme_id),
        str(city or '南京'),
        round(float(origin_pt[0]), 4),
        round(float(origin_pt[1]), 4),
        round(float(dest_pt[0]), 4),
        round(float(dest_pt[1]), 4),
        f"v{_LEG_PLAN_CACHE_VERSION}",
    )

def _routing_stops(stops):
    return list(stops or [])

def _resolve_keywords(activity_query=None):
    query = str(activity_query or "").strip()
    if query:
        return [query], None
    return [], None

def _infer_poi_categories_from_keywords(keywords):
    """根据活动关键词推断 POI 大类；无法推断时返回 None（搜全部）。"""
    cats = set()
    for kw in keywords or []:
        kw_l = str(kw).strip().lower()
        if not kw_l:
            continue
        for hint, categories in _KEYWORD_CATEGORY_HINTS:
            hint_l = hint.lower()
            if hint_l in kw_l or kw_l in hint_l:
                cats.update(categories)
    return cats or None

def _freeze_leg_plan(plan):
    """存入缓存前复制可变字段，避免外部修改污染缓存。"""
    frozen = dict(plan)
    frozen["route_coords"] = list(plan.get("route_coords") or [])
    frozen["route_parts"] = [dict(p) for p in (plan.get("route_parts") or [])]
    return frozen

def _clone_leg_plan(plan):
    out = dict(plan)
    out["route_coords"] = list(plan.get("route_coords") or [])
    out["route_parts"] = [dict(p) for p in (plan.get("route_parts") or [])]
    return out

def find_candidate_pois(
    origin_lon,
    origin_lat,
    keywords,
    city="南京",
    max_radius_m=None,
    limit=8,
    poi_categories=None,
):
    """天地图 POI 周边/关键词检索；以 origin 为锚点按距离排序。"""
    _ = poi_categories
    radius_m = float(max_radius_m) if max_radius_m is not None else float(DEFAULT_POI_SEARCH_RADIUS_M)
    return search_tianditu_pois(
        keywords,
        origin_lon,
        origin_lat,
        city=city,
        max_radius_m=radius_m,
        limit=max(1, int(limit)),
    )

def _poi_coord_key(poi):
    return (round(float(poi["lat"]), 5), round(float(poi["lon"]), 5))

def _merge_candidate_pois(candidate_lists, limit):
    merged = {}
    for item in candidate_lists:
        key = _poi_coord_key(item)
        if key not in merged or item["distance_m"] < merged[key]["distance_m"]:
            merged[key] = item
    ranked = sorted(merged.values(), key=lambda x: (x["distance_m"], -x.get("rating", 0)))
    return ranked[: max(1, int(limit))]

def _stop_coords(stop):
    if not stop:
        return None
    lat = stop.get("lat")
    lng = stop.get("lng")
    if lng is None:
        lng = stop.get("lon")
    try:
        if lat is None or lng is None:
            return None
        return float(lng), float(lat)
    except (TypeError, ValueError):
        return None

def _normalize_preview_stop_type(stop_type):
    st = str(stop_type or "stop").strip().lower()
    if st == "home":
        return "origin"
    return st

def _normalize_preview_stops(preview_stops, origin_lat, origin_lon, work_lat, work_lon, activity_queries=None):
    """规范化路线预览站点；未提供时按 居住→工作→活动 生成默认顺序。"""
    if preview_stops:
        normalized = []
        for stop in preview_stops:
            if not isinstance(stop, dict):
                continue
            stype = _normalize_preview_stop_type(stop.get("type"))
            normalized.append({
                **stop,
                "type": stype,
                "lat": stop.get("lat"),
                "lng": stop.get("lng", stop.get("lon")),
            })
        return normalized

    queries = [str(q).strip() for q in (activity_queries or []) if str(q).strip()]
    stops = [
        {"type": "origin", "lat": origin_lat, "lng": origin_lon, "label": "居住位置"},
    ]
    if work_lat is not None and work_lon is not None:
        stops.append({"type": "work", "lat": work_lat, "lng": work_lon, "label": "工作位置"})
    for q in queries:
        stops.append({"type": "activity", "query": q, "label": q, "lat": None, "lng": None})
    return stops

def _preview_activity_queries(preview_stops):
    out = []
    for stop in preview_stops:
        if _normalize_preview_stop_type(stop.get("type")) != "activity":
            continue
        query = str(stop.get("query") or "").strip()
        if not query:
            label = str(stop.get("label") or "").strip()
            if label.endswith("（待推荐）"):
                query = label[: -len("（待推荐）")].strip()
            else:
                query = label
        if query:
            out.append(query)
    return out

def _preview_has_fixed_anchor(preview_stops, origin_lat, origin_lon):
    """路线预览或 API 起点是否存在可用于链式 POI 检索的锚点。"""
    for stop in preview_stops or []:
        stype = _normalize_preview_stop_type(stop.get("type"))
        if stype in ("origin", "work", "activity") and _stop_coords(stop):
            return True
    return origin_lat is not None and origin_lon is not None

def _find_preview_chained_activity_candidates(
    preview_stops,
    *,
    city,
    max_radius_m,
    limit_per_activity,
    origin_lat=None,
    origin_lon=None,
):
    per_activity = []

    for i, stop in enumerate(preview_stops):
        stype = _normalize_preview_stop_type(stop.get("type"))
        if stype != "activity":
            continue

        query = str(stop.get("query") or "").strip()
        if not query:
            label = str(stop.get("label") or "").strip()
            if label.endswith("（待推荐）"):
                query = label[: -len("（待推荐）")].strip()
            else:
                query = label
        if not query:
            return None, "路线预览", ""

        anchor_points = []
        scope_label = "上一站"
        for j in range(i - 1, -1, -1):
            prev = preview_stops[j]
            prev_type = _normalize_preview_stop_type(prev.get("type"))
            if prev_type == "activity":
                if not per_activity:
                    return None, "路线预览", query
                anchor_points = [
                    (float(c["lon"]), float(c["lat"]))
                    for c in per_activity[-1]["candidates"]
                ]
                scope_label = "上一活动地点"
                break
            prev_coords = _stop_coords(prev)
            if prev_coords and prev_type in ("origin", "work"):
                anchor_points = [prev_coords]
                scope_label = {
                    "origin": "居住位置",
                    "work": "工作位置",
                }.get(prev_type, "上一站")
                break

        if not anchor_points:
            if origin_lat is not None and origin_lon is not None:
                anchor_points = [(float(origin_lon), float(origin_lat))]
                scope_label = "出发活动"
            else:
                return None, "路线预览", query

        fixed_coords = _stop_coords(stop)
        if fixed_coords:
            slon, slat = fixed_coords
            candidates = [{
                "display_name": stop.get("label") or stop.get("name") or query,
                "lat": slat,
                "lon": slon,
                "distance_m": 0,
                "distance_km": 0.0,
                "rating": 4.0,
                "poi_category": "",
                "poi_subtype": "",
                "category_label": "",
            }]
        else:
            keywords, poi_cats = _resolve_keywords(query)
            batch = []
            for slon, slat in anchor_points:
                batch.extend(
                    find_candidate_pois(
                        slon,
                        slat,
                        keywords or [query],
                        city=city,
                        max_radius_m=max_radius_m,
                        limit=limit_per_activity,
                        poi_categories=poi_cats,
                    )
                )
            candidates = _merge_candidate_pois(batch, limit_per_activity)

        if not candidates:
            return None, scope_label, query
        per_activity.append({
            "query": query,
            "candidates": candidates,
            "preview_index": i,
            "anchor_label": scope_label,
        })

    if not per_activity:
        return None, "路线预览", ""
    return per_activity, None, None

def _combo_respects_preview_chain(preview_stops, combo_pois, max_radius_m=None, origin_lat=None, origin_lon=None):
    """校验组合 POI 数量与预览活动数一致（不再限制相邻活动间距）。"""
    _ = (max_radius_m, origin_lat, origin_lon)
    poi_iter = iter(combo_pois)
    for stop in preview_stops:
        if _normalize_preview_stop_type(stop.get("type")) != "activity":
            continue
        try:
            next(poi_iter)
        except StopIteration:
            return False
    return True

def _build_stops_from_preview_and_combo(preview_stops, combo_pois, origin_stop, work_stop):
    """按路线预览顺序组装站点；combo 与预览中每个活动位一一对应（含已选点活动）。"""
    poi_iter = iter(combo_pois)
    seq = []
    for stop in preview_stops:
        stype = _normalize_preview_stop_type(stop.get("type"))
        if stype == "origin":
            seq.append(origin_stop)
            continue
        if stype == "work":
            if work_stop:
                seq.append(work_stop)
            continue
        if stype != "activity":
            continue

        query = str(stop.get("query") or "").strip()
        try:
            poi = next(poi_iter)
        except StopIteration:
            break

        coords = _stop_coords(stop)
        if coords:
            start_time, end_time = _stop_time_fields(stop)
            extra = {
                "name": stop.get("name"),
                "query": query or None,
            }
            if start_time is not None:
                extra["start_time"] = start_time
            if end_time is not None:
                extra["end_time"] = end_time
            seq.append(_stop_dict(
                "activity",
                stop.get("label") or stop.get("name") or query or "活动地点",
                coords[1],
                coords[0],
                **{k: v for k, v in extra.items() if v is not None},
            ))
        else:
            activity_stop = _poi_to_activity_stop(poi, query)
            start_time, end_time = _stop_time_fields(stop)
            if start_time is not None:
                activity_stop["start_time"] = start_time
            if end_time is not None:
                activity_stop["end_time"] = end_time
            seq.append(activity_stop)
    return seq

def _pick_spread_activity_pois(candidates, count, min_separation_m=None):
    """从候选 POI 中选取彼此间距足够的若干活动地点（方案间活动地点应不同）。"""
    if not candidates:
        return []
    min_sep = float(min_separation_m or MIN_DISTANCE_BETWEEN_RECOMMENDED_POIS_M)
    need = max(1, int(count))

    def _coord_key(poi):
        return (round(float(poi["lat"]), 5), round(float(poi["lon"]), 5))

    first = candidates[0]
    selected = [first]
    selected_keys = {_coord_key(first)}

    for poi in candidates[1:]:
        if len(selected) >= need:
            break
        key = _coord_key(poi)
        if key in selected_keys:
            continue
        lat_a, lon_a = float(poi["lat"]), float(poi["lon"])
        if any(
            _haversine_m(lon_a, lat_a, float(ch["lon"]), float(ch["lat"])) < min_sep
            for ch in selected
        ):
            continue
        selected.append(poi)
        selected_keys.add(key)

    if len(selected) < need:
        lon0, lat0 = float(first["lon"]), float(first["lat"])
        rest = sorted(
            candidates,
            key=lambda p: _haversine_m(float(p["lon"]), float(p["lat"]), lon0, lat0),
            reverse=True,
        )
        for poi in rest:
            if len(selected) >= need:
                break
            key = _coord_key(poi)
            if key in selected_keys:
                continue
            selected.append(poi)
            selected_keys.add(key)

    return selected[:need]

def _activity_locations_key_from_stops(stops):
    return tuple(sorted(
        (
            round(float(s["lat"]), 5),
            round(float(s["lon"]), 5),
            str(s.get("query") or s.get("name") or ""),
        )
        for s in (stops or [])
        if s.get("type") == "activity"
    ))

def _activity_locations_key_from_rec(rec):
    places = rec.get("places")
    if places:
        return tuple(sorted(
            (
                round(float(p["lat"]), 5),
                round(float(p["lon"]), 5),
                str(p.get("query") or p.get("name") or ""),
            )
            for p in places
        ))
    return _activity_locations_key_from_stops(rec.get("stops"))

def _dedupe_recommendations_by_activity(recommendations):
    seen = set()
    unique = []
    for rec in recommendations or []:
        key = _activity_locations_key_from_rec(rec)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(rec)
    return unique

TRANSIT_RIDE_MODES = frozenset({"metro", "bus"})

def _should_offer_metro(_calc, p_from, p_to, dest_distance_m):
    """是否推荐地铁：出行距离达到阈值即可（由天地图 API 实际规划）。"""
    dest_m = float(dest_distance_m)
    if dest_m <= 0:
        dest_m = _haversine_m(p_from.x, p_from.y, p_to.x, p_to.y)
    return dest_m >= MIN_DISTANCE_FOR_TRANSIT_METRO_M

def _should_offer_bus(_calc, p_from, p_to, dest_distance_m):
    """是否推荐公交：出行距离达到阈值即可（由天地图 API 实际规划）。"""
    dest_m = float(dest_distance_m)
    if dest_m <= 0:
        dest_m = _haversine_m(p_from.x, p_from.y, p_to.x, p_to.y)
    return dest_m >= MIN_DISTANCE_FOR_TRANSIT_BUS_M

def _stops_need_transit_bus(calc, stops):
    routing = _routing_stops(stops)
    for i in range(len(routing) - 1):
        a, b = routing[i], routing[i + 1]
        p_from = Point(float(a["lon"]), float(a["lat"]))
        p_to = Point(float(b["lon"]), float(b["lat"]))
        leg_m = _haversine_m(p_from.x, p_from.y, p_to.x, p_to.y)
        if _should_offer_bus(calc, p_from, p_to, leg_m):
            return True
    return False

def _stops_need_transit_metro(calc, stops):
    routing = _routing_stops(stops)
    for i in range(len(routing) - 1):
        a, b = routing[i], routing[i + 1]
        p_from = Point(float(a["lon"]), float(a["lat"]))
        p_to = Point(float(b["lon"]), float(b["lat"]))
        leg_m = _haversine_m(p_from.x, p_from.y, p_to.x, p_to.y)
        if _should_offer_metro(calc, p_from, p_to, leg_m):
            return True
    return False

def _metro_eligible_for_itinerary(calc, stops):
    """
    串联行程全程>3km，且至少一段可乘地铁时纳入地铁方案（与公交判定逻辑对齐）。
    """
    routing = _routing_stops(stops)
    if len(routing) < 2:
        return False
    if _sequence_haversine_m(routing) < MIN_DISTANCE_FOR_TRANSIT_METRO_M:
        return False
    return _stops_need_transit_metro(calc, stops)

def _scheme_to_tianditu_travel_mode(scheme_def, origin_pt, dest_pt):
    """出行方案 → 天地图路径规划 travel_mode。"""
    scheme_id = str(scheme_def.get("id") or "")
    modes = list(scheme_def.get("modes") or [])
    straight_m = _haversine_m(origin_pt[0], origin_pt[1], dest_pt[0], dest_pt[1])
    if scheme_id == "low_carbon":
        if straight_m > 2500 and "bike" in modes:
            return "bike"
        if straight_m < 800 and "foot" in modes:
            return "foot"
        return "bike" if "bike" in modes else "foot"
    if scheme_id == "regular":
        return "drive"
    if scheme_id == "transit_metro":
        return "metro"
    if scheme_id == "transit_bus":
        return "bus"
    ui_mode = scheme_def.get("ui_mode") or "walk"
    if ui_mode in ("foot", "walk"):
        return "walk"
    return ui_mode

def _resolve_used_mode(scheme_def, travel_mode):
    scheme_id = str(scheme_def.get("id") or "")
    if scheme_id.startswith("transit_"):
        return scheme_def.get("ui_mode") or travel_mode
    if scheme_id == "low_carbon":
        return "foot" if travel_mode in ("foot", "walk") else "bike"
    if travel_mode == "drive":
        return "drive"
    if travel_mode == "bike":
        return "bike"
    if travel_mode in ("foot", "walk"):
        return "foot"
    return travel_mode

def _icb_at_point(icb_raster, transform, lat, lng):
    import numpy as np

    if icb_raster is None or transform is None:
        return None
    a, b, c, d, e, f = (
        transform.a, transform.b, transform.c,
        transform.d, transform.e, transform.f,
    )
    col = (lng - c) / a
    row = (lat - f) / e
    col_int = int(round(col))
    row_int = int(round(row))
    rows, cols = icb_raster.shape
    if 0 <= row_int < rows and 0 <= col_int < cols:
        value = float(icb_raster[row_int, col_int])
        if np.isfinite(value) and value > 0:
            return value
    return None

def _scheme_defs_eager():
    return [s for s in ROUTE_SCHEME_DEFS if s.get("id") in EAGER_SCHEME_IDS]

def _scheme_defs_lazy_transit(stops, calc=None):
    """可懒加载的公交/地铁方案（按直线距离判断，实际路线由天地图 API 计算）。"""
    _ = calc
    leg_count = max(0, len(_routing_stops(stops)) - 1)
    lazy = []

    if leg_count <= 1:
        routing = _routing_stops(stops)
        if len(routing) >= 2:
            a, b = routing[0], routing[-1]
            p_from = Point(float(a["lon"]), float(a["lat"]))
            p_to = Point(float(b["lon"]), float(b["lat"]))
            leg_m = _haversine_m(p_from.x, p_from.y, p_to.x, p_to.y)
            if _should_offer_metro(None, p_from, p_to, leg_m):
                metro_def = next(
                    (s for s in ROUTE_SCHEME_DEFS if s.get("id") == "transit_metro"),
                    None,
                )
                if metro_def:
                    lazy.append(metro_def)
            if _should_offer_bus(None, p_from, p_to, leg_m):
                bus_def = next(
                    (s for s in ROUTE_SCHEME_DEFS if s.get("id") == "transit_bus"),
                    None,
                )
                if bus_def:
                    lazy.append(bus_def)
        return lazy

    if _stops_need_transit_bus(None, stops):
        bus_def = next(
            (s for s in ROUTE_SCHEME_DEFS if s.get("id") == "transit_bus"),
            None,
        )
        if bus_def:
            lazy.append(bus_def)
    if _metro_eligible_for_itinerary(None, stops):
        metro_def = next(
            (s for s in ROUTE_SCHEME_DEFS if s.get("id") == "transit_metro"),
            None,
        )
        if metro_def:
            lazy.append(metro_def)
    return lazy

def _lazy_transit_placeholder(sdef):
    label = str(sdef.get("label") or "")
    ui_mode = sdef.get("ui_mode") or ""
    return {
        "scheme_id": sdef["id"],
        "scheme_label": label,
        "icon": sdef.get("icon", ""),
        "mode": ui_mode,
        "mode_label": label,
        "lazy": True,
        "distance_m": None,
        "distance_km": None,
        "time_min": None,
        "carbon_g": None,
        "route_coords": [],
        "route_parts": [],
        "legs": [],
        "leg_count": 0,
        "route_point_count": 0,
        "uses_transit_line": str(sdef.get("id", "")).startswith("transit_"),
    }

def _lazy_optimized_placeholder(sdef):
    label = str(sdef.get("label") or "")
    return {
        "scheme_id": sdef["id"],
        "scheme_label": label,
        "icon": sdef.get("icon", ""),
        "mode": "optimized",
        "mode_label": label,
        "lazy": True,
        "distance_m": None,
        "distance_km": None,
        "time_min": None,
        "carbon_g": None,
        "cost_yuan": None,
        "comfort": None,
        "route_coords": [],
        "route_parts": [],
        "legs": [],
        "leg_count": 0,
        "route_point_count": 0,
        "uses_transit_line": False,
    }

def _merge_routes_ordered(eager_plans, lazy_placeholders):
    by_id = {}
    for plan in eager_plans:
        by_id[plan["scheme_id"]] = plan
    for placeholder in lazy_placeholders:
        sid = placeholder.get("scheme_id")
        if sid and sid not in by_id:
            by_id[sid] = placeholder
    return [by_id[s["id"]] for s in ROUTE_SCHEME_DEFS if s.get("id") in by_id]

def _build_itinerary_routes(stops, city="南京", fast_road=True, carbon_budget_g=None):
    plans = _build_optimized_route_plans(
        stops,
        carbon_budget_g=carbon_budget_g,
        city=city,
        fast_road=bool(fast_road),
    )
    by_id = {p["scheme_id"]: p for p in plans}
    ordered = [by_id[s["id"]] for s in OPTIMIZED_SCHEME_DEFS if s["id"] in by_id]
    return ordered, ordered

def _build_route_plan(origin_pt, dest_pt, scheme_def, dist_m_fallback, city="南京", fast_road=False):
    """单段路线：天地图失败则高德，再失败返回 None（不回退本地路网）。"""
    _ = dist_m_fallback, fast_road
    if not has_online_routing_key():
        return None
    cache_key = _leg_plan_cache_key(scheme_def.get("id"), origin_pt, dest_pt, city)
    cached = _LEG_PLAN_CACHE.get(cache_key)
    if cached is not None:
        return _clone_leg_plan(cached)

    from carbon_budget import CARBON_ALPHA_G_PER_KM

    travel_mode = _scheme_to_tianditu_travel_mode(scheme_def, origin_pt, dest_pt)
    tdt_plan = plan_online_route(
        float(origin_pt[0]), float(origin_pt[1]),
        float(dest_pt[0]), float(dest_pt[1]),
        travel_mode,
        city=city,
    )
    if not tdt_plan or not tdt_plan.get("route_coords") or len(tdt_plan["route_coords"]) < 2:
        return None

    route_coords = list(tdt_plan["route_coords"])
    if len(route_coords) > 500:
        route_coords = _simplify_route_coords(route_coords, max_points=500)

    route_parts = [dict(p) for p in (tdt_plan.get("route_parts") or [])]
    if not route_parts:
        profile = tdt_plan.get("profile") or "walking"
        geometry = "transit_ride" if profile == "transit" else "road"
        route_parts = [{
            "type": travel_mode,
            "geometry": geometry,
            "coords": route_coords,
        }]

    used_mode = _resolve_used_mode(scheme_def, travel_mode)
    is_transit_scheme = str(scheme_def.get("id", "")).startswith("transit_")
    dist_m = float(tdt_plan.get("distance_m") or 0)
    # 从路线坐标反算距离作为基准，若 API 返回的距离为 0 或明显不合理则覆盖
    # （同时防御旧缓存返回 distance=0 的结果）
    if route_coords and len(route_coords) >= 2:
        from transport_modes import path_length_from_coords
        coord_dist_m = float(path_length_from_coords(route_coords) or 0)
        if coord_dist_m > 0 and (dist_m <= 0 or dist_m < coord_dist_m * 0.05):
            dist_m = coord_dist_m
    if dist_m > 0:
        time_min = dist_m / max(speed_ms(used_mode), 0.1) / 60.0
    else:
        time_min = float(tdt_plan.get("duration_min") or 0)

    uses_transit_line = bool(tdt_plan.get("uses_transit_line")) or (
        is_transit_scheme and any(
            part.get("geometry") == "transit_ride" and len(part.get("coords") or []) >= 2
            for part in route_parts
        )
    )

    alpha = float(CARBON_ALPHA_G_PER_KM.get(
        used_mode if used_mode in CARBON_ALPHA_G_PER_KM else "bus",
        CARBON_ALPHA_G_PER_KM["bus"],
    ))
    dist_km = dist_m / 1000.0
    carbon_g = dist_km * alpha
    mode_label = MODE_LABELS_ZH.get(used_mode, used_mode)
    result = {
        "scheme_id": scheme_def["id"],
        "scheme_label": _scheme_display_label(scheme_def, used_mode),
        "icon": scheme_def["icon"],
        "mode": used_mode,
        "mode_label": mode_label,
        "distance_m": round(dist_m),
        "distance_km": round(dist_km, 2),
        "time_min": round(time_min, 1),
        "carbon_g": round(carbon_g, 1),
        "cost_yuan": round(_estimate_mode_cost_yuan(used_mode, dist_km, time_min), 2),
        "alpha_g_per_km": alpha,
        "route_coords": route_coords,
        "route_parts": route_parts,
        "route_point_count": len(route_coords),
        "uses_transit_line": uses_transit_line,
        "route_source": tdt_plan.get("source") or "tianditu",
        "walk_fallback": bool(tdt_plan.get("walk_fallback")),
        "lazy": False,
    }
    _LEG_PLAN_CACHE.set(cache_key, _freeze_leg_plan(result))
    return result

def _comparison_copy(best_plan, drive_plan, remaining_g, activity_count=1):
    saved_g = max(0.0, float(drive_plan["carbon_g"]) - float(best_plan["carbon_g"]))
    drive_g = float(drive_plan["carbon_g"])
    saved_pct = (saved_g / drive_g * 100.0) if drive_g > 1e-6 else 0.0
    saved_kg = saved_g / 1000.0
    remaining_kg = max(0.0, float(remaining_g or 0)) / 1000.0
    after_kg = max(0.0, remaining_kg - best_plan["carbon_g"] / 1000.0)
    if int(activity_count or 1) > 1:
        action = f"完成全部 {int(activity_count)} 项活动"
    else:
        action = "前往该地点"
    return {
        "message": (
            f"选择{best_plan['mode_label']}{action}，相对减少 {saved_pct:.0f}% 碳排放，"
            f"为您节省 {saved_kg:.2f} kg 碳预算！"
        ),
        "saved_g": round(saved_g, 1),
        "saved_pct": round(saved_pct, 1),
        "saved_kg": round(saved_kg, 3),
        "remaining_kg": round(remaining_kg, 3),
        "remaining_after_kg": round(after_kg, 3),
    }

def _merge_route_coords(existing, new_coords):
    if not new_coords:
        return existing
    if not existing:
        return list(new_coords)
    merged = list(existing)
    if merged and new_coords and merged[-1] == new_coords[0]:
        merged.extend(new_coords[1:])
    else:
        merged.extend(new_coords)
    return merged

def _stop_dict(stop_type, label, lat, lon, **extra):
    item = {
        "type": stop_type,
        "label": label,
        "lat": float(lat),
        "lon": float(lon),
    }
    item.update(extra)
    return item

def _origin_stop(origin_lat, origin_lon, label="居住位置"):
    return _stop_dict("origin", label, origin_lat, origin_lon)

def _poi_to_activity_stop(poi, query):
    return _stop_dict(
        "activity",
        f"{query}：{poi['display_name']}",
        poi["lat"],
        poi["lon"],
        query=query,
        name=poi["display_name"],
        distance_km=poi.get("distance_km"),
        distance_m=poi.get("distance_m"),
        rating=poi.get("rating"),
        poi_category=poi.get("poi_category"),
        poi_subtype=poi.get("poi_subtype"),
    )

def _sequence_haversine_m(stops):
    total = 0.0
    for i in range(len(stops) - 1):
        a, b = stops[i], stops[i + 1]
        total += _haversine_m(a["lon"], a["lat"], b["lon"], b["lat"])
    return total

def _generate_stop_sequences(origin_stop, work_stop, activity_stops):
    """生成候选站点顺序：从居住点出发，活动顺序可排列，工作点可选前/后。"""
    if not activity_stops:
        seq = [origin_stop]
        if work_stop:
            seq.append(work_stop)
        return [seq]

    sequences = []
    for perm in itertools.permutations(activity_stops):
        ordered = list(perm)
        if work_stop:
            sequences.append([origin_stop, work_stop, *ordered])
            sequences.append([origin_stop, *ordered, work_stop])
        else:
            sequences.append([origin_stop, *ordered])

    seen = set()
    unique = []
    for seq in sequences:
        key = tuple((s["lat"], s["lon"], s.get("type")) for s in seq)
        if key in seen:
            continue
        seen.add(key)
        unique.append(seq)
    return unique

def _transit_leg_eligible(calc, p_from, p_to, straight_m, scheme_id):
    scheme_id = str(scheme_id or "")
    if scheme_id == "transit_metro":
        return _should_offer_metro(calc, p_from, p_to, straight_m)
    if scheme_id == "transit_bus":
        return _should_offer_bus(calc, p_from, p_to, straight_m)
    return True

def _transit_leg_has_ride(leg_plan):
    if not leg_plan:
        return False
    for part in leg_plan.get("route_parts") or []:
        if part.get("geometry") == "transit_ride":
            coords = part.get("coords") or []
            if len(coords) >= 2:
                return True
    return False

def _get_route_calculator(city="南京"):
    try:
        from city_engine import get_shared_calculator
        return get_shared_calculator(city=str(city or "南京").strip() or "南京")
    except Exception:
        return None

def _leg_plan_from_local_transit(origin_pt, dest_pt, scheme_def, city="南京"):
    """天地图公交/地铁不可用时，回退本地线网规划单段。"""
    scheme_id = str(scheme_def.get("id") or "")
    if not scheme_id.startswith("transit_"):
        return None
    mode = "metro" if scheme_id == "transit_metro" else "bus"
    calc = _get_route_calculator(city)
    if calc is None:
        return None

    from carbon_budget import CARBON_ALPHA_G_PER_KM
    from transport_modes import build_transit_path_geometry, path_length_from_coords

    p_from = Point(float(origin_pt[0]), float(origin_pt[1]))
    p_to = Point(float(dest_pt[0]), float(dest_pt[1]))
    geom = build_transit_path_geometry(calc, p_from, p_to, mode)
    if not geom:
        return None

    parts = [
        dict(part)
        for part in (geom.get("parts") or [])
        if len(part.get("coords") or []) >= 2
    ]
    if not _transit_leg_has_ride({"route_parts": parts}):
        return None

    route_coords = list(geom.get("merged") or [])
    if len(route_coords) < 2:
        return None

    dist_m = float(geom.get("total_distance_m") or 0)
    if dist_m <= 0:
        dist_m = float(path_length_from_coords(route_coords) or 0)
    if dist_m <= 0:
        return None

    used_mode = scheme_def.get("ui_mode") or mode
    time_min = dist_m / max(speed_ms(used_mode), 0.1) / 60.0
    alpha = float(CARBON_ALPHA_G_PER_KM.get(
        used_mode if used_mode in CARBON_ALPHA_G_PER_KM else "bus",
        CARBON_ALPHA_G_PER_KM["bus"],
    ))
    dist_km = dist_m / 1000.0
    return {
        "scheme_id": scheme_def["id"],
        "scheme_label": _scheme_display_label(scheme_def, used_mode),
        "icon": scheme_def.get("icon", ""),
        "mode": used_mode,
        "mode_label": MODE_LABELS_ZH.get(used_mode, used_mode),
        "distance_m": round(dist_m),
        "distance_km": round(dist_km, 2),
        "time_min": round(time_min, 1),
        "carbon_g": round(dist_km * alpha, 1),
        "cost_yuan": round(_estimate_mode_cost_yuan(used_mode, dist_km, time_min), 2),
        "alpha_g_per_km": alpha,
        "route_coords": route_coords,
        "route_parts": parts,
        "route_point_count": len(route_coords),
        "uses_transit_line": True,
        "route_source": "local_network",
        "walk_fallback": False,
        "lazy": False,
    }

def _build_walk_leg_plan(origin_pt, dest_pt, straight_m, city="南京", fast_road=False):
    """公交/地铁单段不可乘运时的接驳步行。"""
    leg_plan = _build_route_plan(
        origin_pt,
        dest_pt,
        _WALK_LEG_SCHEME,
        straight_m,
        city=city,
        fast_road=fast_road,
    )
    if not leg_plan:
        return None
    leg_plan = dict(leg_plan)
    leg_plan["walk_fallback"] = True
    return leg_plan

def _transit_leg_walk_access_m(leg_plan):
    """从公交/地铁方案中提取接驳步行总距离（米），用于比较方案优劣。"""
    if not leg_plan:
        return -1.0
    parts = leg_plan.get('route_parts') or []
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

def _build_leg_plan_for_scheme(a, b, scheme_def, city="南京", fast_road=False):
    """单段路线：公交/地铁不可乘运时该段改步行，仍纳入原方案路径。"""
    origin_pt = (float(a["lon"]), float(a["lat"]))
    dest_pt = (float(b["lon"]), float(b["lat"]))
    straight_m = _haversine_m(origin_pt[0], origin_pt[1], dest_pt[0], dest_pt[1])
    scheme_id = str(scheme_def.get("id") or "")

    if not scheme_id.startswith("transit_"):
        leg_plan = _build_route_plan(
            origin_pt, dest_pt, scheme_def, straight_m, city=city, fast_road=fast_road,
        )
        if leg_plan:
            return leg_plan
        if scheme_id in ("low_carbon", "regular", "walk_leg"):
            local_plan = _leg_plan_from_local_road(origin_pt, dest_pt, scheme_def, city=city)
            if local_plan:
                return local_plan
        if scheme_id == "low_carbon":
            return _build_walk_leg_plan(origin_pt, dest_pt, straight_m, city=city, fast_road=fast_road)
        return None

    tdt_leg = _build_route_plan(
        origin_pt, dest_pt, scheme_def, straight_m, city=city, fast_road=fast_road,
    )
    tdt_usable = bool(tdt_leg and _transit_leg_has_ride(tdt_leg))

    # 地铁模式：同时计算本地线网方案，与天地图比较接驳步行距离
    local_leg = None
    if scheme_id == 'transit_metro':
        local_leg = _leg_plan_from_local_transit(origin_pt, dest_pt, scheme_def, city=city)

    if tdt_usable and local_leg:
        tdt_walk = _transit_leg_walk_access_m(tdt_leg)
        local_walk = _transit_leg_walk_access_m(local_leg)
        if local_walk >= 0 and local_walk < tdt_walk:
            return local_leg
        return tdt_leg

    if tdt_usable:
        return tdt_leg

    if local_leg:
        return local_leg

    # 非地铁公交模式：尝试本地线网
    if scheme_id != 'transit_metro':
        leg_plan = _leg_plan_from_local_transit(origin_pt, dest_pt, scheme_def, city=city)
        if leg_plan:
            return leg_plan

    return _build_walk_leg_plan(origin_pt, dest_pt, straight_m, city=city, fast_road=fast_road)

def _leg_plan_from_local_road(origin_pt, dest_pt, scheme_def, city="南京"):
    """天地图不可用时，回退本地路网规划单段（步行/骑行/驾车）。"""
    scheme_id = str(scheme_def.get("id") or "")
    ui_mode = scheme_def.get("ui_mode") or "foot"
    if scheme_id == "regular":
        travel_mode = "drive"
    elif scheme_id == "low_carbon":
        travel_mode = "bike"
    else:
        travel_mode = ui_mode if ui_mode in ("foot", "walk", "bike", "drive") else "foot"

    calc = _get_route_calculator(city)
    if calc is None:
        return None

    from carbon_budget import CARBON_ALPHA_G_PER_KM
    from transport_modes import get_route_geometry_coords, get_route_distance, path_length_from_coords

    p_from = Point(float(origin_pt[0]), float(origin_pt[1]))
    p_to = Point(float(dest_pt[0]), float(dest_pt[1]))
    commute_mode = travel_mode
    if commute_mode in ("walk", "foot"):
        commute_mode = "foot"

    route_coords = get_route_geometry_coords(calc, p_from, p_to, commute_mode)
    if not route_coords or len(route_coords) < 2:
        return None

    dist_m = float(get_route_distance(calc, p_from, p_to, commute_mode) or 0)
    if dist_m <= 0:
        dist_m = float(path_length_from_coords(route_coords) or 0)
    if dist_m <= 0:
        return None

    used_mode = ui_mode if ui_mode in ("foot", "walk", "bike", "drive") else travel_mode
    if used_mode in ("walk", "foot"):
        used_mode = "foot"
    time_min = dist_m / max(speed_ms(used_mode), 0.1) / 60.0
    alpha = float(CARBON_ALPHA_G_PER_KM.get(
        used_mode if used_mode in CARBON_ALPHA_G_PER_KM else "foot",
        CARBON_ALPHA_G_PER_KM["foot"],
    ))
    dist_km = dist_m / 1000.0
    return {
        "scheme_id": scheme_def["id"],
        "scheme_label": _scheme_display_label(scheme_def, used_mode),
        "icon": scheme_def.get("icon", ""),
        "mode": used_mode,
        "mode_label": MODE_LABELS_ZH.get(used_mode, used_mode),
        "distance_m": round(dist_m),
        "distance_km": round(dist_km, 2),
        "time_min": round(time_min, 1),
        "carbon_g": round(dist_km * alpha, 1),
        "cost_yuan": round(_estimate_mode_cost_yuan(used_mode, dist_km, time_min), 2),
        "alpha_g_per_km": alpha,
        "route_coords": route_coords,
        "route_parts": [{
            "type": used_mode,
            "geometry": "road",
            "coords": route_coords,
        }],
        "route_point_count": len(route_coords),
        "uses_transit_line": False,
        "route_source": "local_network",
        "walk_fallback": False,
        "lazy": False,
    }

def _build_multi_stop_route_plan(stops, scheme_def, city="南京", fast_road=False):
    """按站点顺序串联多段路线，返回完整路径与各段摘要（天地图 API）。"""
    if len(stops) < 2:
        return None

    total_dist_m = 0.0
    total_time_min = 0.0
    total_carbon_g = 0.0
    total_cost_yuan = 0.0
    merged_coords = []
    merged_parts = []
    legs = []
    used_mode = scheme_def.get("ui_mode")

    for i in range(len(stops) - 1):
        a, b = stops[i], stops[i + 1]
        leg_plan = _build_leg_plan_for_scheme(a, b, scheme_def, city=city, fast_road=fast_road)
        if not leg_plan:
            return None
        total_dist_m += float(leg_plan["distance_m"])
        total_time_min += float(leg_plan["time_min"])
        total_carbon_g += float(leg_plan["carbon_g"])
        leg_cost_yuan = leg_plan.get("cost_yuan")
        if leg_cost_yuan is None:
            leg_cost_yuan = _estimate_mode_cost_yuan(
                leg_plan.get("mode"),
                leg_plan.get("distance_km"),
                leg_plan.get("time_min"),
            )
        total_cost_yuan += float(leg_cost_yuan)
        merged_coords = _merge_route_coords(merged_coords, leg_plan.get("route_coords") or [])

        for part in leg_plan.get("route_parts") or []:
            part_copy = dict(part)
            part_copy["leg_index"] = i
            part_copy["from_label"] = a.get("label", "")
            part_copy["to_label"] = b.get("label", "")
            merged_parts.append(part_copy)

        legs.append({
            "index": i,
            "from_label": a.get("label", ""),
            "to_label": b.get("label", ""),
            "distance_km": leg_plan.get("distance_km"),
            "time_min": leg_plan.get("time_min"),
            "carbon_g": leg_plan.get("carbon_g"),
            "cost_yuan": round(float(leg_cost_yuan), 2),
            "mode_label": leg_plan.get("mode_label"),
            "route_coords": leg_plan.get("route_coords"),
            "route_parts": leg_plan.get("route_parts"),
            "route_source": leg_plan.get("route_source"),
            "walk_fallback": leg_plan.get("walk_fallback"),
        })
        used_mode = leg_plan.get("mode") or used_mode

    dist_km = total_dist_m / 1000.0
    scheme_ui_mode = scheme_def.get("ui_mode") or used_mode
    if str(scheme_def.get("id", "")).startswith("transit_"):
        display_mode = scheme_ui_mode
    else:
        display_mode = used_mode
    mode_label = MODE_LABELS_ZH.get(display_mode, display_mode)
    uses_transit_line = any(
        part.get("geometry") == "transit_ride" and len(part.get("coords") or []) >= 2
        for part in merged_parts
    )
    if str(scheme_def.get("id", "")).startswith("transit_") and not uses_transit_line:
        return None

    route_sources = {
        leg.get("route_source")
        for leg in legs
        if leg.get("route_source")
    }
    if route_sources == {"local_network"}:
        route_source = "local_network"
    elif "local_network" in route_sources:
        route_source = "mixed"
    elif route_sources == {"amap"}:
        route_source = "amap"
    elif "amap" in route_sources and "tianditu" in route_sources:
        route_source = "mixed"
    elif "amap" in route_sources:
        route_source = "amap"
    else:
        route_source = next(iter(route_sources), "tianditu")

    return {
        "scheme_id": scheme_def["id"],
        "scheme_label": _scheme_display_label(scheme_def, display_mode),
        "icon": scheme_def["icon"],
        "mode": display_mode,
        "mode_label": mode_label,
        "distance_m": round(total_dist_m),
        "distance_km": round(dist_km, 2),
        "time_min": round(total_time_min, 1),
        "carbon_g": round(total_carbon_g, 1),
        "cost_yuan": round(total_cost_yuan, 2),
        "route_coords": merged_coords,
        "route_parts": merged_parts,
        "legs": legs,
        "leg_count": len(legs),
        "route_point_count": len(merged_coords),
        "uses_transit_line": uses_transit_line,
        "route_source": route_source,
        "lazy": False,
    }

def _optimized_scheme_def(scheme_id):
    return next((s for s in OPTIMIZED_SCHEME_DEFS if s["id"] == str(scheme_id or "")), None)

def _mode_comfort(mode, distance_km=None):
    """
    交通方式舒适性得分（越高越舒适）。
    步行/骑行：C(d) = C0 · e^(-β·d)，d 单位 km；
    其他方式：取常数 C0（β=0）。
    """
    m = str(mode or "foot").strip().lower()
    params = MODE_COMFORT_PARAMS.get(m, {"C0": 0.5, "beta": 0.0})
    c0 = float(params.get("C0", 0.5))
    beta = float(params.get("beta", 0.0))
    if beta <= 0 or distance_km is None:
        return max(0.0, c0)
    d_km = max(0.0, float(distance_km or 0.0))
    return max(0.0, c0 * math.exp(-beta * d_km))

def _estimate_mode_cost_yuan(mode, distance_km, time_min=None):
    """Estimate direct out-of-pocket travel cost for optimization objective."""
    m = str(mode or "foot").strip().lower()
    km = max(0.0, float(distance_km or 0.0))
    if m in ("foot", "walk"):
        return 0.0
    if m == "bike":
        minutes = max(0.0, float(time_min or 0.0))
        if minutes <= SHARED_BIKE_INCLUDED_MINUTES:
            return SHARED_BIKE_BASE_FARE_YUAN
        extra_blocks = math.ceil(
            (minutes - SHARED_BIKE_INCLUDED_MINUTES) / SHARED_BIKE_EXTRA_BLOCK_MINUTES
        )
        return SHARED_BIKE_BASE_FARE_YUAN + extra_blocks * SHARED_BIKE_EXTRA_BLOCK_FARE_YUAN
    if m == "bus":
        return 2.0
    if m == "metro":
        if km <= 4:
            return 2.0
        if km <= 9:
            return 3.0
        if km <= 14:
            return 4.0
        if km <= 21:
            return 5.0
        if km <= 28:
            return 6.0
        return 6.0 + math.ceil((km - 28.0) / 9.0)
    if m in ("drive", "car", "private_car"):
        return km * 1.45 + 3.26
    return km * 1.0

def _leg_candidate_sort_key(candidate):
    plan = candidate["plan"]
    return (
        float(plan.get("carbon_g") or 0),
        float(plan.get("time_min") or 0),
        float(candidate.get("cost_yuan") or 0),
    )

def _parse_hhmm_to_minutes(value):
    """解析 HH:MM / H:MM 为当日分钟数；无法解析返回 None。"""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) < 2:
        return None
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except (TypeError, ValueError):
        return None
    if not (0 <= hour < 24 and 0 <= minute < 60):
        return None
    return hour * 60 + minute

def _stop_time_fields(stop):
    """从站点字典提取 start/end 时间（兼容 snake/camel）。"""
    if not isinstance(stop, dict):
        return None, None
    start = stop.get("start_time", stop.get("startTime"))
    end = stop.get("end_time", stop.get("endTime"))
    return start, end

def _leg_travel_budget_min(a, b):
    """
    固定活动时间窗下的路段出行预算（分钟）：
    上一站 end_time → 下一站 start_time。
    缺任一时间则不约束。
    """
    _, end_a = _stop_time_fields(a)
    start_b, _ = _stop_time_fields(b)
    end_min = _parse_hhmm_to_minutes(end_a)
    start_min = _parse_hhmm_to_minutes(start_b)
    if end_min is None or start_min is None:
        return None
    return float(start_min - end_min)

def _filter_leg_candidates_by_time_budget(candidates_by_mode, budget_min):
    """
    优先保留能在时间预算内到达的交通方式；
    若全部超时，回退为耗时最短的一种，避免无解。
    """
    if budget_min is None or not candidates_by_mode:
        return candidates_by_mode
    feasible = {
        mode: cand
        for mode, cand in candidates_by_mode.items()
        if float((cand.get("plan") or {}).get("time_min") or 0) <= float(budget_min) + 1e-6
    }
    if feasible:
        return feasible
    fastest = min(
        candidates_by_mode.values(),
        key=lambda c: float((c.get("plan") or {}).get("time_min") or 1e18),
    )
    return {fastest["mode"]: fastest}

def _scheme_defs_for_optimizer_leg(a, b):
    """按直线距离跳过明显不合适的公交/地铁，减少天地图调用。"""
    p_from = Point(float(a["lon"]), float(a["lat"]))
    p_to = Point(float(b["lon"]), float(b["lat"]))
    straight_m = _haversine_m(p_from.x, p_from.y, p_to.x, p_to.y)
    defs = []
    for sdef in ROUTE_SCHEME_DEFS:
        scheme_id = str(sdef.get("id") or "")
        if scheme_id == "transit_metro" and not _should_offer_metro(None, p_from, p_to, straight_m):
            continue
        if scheme_id == "transit_bus" and not _should_offer_bus(None, p_from, p_to, straight_m):
            continue
        defs.append(sdef)
    return defs

def _build_optimizer_leg_candidates(stops, city="南京", fast_road=False):
    all_leg_candidates = []

    for i in range(len(stops) - 1):
        a, b = stops[i], stops[i + 1]
        mode_scheme_defs = _scheme_defs_for_optimizer_leg(a, b)
        if not mode_scheme_defs:
            return None

        def _plan_one(sdef, origin=a, dest=b):
            return sdef, _build_leg_plan_for_scheme(
                origin, dest, sdef, city=city, fast_road=fast_road
            )

        workers = min(4, max(1, len(mode_scheme_defs)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            planned = list(pool.map(_plan_one, mode_scheme_defs))

        candidates_by_mode = {}
        for sdef, leg_plan in planned:
            if not leg_plan:
                continue

            scheme_id = str(sdef.get("id") or "")
            if scheme_id.startswith("transit_") and not _transit_leg_has_ride(leg_plan):
                continue

            mode_key = str(leg_plan.get("mode") or sdef.get("ui_mode") or scheme_id)
            dist_km = leg_plan.get("distance_km")
            comfort = float(_mode_comfort(mode_key, dist_km))
            cost_yuan = _estimate_mode_cost_yuan(
                mode_key,
                dist_km,
                leg_plan.get("time_min"),
            )
            candidate = {
                "leg_index": i,
                "scheme_id": scheme_id,
                "scheme_label": sdef.get("label"),
                "mode": mode_key,
                "mode_label": leg_plan.get("mode_label") or MODE_LABELS_ZH.get(mode_key, mode_key),
                "cost_yuan": round(cost_yuan, 2),
                "comfort": round(comfort, 3),
                "plan": leg_plan,
            }

            prev = candidates_by_mode.get(mode_key)
            if prev is None or _leg_candidate_sort_key(candidate) < _leg_candidate_sort_key(prev):
                candidates_by_mode[mode_key] = candidate

        if not candidates_by_mode:
            return None

        # 优先满足固定活动时间窗：仅保留能按时到达的方式
        budget_min = _leg_travel_budget_min(a, b)
        candidates_by_mode = _filter_leg_candidates_by_time_budget(
            candidates_by_mode, budget_min
        )
        all_leg_candidates.append(sorted(candidates_by_mode.values(), key=_leg_candidate_sort_key))

    return all_leg_candidates

def _combo_totals(combo):
    distance_m = sum(float(c["plan"].get("distance_m") or 0) for c in combo)
    time_min = sum(float(c["plan"].get("time_min") or 0) for c in combo)
    carbon_g = sum(float(c["plan"].get("carbon_g") or 0) for c in combo)
    cost_yuan = sum(float(c.get("cost_yuan") or 0) for c in combo)
    # 全程舒适性：按各段出行时间加权平均
    if time_min > 1e-9:
        comfort = sum(
            float(
                c.get(
                    "comfort",
                    _mode_comfort(c.get("mode"), c["plan"].get("distance_km")),
                )
            )
            * float(c["plan"].get("time_min") or 0)
            for c in combo
        ) / time_min
    elif combo:
        comfort = sum(
            float(
                c.get(
                    "comfort",
                    _mode_comfort(c.get("mode"), c["plan"].get("distance_km")),
                )
            )
            for c in combo
        ) / len(combo)
    else:
        comfort = 0.0
    return {
        "distance_m": distance_m,
        "time_min": time_min,
        "carbon_g": carbon_g,
        "cost_yuan": cost_yuan,
        "comfort": comfort,
    }

def _normalized_value(value, min_value, max_value):
    span = float(max_value) - float(min_value)
    if span <= 1e-9:
        return 0.0
    return (float(value) - float(min_value)) / span

def _score_combo(totals, ranges, weights):
    """加权目标：碳排与时间越小越好；舒适性越高越好（取反后最小化）。"""
    w_carbon, w_time, w_comfort = weights
    comfort_norm = _normalized_value(totals["comfort"], *ranges["comfort"])
    return (
        float(w_carbon) * _normalized_value(totals["carbon_g"], *ranges["carbon_g"])
        + float(w_time) * _normalized_value(totals["time_min"], *ranges["time_min"])
        + float(w_comfort) * (1.0 - comfort_norm)
    )

def _assemble_optimized_plan(stops, combo, totals, strategy_def, objective_value, budget_g, budget_feasible):
    merged_coords = []
    merged_parts = []
    legs = []
    mode_labels = []
    uses_transit_line = False

    for i, candidate in enumerate(combo):
        leg_plan = candidate["plan"]
        a, b = stops[i], stops[i + 1]
        merged_coords = _merge_route_coords(merged_coords, leg_plan.get("route_coords") or [])
        if candidate["mode_label"] not in mode_labels:
            mode_labels.append(candidate["mode_label"])

        for part in leg_plan.get("route_parts") or []:
            part_copy = dict(part)
            part_copy["leg_index"] = i
            part_copy["from_label"] = a.get("label", "")
            part_copy["to_label"] = b.get("label", "")
            merged_parts.append(part_copy)
            if part_copy.get("geometry") == "transit_ride" and len(part_copy.get("coords") or []) >= 2:
                uses_transit_line = True

        legs.append({
            "index": i,
            "from_label": a.get("label", ""),
            "to_label": b.get("label", ""),
            "distance_km": leg_plan.get("distance_km"),
            "distance_m": leg_plan.get("distance_m"),
            "time_min": leg_plan.get("time_min"),
            "carbon_g": leg_plan.get("carbon_g"),
            "cost_yuan": candidate["cost_yuan"],
            "comfort": candidate.get("comfort"),
            "scheme_id": candidate["scheme_id"],
            "transport_mode": candidate["mode"],
            "mode_label": candidate["mode_label"],
            "route_coords": leg_plan.get("route_coords"),
            "route_parts": leg_plan.get("route_parts"),
            "route_source": leg_plan.get("route_source"),
            "walk_fallback": leg_plan.get("walk_fallback"),
        })

    route_sources = {
        leg.get("route_source")
        for leg in legs
        if leg.get("route_source")
    }
    if route_sources == {"local_network"}:
        route_source = "local_network"
    elif "local_network" in route_sources:
        route_source = "mixed"
    elif "amap" in route_sources and "tianditu" in route_sources:
        route_source = "mixed"
    elif "amap" in route_sources:
        route_source = "amap"
    else:
        route_source = next(iter(route_sources), "tianditu")

    weights = strategy_def["weights"]
    mode_summary = " + ".join(mode_labels) if mode_labels else "组合交通"
    return {
        "scheme_id": strategy_def["id"],
        "scheme_label": strategy_def["label"],
        "icon": strategy_def.get("icon", ""),
        "mode": "optimized",
        "mode_label": strategy_def["label"],
        "mode_summary": mode_summary,
        "distance_m": round(totals["distance_m"]),
        "distance_km": round(totals["distance_m"] / 1000.0, 2),
        "time_min": round(totals["time_min"], 1),
        "carbon_g": round(totals["carbon_g"], 1),
        "cost_yuan": round(totals["cost_yuan"], 2),
        "comfort": round(totals["comfort"], 3),
        "route_coords": merged_coords,
        "route_parts": merged_parts,
        "legs": legs,
        "leg_count": len(legs),
        "route_point_count": len(merged_coords),
        "uses_transit_line": uses_transit_line,
        "route_source": route_source,
        "lazy": False,
        "optimization_model": {
            "type": "carbon_budget_weighted_binary_choice",
            "strategy": strategy_def["strategy"],
            "weights": {
                "carbon": weights[0],
                "time": weights[1],
                "comfort": weights[2],
            },
            "objective_value": round(float(objective_value), 6),
            "carbon_budget_g": round(float(budget_g), 1) if budget_g is not None else None,
            "budget_feasible": bool(budget_feasible),
            "variables": "x_i,k in {0,1}",
            "constraint": "sum(E_i,k * x_i,k) <= B",
        },
    }

def _parse_carbon_budget_g(carbon_budget_g):
    try:
        if carbon_budget_g is None:
            return None
        budget = float(carbon_budget_g)
    except (TypeError, ValueError):
        return None
    return budget if budget > 0 else None

def _build_optimized_route_plan(
    stops,
    strategy_def,
    carbon_budget_g=None,
    city="南京",
    fast_road=False,
    leg_candidates=None,
    combo_space=None,
):
    if len(stops) < 2:
        return None

    if leg_candidates is None:
        leg_candidates = _build_optimizer_leg_candidates(stops, city=city, fast_road=fast_road)
    if not leg_candidates:
        return None

    if combo_space is None:
        combos = []
        for combo in itertools.product(*leg_candidates):
            combos.append((combo, _combo_totals(combo)))
        if not combos:
            return None
        ranges = {}
        for key in ("carbon_g", "time_min", "comfort"):
            values = [totals[key] for _, totals in combos]
            ranges[key] = (min(values), max(values))
        combo_space = (combos, ranges)
    else:
        combos, ranges = combo_space
        if not combos:
            return None

    budget = _parse_carbon_budget_g(carbon_budget_g)
    feasible = [
        (combo, totals)
        for combo, totals in combos
        if budget is None or totals["carbon_g"] <= budget + 1e-6
    ]
    budget_feasible = bool(feasible)
    search_space = feasible if feasible else combos

    weights = strategy_def["weights"]
    best_combo = None
    best_totals = None
    best_score = None
    for combo, totals in search_space:
        score = _score_combo(totals, ranges, weights)
        tie_breaker = (totals["carbon_g"], totals["time_min"])
        if best_score is None or (score, tie_breaker) < (
            best_score,
            (best_totals["carbon_g"], best_totals["time_min"]),
        ):
            best_combo = combo
            best_totals = totals
            best_score = score

    if best_combo is None:
        return None
    return _assemble_optimized_plan(
        stops,
        best_combo,
        best_totals,
        strategy_def,
        best_score,
        budget,
        budget_feasible,
    )

def _build_optimized_route_plans(stops, carbon_budget_g=None, city="南京", fast_road=False):
    """路段候选与组合空间只建一次，两种权重策略复用，避免重复天地图算路。"""
    if len(stops) < 2:
        return []

    leg_candidates = _build_optimizer_leg_candidates(stops, city=city, fast_road=fast_road)
    if not leg_candidates:
        return []

    combos = []
    for combo in itertools.product(*leg_candidates):
        combos.append((combo, _combo_totals(combo)))
    if not combos:
        return []

    ranges = {}
    for key in ("carbon_g", "time_min", "comfort"):
        values = [totals[key] for _, totals in combos]
        ranges[key] = (min(values), max(values))
    combo_space = (combos, ranges)

    plans = []
    for strategy_def in OPTIMIZED_SCHEME_DEFS:
        plan = _build_optimized_route_plan(
            stops,
            strategy_def,
            carbon_budget_g=carbon_budget_g,
            city=city,
            fast_road=fast_road,
            leg_candidates=leg_candidates,
            combo_space=combo_space,
        )
        if plan:
            plans.append(plan)
    return plans

def _pick_model_best_plan(routes, fallback_plans):
    eager = [r for r in (routes or []) if not r.get("lazy") and r.get("carbon_g") is not None]
    model_low = next((r for r in eager if r.get("scheme_id") == "model_low_carbon"), None)
    if model_low:
        return model_low
    if eager:
        return min(eager, key=lambda p: float(p.get("carbon_g") or float("inf")))
    return fallback_plans[-1] if fallback_plans else None

def _build_itinerary_recommendation(stops, remaining_g, activity_count, city="南京", fast_road=True):
    routes, eager_plans = _build_itinerary_routes(
        stops,
        city=city,
        fast_road=fast_road,
        carbon_budget_g=remaining_g,
    )
    if not routes:
        return None

    best_plan = _pick_model_best_plan(routes, eager_plans)
    places = [s for s in stops if s.get("type") == "activity"]
    primary_place = places[0] if places else stops[-1]
    best_route_id = (
        (best_plan or {}).get("scheme_id")
        or OPTIMIZED_SCHEME_DEFS[0]["id"]
    )

    result = {
        "stops": stops,
        "places": places,
        "place": {
            "name": " → ".join(s.get("name") or s.get("label", "") for s in places) or primary_place.get("label", ""),
            "lat": primary_place["lat"],
            "lon": primary_place["lon"],
        },
        "routes": routes,
        "best_route_id": best_route_id,
        "comparison": None,
        "highlight": "",
        "total_distance_km": None,
        "total_time_min": None,
        "total_carbon_g": None,
    }
    if best_plan and best_plan.get("carbon_g") is not None:
        baseline_pool = eager_plans or routes
        baseline_plan = max(
            baseline_pool,
            key=lambda p: float(p.get("carbon_g") or 0),
        )
        comparison = _comparison_copy(
            best_plan, baseline_plan, remaining_g, activity_count=activity_count
        )
        result["comparison"] = comparison
        result["highlight"] = comparison["message"]
        result["total_distance_km"] = best_plan.get("distance_km")
        result["total_time_min"] = best_plan.get("time_min")
        result["total_carbon_g"] = best_plan.get("carbon_g")
    return result

def _parse_stops_for_routing(stops):
    parsed = []
    for idx, stop in enumerate(stops or []):
        lat = stop.get("lat", stop.get("latitude"))
        lng = stop.get("lng", stop.get("lon", stop.get("longitude")))
        if lat is None or lng is None:
            return None, f"第 {idx + 1} 个站点缺少坐标"
        start_time, end_time = _stop_time_fields(stop)
        item = {
            "lat": float(lat),
            "lng": float(lng),
            "label": str(stop.get("label") or f"站点{idx + 1}"),
            "type": str(stop.get("type") or "stop"),
            "name": stop.get("name"),
            "query": stop.get("query"),
        }
        if start_time is not None:
            item["start_time"] = start_time
        if end_time is not None:
            item["end_time"] = end_time
        if stop.get("is_fixed") is not None:
            item["is_fixed"] = bool(stop.get("is_fixed"))
        if stop.get("activity_row_id") is not None:
            item["activity_row_id"] = stop.get("activity_row_id")
        parsed.append(item)
    if len(parsed) < 2:
        return None, "至少需要两个站点"
    return parsed, None

def run_travel_itinerary_scheme_plan(stops, scheme_id, city="南京", carbon_budget_g=None):
    """按站点顺序为指定出行方案计算完整串联路线（绿色出行/最快到达，或内部模式）。"""
    scheme_id = str(scheme_id or "").strip()
    optimized_def = _optimized_scheme_def(scheme_id)
    if optimized_def is not None:
        parsed, err = _parse_stops_for_routing(stops)
        if err:
            return {"success": False, "message": err}

        if not has_online_routing_key():
            return {"success": False, "message": "未配置天地图或高德路径规划 Key"}

        stop_dicts = [
            _stop_dict(
                s.get("type") or "stop",
                s["label"],
                s["lat"],
                s["lng"],
                **{
                    k: s[k]
                    for k in ("name", "query", "start_time", "end_time", "is_fixed", "activity_row_id")
                    if k in s and s[k] is not None
                },
            )
            for s in parsed
        ]
        plan = _build_optimized_route_plan(
            stop_dicts,
            optimized_def,
            carbon_budget_g=carbon_budget_g,
            city=city,
            fast_road=False,
        )
        if not plan:
            return {
                "success": False,
                "message": "该优化方案暂时无法规划路线，请换其他方式或调整站点顺序",
            }
        return {
            "success": True,
            "city": str(city or "南京").strip(),
            "scheme_id": scheme_id,
            "route": plan,
        }

    scheme = next(
        (s for s in ROUTE_SCHEME_DEFS if s.get("id") == scheme_id),
        None,
    )
    if scheme is None:
        return {
            "success": False,
            "message": f"未知出行方案: {scheme_id}",
        }

    parsed, err = _parse_stops_for_routing(stops)
    if err:
        return {"success": False, "message": err}

    if not has_online_routing_key():
        return {"success": False, "message": "未配置天地图或高德路径规划 Key"}

    stop_dicts = [
        _stop_dict(
            s.get("type") or "stop",
            s["label"],
            s["lat"],
            s["lng"],
            **{
                k: s[k]
                for k in ("name", "query", "start_time", "end_time", "is_fixed", "activity_row_id")
                if k in s and s[k] is not None
            },
        )
        for s in parsed
    ]
    plan = _build_multi_stop_route_plan(stop_dicts, scheme, city=city, fast_road=False)
    if not plan:
        return {
            "success": False,
            "message": "该出行方式暂无法规划路线，请换其他方式或调整站点顺序",
        }

    plan = dict(plan)
    plan["lazy"] = False
    return {
        "success": True,
        "city": str(city or "南京").strip(),
        "scheme_id": scheme_id,
        "route": plan,
    }

def _run_multi_activity_recommendation(
    origin_lat,
    origin_lon,
    preview_stops,
    max_distance_km=None,
    companions=None,
    time_preference=None,
    city="南京",
    icb_raster=None,
    transform=None,
    max_itineraries=MAX_MULTI_ITINERARIES,
    work_lat=None,
    work_lon=None,
):
    queries = _preview_activity_queries(preview_stops)
    if len(queries) < 2:
        return {
            "success": False,
            "message": "多活动推荐至少需要两个活动类型",
        }

    max_radius_m = _poi_search_radius_m(max_distance_km)
    per_activity, scope_label, failed_query = _find_preview_chained_activity_candidates(
        preview_stops,
        city=city,
        max_radius_m=max_radius_m,
        limit_per_activity=MAX_CANDIDATES_PER_ACTIVITY_MULTI,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )
    if per_activity is None:
        scope = f"{scope_label}附近" if scope_label else "当前范围"
        return {
            "success": False,
            "message": f"在{scope}未找到「{failed_query}」相关 POI，请换关键词",
        }

    remaining_g = _icb_at_point(icb_raster, transform, origin_lat, origin_lon)
    origin_stop = _origin_stop(origin_lat, origin_lon)
    work_stop = None
    if work_lat is not None and work_lon is not None:
        work_stop = _stop_dict("work", "工作位置", work_lat, work_lon)

    rough_candidates = []
    for combo in itertools.product(*(item["candidates"] for item in per_activity)):
        if not _combo_respects_preview_chain(
            preview_stops, combo, max_radius_m, origin_lat=origin_lat, origin_lon=origin_lon
        ):
            continue
        seq = _build_stops_from_preview_and_combo(
            preview_stops, combo, origin_stop, work_stop
        )
        if len(seq) < 2:
            continue
        rough_candidates.append((_sequence_haversine_m(seq), seq))

    rough_candidates.sort(key=lambda x: x[0])
    seen_route_keys = set()
    seen_activity_keys = set()
    finalists = []
    for _, seq in rough_candidates:
        route_key = tuple(
            (s.get("type"), round(s["lat"], 5), round(s["lon"], 5))
            for s in seq
        )
        activity_key = _activity_locations_key_from_stops(seq)
        if not activity_key:
            continue
        if activity_key in seen_activity_keys:
            continue
        if route_key in seen_route_keys:
            continue
        seen_activity_keys.add(activity_key)
        seen_route_keys.add(route_key)
        finalists.append(seq)
        if len(finalists) >= max(int(max_itineraries), ROUGH_ITINERARY_CANDIDATES):
            break

    recommendations = []
    for stops in finalists:
        rec = _build_itinerary_recommendation(
            stops, remaining_g, activity_count=len(queries), city=city,
        )
        if not rec:
            rec = _build_itinerary_recommendation(
                stops, remaining_g, activity_count=len(queries), city=city, fast_road=False,
            )
        if rec:
            recommendations.append(rec)

    recommendations.sort(key=lambda r: float(r.get("total_carbon_g") or float("inf")))
    recommendations = _dedupe_recommendations_by_activity(recommendations)
    recommendations = recommendations[: max(1, int(max_itineraries))]

    if not recommendations:
        if not rough_candidates:
            return {
                "success": False,
                "message": "未找到可串联的 POI 组合，请换关键词",
            }
        return {
            "success": False,
            "message": "未能生成有效的多活动串联路线，请调整活动类型",
        }

    label = "、".join(f"「{q}」" for q in queries)
    return {
        "success": True,
        "multi_activity": True,
        "message": (
            f"已按路线预览顺序（活动以预览中上一站为基准）为您规划 "
            f"{len(recommendations)} 条串联 {label} 的最优出行路线"
        ),
        "constraints": {
            "companions": companions,
            "time_preference": time_preference,
            "activity_queries": queries,
            "preview_stops": preview_stops,
        },
        "origin": {"lat": origin_lat, "lon": origin_lon},
        "initial_budget_g": remaining_g,
        "initial_budget_kg": round(remaining_g / 1000.0, 3) if remaining_g else None,
        "remaining_budget_g": remaining_g,
        "remaining_budget_kg": round(remaining_g / 1000.0, 3) if remaining_g else None,
        "recommendations": recommendations,
    }

def _is_fixed_activity_chain(preview_stops):
    """路线预览中所有站点均已有坐标，且至少两个站点（居住点不参与路线时亦同）。"""
    if not preview_stops or len(preview_stops) < 2:
        return False
    for stop in preview_stops:
        if _stop_coords(stop) is None:
            return False
    return True

def _preview_has_origin_stop(preview_stops):
    return any(
        _normalize_preview_stop_type(s.get("type")) == "origin"
        for s in (preview_stops or [])
    )

def _preview_to_chain_stops(preview_stops):
    stops = []
    for i, stop in enumerate(preview_stops):
        coords = _stop_coords(stop)
        if not coords:
            continue
        slon, slat = coords
        stype = _normalize_preview_stop_type(stop.get("type"))
        if i == 0:
            stype = "origin"
        elif stype == "origin":
            stype = "activity"
        label = str(stop.get("label") or stop.get("name") or f"活动 {i + 1}")
        start_time, end_time = _stop_time_fields(stop)
        extra = {
            "name": stop.get("name"),
            "query": stop.get("query"),
        }
        if start_time is not None:
            extra["start_time"] = start_time
        if end_time is not None:
            extra["end_time"] = end_time
        if stop.get("is_fixed") is not None:
            extra["is_fixed"] = bool(stop.get("is_fixed"))
        stops.append(_stop_dict(
            stype,
            label,
            slat,
            slon,
            **{k: v for k, v in extra.items() if v is not None},
        ))
    return stops

def _run_fixed_chain_recommendation(
    preview_stops,
    *,
    origin_lat,
    origin_lon,
    city="南京",
    icb_raster=None,
    transform=None,
    companions=None,
    time_preference=None,
):
    """按已确认的活动安排串联规划出行方案（站点坐标固定）。"""
    chain_stops = _preview_to_chain_stops(preview_stops)
    if len(chain_stops) < 2:
        return {"success": False, "message": "至少需要两个已确认的活动地点"}

    remaining_g = _icb_at_point(icb_raster, transform, origin_lat, origin_lon)
    activity_count = max(1, len(chain_stops) - 1)
    rec = _build_itinerary_recommendation(
        chain_stops,
        remaining_g,
        activity_count=activity_count,
        city=city,
    )
    if not rec:
        return {
            "success": False,
            "message": "无法规划活动间出行路线，请检查地点坐标",
        }

    labels = [s.get("label") or s.get("name") or "" for s in chain_stops if s.get("label") or s.get("name")]
    route_label = " → ".join(labels) if labels else "活动出行"
    return {
        "success": True,
        "multi_activity": len(chain_stops) > 2,
        "message": f"已按选定活动地点规划串联出行路线：{route_label}",
        "constraints": {
            "companions": companions,
            "time_preference": time_preference,
            "preview_stops": preview_stops,
            "fixed_chain": True,
        },
        "origin": {"lat": origin_lat, "lon": origin_lon},
        "initial_budget_g": remaining_g,
        "initial_budget_kg": round(remaining_g / 1000.0, 3) if remaining_g else None,
        "remaining_budget_g": remaining_g,
        "remaining_budget_kg": round(remaining_g / 1000.0, 3) if remaining_g else None,
        "recommendations": [rec],
    }

def run_travel_recommendation(
    origin_lat,
    origin_lon,
    activity_query=None,
    activity_queries=None,
    activity_preset_id=None,  # 已废弃，保留参数以兼容旧版 app 调用
    preview_stops=None,
    max_distance_km=None,
    companions=None,
    time_preference=None,
    city="南京",
    icb_raster=None,
    transform=None,
    max_places=4,
    max_itineraries=MAX_MULTI_ITINERARIES,
    work_lat=None,
    work_lon=None,
):
    """
    执行出行推荐主流程。
    - 活动 POI 按路线预览顺序检索：每个活动以其在预览中的上一站为锚点（居住/工作/上一活动）。
    - 单个活动：返回最多 max_places 个目的地及 OD 路线。
    - 多个活动：按预览顺序串联，返回最多 max_itineraries 条完整路线。
    companions / time_preference 预留字段，当前版本参与展示不参与排序。
    """
    try:
        require_tianditu_api_key()
    except ValueError as exc:
        return {"success": False, "message": str(exc)}

    queries = [str(q).strip() for q in (activity_queries or []) if str(q).strip()]
    if not queries and activity_query:
        queries = [str(activity_query).strip()]

    preview = _normalize_preview_stops(
        preview_stops, origin_lat, origin_lon, work_lat, work_lon, activity_queries=queries
    )

    if _is_fixed_activity_chain(preview):
        return _run_fixed_chain_recommendation(
            preview,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            city=city,
            icb_raster=icb_raster,
            transform=transform,
            companions=companions,
            time_preference=time_preference,
        )

    preview_queries = _preview_activity_queries(preview)
    if not preview_queries:
        return {
            "success": False,
            "message": "请输入活动类型关键词，如：咖啡、公园、超市",
        }

    if not _preview_has_fixed_anchor(preview, origin_lat, origin_lon):
        return {
            "success": False,
            "message": "请至少为第一个活动选择地点，作为 POI 推荐起点",
        }

    if len(preview_queries) >= 2:
        return _run_multi_activity_recommendation(
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            preview_stops=preview,
            max_distance_km=max_distance_km,
            companions=companions,
            time_preference=time_preference,
            city=city,
            icb_raster=icb_raster,
            transform=transform,
            max_itineraries=max_itineraries,
            work_lat=work_lat,
            work_lon=work_lon,
        )

    activity_query = preview_queries[0]
    keywords, poi_cats = _resolve_keywords(activity_query)
    if not keywords:
        return {
            "success": False,
            "message": "请输入活动类型关键词，如：咖啡、公园、超市",
        }

    max_radius_m = _poi_search_radius_m(max_distance_km)
    place_limit = max(1, min(int(max_places), MAX_SINGLE_PLACES))
    per_activity, scope_label, failed_query = _find_preview_chained_activity_candidates(
        preview,
        city=city,
        max_radius_m=max_radius_m,
        limit_per_activity=max(place_limit * 6, 12),
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )
    if per_activity is None:
        scope = f"{scope_label}附近" if scope_label else "当前范围"
        return {
            "success": False,
            "message": f"在{scope}未找到「{failed_query}」相关 POI，请换关键词",
        }

    candidates = per_activity[0]["candidates"]
    anchor_label = per_activity[0].get("anchor_label") or "上一站"
    if not candidates:
        return {
            "success": False,
            "message": f"在{anchor_label}附近未找到匹配的 POI，请换关键词",
        }

    selected_pois = _pick_spread_activity_pois(candidates, place_limit)
    if len(selected_pois) < place_limit:
        return {
            "success": False,
            "message": (
                f"在{anchor_label}附近未找到足够分散的 {place_limit} 个活动地点，"
                "请换关键词"
            ),
        }

    remaining_g = _icb_at_point(icb_raster, transform, origin_lat, origin_lon)
    origin_stop = _origin_stop(origin_lat, origin_lon)
    work_stop = None
    if work_lat is not None and work_lon is not None:
        work_stop = _stop_dict("work", "工作位置", work_lat, work_lon)

    recommendations = []
    for poi in selected_pois:
        activity_stop = _poi_to_activity_stop(poi, activity_query)
        stops = _build_stops_from_preview_and_combo(
            preview, [poi], origin_stop, work_stop
        )
        if len(stops) < 2:
            # 居住点仅用于碳预算，默认不参与路线；仅当预览中显式包含 origin 时才补上
            if not _preview_has_origin_stop(preview):
                continue
            stops = [origin_stop]
            if work_stop:
                stops.append(work_stop)
            stops.append(activity_stop)

        if len(stops) < 2:
            continue

        routes, eager_plans = _build_itinerary_routes(stops, city=city)
        if not routes:
            continue

        best_plan = _pick_model_best_plan(routes, eager_plans)
        best_route_id = (
            (best_plan or {}).get("scheme_id")
            or OPTIMIZED_SCHEME_DEFS[0]["id"]
        )
        comparison = None
        highlight = ""
        total_distance_km = None
        total_time_min = None
        total_carbon_g = None
        if best_plan and best_plan.get("carbon_g") is not None:
            baseline_pool = eager_plans or routes
            baseline_plan = max(
                baseline_pool,
                key=lambda p: float(p.get("carbon_g") or 0),
            )
            comparison = _comparison_copy(best_plan, baseline_plan, remaining_g, activity_count=1)
            highlight = comparison["message"]
            total_distance_km = best_plan.get("distance_km")
            total_time_min = best_plan.get("time_min")
            total_carbon_g = best_plan.get("carbon_g")

        recommendations.append({
            "stops": stops,
            "places": [activity_stop],
            "place": {
                "name": poi["display_name"],
                "lat": poi["lat"],
                "lon": poi["lon"],
                "distance_km": poi["distance_km"],
                "distance_m": poi["distance_m"],
                "rating": poi.get("rating"),
                "poi_category": poi.get("poi_category"),
                "poi_subtype": poi.get("poi_subtype"),
            },
            "routes": routes,
            "best_route_id": best_route_id,
            "comparison": comparison,
            "highlight": highlight,
            "total_distance_km": total_distance_km,
            "total_time_min": total_time_min,
            "total_carbon_g": total_carbon_g,
        })

    recommendations = _dedupe_recommendations_by_activity(recommendations)

    activity_label = str(activity_query or "").strip()
    if not recommendations:
        if not _preview_has_origin_stop(preview):
            return {
                "success": False,
                "message": "请至少选择两个活动地点以规划出行路线（居住点仅用于碳预算，不参与路线）",
            }
        return {
            "success": False,
            "message": f"未能规划到「{activity_label}」相关目的地的出行路线，请换地点后重试",
        }

    return {
        "success": True,
        "multi_activity": False,
        "message": (
            f"已按路线预览顺序为您推荐 {len(recommendations)} 个"
            f"「{activity_label}」相关目的地"
        ),
        "constraints": {
            "companions": companions,
            "time_preference": time_preference,
            "activity_query": activity_query,
            "activity_queries": [activity_query],
            "preview_stops": preview,
        },
        "origin": {"lat": origin_lat, "lon": origin_lon},
        "initial_budget_g": remaining_g,
        "initial_budget_kg": round(remaining_g / 1000.0, 3) if remaining_g else None,
        "remaining_budget_g": remaining_g,
        "remaining_budget_kg": round(remaining_g / 1000.0, 3) if remaining_g else None,
        "recommendations": recommendations,
    }

def run_travel_chain_routes(stops, city="南京", scheme_id="low_carbon"):
    if not stops or len(stops) < 2:
        return {
            "success": False,
            "message": "出行链至少需要两个站点",
        }

    if not has_online_routing_key():
        return {"success": False, "message": "未配置天地图或高德路径规划 Key"}

    parsed, err = _parse_stops_for_routing(stops)
    if err:
        return {"success": False, "message": err}

    scheme = next(
        (s for s in ROUTE_SCHEME_DEFS if s.get("id") == str(scheme_id or "low_carbon")),
        ROUTE_SCHEME_DEFS[0],
    )
    stop_dicts = [
        _stop_dict(
            s.get("type") or "stop",
            s["label"],
            s["lat"],
            s["lng"],
        )
        for s in parsed
    ]
    plan = _build_multi_stop_route_plan(stop_dicts, scheme, city=city)
    if not plan:
        return {
            "success": False,
            "message": "出行链路线规划失败",
        }

    legs = plan.get("legs") or []
    return {
        "success": True,
        "city": city,
        "scheme_id": scheme.get("id"),
        "route_coords": plan.get("route_coords"),
        "route_parts": plan.get("route_parts"),
        "legs": legs,
    }
