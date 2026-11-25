"""
天地图路径规划：步行 / 驾车 / 公交地铁。
坐标输入输出均为 WGS-84 / CGCS2000（与本地 OSM 路网、Leaflet 一致）。
"""
from __future__ import annotations

import json
import math
import os
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict

_ROUTE_CACHE_MAX = 512
_ROUTE_CACHE: OrderedDict = {}
_HTTP_HEADERS = {
    'User-Agent': 'IATP/1.0 (tianditu-routing)',
    'Accept': 'application/json, text/xml, */*',
}
# 驾车可在 1s 内返回；公交规划对本服务端 Key 常整段挂起，不宜长等。
# 实测：/v2/search、/geocoder、/drive 正常，/transit 无 Referer 时读超时，
# 带天地图 Referer 则立刻 403（Key 类型为服务器端）。
_DRIVE_TIMEOUT = (5, 20)
_TRANSIT_TIMEOUT = (5, 8)
_DRIVE_RETRIES = 1
_TRANSIT_RETRIES = 0


def _http_get(url, params, timeout, retries, label):
    """GET 天地图接口；超时/断连时重试。"""
    import requests
    from requests.exceptions import ConnectionError as ReqConnectionError
    from requests.exceptions import Timeout as ReqTimeout

    last_exc = None
    attempts = max(1, int(retries) + 1)
    for attempt in range(1, attempts + 1):
        try:
            t0 = time.time()
            resp = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers=_HTTP_HEADERS,
            )
            print(
                f'[tianditu_route] {label} HTTP {resp.status_code} '
                f'{time.time() - t0:.1f}s attempt={attempt}/{attempts}',
                flush=True,
            )
            return resp
        except (ReqTimeout, ReqConnectionError) as exc:
            last_exc = exc
            print(
                f'[tianditu_route] {label} 第{attempt}/{attempts}次失败: {exc}',
                flush=True,
            )
            if attempt < attempts:
                time.sleep(min(1.2 * attempt, 3.0))
    raise last_exc

def _parse_lonlat_polyline(polyline_str):
    """天地图 lon,lat 折线 → [[lat, lng], ...]。"""
    if not polyline_str:
        return []
    coords = []
    for pair in str(polyline_str).split(';'):
        pair = pair.strip()
        if not pair:
            continue
        parts = pair.split(',')
        if len(parts) < 2:
            continue
        try:
            lng = float(parts[0])
            lat = float(parts[1])
        except (TypeError, ValueError):
            continue
        coords.append([lat, lng])
    return coords

def _dedupe_route_coords(coords, min_dist_m=3.0):
    if not coords:
        return []
    out = [coords[0]]
    for lat, lng in coords[1:]:
        prev = out[-1]
        dlat = (lat - prev[0]) * 111320.0
        dlng = (lng - prev[1]) * 111320.0 * max(math.cos(math.radians(prev[0])), 0.2)
        if math.hypot(dlat, dlng) >= min_dist_m:
            out.append([lat, lng])
    if len(out) == 1 and len(coords) > 1:
        out.append(coords[-1])
    return out

def _merge_route_coords(existing, new_coords):
    if not new_coords:
        return existing
    if not existing:
        return list(new_coords)
    merged = list(existing)
    for coord in new_coords:
        if not merged:
            merged.append(coord)
            continue
        prev = merged[-1]
        if abs(prev[0] - coord[0]) > 1e-6 or abs(prev[1] - coord[1]) > 1e-6:
            merged.append(coord)
    return merged

def _travel_mode_to_profile(travel_mode, force_profile=None):
    from transport_modes import normalize_mode_key

    if force_profile:
        return force_profile
    key = normalize_mode_key(travel_mode)
    if key in ('walk', 'foot'):
        return 'walking'
    if key == 'bike':
        return 'bicycling'
    if key in ('car', 'drive'):
        return 'driving'
    if key in ('bus', 'metro', 'subway'):
        return 'transit'
    return 'walking'

def _is_transit_travel_mode(travel_mode):
    from transport_modes import normalize_mode_key

    return normalize_mode_key(travel_mode) in ('bus', 'metro', 'subway')

def _line_is_metro(line_info):
    try:
        line_type = int(line_info.get('lineType') or 0)
        if line_type == 2:
            return True
        if line_type == 1:
            return False
    except (TypeError, ValueError):
        pass
    name = str(line_info.get('lineName') or line_info.get('segmentName') or '')
    direction = str(line_info.get('direction') or '')
    metro_keywords = ('地铁', '轨道交通', '轻轨', '有轨', '磁浮', '市域', 'S1', 'S2', 'S3')
    return any(kw in name or kw in direction for kw in metro_keywords)

def _segment_ride_kind(segment):
    """
    解析分段乘运类型。天地图 segmentType：1=步行，2=公交，3=地铁（4=磁浮等按地铁处理）。
    步行段即使 segmentLine 含空 lineName，也不应判为公交。
    """
    try:
        segment_type = int(segment.get('segmentType') or 0)
    except (TypeError, ValueError):
        segment_type = 0
    if segment_type == 1:
        return None
    if segment_type == 2:
        return 'bus'
    if segment_type in (3, 4):
        return 'metro'

    has_metro = False
    has_bus = False
    for line in _segment_line_items(segment):
        line_coords = _parse_lonlat_polyline(line.get('linePoint'))
        if len(line_coords) < 2:
            continue
        line_name = str(line.get('lineName') or line.get('segmentName') or '').strip()
        if not line_name:
            continue
        if _line_is_metro(line):
            has_metro = True
        else:
            has_bus = True
    if has_metro:
        return 'metro'
    if has_bus:
        return 'bus'
    return None

def _segment_has_metro_ride(segment):
    return _segment_ride_kind(segment) == 'metro'

def _segment_has_bus_ride(segment):
    return _segment_ride_kind(segment) == 'bus'

def _segment_line_items(segment):
    block = segment.get('segmentLine')
    if isinstance(block, list):
        return [item for item in block if isinstance(item, dict)]
    if isinstance(block, dict):
        return [block]
    return []

def _transit_plan_has_required_ride(line, travel_mode):
    from transport_modes import normalize_mode_key

    key = normalize_mode_key(travel_mode)
    segments = line.get('segments') or []
    for segment in segments:
        if key in ('metro', 'subway'):
            if _segment_has_metro_ride(segment):
                return True
        elif key == 'bus':
            if _segment_has_bus_ride(segment):
                return True
    return False

def _select_transit_line(raw_data, travel_mode):
    from transport_modes import normalize_mode_key

    results = raw_data.get('results') or []
    candidates = []
    for result in results:
        for line in result.get('lines') or []:
            if _transit_plan_has_required_ride(line, travel_mode):
                candidates.append(line)
    if not candidates:
        for result in results:
            for line in result.get('lines') or []:
                if line.get('segments'):
                    candidates.append(line)
    if not candidates:
        return None

    key = normalize_mode_key(travel_mode)
    if key in ('metro', 'subway', 'bus'):
        pure = []
        mixed = []
        for line in candidates:
            has_metro = any(_segment_has_metro_ride(seg) for seg in (line.get('segments') or []))
            has_bus = any(_segment_has_bus_ride(seg) for seg in (line.get('segments') or []))
            if key == 'bus':
                (pure if has_bus and not has_metro else mixed).append(line)
            else:
                (pure if has_metro and not has_bus else mixed).append(line)
        if pure:
            return pure[0]
        if mixed:
            return mixed[0]
        return None
    return candidates[0]

def _transit_route_parts_connected(route_parts, max_gap_m=300.0):
    from transport_modes import haversine_m

    prev_end = None
    for part in route_parts or []:
        coords = part.get('coords') or []
        if len(coords) < 2:
            continue
        start = coords[0]
        if prev_end is not None:
            gap = haversine_m(prev_end[1], prev_end[0], start[1], start[0])
            if gap > max_gap_m:
                return False
        prev_end = coords[-1]
    return True

def _extract_tianditu_transit_route(raw_data, travel_mode):
    from transport_modes import normalize_mode_key

    best = _select_transit_line(raw_data, travel_mode)
    if best is None:
        return None

    mode_key = normalize_mode_key(travel_mode)
    merged_coords = []
    route_parts = []

    for segment in best.get('segments') or []:
        ride_kind = _segment_ride_kind(segment)
        kept_rides = []
        if ride_kind is not None:
            if mode_key in ('metro', 'subway') and ride_kind != 'metro':
                ride_kind = None
            elif mode_key == 'bus' and ride_kind != 'bus':
                ride_kind = None

        if ride_kind is not None:
            for line in _segment_line_items(segment):
                line_coords = _parse_lonlat_polyline(line.get('linePoint'))
                if len(line_coords) < 2:
                    continue
                kept_rides.append({
                    'type': ride_kind,
                    'geometry': 'transit_ride',
                    'coords': line_coords,
                    'line_name': str(line.get('lineName') or line.get('segmentName') or ''),
                })

        if ride_kind is None:
            walk_coords = []
            for line in _segment_line_items(segment):
                walk_coords = _merge_route_coords(
                    walk_coords, _parse_lonlat_polyline(line.get('linePoint')),
                )
            start = segment.get('stationStart') or {}
            end = segment.get('stationEnd') or {}
            if len(walk_coords) < 2:
                start_pt = _parse_lonlat_polyline(start.get('lonlat'))
                end_pt = _parse_lonlat_polyline(end.get('lonlat'))
                if start_pt and end_pt:
                    walk_coords = [start_pt[0], end_pt[-1]]
            if len(walk_coords) >= 2:
                route_parts.append({
                    'type': 'walk',
                    'geometry': 'transit_access',
                    'coords': walk_coords,
                })
                merged_coords = _merge_route_coords(merged_coords, walk_coords)
        elif kept_rides:
            for ride in kept_rides:
                route_parts.append(ride)
                merged_coords = _merge_route_coords(merged_coords, ride['coords'])

    merged_coords = _dedupe_route_coords(merged_coords)
    if len(merged_coords) < 2:
        return None

    uses_transit_line = any(
        p.get('geometry') == 'transit_ride' and len(p.get('coords') or []) >= 2
        for p in route_parts
    )

    distance_m = 0.0
    duration_sec = 0.0
    for segment in best.get('segments') or []:
        for line in _segment_line_items(segment):
            try:
                dist = abs(float(line.get('segmentDistance') or 0))
                if dist > 0:
                    distance_m += dist
            except (TypeError, ValueError):
                pass
            try:
                t = abs(float(line.get('segmentTime') or 0))
                if t > 0:
                    duration_sec += t * 60.0
            except (TypeError, ValueError):
                pass

    return {
        'route_coords': merged_coords,
        'route_parts': route_parts,
        'distance_m': distance_m,
        'duration_sec': duration_sec,
        'uses_transit_line': uses_transit_line,
    }

def _parse_drive_payload(payload):
    """解析驾车/步行规划响应（JSON 或 XML）。"""
    if payload is None:
        return None
    if isinstance(payload, dict):
        data = payload
    else:
        text = str(payload).strip()
        if not text:
            return None
        if text.startswith('{'):
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                return None
        else:
            try:
                root = ET.fromstring(text)
            except ET.ParseError:
                return None
            data = {}
            for tag in ('distance', 'duration', 'routelatlon'):
                node = root.find(f'.//{tag}')
                if node is not None and node.text:
                    data[tag] = node.text.strip()
            if not data.get('routelatlon'):
                simple_items = root.findall('.//simple/item')
                coords = []
                for item in simple_items:
                    street = item.findtext('streetLatLon') or item.findtext('turnlatlon') or ''
                    coords = _merge_route_coords(coords, _parse_lonlat_polyline(street))
                if coords:
                    data['routelatlon'] = ';'.join(f'{c[1]:.6f},{c[0]:.6f}' for c in coords) + ';'

    routelatlon = data.get('routelatlon') or data.get('routelatLon') or ''
    coords = _parse_lonlat_polyline(routelatlon)
    if len(coords) < 2:
        # 尝试嵌套结构：result.routes[0] / result / routes / route
        result = data.get('result') or {}
        if isinstance(result, dict):
            routes_list = result.get('routes') or []
            if isinstance(routes_list, list) and routes_list:
                result = routes_list[0]
            routelatlon = result.get('routelatlon') or result.get('routelatLon') or ''
            coords = _parse_lonlat_polyline(routelatlon)
    if len(coords) < 2:
        routes = data.get('routes') or data.get('route') or {}
        if isinstance(routes, dict):
            routelatlon = routes.get('routelatlon') or routes.get('routelatLon') or ''
            coords = _parse_lonlat_polyline(routelatlon)
    if len(coords) < 2:
        return None

    def _extract_nested(obj, *keys):
        if isinstance(obj, dict):
            for k in keys:
                v = obj.get(k)
                if v is not None and str(v) != '':
                    return v
        return None

    result_obj = data.get('result') or {}
    if isinstance(result_obj, dict):
        routes_list = result_obj.get('routes') or []
        if isinstance(routes_list, list) and routes_list:
            result_obj = routes_list[0]
    else:
        result_obj = {}

    distance_raw = (
        _extract_nested(data, 'distance')
        or _extract_nested(result_obj, 'distance')
    )
    try:
        distance_m = float(distance_raw or 0)
    except (TypeError, ValueError):
        distance_m = 0.0

    # 从路线坐标反算距离作为基准，若 API 返回的距离明显不合理（< 坐标距离的 10%）
    # 则使用坐标距离，防止 API 返回错误的小数值（如 0.02）或字段缺失
    if len(coords) >= 2:
        from transport_modes import path_length_from_coords
        coord_dist_m = float(path_length_from_coords(coords) or 0)
        if coord_dist_m > 0 and (distance_m <= 0 or distance_m < coord_dist_m * 0.1):
            distance_m = coord_dist_m
    elif distance_m <= 0:
        distance_m = 0.0

    duration_raw = (
        _extract_nested(data, 'duration')
        or _extract_nested(result_obj, 'duration')
    )
    try:
        duration_sec = float(duration_raw or 0)
    except (TypeError, ValueError):
        duration_sec = 0.0

    return {
        'route_coords': _dedupe_route_coords(coords),
        'distance_m': distance_m,
        'duration_sec': duration_sec,
    }

def _request_tianditu_drive(origin_lng, origin_lat, dest_lng, dest_lat, style, tk):
    post_str = json.dumps({
        'orig': f'{float(origin_lng):.6f},{float(origin_lat):.6f}',
        'dest': f'{float(dest_lng):.6f},{float(dest_lat):.6f}',
        'style': str(style),
    }, ensure_ascii=False)
    try:
        resp = _http_get(
            'https://api.tianditu.gov.cn/drive',
            {'postStr': post_str, 'type': 'search', 'tk': tk},
            timeout=_DRIVE_TIMEOUT,
            retries=_DRIVE_RETRIES,
            label=f'drive style={style}',
        )
        return _parse_drive_payload(resp.text)
    except Exception as exc:
        print(f'[tianditu_route] 驾车/步行请求失败 style={style}: {exc}', flush=True)
        return None

def _transit_linetype_param(travel_mode):
    """
    天地图 linetype 为策略位掩码（非出行方式）：
    1=较快捷, 2=少换乘, 4=少步行, 8=不坐地铁。
    传 0 会导致 results 为空；地铁模式用 7（不含「不坐地铁」）。
    """
    from transport_modes import normalize_mode_key

    key = normalize_mode_key(travel_mode)
    if key in ('metro', 'subway'):
        return 7
    if key == 'bus':
        return 7
    return 1

def _request_tianditu_transit(origin_lng, origin_lat, dest_lng, dest_lat, travel_mode, tk):
    post_str = json.dumps({
        'startposition': f'{float(origin_lng):.6f},{float(origin_lat):.6f}',
        'endposition': f'{float(dest_lng):.6f},{float(dest_lat):.6f}',
        'linetype': _transit_linetype_param(travel_mode),
    }, ensure_ascii=False)
    try:
        resp = _http_get(
            'https://api.tianditu.gov.cn/transit',
            {'type': 'busline', 'postStr': post_str, 'tk': tk},
            timeout=_TRANSIT_TIMEOUT,
            retries=_TRANSIT_RETRIES,
            label=f'transit mode={travel_mode}',
        )
        data = resp.json()
    except Exception as exc:
        print(f'[tianditu_route] 公交/地铁请求失败: {exc}', flush=True)
        print(
            '[tianditu_route] 搜索/驾车接口可用，但公交规划 /transit 无响应。'
            '请在天地图控制台确认该服务端 Key 已开通「公交规划」并配置本机 IP 白名单；'
            '本次将回退本地地铁/公交线网。',
            flush=True,
        )
        return None, None

    if str(data.get('resultCode', data.get('ret_code', '0'))) not in ('0', '1000'):
        print(
            f'[tianditu_route] 公交/地铁 API 失败 '
            f'(code={data.get("resultCode", data.get("ret_code"))}, '
            f'msg={data.get("remark", data.get("msg", "unknown"))})',
            flush=True,
        )
        return None, data

    parsed = _extract_tianditu_transit_route(data, travel_mode)
    if not parsed or len(parsed.get('route_coords') or []) < 2:
        print('[tianditu_route] 公交/地铁未解析到有效路径', flush=True)
        return None, data
    return parsed, data

def _build_plan_result(parsed, profile, walk_fallback=False, requested_mode=None):
    duration_sec = float(parsed.get('duration_sec') or 0)
    distance_m = float(parsed.get('distance_m') or 0)
    return {
        'route_coords': parsed['route_coords'],
        'distance_m': distance_m,
        'duration_min': round(duration_sec / 60.0, 2) if duration_sec > 0 else 0.0,
        'duration_sec': duration_sec,
        'source': 'tianditu',
        'profile': profile,
        'walk_fallback': bool(walk_fallback),
        'requested_mode': requested_mode,
        'route_parts': parsed.get('route_parts') or [],
        'uses_transit_line': bool(parsed.get('uses_transit_line')),
    }

def plan_tianditu_route(origin_lng, origin_lat, dest_lng, dest_lat, travel_mode, city='南京'):
    """
    调用天地图路径规划 API。

    返回 dict 或 None:
        route_coords: [[lat, lng], ...] WGS-84
        distance_m, duration_min, source='tianditu', profile
    """
    _ = city
    tk = os.environ.get('TIANDITU_API_KEY', '').strip()
    if not tk:
        print('[tianditu_route] 未配置 TIANDITU_API_KEY，跳过天地图路径规划', flush=True)
        return None

    profile = _travel_mode_to_profile(travel_mode)
    cache_key = (
        round(float(origin_lng), 5), round(float(origin_lat), 5),
        round(float(dest_lng), 5), round(float(dest_lat), 5),
        profile, str(city or '南京'), str(travel_mode), 'v5',
    )
    cached = _ROUTE_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    parsed, raw = None, None
    if profile == 'bicycling':
        # 天地图 Web 服务暂无骑行规划，交由上层回退本地路网
        return None
    if profile == 'transit':
        parsed, raw = _request_tianditu_transit(
            origin_lng, origin_lat, dest_lng, dest_lat, travel_mode, tk,
        )
        if parsed is None:
            return None
        has_line_geom = bool(parsed.get('uses_transit_line'))
        parts_connected = _transit_route_parts_connected(parsed.get('route_parts') or [])
        if not has_line_geom or not parts_connected:
            print('[tianditu_route] 公交/地铁方案折线不可用，返回空结果供上层回退', flush=True)
            return None
    elif profile == 'walking':
        parsed = _request_tianditu_drive(
            origin_lng, origin_lat, dest_lng, dest_lat, style='3', tk=tk,
        )
    else:
        parsed = _request_tianditu_drive(
            origin_lng, origin_lat, dest_lng, dest_lat, style='0', tk=tk,
        )

    if parsed is None or len(parsed.get('route_coords') or []) < 2:
        return None

    result = _build_plan_result(parsed, profile, requested_mode=travel_mode)
    _ROUTE_CACHE[cache_key] = dict(result)
    if len(_ROUTE_CACHE) > _ROUTE_CACHE_MAX:
        _ROUTE_CACHE.popitem(last=False)
    return result
