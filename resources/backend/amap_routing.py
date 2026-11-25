# -*- coding: utf-8 -*-
"""
高德路径规划（天地图失败时的回退）。
输入输出均为 WGS-84；请求前转为 GCJ-02，折线再转回 WGS-84。
"""
from __future__ import annotations

import math
import os
from collections import OrderedDict

_ROUTE_CACHE_MAX = 512
_ROUTE_CACHE: OrderedDict = {}
_AMAP_TIMEOUT = (5, 12)
_HTTP_HEADERS = {
    'User-Agent': 'IATP/1.0 (amap-routing)',
    'Accept': 'application/json',
}

_A = 6378245.0
_EE = 0.00669342162296594323


def _amap_key():
    return (
        os.environ.get('AMAP_API_KEY', '').strip()
        or os.environ.get('VITE_AMAP_API_KEY', '').strip()
    )


def has_amap_key():
    return bool(_amap_key())


def _out_of_china(lng, lat):
    return not (72.004 <= lng <= 137.8347 and 0.8293 <= lat <= 55.8271)


def _transform_lat(lng, lat):
    ret = (
        -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat
        + 0.1 * lng * lat + 0.2 * math.sqrt(abs(lng))
    )
    ret += (20.0 * math.sin(6.0 * lng * math.pi) + 20.0 * math.sin(2.0 * lng * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * math.pi) + 40.0 * math.sin(lat / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * math.pi) + 320.0 * math.sin(lat * math.pi / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lng(lng, lat):
    ret = (
        300.0 + lng + 2.0 * lat + 0.1 * lng * lng
        + 0.1 * lng * lat + 0.1 * math.sqrt(abs(lng))
    )
    ret += (20.0 * math.sin(6.0 * lng * math.pi) + 20.0 * math.sin(2.0 * lng * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * math.pi) + 40.0 * math.sin(lng / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * math.pi) + 300.0 * math.sin(lng / 30.0 * math.pi)) * 2.0 / 3.0
    return ret


def wgs84_to_gcj02(lng, lat):
    lng, lat = float(lng), float(lat)
    if _out_of_china(lng, lat):
        return lng, lat
    dlat = _transform_lat(lng - 105.0, lat - 35.0)
    dlng = _transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * math.pi
    magic = math.sin(radlat)
    magic = 1 - _EE * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((_A * (1 - _EE)) / (magic * sqrtmagic) * math.pi)
    dlng = (dlng * 180.0) / (_A / sqrtmagic * math.cos(radlat) * math.pi)
    return lng + dlng, lat + dlat


def gcj02_to_wgs84(lng, lat):
    lng, lat = float(lng), float(lat)
    if _out_of_china(lng, lat):
        return lng, lat
    glng, glat = wgs84_to_gcj02(lng, lat)
    return lng * 2.0 - glng, lat * 2.0 - glat


def _parse_amap_polyline(polyline_str):
    """高德 lng,lat;lng,lat → [[lat, lng], ...]（WGS-84）。"""
    if not polyline_str or polyline_str in ('[]', '[]'):
        return []
    if isinstance(polyline_str, list):
        polyline_str = ';'.join(str(x) for x in polyline_str)
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
        wlng, wlat = gcj02_to_wgs84(lng, lat)
        coords.append([wlat, wlng])
    return coords


def _merge_coords(existing, new_coords):
    if not new_coords:
        return existing or []
    if not existing:
        return list(new_coords)
    merged = list(existing)
    for coord in new_coords:
        prev = merged[-1]
        if abs(prev[0] - coord[0]) > 1e-6 or abs(prev[1] - coord[1]) > 1e-6:
            merged.append(coord)
    return merged


def _dedupe_coords(coords, min_dist_m=3.0):
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


def _mode_profile(travel_mode):
    from transport_modes import normalize_mode_key

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


def _is_metro_line(line):
    text = f"{line.get('type') or ''}{line.get('name') or ''}"
    return any(k in text for k in ('地铁', '轨道交通', '轻轨', '有轨', '磁浮', '市域'))


def _http_get(url, params):
    import requests

    resp = requests.get(url, params=params, timeout=_AMAP_TIMEOUT, headers=_HTTP_HEADERS)
    resp.raise_for_status()
    return resp.json()


def _extract_walk_drive_path(data, profile):
    paths = ((data.get('route') or {}).get('paths') or []) if isinstance(data.get('route'), dict) else []
    if not paths and isinstance(data.get('data'), dict):
        paths = data['data'].get('paths') or []
    if not paths:
        return None
    path = paths[0]
    coords = []
    for step in path.get('steps') or []:
        coords = _merge_coords(coords, _parse_amap_polyline(step.get('polyline')))
    if len(coords) < 2:
        coords = _parse_amap_polyline(path.get('polyline'))
    coords = _dedupe_coords(coords)
    if len(coords) < 2:
        return None
    try:
        distance_m = float(path.get('distance') or 0)
    except (TypeError, ValueError):
        distance_m = 0.0
    try:
        duration_sec = float(path.get('duration') or 0)
    except (TypeError, ValueError):
        duration_sec = 0.0
    if distance_m <= 0 and len(coords) >= 2:
        from transport_modes import path_length_from_coords
        distance_m = float(path_length_from_coords(coords) or 0)
    return {
        'route_coords': coords,
        'route_parts': [{
            'type': 'walk' if profile == 'walking' else ('bike' if profile == 'bicycling' else 'drive'),
            'geometry': 'road',
            'coords': coords,
        }],
        'distance_m': distance_m,
        'duration_sec': duration_sec,
        'uses_transit_line': False,
    }


def _buslines_of(segment):
    bus = segment.get('bus') or {}
    lines = bus.get('buslines') if isinstance(bus, dict) else None
    if isinstance(lines, list):
        return [x for x in lines if isinstance(x, dict)]
    if isinstance(lines, dict):
        return [lines]
    return []


def _transit_has_metro(transit):
    for seg in transit.get('segments') or []:
        for line in _buslines_of(seg):
            if _is_metro_line(line):
                return True
    return False


def _transit_has_bus(transit):
    for seg in transit.get('segments') or []:
        for line in _buslines_of(seg):
            if line.get('name') and not _is_metro_line(line):
                return True
    return False


def _select_transit(data, travel_mode):
    from transport_modes import normalize_mode_key

    route = data.get('route') if isinstance(data.get('route'), dict) else {}
    transits = data.get('transits') or route.get('transits') or []
    if not isinstance(transits, list):
        return None
    key = normalize_mode_key(travel_mode)
    scored = []
    for item in transits:
        if not isinstance(item, dict):
            continue
        has_m = _transit_has_metro(item)
        has_b = _transit_has_bus(item)
        if key in ('metro', 'subway'):
            if not has_m:
                continue
            rank = 0 if has_m and not has_b else 1
        elif key == 'bus':
            if not has_b and not has_m:
                continue
            rank = 0 if has_b and not has_m else 1
        else:
            rank = 2
        scored.append((rank, item))
    if not scored:
        return transits[0] if transits else None
    scored.sort(key=lambda x: x[0])
    return scored[0][1]


def _walking_coords(walking):
    if not isinstance(walking, dict):
        return []
    coords = []
    for step in walking.get('steps') or []:
        coords = _merge_coords(coords, _parse_amap_polyline(step.get('polyline')))
    if len(coords) < 2:
        coords = _parse_amap_polyline(walking.get('polyline'))
    if len(coords) < 2:
        orig = str(walking.get('origin') or '')
        dest = str(walking.get('destination') or '')
        pts = _parse_amap_polyline(';'.join(p for p in (orig, dest) if p))
        if len(pts) >= 2:
            coords = pts
    return coords


def _extract_transit(data, travel_mode):
    best = _select_transit(data, travel_mode)
    if not best:
        return None
    merged = []
    parts = []
    for segment in best.get('segments') or []:
        if not isinstance(segment, dict):
            continue
        walk_coords = _walking_coords(segment.get('walking') or {})
        if len(walk_coords) >= 2:
            parts.append({
                'type': 'walk',
                'geometry': 'transit_access',
                'coords': walk_coords,
            })
            merged = _merge_coords(merged, walk_coords)
        for line in _buslines_of(segment):
            line_coords = _parse_amap_polyline(line.get('polyline'))
            if len(line_coords) < 2:
                continue
            kind = 'metro' if _is_metro_line(line) else 'bus'
            parts.append({
                'type': kind,
                'geometry': 'transit_ride',
                'coords': line_coords,
                'line_name': str(line.get('name') or ''),
            })
            merged = _merge_coords(merged, line_coords)
        exit_walk = segment.get('exit') if isinstance(segment.get('exit'), dict) else {}
        # entrance/exit 无折线时忽略
        _ = exit_walk
    merged = _dedupe_coords(merged)
    if len(merged) < 2:
        return None
    uses_transit_line = any(
        p.get('geometry') == 'transit_ride' and len(p.get('coords') or []) >= 2
        for p in parts
    )
    try:
        distance_m = float(best.get('distance') or 0)
    except (TypeError, ValueError):
        distance_m = 0.0
    try:
        duration_sec = float(best.get('duration') or 0)
    except (TypeError, ValueError):
        duration_sec = 0.0
    if distance_m <= 0:
        from transport_modes import path_length_from_coords
        distance_m = float(path_length_from_coords(merged) or 0)
    return {
        'route_coords': merged,
        'route_parts': parts,
        'distance_m': distance_m,
        'duration_sec': duration_sec,
        'uses_transit_line': uses_transit_line,
    }


def _build_plan(parsed, profile, travel_mode):
    duration_sec = float(parsed.get('duration_sec') or 0)
    distance_m = float(parsed.get('distance_m') or 0)
    return {
        'route_coords': parsed['route_coords'],
        'distance_m': distance_m,
        'duration_min': round(duration_sec / 60.0, 2) if duration_sec > 0 else 0.0,
        'duration_sec': duration_sec,
        'source': 'amap',
        'profile': profile,
        'walk_fallback': False,
        'requested_mode': travel_mode,
        'route_parts': parsed.get('route_parts') or [],
        'uses_transit_line': bool(parsed.get('uses_transit_line')),
    }


def plan_amap_route(origin_lng, origin_lat, dest_lng, dest_lat, travel_mode, city='南京'):
    """
    调用高德路径规划。成功返回与天地图相同结构的 dict，失败返回 None。
    """
    key = _amap_key()
    if not key:
        print('[amap_route] 未配置 AMAP_API_KEY，跳过高德路径规划', flush=True)
        return None

    profile = _mode_profile(travel_mode)
    cache_key = (
        round(float(origin_lng), 5), round(float(origin_lat), 5),
        round(float(dest_lng), 5), round(float(dest_lat), 5),
        profile, str(city or '南京'), str(travel_mode), 'amap_v1',
    )
    cached = _ROUTE_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    o_lng, o_lat = wgs84_to_gcj02(origin_lng, origin_lat)
    d_lng, d_lat = wgs84_to_gcj02(dest_lng, dest_lat)
    origin = f'{o_lng:.6f},{o_lat:.6f}'
    destination = f'{d_lng:.6f},{d_lat:.6f}'
    city_name = str(city or '南京').strip() or '南京'

    try:
        if profile == 'walking':
            data = _http_get('https://restapi.amap.com/v3/direction/walking', {
                'key': key, 'origin': origin, 'destination': destination,
            })
            parsed = _extract_walk_drive_path(data, profile) if str(data.get('status')) == '1' else None
        elif profile == 'driving':
            data = _http_get('https://restapi.amap.com/v3/direction/driving', {
                'key': key, 'origin': origin, 'destination': destination, 'extensions': 'base',
            })
            parsed = _extract_walk_drive_path(data, profile) if str(data.get('status')) == '1' else None
        elif profile == 'bicycling':
            data = _http_get('https://restapi.amap.com/v4/direction/bicycling', {
                'key': key, 'origin': origin, 'destination': destination,
            })
            ok = str(data.get('errcode', data.get('status', '0'))) in ('0', '1')
            parsed = _extract_walk_drive_path(data, profile) if ok else None
        else:
            data = _http_get('https://restapi.amap.com/v3/direction/transit/integrated', {
                'key': key,
                'origin': origin,
                'destination': destination,
                'city': city_name,
                'cityd': city_name,
                'strategy': '0',
                'nightflag': '0',
            })
            if str(data.get('status')) != '1':
                print(
                    f"[amap_route] 公交/地铁失败 infocode={data.get('infocode')} "
                    f"info={data.get('info')}",
                    flush=True,
                )
                return None
            parsed = _extract_transit(data, travel_mode)
            if parsed and not parsed.get('uses_transit_line'):
                print('[amap_route] 高德方案无地铁/公交乘运段', flush=True)
                return None
    except Exception as exc:
        print(f'[amap_route] 请求失败: {exc}', flush=True)
        return None

    if parsed is None or len(parsed.get('route_coords') or []) < 2:
        print('[amap_route] 未解析到有效路径', flush=True)
        return None

    result = _build_plan(parsed, profile, travel_mode)
    _ROUTE_CACHE[cache_key] = dict(result)
    if len(_ROUTE_CACHE) > _ROUTE_CACHE_MAX:
        _ROUTE_CACHE.popitem(last=False)
    print(
        f"[amap_route] 成功 mode={travel_mode} dist_m={result['distance_m']:.0f} "
        f"parts={len(result['route_parts'])}",
        flush=True,
    )
    return result
