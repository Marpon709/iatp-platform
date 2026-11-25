"""
天地图 POI 检索与地理编码（周边搜索 / 关键词搜索 / 地址解析）。
坐标输入输出均为 WGS-84 / CGCS2000（与本地 OSM 路网、Leaflet 一致）。
"""
from __future__ import annotations

import json
import math
import os
from collections import OrderedDict

_EARTH_RADIUS_M = 6371000.0
_POI_CACHE_MAX = 256
_POI_CACHE: OrderedDict = {}

CITY_SPECIFY = {
    '南京': '156320100',
    '北京': '156110000',
    '上海': '156310000',
    '武汉': '156420100',
}

CHINA_MAP_BOUND = '73.66,3.86,135.05,53.55'

def require_tianditu_api_key():
    key = os.environ.get('TIANDITU_API_KEY', '').strip()
    if not key:
        raise ValueError('未配置 TIANDITU_API_KEY 环境变量，无法使用出行推荐')
    return key

def _city_specify(city='南京'):
    city = str(city or '南京').strip()
    return CITY_SPECIFY.get(city, city or '156000000')

def _haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(float, (lon1, lat1, lon2, lat2))
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))

def _keywords_to_query(keywords):
    if not keywords:
        return ''
    if isinstance(keywords, str):
        return keywords.strip()
    parts = [str(k).strip() for k in keywords if str(k).strip()]
    return ' '.join(parts)

def _normalize_geocode_query(keywords, city='南京'):
    query = _keywords_to_query(keywords)
    city = str(city or '').strip()
    if city and query.startswith(city):
        query = query[len(city):].strip(' ,，、')
    return query or _keywords_to_query(keywords)

def _parse_lonlat(location):
    location = str(location or '')
    if ',' not in location:
        return None
    try:
        lng, lat = (float(x) for x in location.split(',')[:2])
    except (TypeError, ValueError):
        return None
    return lng, lat

def _parse_location_item(name, address, location, source='tianditu', category='天地图检索', extra=None):
    parsed = _parse_lonlat(location)
    if not parsed:
        return None
    lng, lat = parsed
    display = str(name or '').strip()
    addr = str(address or '').strip()
    if addr and display and addr != display:
        display = f'{display} - {addr}'
    elif addr:
        display = addr
    if not display:
        return None
    item = {
        'display_name': display,
        'lat': round(lat, 6),
        'lon': round(lng, 6),
        'source': source,
        'category': category,
        'poi_category': '',
        'poi_subtype': '',
    }
    if extra:
        item.update(extra)
    return item

def _parse_poi_item(poi, source='tianditu', category='天地图POI检索'):
    name = str(poi.get('name') or '').strip()
    address = str(poi.get('address') or '').strip()
    subtype = str(poi.get('typeName') or '').strip()
    return _parse_location_item(
        name,
        address,
        poi.get('lonlat'),
        source=source,
        category=category,
        extra={
            'poi_category': str(poi.get('typeCode') or ''),
            'poi_subtype': subtype,
        },
    )

def _request_tianditu(url, params, timeout=8):
    import requests

    resp = requests.get(url, params=params, timeout=timeout)
    try:
        data = resp.json()
    except ValueError:
        print(f'[tianditu_poi] 非 JSON 响应 status={resp.status_code}', flush=True)
        return None

    if data.get('code') in (301012, 18):
        print(
            f'[tianditu_poi] 密钥类型不匹配: {data.get("msg")} — {data.get("resolve")} '
            f'（后端需使用「服务端」类型 Key）',
            flush=True,
        )
        return None

    status = data.get('status') or {}
    if isinstance(status, dict):
        infocode = status.get('infocode')
        if infocode not in (None, 1000, '1000'):
            print(
                f'[tianditu_poi] API 失败: {status.get("cndesc", "unknown")} '
                f'({infocode})',
                flush=True,
            )
            return None
    return data

def _search_v2(post_obj, tk):
    return _request_tianditu(
        'https://api.tianditu.gov.cn/v2/search',
        {
            'postStr': json.dumps(post_obj, ensure_ascii=False),
            'type': 'query',
            'tk': tk,
        },
    )

def search_tianditu_geocode_candidates(keywords, city='南京', limit=5):
    """
    地址/POI 检索：行政区划搜索 → 普通搜索 → 地理编码。
    返回 geocode API 兼容的结果列表。
    """
    tk = os.environ.get('TIANDITU_API_KEY', '').strip()
    if not tk:
        return [], None

    city = str(city or '南京').strip()
    raw_query = _keywords_to_query(keywords)
    query = _normalize_geocode_query(keywords, city) or raw_query
    if not query:
        return [], None

    limit = max(1, int(limit))
    seen = set()
    results = []
    geocode_error = None
    specify = _city_specify(city)

    def _append(item):
        if not item:
            return
        key = (round(float(item['lat']), 5), round(float(item['lon']), 5))
        if key in seen:
            return
        seen.add(key)
        results.append(item)

    # 策略1：行政区划区域搜索
    try:
        data = _search_v2({
            'keyWord': query,
            'queryType': 12,
            'start': 0,
            'count': min(limit * 2, 20),
            'specify': specify,
        }, tk)
        if data:
            for poi in (data.get('pois') or [])[:limit]:
                _append(_parse_poi_item(poi, source='tianditu'))
    except Exception as exc:
        print(f'[geocode] 天地图行政区划搜索异常: {exc}', flush=True)

    # 策略2：普通搜索（含建议词）
    if len(results) < limit:
        try:
            data = _search_v2({
                'keyWord': query,
                'queryType': 1,
                'start': 0,
                'count': min(limit * 2, 20),
                'level': 12,
                'mapBound': CHINA_MAP_BOUND,
                'specify': specify,
            }, tk)
            if data:
                for poi in (data.get('pois') or []):
                    _append(_parse_poi_item(poi, source='tianditu'))
                    if len(results) >= limit:
                        break
                for suggest in (data.get('prompt') or {}).get('admins') or []:
                    if len(results) >= limit:
                        break
                    _append(_parse_location_item(
                        suggest.get('adminName') or suggest.get('keyword') or query,
                        '',
                        suggest.get('lonlat'),
                        source='tianditu_suggest',
                        category='天地图建议',
                    ))
        except Exception as exc:
            print(f'[geocode] 天地图普通搜索异常: {exc}', flush=True)

    # 策略3：正向地理编码
    if len(results) < limit:
        import requests

        for address in dict.fromkeys([query, raw_query]):
            if not address:
                continue
            try:
                resp = requests.get(
                    'https://api.tianditu.gov.cn/geocoder',
                    params={
                        'ds': json.dumps({'keyWord': address}, ensure_ascii=False),
                        'tk': tk,
                    },
                    timeout=5,
                )
                data = resp.json()
                if str(data.get('status')) == '0' and data.get('location'):
                    loc = data['location']
                    _append(_parse_location_item(
                        address,
                        '',
                        f"{loc.get('lon')},{loc.get('lat')}",
                        source='tianditu',
                        category='天地图地理编码',
                    ))
                    break
                geocode_error = data.get('msg', geocode_error or '未知错误')
            except Exception as exc:
                print(f'[geocode] 天地图地理编码请求异常: {exc}', flush=True)

    if results:
        print(f'[geocode] 天地图检索成功 {len(results)} 条（query={query!r}）', flush=True)
    return results[:limit], geocode_error if not results else None

def _parse_poi_search_item(poi, anchor_lon, anchor_lat, max_radius_m):
    parsed = _parse_lonlat(poi.get('lonlat'))
    if not parsed:
        return None
    lng, lat = parsed
    dist_m = _haversine_m(anchor_lon, anchor_lat, lng, lat)
    if dist_m > float(max_radius_m):
        return None

    name = str(poi.get('name') or '').strip()
    address = str(poi.get('address') or '').strip()
    display = name or address or 'POI'
    if address and name and address != name:
        display = f'{name} - {address}'

    distance_raw = poi.get('distance')
    if distance_raw not in (None, ''):
        try:
            dist_m = float(str(distance_raw).replace('km', '').replace('m', '').strip())
            if 'km' in str(distance_raw):
                dist_m *= 1000.0
        except (TypeError, ValueError):
            pass

    subtype = str(poi.get('typeName') or '').strip()
    return {
        'display_name': display,
        'lat': lat,
        'lon': lng,
        'distance_m': round(dist_m),
        'distance_km': round(dist_m / 1000.0, 2),
        'rating': 4.0,
        'poi_category': '',
        'poi_subtype': subtype,
        'category_label': subtype or 'POI',
        'source': 'tianditu',
        'tianditu_id': str(poi.get('hotPointID') or ''),
    }

def search_tianditu_pois(
    keywords,
    origin_lon,
    origin_lat,
    city='南京',
    max_radius_m=50_000,
    limit=8,
):
    tk = require_tianditu_api_key()
    query = _keywords_to_query(keywords)
    if not query:
        return []

    origin_lon = float(origin_lon)
    origin_lat = float(origin_lat)
    city = str(city or '南京').strip()
    max_radius_m = max(500.0, float(max_radius_m))
    limit = max(1, int(limit))
    specify = _city_specify(city)

    cache_key = (
        round(origin_lon, 4), round(origin_lat, 4),
        query, city, int(max_radius_m), limit,
    )
    cached = _POI_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)

    seen = set()
    candidates = []
    search_radius = min(int(max_radius_m), 10_000)

    around_data = _search_v2({
        'keyWord': query,
        'queryType': 3,
        'start': 0,
        'count': min(max(limit * 3, 10), 25),
        'level': 12,
        'pointLonlat': f'{origin_lon:.6f},{origin_lat:.6f}',
        'queryRadius': str(search_radius),
    }, tk)
    if around_data:
        for poi in around_data.get('pois') or []:
            item = _parse_poi_search_item(poi, origin_lon, origin_lat, max_radius_m)
            if not item:
                continue
            key = (round(item['lat'], 5), round(item['lon'], 5))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(item)

    if len(candidates) < limit:
        region_data = _search_v2({
            'keyWord': query,
            'queryType': 12,
            'start': 0,
            'count': min(max(limit * 3, 10), 25),
            'specify': specify,
        }, tk)
        if region_data:
            for poi in region_data.get('pois') or []:
                item = _parse_poi_search_item(poi, origin_lon, origin_lat, max_radius_m)
                if not item:
                    continue
                key = (round(item['lat'], 5), round(item['lon'], 5))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(item)

    candidates.sort(key=lambda x: (x['distance_m'], -x.get('rating', 0)))
    result = candidates[:limit]
    _POI_CACHE[cache_key] = list(result)
    if len(_POI_CACHE) > _POI_CACHE_MAX:
        _POI_CACHE.popitem(last=False)
    return result
