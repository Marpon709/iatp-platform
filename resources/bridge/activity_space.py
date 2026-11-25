"""Validate and present the engine's two-anchor activity-space results."""
import math


def validate_request(payload, activity_types):
    anchors = []
    for key, label in [('origin', '出发地'), ('destination', '目的地')]:
        item = payload.get(key) or {}
        coordinates = item.get('coordinate') or []
        if len(coordinates) != 2:
            raise ValueError(f'请设置{label}。')
        longitude, latitude = map(float, coordinates)
        if not (math.isfinite(longitude) and math.isfinite(latitude)):
            raise ValueError(f'{label}坐标无效。')
        if not (118 <= longitude <= 120 and 30.8 <= latitude <= 33):
            raise ValueError('请选择南京及周边的地点。')
        anchors.append({'label': str(item.get('label') or label)[:80],
                        'coordinate': [longitude, latitude]})

    def minutes(value):
        hour, minute = map(int, str(value).split(':'))
        if not (0 <= hour < 24 and 0 <= minute < 60):
            raise ValueError('请输入有效的出发与到达时间。')
        return hour * 60 + minute

    start = minutes(payload.get('departure', '17:00'))
    end = minutes(payload.get('arrival', '18:30'))
    if end <= start:
        raise ValueError('请将最晚到达时间设置在当天出发时间之后。')
    stay = float(payload.get('stayMin', 20))
    if not math.isfinite(stay) or not 1 <= stay <= 240:
        raise ValueError('活动停留时间须在1至240分钟之间。')
    if stay >= end - start:
        raise ValueError('请为前往目的地及活动地点留出通行时间。')
    mode = str(payload.get('mode', 'walk'))
    mode = {'foot': 'walk', 'subway': 'metro', 'car': 'drive'}.get(mode, mode)
    if mode not in ['walk', 'bike', 'bus', 'metro', 'drive']:
        raise ValueError('请选择有效的交通方式。')
    known = {item['id'] for item in activity_types}
    selected = list(dict.fromkeys(payload.get('selectedTypes') or []))
    if not selected or any(item not in known for item in selected):
        raise ValueError('请至少选择一种有效的设施类型。')
    return {'origin': anchors[0], 'destination': anchors[1],
            'departure': f'{start // 60:02}:{start % 60:02}',
            'arrival': f'{end // 60:02}:{end % 60:02}',
            'stayMin': stay, 'mode': mode, 'selectedTypes': selected}


def engine_request(parameters, activity_types):
    start, end = parameters['departure'], parameters['arrival']
    activities = []
    for key, time in [('origin', start), ('destination', end)]:
        point = parameters[key]
        activities.append({'name': point['label'], 'lng': point['coordinate'][0],
                           'lat': point['coordinate'][1], 'start_time': time, 'end_time': time})
    # A shared stay duration makes the geometry independent of the selected types.
    # Compute each type once, then allow instant preference filtering in the viewer.
    types = [item['id'] for item in activity_types]
    return {'city': '南京', 'activities': activities,
            'commute_modes': [parameters['mode']], 'travel_mode': parameters['mode'],
            'selected_poi_types': types,
            'poi_activity_times': {key: parameters['stayMin'] for key in types}}


def present_result(raw, parameters, activity_types):
    if not raw.get('success'):
        return {'success': False, 'message': raw.get('message') or '活动范围计算失败。'}
    spaces = raw.get('activity_spaces') or []
    routes = []
    for space in spaces:
        route = space.get('commute_route') or []
        if len(route) >= 2:
            routes.append([[float(lon), float(lat)] for lat, lon in route])
    areas = [space['geometry'] for space in raw.get('evaluation_spaces') or []]
    polygons = []
    for area in areas:
        components = [area['coordinates']] if area['type'] == 'Polygon' else area['coordinates']
        polygons.extend(component[0] for component in components)
    for polygon in ([] if areas else raw.get('reachable_polygons') or []):
        coordinates = [[float(lon), float(lat)] for lon, lat in polygon.get('coords', [])]
        if len(coordinates) >= 3:
            if coordinates[0] != coordinates[-1]:
                coordinates.append(coordinates[0])
            polygons.append(coordinates)
            areas.append({'type': 'Polygon', 'coordinates': [coordinates]})
    categories = []
    for item in activity_types:
        detail = (raw.get('accessibility_details') or {}).get(item['id']) or {}
        categories.append({'id': item['id'], 'label': item['label'],
                           'score': float(detail.get('raw_score') or 0),
                           'count': int(detail.get('poi_count') or 0)})
    pois = []
    seen = set()
    scored_pois = [poi for space in raw.get('evaluation_spaces') or [] for poi in space['scored_pois']]
    for poi in (scored_pois if raw.get('evaluation_spaces') else raw.get('poi_markers') or []):
        if 'type' in poi:
            pois.append({'type': poi['type'], 'name': str(poi.get('name') or '未命名场所'),
                         'coordinate': [float(poi['coords'][1]), float(poi['coords'][0])],
                         'contribution': poi['contribution'], 'distance': poi['distance']})
            continue
        key = (poi.get('category'), poi.get('name'), poi.get('lng'), poi.get('lat'))
        if key in seen:
            continue
        seen.add(key)
        pois.append({'type': poi['category'], 'name': str(poi.get('name') or '未命名场所'),
                     'coordinate': [float(poi['lng']), float(poi['lat'])]})
    gap = (raw.get('gap_details') or [{}])[0]
    return {'success': True, 'parameters': parameters, 'routes': routes, 'areas': areas,
            'polygons': polygons, 'categories': categories, 'pois': pois,
            'gap': gap, 'hasActivitySpace': bool(polygons),
            'scoreMethod': 'selected_facility_union_distance_decay_sum',
            'source': {'kind': 'engine_daily_activity_range',
                       'routeSource': gap.get('route_source'), 'method': gap.get('method')},
            'message': gap.get('message') or ''}
