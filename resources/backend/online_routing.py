# -*- coding: utf-8 -*-
"""在线路径规划：天地图优先，失败则高德。"""
from __future__ import annotations

import os

from amap_routing import has_amap_key, plan_amap_route
from tianditu_routing import _is_transit_travel_mode, plan_tianditu_route


def has_online_routing_key():
    tdt = (
        os.environ.get('TIANDITU_API_KEY', '').strip()
        or os.environ.get('VITE_TIANDITU_API_KEY', '').strip()
    )
    return bool(tdt) or has_amap_key()


def online_plan_usable(plan, travel_mode):
    if not plan or len(plan.get('route_coords') or []) < 2:
        return False
    if not _is_transit_travel_mode(travel_mode):
        return True
    if plan.get('walk_fallback') or plan.get('uses_walk_fallback'):
        return False
    if plan.get('uses_transit_line'):
        return True
    return any(
        part.get('geometry') == 'transit_ride' and len(part.get('coords') or []) >= 2
        for part in (plan.get('route_parts') or [])
    )


def plan_online_route(origin_lng, origin_lat, dest_lng, dest_lat, travel_mode, city='南京'):
    """
    天地图 → 高德。两者都不可用时返回 None，由上层回退本地线网。
    """
    tdt = plan_tianditu_route(
        origin_lng, origin_lat, dest_lng, dest_lat, travel_mode, city=city,
    )
    if online_plan_usable(tdt, travel_mode):
        return tdt

    if has_amap_key():
        print('[online_route] 天地图路线不可用，改用高德路径规划', flush=True)
        amap = plan_amap_route(
            origin_lng, origin_lat, dest_lng, dest_lat, travel_mode, city=city,
        )
        if online_plan_usable(amap, travel_mode):
            return amap
        print('[online_route] 高德路线也不可用', flush=True)
    else:
        print('[online_route] 未配置 AMAP_API_KEY，无法回退高德', flush=True)
    return None
