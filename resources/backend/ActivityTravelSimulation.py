"""
活动出行模拟后端模块：个人活动出行链分段碳排放计算。
"""

def _scheme_for_transport_mode(mode: str) -> dict:
    """将前端交通方式映射为路线方案（用于单段碳排计算）。"""
    from travel_recommendation import ROUTE_SCHEME_DEFS, _WALK_LEG_SCHEME

    key = str(mode or "foot").strip().lower()
    if key in ("subway", "metro"):
        return next(s for s in ROUTE_SCHEME_DEFS if s["id"] == "transit_metro")
    if key == "bus":
        return next(s for s in ROUTE_SCHEME_DEFS if s["id"] == "transit_bus")
    if key == "drive":
        return next(s for s in ROUTE_SCHEME_DEFS if s["id"] == "regular")
    if key == "bike":
        return next(s for s in ROUTE_SCHEME_DEFS if s["id"] == "low_carbon")
    return _WALK_LEG_SCHEME

def _resolve_leg_plan(prev, curr, mode, city="南京"):
    """单段路线规划：天地图/高德 → 本地线网/路网 → 步行 → 直线估算。"""
    from travel_recommendation import (
        _WALK_LEG_SCHEME,
        _build_leg_plan_for_scheme,
        _haversine_m,
        MODE_LABELS_ZH,
    )
    from carbon_budget import CARBON_ALPHA_G_PER_KM
    from transport_modes import speed_ms

    scheme = _scheme_for_transport_mode(mode)
    leg_plan = _build_leg_plan_for_scheme(prev, curr, scheme, city=city)
    if leg_plan:
        return leg_plan, None

    leg_plan = _build_leg_plan_for_scheme(prev, curr, _WALK_LEG_SCHEME, city=city)
    if leg_plan:
        leg_plan = dict(leg_plan)
        leg_plan["walk_fallback"] = True
        return leg_plan, "route_walk_fallback"

    origin_pt = (float(prev["lon"]), float(prev["lat"]))
    dest_pt = (float(curr["lon"]), float(curr["lat"]))
    dist_m = float(_haversine_m(origin_pt[0], origin_pt[1], dest_pt[0], dest_pt[1]) or 0)
    if dist_m <= 0:
        return None, "route_unavailable"

    used_mode = str(mode or "foot").strip().lower()
    if used_mode in ("walk", "foot"):
        used_mode = "foot"
    elif used_mode in ("subway", "metro"):
        used_mode = "metro"
    alpha = float(CARBON_ALPHA_G_PER_KM.get(
        used_mode if used_mode in CARBON_ALPHA_G_PER_KM else "foot",
        CARBON_ALPHA_G_PER_KM["foot"],
    ))
    dist_km = dist_m / 1000.0
    time_min = dist_m / max(speed_ms(used_mode), 0.1) / 60.0
    return {
        "mode": used_mode,
        "mode_label": MODE_LABELS_ZH.get(used_mode, used_mode),
        "distance_m": round(dist_m),
        "distance_km": round(dist_km, 2),
        "time_min": round(time_min, 1),
        "carbon_g": round(dist_km * alpha, 1),
        "alpha_g_per_km": alpha,
        "route_coords": [[float(prev["lat"]), float(prev["lon"])], [float(curr["lat"]), float(curr["lon"])]],
        "route_parts": [],
        "route_source": "straight_line",
        "walk_fallback": True,
    }, "route_straight_line_fallback"

def run_personal_activity_carbon(home=None, activities=None, stops=None, city="南京"):
    """
    按出行链分段计算碳排放。
    - stops：按时间顺序的活动站点列表（活动1→活动2→…），每段使用对应站点 mode；
    - home + activities：兼容旧版「居住位置→各出行点」。
    """
    from online_routing import has_online_routing_key
    from travel_recommendation import _merge_route_coords, _stop_dict

    if not has_online_routing_key():
        return {"success": False, "message": "未配置天地图或高德路径规划 Key"}

    chain = []
    if stops:
        if not isinstance(stops, list) or len(stops) < 2:
            return {"success": False, "message": "请至少提供两个已确认的活动地点"}
        for i, stop in enumerate(stops):
            lat = stop.get("lat", stop.get("latitude"))
            lng = stop.get("lng", stop.get("lon", stop.get("longitude")))
            if lat is None or lng is None:
                return {"success": False, "message": f"活动 {i + 1} 缺少地点坐标"}
            label = str(stop.get("label") or stop.get("name") or f"活动 {i + 1}")
            chain.append({
                "lat": lat,
                "lng": lng,
                "label": label,
                "mode": stop.get("mode") or stop.get("transport_mode") or "foot",
                "time": stop.get("time"),
            })
    else:
        if not home:
            return {"success": False, "message": "请先确认今日固定活动安排"}
        if not activities:
            return {"success": False, "message": "请至少填写一条完整出行信息"}
        home_lat = home.get("lat", home.get("latitude"))
        home_lng = home.get("lng", home.get("lon", home.get("longitude")))
        if home_lat is None or home_lng is None:
            return {"success": False, "message": "居住位置坐标无效"}
        chain.append({
            "lat": home_lat,
            "lng": home_lng,
            "label": "居住位置",
            "mode": (activities[0] or {}).get("mode") or "foot",
        })
        for i, act in enumerate(activities):
            lat = act.get("lat", act.get("latitude"))
            lng = act.get("lng", act.get("lon", act.get("longitude")))
            if lat is None or lng is None:
                return {"success": False, "message": f"出行 {i + 1} 缺少地点坐标"}
            chain.append({
                "lat": lat,
                "lng": lng,
                "label": str(act.get("label") or f"出行 {i + 1}"),
                "mode": act.get("mode") or act.get("transport_mode") or "foot",
                "time": act.get("time"),
            })

    prev = _stop_dict("origin", chain[0]["label"], chain[0]["lat"], chain[0]["lng"])
    legs = []
    warnings = []
    total_carbon_g = 0.0
    total_dist_m = 0.0
    total_time_min = 0.0
    merged_coords = []

    for i in range(1, len(chain)):
        act = chain[i]
        label = act["label"]
        mode = act.get("mode") or "foot"
        curr = _stop_dict("activity", label, act["lat"], act["lng"], time=act.get("time"))

        leg_plan, warning = _resolve_leg_plan(prev, curr, mode, city=city)
        if not leg_plan:
            return {
                "success": False,
                "message": f"{prev.get('label', '')} → {label} 路线规划失败",
                "legs": legs,
            }
        if warning:
            warnings.append(f"{prev.get('label', '')} → {label}: {warning}")

        # 校验碳排放：若 leg_plan 返回 carbon_g=0 但实际为非零碳出行方式，
        # 则根据距离和该方式的碳排放因子重新计算（防御路线回退为步行导致碳排为 0）
        carbon_g = float(leg_plan.get("carbon_g") or 0)
        dist_km = float(leg_plan.get("distance_km") or 0)
        used_mode = str(leg_plan.get("mode") or mode or "foot").strip().lower()
        walk_fallback = bool(leg_plan.get("walk_fallback"))
        if carbon_g <= 0 and dist_km > 0 and not walk_fallback:
            from carbon_budget import CARBON_ALPHA_G_PER_KM
            alpha = float(CARBON_ALPHA_G_PER_KM.get(
                used_mode if used_mode in CARBON_ALPHA_G_PER_KM else "foot",
                CARBON_ALPHA_G_PER_KM.get("foot", 0),
            ))
            if alpha > 0:
                carbon_g = dist_km * alpha
        elif walk_fallback and carbon_g <= 0 and dist_km > 0:
            # 步行回退时按原始请求方式重新估算碳排
            from carbon_budget import CARBON_ALPHA_G_PER_KM
            req_mode = str(mode or "foot").strip().lower()
            alpha = float(CARBON_ALPHA_G_PER_KM.get(
                req_mode if req_mode in CARBON_ALPHA_G_PER_KM else "foot",
                CARBON_ALPHA_G_PER_KM.get("foot", 0),
            ))
            if alpha > 0:
                carbon_g = dist_km * alpha

        leg_entry = {
            "index": i - 1,
            "from_label": prev.get("label", ""),
            "to_label": label,
            "time": act.get("time"),
            "transport_mode": mode,
            "mode_label": leg_plan.get("mode_label"),
            "distance_km": round(dist_km, 2),
            "distance_m": leg_plan.get("distance_m"),
            "time_min": leg_plan.get("time_min"),
            "carbon_g": round(carbon_g, 1),
            "alpha_g_per_km": leg_plan.get("alpha_g_per_km"),
            "route_coords": leg_plan.get("route_coords"),
            "route_parts": leg_plan.get("route_parts"),
            "route_source": leg_plan.get("route_source"),
            "walk_fallback": walk_fallback,
        }
        legs.append(leg_entry)
        total_carbon_g += carbon_g
        total_dist_m += float(leg_plan.get("distance_m") or 0)
        total_time_min += float(leg_plan.get("time_min") or 0)
        merged_coords = _merge_route_coords(merged_coords, leg_plan.get("route_coords") or [])
        prev = curr

    first = chain[0]
    return {
        "success": True,
        "city": city,
        "legs": legs,
        "leg_count": len(legs),
        "total_carbon_g": round(total_carbon_g, 1),
        "total_carbon_kg": round(total_carbon_g / 1000.0, 3),
        "total_distance_km": round(total_dist_m / 1000.0, 2),
        "total_time_min": round(total_time_min, 1),
        "route_coords": merged_coords,
        "home": {"lat": float(first["lat"]), "lng": float(first["lng"])},
        "warnings": warnings,
    }
