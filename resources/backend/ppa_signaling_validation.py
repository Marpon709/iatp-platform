# -*- coding: utf-8 -*-
"""Held-out A–X–B PPA validation on processed Nanjing signaling data."""
from __future__ import annotations

import io
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from datetime import datetime

import numpy as np
from pyproj import Transformer
from shapely.geometry import Point, LineString
from shapely.ops import transform as shp_transform

SRC = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "POI", "Nanjing", "Users_signaling.geojson",
)
OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "POI", "Nanjing", "ppa_validation_results",
)
CKPT_PATH = os.path.join(OUT_DIR, "ppa_validation_checkpoint.jsonl")

EARTH_R = 6371000.0
MIN_STAY_MIN = 5.0
MIN_SEP_M = 80.0
POS_TOL_M = 100.0
BUFFER_M_LIST = (500.0, 1000.0)
N_BOOT = 1000
RNG = np.random.default_rng(42)
PPA_TOL_M = 100.0
_TO_UTM50 = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)

MODE_MAP = {
    "slow": "walk",
    "walk": "walk",
    "foot": "walk",
    "bike": "bike",
    "metro": "metro",
    "subway": "metro",
    "car": "drive",
    "drive": "drive",
    "public": "bus",
    "bus": "bus",
}
# km/h, aligned with transport_modes.CARBON_BUDGET_MODE_SPEEDS_KMH
MODE_SPEED_KMH = {
    "walk": 4.8,
    "bike": 12.0,
    "bus": 25.0,
    "metro": 30.0,
    "drive": 50.0,
    "car": 50.0,
}
MODE_VMAX_KMH = {
    "walk": 8.0,
    "bike": 20.0,
    "bus": 40.0,
    "metro": 45.0,
    "drive": 80.0,
}
POI_CAT = {
    "SHOPPING&DINING3": "shopping",
    "CULTURE4": "recreation",
    "SCHOOLING5": "education",
    "MEDICAL6": "medical",
    "home": "home",
    "second": "second",
    "work": "work",
    "No POI found": None,
}


def parse_val(line: str) -> str:
    return line.split(":", 1)[1].strip().rstrip(",").strip().strip('"')


def haversine_m(lon1, lat1, lon2, lat2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(min(1.0, math.sqrt(a)))


def parse_dt(date_s, time_s):
    return datetime.strptime(f"{date_s} {time_s}", "%Y-%m-%d %H:%M:%S")


def infer_mode(raw_mode, velocity_ms):
    key = MODE_MAP.get(str(raw_mode).strip().lower(), "walk")
    if key == "walk" and velocity_ms is not None:
        try:
            v = float(velocity_ms)
            if v >= 3.0:
                return "bike"
        except (TypeError, ValueError):
            pass
    return key


def map_poi(raw):
    if raw is None:
        return None
    return POI_CAT.get(raw, None)


def load_trips(path):
    trips = []
    cur = {}
    coords = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.lstrip()
            if s.startswith('"coordinates"'):
                coords = []
            elif coords is not None and s[:1].isdigit() or (coords is not None and s.startswith("-")):
                try:
                    coords.append(float(s.strip().rstrip(",")))
                except ValueError:
                    pass
                if len(coords) >= 2:
                    cur["lon"], cur["lat"] = coords[0], coords[1]
                    coords = None
            elif s.startswith('"userId"'):
                cur["userId"] = parse_val(s)
            elif s.startswith('"type"'):
                v = parse_val(s)
                if v in ("origin", "destination"):
                    cur["kind"] = v
            elif s.startswith('"date"'):
                cur["date"] = parse_val(s)
            elif s.startswith('"time"'):
                cur["time"] = parse_val(s)
            elif s.startswith('"poi_type"'):
                cur["poi_type"] = parse_val(s)
            elif s.startswith('"travel_mode"'):
                cur["travel_mode"] = parse_val(s)
            elif s.startswith('"travel_time"'):
                try:
                    cur["travel_time"] = float(parse_val(s))
                except ValueError:
                    cur["travel_time"] = None
            elif s.startswith('"velocity"'):
                try:
                    cur["velocity"] = float(parse_val(s))
                except ValueError:
                    cur["velocity"] = None
            elif s.startswith('"total_distance"'):
                try:
                    cur["total_distance"] = float(parse_val(s))
                except ValueError:
                    cur["total_distance"] = None
            elif s.startswith("}") and "kind" in cur and "userId" in cur:
                trips.append(cur)
                cur = {}
    return trips


def pair_od(trips):
    """Pair consecutive origin-destination records per user (file order)."""
    by_user = defaultdict(list)
    for rec in trips:
        by_user[rec["userId"]].append(rec)
    od = []
    for uid, recs in by_user.items():
        i = 0
        while i < len(recs) - 1:
            a, b = recs[i], recs[i + 1]
            if a.get("kind") == "origin" and b.get("kind") == "destination":
                od.append((a, b))
                i += 2
            else:
                i += 1
    return od


def build_day_stops(od_pairs):
    """Merge OD into stay nodes per (user, date)."""
    days = defaultdict(list)
    for o, d in od_pairs:
        uid = o["userId"]
        date = o.get("date") or d.get("date")
        days[(uid, date)].append((o, d))

    out = {}
    for key, pairs in days.items():
        pairs.sort(key=lambda x: parse_dt(x[0]["date"], x[0]["time"]))
        stops = []
        for i, (o, d) in enumerate(pairs):
            o_dt = parse_dt(o["date"], o["time"])
            d_dt = parse_dt(d["date"], d["time"])
            mode = infer_mode(o.get("travel_mode") or d.get("travel_mode"), o.get("velocity"))
            if i == 0:
                stops.append({
                    "lon": o["lon"], "lat": o["lat"],
                    "poi": o.get("poi_type"),
                    "arrive": None,
                    "leave": o_dt,
                    "mode_out": mode,
                })
            else:
                prev = stops[-1]
                dist = haversine_m(prev["lon"], prev["lat"], o["lon"], o["lat"])
                if dist <= MIN_SEP_M:
                    prev["leave"] = o_dt
                    prev["mode_out"] = mode
                    if prev.get("poi") in (None, "No POI found"):
                        prev["poi"] = o.get("poi_type")
                else:
                    stops.append({
                        "lon": o["lon"], "lat": o["lat"],
                        "poi": o.get("poi_type"),
                        "arrive": prev["leave"],
                        "leave": o_dt,
                        "mode_out": mode,
                    })
            stops.append({
                "lon": d["lon"], "lat": d["lat"],
                "poi": d.get("poi_type"),
                "arrive": d_dt,
                "leave": None,
                "mode_in": mode,
                "travel_time_s": o.get("travel_time"),
                "velocity": o.get("velocity"),
            })
        # merge dest with next origin if same place
        merged = []
        for st in stops:
            if merged:
                last = merged[-1]
                dist = haversine_m(last["lon"], last["lat"], st["lon"], st["lat"])
                if dist <= MIN_SEP_M:
                    if st.get("arrive") and (last.get("arrive") is None or st["arrive"] < last["arrive"]):
                        last["arrive"] = st["arrive"]
                    if st.get("leave"):
                        last["leave"] = st["leave"]
                    if st.get("mode_out"):
                        last["mode_out"] = st["mode_out"]
                    if st.get("mode_in"):
                        last["mode_in"] = st["mode_in"]
                    if last.get("poi") in (None, "No POI found") and st.get("poi"):
                        last["poi"] = st["poi"]
                    continue
            merged.append(st)
        out[key] = merged
    return out


def point_to_segment_m(px, py, ax, ay, bx, by):
    """Approximate meters from P to segment AB using local equirectangular."""
    lat0 = math.radians((ay + by + py) / 3.0)
    kx = EARTH_R * math.cos(lat0) * math.pi / 180.0
    ky = EARTH_R * math.pi / 180.0
    axm, aym = ax * kx, ay * ky
    bxm, bym = bx * kx, by * ky
    pxm, pym = px * kx, py * ky
    vx, vy = bxm - axm, bym - aym
    wx, wy = pxm - axm, pym - aym
    vv = vx * vx + vy * vy
    t = 0.0 if vv <= 1e-9 else max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
    dx, dy = pxm - (axm + t * vx), pym - (aym + t * vy)
    return math.hypot(dx, dy)


def ellipse_area_m2(budget_m, d_ab):
    a = budget_m / 2.0
    c = d_ab / 2.0
    if a <= c + 1e-6:
        return 0.0
    b = math.sqrt(max(0.0, a * a - c * c))
    return math.pi * a * b


def buffer_area_m2(d_ab, width_m):
    return d_ab * 2.0 * width_m + math.pi * width_m * width_m


HOME_POI = "home"
WORK_POI = "work"  # 停留超过 4 小时的 second 已改标为 work
_MAX_USERS = int(os.environ.get("PPA_MAX_USERS", "0") or 0)


def extract_triples(day_stops):
    """仅保留 居住地(home) → 中间活动 X → 工作地(work) 样本。
    A 取 X 之前最近的 home，B 取 X 之后最近的 work；X 本身不是 home/work。
    """
    triples = []
    n_users = len({k[0] for k in day_stops})
    n_user_days = len(day_stops)
    n_chains = 0
    n_stops_ok = 0
    n_home_work_days = 0
    for (uid, date), stops in day_stops.items():
        if len(stops) >= 3:
            n_stops_ok += 1
            n_chains += 1
        has_home = any(st.get("poi") == HOME_POI for st in stops)
        has_work = any(st.get("poi") == WORK_POI for st in stops)
        if has_home and has_work:
            n_home_work_days += 1
        n = len(stops)
        for j in range(1, n - 1):
            X = stops[j]
            if X.get("poi") in (HOME_POI, WORK_POI):
                continue
            A = None
            for i in range(j - 1, -1, -1):
                if stops[i].get("poi") == HOME_POI:
                    A = stops[i]
                    break
            B = None
            for k in range(j + 1, n):
                if stops[k].get("poi") == WORK_POI:
                    B = stops[k]
                    break
            if A is None or B is None:
                continue
            if A.get("leave") is None or B.get("arrive") is None or X.get("arrive") is None:
                continue
            d_ax = haversine_m(A["lon"], A["lat"], X["lon"], X["lat"])
            d_xb = haversine_m(X["lon"], X["lat"], B["lon"], B["lat"])
            d_ab = haversine_m(A["lon"], A["lat"], B["lon"], B["lat"])
            if min(d_ax, d_xb, d_ab) < MIN_SEP_M:
                continue
            t_avail = (B["arrive"] - A["leave"]).total_seconds()
            if t_avail <= 0:
                continue
            mode = X.get("mode_in") or A.get("mode_out") or "walk"
            v_ms = MODE_SPEED_KMH[mode] / 3.6
            budget_m = v_ms * t_avail
            stay_s = None
            if X.get("leave") and X.get("arrive"):
                stay_s = (X["leave"] - X["arrive"]).total_seconds()
            t_ax = (X["arrive"] - A["leave"]).total_seconds()
            t_xb = None
            if X.get("leave"):
                t_xb = (B["arrive"] - X["leave"]).total_seconds()
            cat = map_poi(X.get("poi"))
            identifiable = cat in ("shopping", "recreation", "education", "medical")

            in_ppa = (d_ax + d_xb) <= (budget_m + POS_TOL_M)
            in_ppa_notol = (d_ax + d_xb) <= budget_m
            circuity = 1.3
            in_ppa_net = (d_ax + d_xb) * circuity <= (budget_m + POS_TOL_M)
            in_buf = {w: point_to_segment_m(X["lon"], X["lat"], A["lon"], A["lat"], B["lon"], B["lat"]) <= w
                      for w in BUFFER_M_LIST}

            # independent feasibility
            vmax = MODE_VMAX_KMH[mode] / 3.6
            time_ok = True
            if t_ax <= 0:
                time_ok = False
            if t_xb is not None and t_xb <= 0:
                time_ok = False
            stay_ok = stay_s is not None and stay_s >= MIN_STAY_MIN * 60.0
            speed_ok = True
            if t_ax > 0 and d_ax / t_ax > vmax * 1.5:
                speed_ok = False
            if t_xb is not None and t_xb > 0 and d_xb / t_xb > vmax * 1.5:
                speed_ok = False
            feasible = time_ok and stay_ok and speed_ok

            triples.append({
                "userId": uid,
                "date": date,
                "mode": mode,
                "poi_A": A.get("poi"),
                "poi_B": B.get("poi"),
                "poi_raw": X.get("poi"),
                "poi_cat": cat,
                "identifiable": identifiable,
                "t_avail_s": t_avail,
                "stay_s": stay_s,
                "d_ax": d_ax, "d_xb": d_xb, "d_ab": d_ab,
                "budget_m": budget_m,
                "in_ppa": in_ppa,
                "in_ppa_notol": in_ppa_notol,
                "in_ppa_net": in_ppa_net,
                "in_buf500": in_buf[500.0],
                "in_buf1000": in_buf[1000.0],
                "feasible": feasible,
                "ppa_area_m2": ellipse_area_m2(budget_m, d_ab),
                "buf500_area_m2": buffer_area_m2(d_ab, 500.0),
                "buf1000_area_m2": buffer_area_m2(d_ab, 1000.0),
            })
    return triples, n_users, n_user_days, n_chains, n_stops_ok, n_home_work_days


def rate(xs):
    xs = list(xs)
    return float(np.mean(xs)) if xs else float("nan")


def cluster_bootstrap(triples, field, n_boot=N_BOOT):
    by_user = defaultdict(list)
    for t in triples:
        by_user[t["userId"]].append(1.0 if t[field] else 0.0)
    users = list(by_user)
    if not users:
        return (float("nan"),) * 3
    stats = []
    for _ in range(n_boot):
        samp = RNG.choice(users, size=len(users), replace=True)
        vals = []
        for u in samp:
            vals.extend(by_user[u])
        stats.append(float(np.mean(vals)))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    point = rate(t[field] for t in triples)
    return point, float(lo), float(hi)


def summarize(triples, n_users_raw, n_users_stops, n_user_days, n_chains, n_home_work_days=0):
    users = {t["userId"] for t in triples}
    s = {
        "n_users_in_file": n_users_raw,
        "n_user_days": n_user_days,
        "n_user_days_with_3plus_stops": n_chains,
        "n_user_days_with_home_and_work": n_home_work_days,
        "sample_definition": "home -> held-out X -> work",
        "n_valid_triples": len(triples),
        "n_users_with_valid_triples": len(users),
    }
    if not triples:
        return s

    p, lo, hi = cluster_bootstrap(triples, "in_ppa")
    s["ppa_containment"] = {"rate": p, "ci95": [lo, hi]}
    p2, lo2, hi2 = cluster_bootstrap(triples, "in_ppa_notol")
    s["ppa_containment_no_tol"] = {"rate": p2, "ci95": [lo2, hi2]}
    p3, lo3, hi3 = cluster_bootstrap(triples, "in_ppa_net")
    s["ppa_containment_network_circuity_1_3"] = {"rate": p3, "ci95": [lo3, hi3]}
    b5, l5, h5 = cluster_bootstrap(triples, "in_buf500")
    s["buffer500_containment"] = {"rate": b5, "ci95": [l5, h5]}
    b10, l10, h10 = cluster_bootstrap(triples, "in_buf1000")
    s["buffer1000_containment"] = {"rate": b10, "ci95": [l10, h10]}
    f, fl, fh = cluster_bootstrap(triples, "feasible")
    s["feasibility"] = {"rate": f, "ci95": [fl, fh]}

    areas = [t["ppa_area_m2"] for t in triples]
    s["ppa_area_km2"] = {
        "mean": float(np.mean(areas)) / 1e6,
        "median": float(np.median(areas)) / 1e6,
        "p25": float(np.percentile(areas, 25)) / 1e6,
        "p75": float(np.percentile(areas, 75)) / 1e6,
    }
    s["buffer500_area_km2_mean"] = float(np.mean([t["buf500_area_m2"] for t in triples])) / 1e6
    s["buffer1000_area_km2_mean"] = float(np.mean([t["buf1000_area_m2"] for t in triples])) / 1e6

    ident = [t for t in triples if t["identifiable"]]
    cats_all = {t["poi_cat"] for t in ident}
    cats_in = {t["poi_cat"] for t in ident if t["in_ppa"]}
    s["poi_type_match"] = {
        "identifiable_samples": len(ident),
        "categories_identifiable": sorted(cats_all),
        "categories_inside_ppa": sorted(cats_in),
        "rate": (len(cats_in) / len(cats_all)) if cats_all else float("nan"),
        "sample_containment_among_identifiable": rate(t["in_ppa"] for t in ident),
    }

    by_mode = {}
    for mode in sorted({t["mode"] for t in triples}):
        sub = [t for t in triples if t["mode"] == mode]
        by_mode[mode] = {
            "n": len(sub),
            "ppa_containment": rate(t["in_ppa"] for t in sub),
            "buffer500_containment": rate(t["in_buf500"] for t in sub),
            "feasibility": rate(t["in_ppa"] and t["feasible"] for t in sub) if False else rate(t["feasible"] for t in sub),
            "mean_ppa_area_km2": float(np.mean([t["ppa_area_m2"] for t in sub])) / 1e6,
        }
    s["by_mode"] = by_mode

    by_cat = {}
    for cat in sorted({t["poi_cat"] for t in ident}):
        sub = [t for t in ident if t["poi_cat"] == cat]
        by_cat[cat] = {
            "n": len(sub),
            "ppa_containment": rate(t["in_ppa"] for t in sub),
            "buffer500_containment": rate(t["in_buf500"] for t in sub),
        }
    s["by_poi_category"] = by_cat

    # containment vs area: quartile bins of PPA area
    qs = np.quantile(areas, [0, 0.25, 0.5, 0.75, 1.0])
    bins = []
    for i in range(4):
        lo_a, hi_a = qs[i], qs[i + 1]
        sub = [t for t in triples if lo_a <= t["ppa_area_m2"] <= hi_a] if i == 3 else [
            t for t in triples if lo_a <= t["ppa_area_m2"] < hi_a
        ]
        bins.append({
            "area_km2_range": [float(lo_a) / 1e6, float(hi_a) / 1e6],
            "n": len(sub),
            "ppa_containment": rate(t["in_ppa"] for t in sub),
            "buffer500_containment": rate(t["in_buf500"] for t in sub),
        })
    s["containment_by_ppa_area_quartile"] = bins
    return s


def _windows_axb(stops, poi_a, poi_b):
    out = []
    n = len(stops)
    i = 0
    while i < n:
        if stops[i].get("poi") != poi_a:
            i += 1
            continue
        j = i + 1
        while j < n and stops[j].get("poi") != poi_b:
            j += 1
        if j >= n:
            break
        for k in range(i + 1, j):
            x = stops[k]
            if x.get("poi") in (HOME_POI, WORK_POI):
                continue
            if stops[i].get("leave") is None or x.get("arrive") is None or stops[j].get("arrive") is None:
                continue
            out.append((stops[i], x, stops[j]))
        i = j
    return out


def _median_xy(pts):
    return float(np.median([p[0] for p in pts])), float(np.median([p[1] for p in pts]))


def _poly_area_m2(poly):
    if poly is None or getattr(poly, "is_empty", True):
        return 0.0
    geom_m = shp_transform(_TO_UTM50.transform, poly)
    if geom_m is None or geom_m.is_empty:
        return 0.0
    return float(geom_m.area)


def _as_poly(result):
    if result is None:
        return None
    if hasattr(result, "geom_type"):
        return result
    if isinstance(result, dict):
        return result.get("polygon")
    if isinstance(result, (tuple, list)) and result:
        return _as_poly(result[0])
    return None


def _contains_tol(poly, lon, lat, tol_m=PPA_TOL_M):
    if poly is None or getattr(poly, "is_empty", True):
        return False, False
    pt = Point(lon, lat)
    strict = bool(poly.covers(pt))
    if tol_m <= 0:
        return strict, strict
    geom_m = shp_transform(_TO_UTM50.transform, poly)
    pt_m = shp_transform(_TO_UTM50.transform, pt)
    return bool(geom_m.buffer(tol_m).covers(pt_m)), strict


def _majority_mode(windows):
    counts = {}
    for a, x, _b in windows:
        m = x.get("mode_in") or a.get("mode_out") or "walk"
        m = str(m).strip().lower() or "walk"
        if m not in MODE_SPEED_KMH:
            m = "walk"
        counts[m] = counts.get(m, 0) + 1
    return max(counts, key=counts.get) if counts else "walk"


def _dt_seconds(a, b):
    if a is None or b is None:
        return None
    return (b - a).total_seconds()


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def _load_checkpoint():
    done = set()
    triples = []
    n_hw = n_ppa = n_fail = 0
    if not os.path.isfile(CKPT_PATH):
        return done, triples, n_hw, n_ppa, n_fail
    with open(CKPT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            uid = rec.get("uid")
            if uid is None:
                continue
            done.add(uid)
            n_hw += int(rec.get("hw") or 0)
            n_ppa += int(rec.get("ppa") or 0)
            n_fail += int(rec.get("fail") or 0)
            triples.extend(rec.get("rows") or [])
    print(f"resume checkpoint: users={len(done)} samples={len(triples)}", flush=True)
    return done, triples, n_hw, n_ppa, n_fail


def _append_checkpoint(rec):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CKPT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(rec), ensure_ascii=False) + "\n")


def build_user_ppas(day_stops):
    """每个用户一个 home–work 或 work–home 网络时空棱柱 PPA。"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from city_engine import get_shared_calculator
    from PersonalAccessibility import _build_ppa_polygon
    from transport_modes import get_route_geometry_coords, normalize_route_mode_for_calc

    print("loading shared CarbonBudgetCalculator (南京)…", flush=True)
    calc = get_shared_calculator("南京")
    if calc is None:
        raise RuntimeError("无法加载南京 CarbonBudgetCalculator")

    by_user = defaultdict(list)
    for (uid, date), stops in day_stops.items():
        by_user[uid].append((date, stops))

    done, triples, n_users_hw, n_users_ppa, n_fail = _load_checkpoint()
    uids = sorted(by_user)
    if _MAX_USERS > 0:
        uids = uids[:_MAX_USERS]
    n_all = len(uids)
    t0 = time.time()

    for ui, uid in enumerate(uids, 1):
        if uid in done:
            if ui % 50 == 0 or ui == n_all:
                print(
                    f"  users {ui}/{n_all}  (skip done) hw={n_users_hw} ppa={n_users_ppa} "
                    f"samples={len(triples)} fail={n_fail}  {time.time() - t0:.0f}s",
                    flush=True,
                )
            continue
        homes, works = [], []
        hw, wh = [], []
        user_rows = []
        hw_flag = ppa_flag = fail_flag = 0
        for date, stops in by_user[uid]:
            for s in stops:
                s["date"] = date
                if s.get("poi") == HOME_POI:
                    homes.append((s["lon"], s["lat"]))
                elif s.get("poi") == WORK_POI:
                    works.append((s["lon"], s["lat"]))
            hw.extend(_windows_axb(stops, HOME_POI, WORK_POI))
            wh.extend(_windows_axb(stops, WORK_POI, HOME_POI))

        if not homes or not works:
            _append_checkpoint({"uid": uid, "hw": 0, "ppa": 0, "fail": 0, "rows": []})
            continue
        n_users_hw += 1
        hw_flag = 1
        if len(wh) > len(hw):
            windows, poi_a, poi_b = wh, WORK_POI, HOME_POI
            origin_xy, dest_xy = _median_xy(works), _median_xy(homes)
        else:
            windows, poi_a, poi_b = hw, HOME_POI, WORK_POI
            origin_xy, dest_xy = _median_xy(homes), _median_xy(works)
        if not windows:
            _append_checkpoint({"uid": uid, "hw": 1, "ppa": 0, "fail": 0, "rows": []})
            continue

        raw_mode = _majority_mode(windows)
        calc_mode = normalize_route_mode_for_calc(raw_mode)
        t_totals = []
        for a, _x, b in windows:
            dt = _dt_seconds(a.get("leave"), b.get("arrive"))
            if dt is not None and dt > 0:
                t_totals.append(dt)
        if not t_totals:
            _append_checkpoint({"uid": uid, "hw": 1, "ppa": 0, "fail": 0, "rows": []})
            continue
        t_total_s = float(np.median(t_totals))
        t_total_min = t_total_s / 60.0

        origin_pt = Point(origin_xy[0], origin_xy[1])
        dest_pt = Point(dest_xy[0], dest_xy[1])
        t_commute_min = None
        try:
            tt, _dist = calc._calculate_travel_time_and_distance(origin_pt, dest_pt, calc_mode)
            if tt is not None and np.isfinite(tt) and 0 < tt < 1e8:
                t_commute_min = float(tt)
        except Exception:
            t_commute_min = None
        if t_commute_min is None:
            d_ab0 = haversine_m(origin_xy[0], origin_xy[1], dest_xy[0], dest_xy[1])
            v0 = MODE_SPEED_KMH.get(raw_mode, 4.8) * 1000.0 / 3600.0
            t_commute_min = (d_ab0 / v0 / 60.0) if v0 > 0 else 0.0

        elastic_min = max(0.0, t_total_min - t_commute_min)
        poly = None
        try:
            route_coords = get_route_geometry_coords(calc, origin_pt, dest_pt, calc_mode)
            buf = io.StringIO()
            with redirect_stdout(buf):
                result = _build_ppa_polygon(
                    calc,
                    origin_pt,
                    elastic_min,
                    calc_mode,
                    work_pt=dest_pt,
                    route_coords=route_coords if route_coords and len(route_coords) >= 2 else None,
                    total_travel_budget_min=t_total_min,
                    commute_time_min=t_commute_min,
                )
            poly = _as_poly(result)
        except Exception as e:
            n_fail += 1
            if n_fail <= 8:
                print(f"  ppa fail {uid} {type(e).__name__}: {e}", flush=True)
            fail_flag = 1

        if poly is None or getattr(poly, "is_empty", True):
            n_fail += 1
            fail_flag = 1
            _append_checkpoint({"uid": uid, "hw": 1, "ppa": 0, "fail": fail_flag, "rows": []})
            continue
        n_users_ppa += 1
        ppa_flag = 1
        area_m2 = _poly_area_m2(poly)
        d_ab = haversine_m(origin_xy[0], origin_xy[1], dest_xy[0], dest_xy[1])
        v = MODE_SPEED_KMH.get(raw_mode, 4.8) * 1000.0 / 3600.0
        vmax = MODE_VMAX_KMH.get(raw_mode, 8.0) / 3.6

        for a, x, b in windows:
            t_avail = _dt_seconds(a.get("leave"), b.get("arrive"))
            if t_avail is None or t_avail <= 0:
                continue
            d_ax = haversine_m(a["lon"], a["lat"], x["lon"], x["lat"])
            d_xb = haversine_m(x["lon"], x["lat"], b["lon"], b["lat"])
            if min(d_ax, d_xb, d_ab) < MIN_SEP_M:
                continue
            in_ppa, in_strict = _contains_tol(poly, x["lon"], x["lat"], PPA_TOL_M)
            in500 = point_to_segment_m(x["lon"], x["lat"], origin_xy[0], origin_xy[1], dest_xy[0], dest_xy[1]) <= 500.0
            in1000 = point_to_segment_m(x["lon"], x["lat"], origin_xy[0], origin_xy[1], dest_xy[0], dest_xy[1]) <= 1000.0
            stay_s = _dt_seconds(x.get("arrive"), x.get("leave"))
            t_ax = _dt_seconds(a.get("leave"), x.get("arrive"))
            t_xb = _dt_seconds(x.get("leave"), b.get("arrive"))
            stay_ok = stay_s is not None and stay_s >= MIN_STAY_MIN * 60.0
            time_ok = (t_ax is not None and t_ax > 0) and (t_xb is None or t_xb > 0)
            speed_ok = True
            if t_ax and t_ax > 0 and d_ax / t_ax > vmax * 1.5:
                speed_ok = False
            if t_xb and t_xb > 0 and d_xb / t_xb > vmax * 1.5:
                speed_ok = False
            cat = map_poi(x.get("poi"))
            identifiable = cat in ("shopping", "recreation", "education", "medical")
            rec = {
                "userId": uid,
                "date": a.get("date") or str(a.get("leave")),
                "mode": raw_mode,
                "calc_mode": calc_mode,
                "direction": f"{poi_a}-{poi_b}",
                "poi_A": poi_a,
                "poi_B": poi_b,
                "poi_X": x.get("poi"),
                "poi_raw": x.get("poi"),
                "poi_cat": cat,
                "identifiable": identifiable,
                "t_avail_s": t_avail,
                "t_total_s": t_total_s,
                "t_commute_s": t_commute_min * 60.0,
                "elastic_s": elastic_min * 60.0,
                "stay_s": stay_s if stay_s is not None else float("nan"),
                "d_ax": d_ax,
                "d_xb": d_xb,
                "d_ab": d_ab,
                "budget_m": v * t_total_s,
                "in_ppa": in_ppa,
                "in_ppa_notol": in_strict,
                "in_ppa_net": in_ppa,
                "in_buf500": in500,
                "in_buf1000": in1000,
                "feasible": bool(stay_ok and time_ok and speed_ok),
                "ppa_area_m2": area_m2,
                "buf500_area_m2": buffer_area_m2(d_ab, 500.0),
                "buf1000_area_m2": buffer_area_m2(d_ab, 1000.0),
            }
            user_rows.append(rec)
            triples.append(rec)

        _append_checkpoint({
            "uid": uid, "hw": hw_flag, "ppa": ppa_flag, "fail": fail_flag, "rows": user_rows,
        })
        if ui % 10 == 0 or ui == n_all or ui <= 3:
            print(
                f"  users {ui}/{n_all}  hw={n_users_hw} ppa={n_users_ppa} "
                f"samples={len(triples)} fail={n_fail}  {time.time() - t0:.0f}s",
                flush=True,
            )

    return triples, n_users_hw, n_users_ppa, n_fail


def main():
    print("loading", SRC, flush=True)
    trips = load_trips(SRC)
    print("records", len(trips), flush=True)
    od = pair_od(trips)
    print("od_pairs", len(od), flush=True)
    users_raw = {o["userId"] for o, _ in od}
    day_stops = build_day_stops(od)
    n_user_days = len(day_stops)
    n_stops_ok = sum(1 for v in day_stops.values() if len(v) >= 3)
    triples, n_users_hw, n_users_ppa, n_fail = build_user_ppas(day_stops)
    print(
        f"samples {len(triples)} users_hw {n_users_hw} users_ppa {n_users_ppa} fail {n_fail}",
        flush=True,
    )
    summary = summarize(triples, len(users_raw), n_users_ppa, n_user_days, n_stops_ok, n_users_hw)
    summary["sample_definition"] = "one PPA per user (home-work or work-home); X held-out on that direction"
    summary["n_users_with_home_and_work"] = n_users_hw
    summary["n_users_with_platform_ppa"] = n_users_ppa
    summary["n_ppa_build_fail"] = n_fail
    by_uid = {}
    for t in triples:
        by_uid[t["userId"]] = t["ppa_area_m2"]
    if by_uid:
        uareas = np.array(list(by_uid.values()), dtype=float)
        summary["ppa_area_km2_per_user"] = {
            "n": int(len(uareas)),
            "mean": float(np.mean(uareas)) / 1e6,
            "median": float(np.median(uareas)) / 1e6,
            "p25": float(np.percentile(uareas, 25)) / 1e6,
            "p75": float(np.percentile(uareas, 75)) / 1e6,
        }
    dirs = Counter(t["direction"] for t in triples)
    summary["n_samples_by_direction"] = dict(dirs)
    summary["generated_at"] = datetime.now().isoformat(timespec="seconds")
    summary["source"] = SRC
    summary["notes"] = {
        "sample": "每用户一个 PPA：home→work 或 work→home（中间停驻更多的一侧）；X 为该方向上的留出停驻",
        "ppa_definition": "平台网络时空棱柱 PersonalAccessibility._build_ppa_polygon（必经通勤路径 + 弹性时间侧向拓展）",
        "anchor": "用户全部 home / work 停驻点的中位数坐标",
        "T_total": "该方向各次 A.leave→B.arrive 的中位数",
        "t_commute": "路网最短路时间；失败则用哈弗赛因/模式速度",
        "mode": "该方向中间停驻 mode_in（否则 A.mode_out）众数，经 normalize_route_mode_for_calc",
        "v_mode_kmh": MODE_SPEED_KMH,
        "baseline": "AB 线段 500m / 1000m 缓冲区（对照）",
        "feasibility": "时间有序、停留>=5min、隐含速度 <= 1.5 * 模式 vmax",
        "bootstrap": "user-clustered, 1000 resamples, 95% percentile CI",
        "containment_tolerance_m": PPA_TOL_M,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    sum_path = os.path.join(OUT_DIR, "ppa_validation_summary.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    csv_path = os.path.join(OUT_DIR, "ppa_validation_triples.csv")
    cols = [
        "userId", "date", "mode", "calc_mode", "direction", "poi_A", "poi_B", "poi_X", "poi_raw",
        "poi_cat", "identifiable", "t_avail_s", "t_total_s", "t_commute_s", "elastic_s", "stay_s",
        "d_ax", "d_xb", "d_ab", "budget_m",
        "in_ppa", "in_ppa_notol", "in_ppa_net", "in_buf500", "in_buf1000",
        "feasible", "ppa_area_m2", "buf500_area_m2", "buf1000_area_m2",
    ]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for t in triples:
            row = []
            for c in cols:
                v = t[c]
                if isinstance(v, bool):
                    row.append("1" if v else "0")
                elif v is None:
                    row.append("")
                elif isinstance(v, float):
                    row.append(f"{v:.6f}")
                else:
                    row.append(str(v))
            f.write(",".join(row) + "\n")
    print("wrote", sum_path, flush=True)
    print("wrote", csv_path, flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
