"""个人公平性与社区生活圈公平性分析。"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.neighbors import BallTree

from fairness_analysis import (
    CITY_FOLDER_MAP,
    DEFAULT_RESOLUTION,
    find_poi_file,
    load_poi_npz,
    points_to_raster,
)

COMMUNITY_REPORT_CATEGORIES = [
    "work",
    "education",
    "medical",
    "shopping",
    "catering",
    "recreation",
    "transportation_station",
]

CATEGORY_LABELS_ZH = {
    "work": "工作就业",
    "education": "教育资源",
    "medical": "医疗服务",
    "shopping": "购物便利",
    "catering": "餐饮服务",
    "recreation": "休闲娱乐",
    "transportation_station": "交通出行",
}

TRAVEL_SPEEDS: Dict[str, float] = {
    "walk": 5.0,
    "bike": 12.0,
    "bus": 20.0,
    "metro": 30.0,
    "car": 40.0,
}

TRAVEL_MODE_LABELS_ZH: Dict[str, str] = {
    "walk": "步行",
    "bike": "骑行",
    "bus": "公交",
    "metro": "地铁",
    "car": "驾车",
}

GRADE_LABELS_ZH: Dict[str, str] = {
    "excellent": "资源充足",
    "good": "相对便利",
    "fair": "中等水平",
    "limited": "服务不足",
    "poor": "严重不足",
}

_balltree_cache: Dict[Tuple[str, str], Tuple[BallTree, np.ndarray, int]] = {}

def compute_community_equity_report(
    lat: float,
    lng: float,
    city: str = "南京",
    resolution: float = DEFAULT_RESOLUTION,
) -> Dict[str, Any]:
    city_name = str(city or "南京").strip()
    poi_folder = CITY_FOLDER_MAP.get(city_name, "Nanjing")
    base_dir = Path(__file__).parent
    poi_base = base_dir / "POI" / poi_folder / "Location-based"
    categories_result = []
    valid_percentiles = []
    valid_accessibility = []
    for cat_key in COMMUNITY_REPORT_CATEGORIES:
        cat_info = {
            "key": cat_key,
            "label_zh": CATEGORY_LABELS_ZH.get(cat_key, cat_key),
            "local_accessibility": None,
            "city_mean": 0.0,
            "city_median": 0.0,
            "percentile": None,
            "level": "no_data",
        }
        try:
            file_path = find_poi_file(poi_base, cat_key, subcategory=None)
            if file_path is None or not file_path.exists():
                categories_result.append(cat_info)
                continue
            poi_data = load_poi_npz(str(file_path))
            if poi_data is None:
                categories_result.append(cat_info)
                continue
            raster, transform, _bounds_tuple = points_to_raster(
                poi_data["demand_coords"], poi_data["ai"], resolution=resolution
            )
            if raster.size == 0:
                categories_result.append(cat_info)
                continue
            h, w = raster.shape
            min_lon, min_lat, max_lon, max_lat = _bounds_tuple
            col = int((lng - min_lon) / resolution)
            row = int((max_lat - lat) / resolution)
            if not (0 <= col < w and 0 <= row < h):
                categories_result.append(cat_info)
                continue
            local_val = float(raster[row, col])
            if np.isnan(local_val) or local_val < 0:
                categories_result.append(cat_info)
                continue
            cat_info["local_accessibility"] = round(local_val, 4)
            all_valid = raster[np.isfinite(raster) & (raster >= 0)]
            if len(all_valid) < 10:
                categories_result.append(cat_info)
                continue
            city_mean = float(np.mean(all_valid))
            city_median = float(np.median(all_valid))
            cat_info["city_mean"] = round(city_mean, 4)
            cat_info["city_median"] = round(city_median, 4)
            percentile = float(np.mean(all_valid <= local_val)) * 100.0
            cat_info["percentile"] = round(percentile, 1)
            valid_percentiles.append(percentile)
            valid_accessibility.append(local_val)
            if percentile >= 80:
                cat_info["level"] = "high"
            elif percentile >= 50:
                cat_info["level"] = "above_avg"
            elif percentile >= 20:
                cat_info["level"] = "below_avg"
            else:
                cat_info["level"] = "low"
        except Exception as e:
            print(f"[community_equity] 处理品类 {cat_key} 失败: {e}")
            categories_result.append(cat_info)
    covered = sum((1 for c in categories_result if c["level"] in ("high", "above_avg")))
    total = len([c for c in categories_result if c["level"] != "no_data"])
    coverage_pct = round(covered / total * 100, 1) if total > 0 else 0.0
    if coverage_pct >= 80:
        life_grade = "excellent"
    elif coverage_pct >= 60:
        life_grade = "good"
    elif coverage_pct >= 40:
        life_grade = "fair"
    else:
        life_grade = "limited"
    if valid_percentiles:
        overall_score = round(float(np.mean(valid_percentiles)), 1)
    else:
        overall_score = 0.0
    insights = _generate_community_insights(categories_result, life_grade, overall_score, coverage_pct)
    return {
        "success": True,
        "location": {"lat": round(lat, 6), "lng": round(lng, 6)},
        "city": city_name,
        "categories": categories_result,
        "life_circle": {
            "covered_count": covered,
            "total_count": total,
            "coverage_pct": coverage_pct,
            "grade": life_grade,
            "grade_label": {
                "excellent": "优秀",
                "good": "良好",
                "fair": "一般",
                "limited": "待改善",
            }.get(life_grade, life_grade),
        },
        "overall_score": overall_score,
        "insights": insights,
    }

def _generate_community_insights(
    categories: List[Dict],
    life_grade: str,
    overall_score: float,
    coverage_pct: float,
) -> List[str]:
    insights = []
    high_cats = [c for c in categories if c["level"] == "high"]
    low_cats = [c for c in categories if c["level"] in ("low", "below_avg")]
    if life_grade == "excellent":
        insights.append("该位置公共服务配套完善，各项设施可达性均处于全市领先水平。")
    elif life_grade == "good":
        insights.append("该位置生活配套较为齐全，多数设施可达性高于全市平均水平。")
    elif life_grade == "fair":
        insights.append("该位置生活配套水平一般，部分设施可达性有待提升。")
    else:
        insights.append("该位置公共服务覆盖不足，多数设施可达性低于全市平均水平。")
    if high_cats:
        names = "、".join([c["label_zh"] for c in high_cats[:3]])
        insights.append(f"优势资源：{names}方面表现突出，在全市排名靠前。")
    if low_cats:
        names = "、".join([c["label_zh"] for c in low_cats[:3]])
        insights.append(f"改善方向：{names}方面相对薄弱，建议关注相关配套建设规划。")
    if overall_score >= 60:
        insights.append(f"综合公平性评分 {overall_score:.0f} 分，整体居住品质良好。")
    else:
        insights.append(f"综合公平性评分 {overall_score:.0f} 分，建议结合个人出行能力综合判断。")
    return insights

def _get_poi_balltree(city_poi_dir: str, cat_key: str) -> Optional[Tuple[BallTree, np.ndarray, int]]:
    cache_key = (str(city_poi_dir), cat_key)
    cached = _balltree_cache.get(cache_key)
    if cached is not None:
        return cached
    try:
        from Data_loader import load_poi_from_geojson

        poi_dict = load_poi_from_geojson(poi_dir=city_poi_dir, use_cache=True)
        gdf = poi_dict.get(cat_key)
        if gdf is None or len(gdf) == 0:
            _balltree_cache[cache_key] = None
            return None
        coords = np.column_stack([
            gdf.geometry.y.values.astype(np.float64),
            gdf.geometry.x.values.astype(np.float64),
        ])
        coords_rad = np.deg2rad(coords)
        tree = BallTree(coords_rad, metric="haversine")
        result = (tree, coords_rad, len(coords))
        _balltree_cache[cache_key] = result
        print(f"[BallTree] 已为 {cat_key} 建立空间索引（{len(coords)} 个POI）")
        return result
    except Exception as e:
        print(f"[BallTree] 品类 {cat_key} 建树失败: {e}")
        _balltree_cache[cache_key] = None
        return None

def _count_pois_in_radius(tree: BallTree, center_rad: np.ndarray, radius_m: float) -> int:
    radius_rad = radius_m / 6371000.0
    indices = tree.query_radius(center_rad, r=radius_rad)
    return int(len(indices[0])) if len(indices) > 0 else 0

def _build_city_distribution(
    city_poi_dir: str,
    cat_key: str,
    effective_radius_m: float,
    sample_size: int = 300,
) -> Optional[np.ndarray]:
    tree_result = _get_poi_balltree(city_poi_dir, cat_key)
    if tree_result is None:
        return None
    tree, coords_rad, n_pois = tree_result
    if n_pois < 10:
        return None
    if n_pois <= sample_size:
        indices = np.arange(n_pois)
    else:
        rng = np.random.RandomState(42)
        indices = rng.choice(n_pois, size=sample_size, replace=False)
    sample_rad = coords_rad[indices]
    counts = np.array(
        [_count_pois_in_radius(tree, sample_rad[i:i + 1], effective_radius_m) for i in range(len(sample_rad))],
        dtype=np.float64,
    )
    return counts

def _resolve_city_poi_dir(city: str) -> str:
    city_name = str(city or "南京").strip()
    poi_folder = CITY_FOLDER_MAP.get(city_name, "Nanjing")
    base_dir = Path(__file__).parent
    return str(base_dir / "POI" / poi_folder)

def compute_personal_fairness_ranking(
    lat: float,
    lng: float,
    city: str = "南京",
    travel_mode: str = "bus",
    time_budget_minutes: int = 30,
    resolution: float = DEFAULT_RESOLUTION,
) -> Dict[str, Any]:
    city_poi_dir = _resolve_city_poi_dir(city)
    speed = TRAVEL_SPEEDS.get(travel_mode, 20.0)
    effective_radius_m = speed * (time_budget_minutes / 60.0) * 1000.0
    user_rad = np.deg2rad([[lat, lng]])
    categories_result = []
    valid_percentiles = []
    for cat_key in COMMUNITY_REPORT_CATEGORIES:
        cat_info = {
            "key": cat_key,
            "label_zh": CATEGORY_LABELS_ZH.get(cat_key, cat_key),
            "city_percentile": None,
            "local_poi_count": 0,
            "grade": "no_data",
            "grade_label": "无数据",
        }
        try:
            tree_result = _get_poi_balltree(city_poi_dir, cat_key)
            if tree_result is None:
                categories_result.append(cat_info)
                continue
            tree, _, total_pois = tree_result
            local_count = _count_pois_in_radius(tree, user_rad, effective_radius_m)
            cat_info["local_poi_count"] = local_count
            if local_count == 0 and effective_radius_m < 2000:
                categories_result.append(cat_info)
                continue
            dist = _build_city_distribution(city_poi_dir, cat_key, effective_radius_m)
            if dist is None or len(dist) < 10:
                categories_result.append(cat_info)
                continue
            percentile = float(np.mean(dist <= local_count)) * 100.0
            cat_info["city_percentile"] = round(percentile, 1)
            valid_percentiles.append(percentile)
            if percentile >= 80:
                cat_info["grade"] = "excellent"
                cat_info["grade_label"] = "资源充足"
            elif percentile >= 55:
                cat_info["grade"] = "good"
                cat_info["grade_label"] = "相对便利"
            elif percentile >= 30:
                cat_info["grade"] = "fair"
                cat_info["grade_label"] = "中等水平"
            elif percentile >= 15:
                cat_info["grade"] = "limited"
                cat_info["grade_label"] = "服务不足"
            else:
                cat_info["grade"] = "poor"
                cat_info["grade_label"] = "严重不足"
            categories_result.append(cat_info)
        except Exception as e:
            print(f"[personal_ranking] 品类 {cat_key} 处理失败: {e}")
            categories_result.append(cat_info)
    if valid_percentiles:
        overall_pct = round(float(np.mean(valid_percentiles)), 1)
    else:
        overall_pct = 0.0
    if overall_pct >= 80:
        overall_grade = "excellent"
    elif overall_pct >= 55:
        overall_grade = "good"
    elif overall_pct >= 30:
        overall_grade = "fair"
    elif overall_pct >= 15:
        overall_grade = "limited"
    else:
        overall_grade = "poor"
    insights = _generate_personal_insights(
        categories_result,
        overall_grade,
        travel_mode,
        time_budget_minutes,
        effective_radius_m,
    )
    return {
        "success": True,
        "personal_ranking": {
            "overall_percentile": overall_pct,
            "overall_grade": overall_grade,
            "overall_grade_label": GRADE_LABELS_ZH.get(overall_grade, overall_grade),
            "travel_mode_label": TRAVEL_MODE_LABELS_ZH.get(travel_mode, travel_mode),
            "time_budget_minutes": time_budget_minutes,
            "effective_radius_km": round(effective_radius_m / 1000.0, 1),
            "location": {"lat": round(lat, 6), "lng": round(lng, 6)},
            "categories": categories_result,
            "insights": insights,
        },
    }

def _generate_personal_insights(
    categories: List[Dict],
    overall_grade: str,
    travel_mode: str,
    time_budget_minutes: int,
    effective_radius_m: float,
) -> List[str]:
    mode_label = TRAVEL_MODE_LABELS_ZH.get(travel_mode, travel_mode)
    radius_km = effective_radius_m / 1000.0
    insights = []
    excellent = [c for c in categories if c["grade"] == "excellent"]
    poor = [c for c in categories if c["grade"] in ("limited", "poor")]
    if overall_grade == "excellent":
        insights.append(
            f"{mode_label}{time_budget_minutes}分钟（约{radius_km:.1f}km范围）内，"
            "您的生活配套资源处于全市领先水平。"
        )
    elif overall_grade == "good":
        insights.append(
            f"{mode_label}{time_budget_minutes}分钟（约{radius_km:.1f}km范围）内，"
            "您的日常出行便利度高于全市平均水平。"
        )
    elif overall_grade == "fair":
        insights.append(
            f"{mode_label}{time_budget_minutes}分钟（约{radius_km:.1f}km范围）内，"
            "部分资源可达性处于中等，可考虑延长出行时间或更换出行方式。"
        )
    elif overall_grade == "limited":
        insights.append(
            f"{mode_label}{time_budget_minutes}分钟（约{radius_km:.1f}km范围）内，"
            "多数资源可达性低于全市平均水平。"
        )
    else:
        insights.append(f"您当前出行方式下，{time_budget_minutes}分钟内多项生活资源可达性严重不足。")
    if excellent:
        names = "、".join([c["label_zh"] for c in excellent[:3]])
        insights.append(f"优势资源：{names}方面资源充足。")
    if poor:
        names = "、".join([c["label_zh"] for c in poor[:3]])
        insights.append(f"短板资源：{names}方面明显不足（有效半径内覆盖POI较少）。")
    if overall_grade in ("limited", "poor"):
        if travel_mode == "walk":
            insights.append("步行范围有限，建议尝试骑行或公交出行以大幅扩大可达范围。")
        elif travel_mode in ("bike", "bus"):
            insights.append("建议尝试地铁出行以进一步扩大可达范围，或考虑搬迁至公共资源更集中的区域。")
        elif travel_mode == "metro":
            insights.append("即乘地铁仍有多项资源不足，当前居住地公共服务覆盖较弱，建议关注周边城区住房选择。")
        else:
            insights.append("建议考虑搬迁至市中心或地铁沿线区域以提升生活便利度。")
    return insights

def compute_location_comparison(
    origin: Dict[str, float],
    target: Dict[str, float],
    city: str = "南京",
    travel_mode: str = "bus",
    time_budget_minutes: int = 30,
    resolution: float = DEFAULT_RESOLUTION,
) -> Dict[str, Any]:
    origin_result = compute_personal_fairness_ranking(
        lat=origin["lat"],
        lng=origin["lng"],
        city=city,
        travel_mode=travel_mode,
        time_budget_minutes=time_budget_minutes,
        resolution=resolution,
    )
    target_result = compute_personal_fairness_ranking(
        lat=target["lat"],
        lng=target["lng"],
        city=city,
        travel_mode=travel_mode,
        time_budget_minutes=time_budget_minutes,
        resolution=resolution,
    )
    if not origin_result.get("success") or not target_result.get("success"):
        return {"success": False, "message": "位置信息无效，无法完成对比"}
    o_rank = origin_result["personal_ranking"]
    t_rank = target_result["personal_ranking"]
    overall_diff = round(t_rank["overall_percentile"] - o_rank["overall_percentile"], 1)
    o_cats = {c["key"]: c for c in o_rank["categories"]}
    t_cats = {c["key"]: c for c in t_rank["categories"]}
    category_diffs = []
    for cat_key in COMMUNITY_REPORT_CATEGORIES:
        o_c = o_cats.get(cat_key, {})
        t_c = t_cats.get(cat_key, {})
        o_pct = o_c.get("city_percentile")
        t_pct = t_c.get("city_percentile")
        diff = round(t_pct - o_pct, 1) if o_pct is not None and t_pct is not None else None
        category_diffs.append({
            "key": cat_key,
            "label_zh": CATEGORY_LABELS_ZH.get(cat_key, cat_key),
            "origin_percentile": o_pct,
            "target_percentile": t_pct,
            "diff": diff,
        })
    comparison_insights = []
    mode_label = TRAVEL_MODE_LABELS_ZH.get(travel_mode, travel_mode)
    if overall_diff > 5:
        comparison_insights.append(
            f"{mode_label}{time_budget_minutes}分钟：目标位置综合公平性比当前位置高 "
            f"{overall_diff:.0f} 个百分点，建议搬迁。"
        )
    elif overall_diff > 0:
        comparison_insights.append(
            f"{mode_label}{time_budget_minutes}分钟：目标位置综合公平性略高 {overall_diff:.0f} 个百分点。"
        )
    elif overall_diff > -5:
        comparison_insights.append("两处位置综合公平性基本持平。")
    else:
        comparison_insights.append(
            f"目标位置综合公平性比当前位置低 {abs(overall_diff):.0f} 个百分点。"
        )
    improved = [d for d in category_diffs if d["diff"] is not None and d["diff"] > 5]
    declined = [d for d in category_diffs if d["diff"] is not None and d["diff"] < -5]
    if improved:
        names = "、".join([d["label_zh"] for d in improved[:3]])
        comparison_insights.append(f"迁入后显著提升：{names}。")
    if declined:
        names = "、".join([d["label_zh"] for d in declined[:3]])
        comparison_insights.append(f"需要注意可能下降：{names}。")
    return {
        "success": True,
        "origin_ranking": o_rank,
        "target_ranking": t_rank,
        "comparison": {
            "overall_diff": overall_diff,
            "category_diffs": category_diffs,
            "insights": comparison_insights,
        },
    }

def generate_improvement_suggestions(
    lat: float,
    lng: float,
    city: str = "南京",
    current_travel_mode: str = "bus",
    time_budget_minutes: int = 30,
    resolution: float = DEFAULT_RESOLUTION,
) -> Dict[str, Any]:
    current = compute_personal_fairness_ranking(
        lat=lat,
        lng=lng,
        city=city,
        travel_mode=current_travel_mode,
        time_budget_minutes=time_budget_minutes,
        resolution=resolution,
    )
    mode_comparisons = []
    best_mode = current_travel_mode
    best_score = current["personal_ranking"]["overall_percentile"] if current.get("success") else 0.0
    for mode in ["walk", "bike", "bus", "metro", "car"]:
        result = compute_personal_fairness_ranking(
            lat=lat,
            lng=lng,
            city=city,
            travel_mode=mode,
            time_budget_minutes=time_budget_minutes,
            resolution=resolution,
        )
        if result.get("success"):
            score = result["personal_ranking"]["overall_percentile"]
            mode_comparisons.append({
                "mode": mode,
                "mode_label": TRAVEL_MODE_LABELS_ZH.get(mode, mode),
                "overall_percentile": score,
                "effective_radius_km": result["personal_ranking"].get("effective_radius_km", 0),
                "grade_label": result["personal_ranking"]["overall_grade_label"],
            })
            if score > best_score:
                best_score = score
                best_mode = mode
    weak_categories = []
    if current.get("success"):
        for c in current["personal_ranking"]["categories"]:
            if c["grade"] in ("poor", "limited", "fair"):
                weak_categories.append(c)
    suggestions = []
    if best_mode != current_travel_mode:
        mode_label = TRAVEL_MODE_LABELS_ZH.get(best_mode, best_mode)
        current_label = TRAVEL_MODE_LABELS_ZH.get(current_travel_mode, current_travel_mode)
        gain = round(best_score - current["personal_ranking"]["overall_percentile"], 1) if current.get("success") else 0
        if gain > 3:
            suggestions.append({
                "type": "mode_change",
                "title": f"切换出行方式：{current_label} → {mode_label}",
                "detail": (
                    f"改乘{mode_label}可将综合公平性从 {current['personal_ranking']['overall_percentile']} "
                    f"分提升至 {best_score} 分（+{gain} 个百分点）。{mode_label}的有效活动范围更大，"
                    "能覆盖更多公共服务资源。"
                ),
                "gain_pct": gain,
            })
    for wc in weak_categories[:2]:
        suggestions.append({
            "type": "target_weakness",
            "title": f"关注薄弱环节：{wc['label_zh']}",
            "detail": (
                f"{wc['label_zh']}方面仅超过全市 {wc['city_percentile']:.0f}% 的位置，"
                "该品类POI在您的有效活动半径内覆盖较少。可考虑前往该资源丰富的区域工作或活动。"
            ),
            "target_category": wc["key"],
        })
    if current.get("success") and current["personal_ranking"]["overall_grade"] in (
        "poor",
        "limited",
    ):
        suggestions.append({
            "type": "relocate",
            "title": "考虑搬迁至城市核心区",
            "detail": (
                f"即使用{best_mode}出行，当前居住地多项资源仍低于全市平均水平。"
                "建议考虑搬迁至市中心或地铁沿线区域以大幅提升生活便利度。"
            ),
        })
    elif current.get("success") and current["personal_ranking"]["overall_grade"] == "fair":
        suggestions.append({
            "type": "time_budget",
            "title": "尝试延长出行时间预算",
            "detail": (
                f"将时间预算从 {time_budget_minutes} 分钟延长至 "
                f"{min(120, time_budget_minutes * 2)} 分钟可显著扩大活动范围，可能带来公平性评级的提升。"
            ),
        })
    return {
        "success": True,
        "current_mode_label": TRAVEL_MODE_LABELS_ZH.get(current_travel_mode, current_travel_mode),
        "time_budget_minutes": time_budget_minutes,
        "current_ranking": (current.get("personal_ranking") if current.get("success") else None),
        "mode_comparisons": mode_comparisons,
        "best_mode": best_mode,
        "best_mode_label": TRAVEL_MODE_LABELS_ZH.get(best_mode, best_mode),
        "suggestions": suggestions,
    }
