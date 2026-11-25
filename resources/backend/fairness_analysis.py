import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from rasterio.transform import Affine
from sklearn.neighbors import BallTree

DEGREE_TO_METER = 111000.0
DEFAULT_ACCESSIBILITY_RESOLUTION_M = 1000
DEFAULT_RESOLUTION = DEFAULT_ACCESSIBILITY_RESOLUTION_M / DEGREE_TO_METER

CITY_FOLDER_MAP = {
    '南京': 'Nanjing', 'Nanjing': 'Nanjing',
    '武汉': 'Wuhan', 'Wuhan': 'Wuhan',
    '北京': 'Beijing', 'Beijing': 'Beijing',
    '上海': 'Shanghai', 'Shanghai': 'Shanghai',
}

CLUSTER_LABELS = {
    0: 'No data', 1: 'Low-Low (LL)', 2: 'Low-High (LH)', 3: 'High-Low (HL)', 4: 'High-High (HH)',
}

MORAN_COLORS = [
    '#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7',
    '#fddbc7', '#f4a582', '#d6604d', '#b2182b',
]

CLUSTER_COLORS = {
    0: (0, 0, 0, 0),
    1: (0.173, 0.482, 0.714, 0.8),
    2: (0.671, 0.851, 0.914, 0.8),
    3: (0.992, 0.682, 0.380, 0.8),
    4: (0.843, 0.098, 0.110, 0.8),
}

MORAN_SPATIAL_WEIGHT_PARAMS: Dict[str, Dict[str, float]] = {
    'shopping': {'lambda_mix': 0.9, 'beta': 0.005, 'alpha': 1.0, 'd_max': 3000.0},
    'catering': {'lambda_mix': 0.9, 'beta': 0.005, 'alpha': 1.0, 'd_max': 3000.0},
    'transportation_station': {'lambda_mix': 0.9, 'beta': 0.005, 'alpha': 1.0, 'd_max': 3000.0},
    'education': {'lambda_mix': 0.9, 'beta': 0.005, 'alpha': 1.0, 'd_max': 3000.0},
    'work': {'lambda_mix': 0.3, 'beta': 0.001, 'alpha': 1.5, 'd_max': 10000.0},
    'recreation': {'lambda_mix': 0.3, 'beta': 0.001, 'alpha': 1.5, 'd_max': 10000.0},
    'medical': {'lambda_mix': 0.3, 'beta': 0.001, 'alpha': 1.5, 'd_max': 10000.0},
}

MORAN_MAX_CELLS_FOR_FULL_NETWORK = 220
DEFAULT_SEARCH_RADIUS = {k: float(v['d_max']) for k, v in MORAN_SPATIAL_WEIGHT_PARAMS.items()}

def haversine_distance_matrix(coords: np.ndarray) -> np.ndarray:
    coords_rad = np.deg2rad(coords)
    tree = BallTree(coords_rad, metric="haversine")
    dist_mat, _ = tree.query(coords_rad, k=min(len(coords), 100))
    n = len(coords)
    full_dist = np.zeros((n, n), dtype=np.float64)
    for i, (dists, indices) in enumerate(zip(dist_mat, tree.query(coords_rad, k=n))):
        full_dist[i, indices] = dists * 6371000
        full_dist[indices, i] = dists * 6371000
    return full_dist

def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = (np.radians(lat1), np.radians(lat2))
    dphi = np.radians(lat2 - lat1)
    dlamb = np.radians(lon2 - lon1)
    a = np.sin(dphi * 0.5)**2 + np.cos(p1) * np.cos(p2) * np.sin(dlamb * 0.5)**2
    return float(r * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))

def _grid_chebyshev_neighbor_weights(valid_rows: np.ndarray, valid_cols: np.ndarray, chebyshev_k: int) -> Tuple[List[int], List[int], List[float]]:
    if chebyshev_k < 1:
        return ([], [], [])
    pos: Dict[Tuple[int, int], int] = {}
    for i, (r, c) in enumerate(zip(valid_rows, valid_cols)):
        pos[int(r), int(c)] = i
    rows_list: List[int] = []
    cols_list: List[int] = []
    weights_list: List[float] = []
    for i, (r, c) in enumerate(zip(valid_rows, valid_cols)):
        r0, c0 = (int(r), int(c))
        for dr in range(-chebyshev_k, chebyshev_k + 1):
            for dc in range(-chebyshev_k, chebyshev_k + 1):
                if dr == 0 and dc == 0:
                    continue
                if max(abs(dr), abs(dc)) > chebyshev_k:
                    continue
                j = pos.get((r0 + dr, c0 + dc))
                if j is not None:
                    rows_list.append(i)
                    cols_list.append(j)
                    weights_list.append(1.0)
    return (rows_list, cols_list, weights_list)

def combined_spatial_weight_raw(d_m: float, lambda_mix: float, beta: float, alpha: float, d_max: float, d_floor_m: float = 1.0) -> float:
    if d_m <= 0 or not np.isfinite(d_m) or d_m > d_max:
        return 0.0
    lam = float(np.clip(lambda_mix, 0.0, 1.0))
    exp_part = float(np.exp(-float(beta) * d_m))
    d_eff = max(float(d_m), float(d_floor_m))
    pow_part = float(d_eff**(-float(alpha)))
    return lam * exp_part + (1.0 - lam) * pow_part

def try_init_route_calculator(city: str) -> Any:
    city_key = str(city or "南京").strip()
    try:
        from city_engine import get_shared_calculator
    except Exception as e:
        print(f"[fairness] 路网距离模块导入失败: {e}")
        return None
    try:
        calc = get_shared_calculator(city=city_key)
        if calc is None:
            print("[fairness] 共享计算器不可用，将使用直线距离")
        return calc
    except Exception as e:
        print(f"[fairness] 路网初始化失败（将使用直线距离）: {e}")
        return None

def _pick_route_mode(calc: Any) -> Optional[str]:
    if calc is None:
        return None
    for m in ("bike", "car", "bus", "metro"):
        if m in getattr(calc, "network_trees", {}):
            return m
    return None

def _pair_distance_m(i_idx: int, j_idx: int, lons: np.ndarray, lats: np.ndarray, route_calc: Any, route_mode: Optional[str], use_network: bool) -> float:
    lon1, lat1 = (float(lons[i_idx]), float(lats[i_idx]))
    lon2, lat2 = (float(lons[j_idx]), float(lats[j_idx]))
    if use_network and route_calc is not None and (route_mode is not None):
        try:
            from shapely.geometry import Point
            origin = Point(lon1, lat1)
            dest = Point(lon2, lat2)
            _t, d_m = route_calc._calculate_travel_time_and_distance(origin, dest, route_mode)
            if d_m is not None and np.isfinite(d_m) and (float(d_m) > 0):
                return float(d_m)
        except Exception:
            pass
    return haversine_distance_m(lat1, lon1, lat2, lon2)

def load_poi_npz(file_path: str) -> Optional[Dict]:
    try:
        data = np.load(file_path)
        coords = np.array(data["demand_coords"], dtype=np.float64)
        ai = np.array(data["ai"], dtype=np.float32)
        return {"demand_coords": coords, "ai": ai}
    except Exception as e:
        print(f"[fairness] 加载文件失败 {file_path}: {e}")
        return None

def find_poi_file(poi_dir: Path, poi_category: str, subcategory: str = None) -> Optional[Path]:
    resol_dirs = sorted(poi_dir.glob("resol_*"))
    if not resol_dirs:
        resol_dirs = [poi_dir]
    all_npz = []
    for rdir in resol_dirs:
        all_npz.extend(rdir.glob("*.npz"))
    if subcategory and subcategory.strip():
        sc = subcategory.strip()
        for f in all_npz:
            if f.stem.startswith(f"{poi_category}_{sc}_"):
                return f
        for f in all_npz:
            if sc in f.stem and f.stem.startswith(f"{poi_category}_"):
                return f
    for f in all_npz:
        stem = f.stem
        if stem == f"{poi_category}_distance10000" or stem == f"{poi_category}_distance100000":
            return f
    for f in all_npz:
        if f.stem.startswith(f"{poi_category}_"):
            return f
    return None

def points_to_raster(demand_coords: np.ndarray, ai_values: np.ndarray, resolution: float = DEFAULT_RESOLUTION, bounds: Tuple[float, float, float, float] = None) -> Tuple[np.ndarray, Affine, Tuple[float, float, float, float]]:
    if len(demand_coords) == 0:
        return (np.array([]), None, None)
    lons = demand_coords[:, 0]
    lats = demand_coords[:, 1]
    if bounds is None:
        min_lon, max_lon = (np.min(lons), np.max(lons))
        min_lat, max_lat = (np.min(lats), np.max(lats))
    else:
        min_lon, min_lat, max_lon, max_lat = bounds
    width = max(1, int(np.ceil((max_lon - min_lon) / resolution)))
    height = max(1, int(np.ceil((max_lat - min_lat) / resolution)))
    raster = np.full((height, width), np.nan, dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)
    for (lon, lat), ai in zip(demand_coords, ai_values):
        col = int((lon - min_lon) / resolution)
        row = int((max_lat - lat) / resolution)
        if 0 <= col < width and 0 <= row < height:
            idx = (row, col)
            if np.isnan(raster[idx]):
                raster[idx] = ai
                count[idx] = 1
            else:
                raster[idx] = (raster[idx] * count[idx] + ai) / (count[idx] + 1)
                count[idx] += 1
    transform = Affine(resolution, 0, min_lon, 0, -resolution, max_lat)
    return (raster, transform, (min_lon, min_lat, max_lon, max_lat))

LISA_PERMUTATION_COUNT = 999

def _report_analysis_progress(progress_callback: Optional[Any], pct: float, status: str) -> None:
    if not progress_callback:
        return
    try:
        progress_callback(int(max(0, min(100, round(pct)))), str(status))
    except Exception:
        pass

def _wrap_sub_progress(progress_callback: Optional[Any], base: float, span: float):
    def inner(pct: float, status: str) -> None:
        mapped = base + (float(pct) / 100.0) * span
        _report_analysis_progress(progress_callback, mapped, status)
    return inner

def _local_moran_values_and_lag(valid_vals: np.ndarray, W_norm: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """给定属性向量与行标准化权重矩阵，计算局部 Moran I、Z 与空间滞后。"""
    valid_vals = np.asarray(valid_vals, dtype=np.float64)
    n = len(valid_vals)
    if n == 0:
        return (np.array([]), np.array([]), np.array([]))
    mean_val = float(np.mean(valid_vals))
    s_sq = float(np.var(valid_vals, ddof=0))
    if not np.isfinite(s_sq) or s_sq < 1e-15:
        zeros = np.zeros(n, dtype=np.float64)
        return (zeros, zeros, zeros)
    deviations = valid_vals - mean_val
    weighted_lag = np.asarray(W_norm.dot(deviations)).flatten()
    moran_vals = deviations / s_sq * weighted_lag
    m_mean = float(np.nanmean(moran_vals))
    m_std = float(np.nanstd(moran_vals))
    if m_std > 1e-12:
        z_vals = (moran_vals - m_mean) / m_std
    else:
        z_vals = np.zeros_like(moran_vals)
    return (moran_vals, z_vals, weighted_lag)

def _build_moran_weight_matrix(
    valid_rows: np.ndarray,
    valid_cols: np.ndarray,
    transform: Affine,
    lambda_mix: float,
    beta: float,
    alpha: float,
    d_max: float,
    route_calculator: Any = None,
) -> Tuple[Optional[Any], Dict[str, Any]]:
    """构建行标准化空间权重矩阵 W̃（固定地理邻接，用于置换检验）。"""
    from scipy.sparse import csr_matrix

    n_cells = len(valid_rows)
    a, b, c_lon, d, e, f_lat = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    cell_lats = np.zeros(n_cells)
    cell_lons = np.zeros(n_cells)
    for i, (r, c) in enumerate(zip(valid_rows, valid_cols)):
        cell_lons[i] = c_lon + a * (c + 0.5) + b * (r + 0.5)
        cell_lats[i] = f_lat + d * (c + 0.5) + e * (r + 0.5)
    coords = np.column_stack([cell_lats, cell_lons])
    use_network = route_calculator is not None and n_cells <= MORAN_MAX_CELLS_FOR_FULL_NETWORK
    route_mode = _pick_route_mode(route_calculator) if use_network else None
    if use_network and route_mode is None:
        use_network = False
    coords_rad = np.deg2rad(coords)
    tree = BallTree(coords_rad, metric="haversine")
    r_search_m = float(d_max) * (1.25 if use_network else 1.0)
    radius_rad = r_search_m / 6371000.0
    distance_metric = "network_shortest_path" if use_network else "haversine"
    if not use_network and route_calculator is not None and (n_cells > MORAN_MAX_CELLS_FOR_FULL_NETWORK):
        print(
            f"[fairness] 有效像元 {n_cells} > {MORAN_MAX_CELLS_FOR_FULL_NETWORK}，d_ij 改用大圆距离近似（可下调 MORAN_MAX_CELLS_FOR_FULL_NETWORK 以强制路网）"
        )
    batch_size = 400
    rows_list, cols_list, weights_list = ([], [], [])
    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)
        query_coords = coords_rad[start:end]
        dist_list, ind_list = tree.query_radius(query_coords, r=radius_rad, return_distance=True, sort_results=False)
        for local_i, (dists, indices) in enumerate(zip(dist_list, ind_list)):
            global_i = start + local_i
            for d_rad, j in zip(dists, indices):
                j = int(j)
                if j == global_i:
                    continue
                d_hav = float(d_rad) * 6371000.0
                if d_hav <= 0 or d_hav > d_max or (not np.isfinite(d_hav)):
                    continue
                d_ij = _pair_distance_m(
                    global_i,
                    j,
                    cell_lons,
                    cell_lats,
                    route_calculator,
                    route_mode,
                    use_network,
                )
                if d_ij <= 0 or d_ij > d_max or (not np.isfinite(d_ij)):
                    continue
                w_raw = combined_spatial_weight_raw(d_ij, lambda_mix, beta, alpha, d_max)
                if w_raw > 0:
                    rows_list.append(global_i)
                    cols_list.append(j)
                    weights_list.append(w_raw)
    meta: Dict[str, Any] = {
        "distance_metric": distance_metric,
        "row_standardized_W": True,
        "d_max_m": float(d_max),
        "lambda_mix": float(lambda_mix),
        "beta": float(beta),
        "alpha": float(alpha),
        "n_valid_cells": int(n_cells),
        "weight_scheme": "distance_decay",
    }
    if not rows_list:
        print(
            f"[fairness] 距离带内无邻接边（n_valid={n_cells}, d_max_m={d_max}, metric={distance_metric}），尝试栅格邻域权重（Queen 及更大邻域）"
        )
        for k in (1, 2, 3, 4, 5):
            rows_list, cols_list, weights_list = _grid_chebyshev_neighbor_weights(valid_rows, valid_cols, k)
            if rows_list:
                meta["weight_scheme"] = f"grid_chebyshev_k{k}_fallback"
                meta["distance_metric_note"] = f"原 {distance_metric} 在 d_max={d_max:.0f}m 下无邻接；已改用栅格 Chebyshev≤{k} 邻域"
                print(f"[fairness] 栅格邻域 k={k} 得到 {len(rows_list)} 条有向边")
                break
    if not rows_list:
        print("[fairness] Queen 邻接仍为空（多为单像元或像元互不相邻），无法计算局部 Moran")
        return (None, meta)
    W_raw = csr_matrix((weights_list, (rows_list, cols_list)), shape=(n_cells, n_cells))
    row_sums = np.array(W_raw.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    W_norm = W_raw.multiply(1.0 / row_sums[:, np.newaxis])
    return (W_norm, meta)

def compute_lisa_pvalue_raster(
    accessibility_raster: np.ndarray,
    moran_raster: np.ndarray,
    W_norm: Any,
    valid_rows: np.ndarray,
    valid_cols: np.ndarray,
    n_permutations: int = LISA_PERMUTATION_COUNT,
    random_seed: int = 42,
    progress_callback: Optional[Any] = None,
    progress_base: float = 50.0,
    progress_span: float = 38.0,
) -> np.ndarray:
    import time

    h, w = accessibility_raster.shape
    pvalue_raster = np.full((h, w), np.nan, dtype=np.float64)
    if W_norm is None:
        return pvalue_raster

    valid_vals = accessibility_raster[valid_rows, valid_cols].astype(np.float64)
    observed_moran = moran_raster[valid_rows, valid_cols].astype(np.float64)
    n = len(valid_vals)
    if n == 0:
        return pvalue_raster

    extreme_count = np.zeros(n, dtype=np.int64)
    abs_obs = np.abs(observed_moran)
    rng = np.random.default_rng(random_seed)
    n_perm = max(1, int(n_permutations))

    t0 = time.time()
    report_every = max(1, n_perm // 20)
    _report_analysis_progress(
        progress_callback,
        progress_base,
        f"正在进行 LISA 置换检验（0/{n_perm}）...",
    )
    for perm_i in range(n_perm):
        perm_vals = rng.permutation(valid_vals)
        perm_moran, _, _ = _local_moran_values_and_lag(perm_vals, W_norm)
        extreme_count += (np.abs(perm_moran) >= abs_obs)
        if progress_callback and ((perm_i + 1) % report_every == 0 or perm_i + 1 == n_perm):
            pct = progress_base + ((perm_i + 1) / n_perm) * progress_span
            _report_analysis_progress(
                progress_callback,
                pct,
                f"正在进行 LISA 置换检验（{perm_i + 1}/{n_perm}）...",
            )

    p_values = (extreme_count.astype(np.float64) + 1.0) / (n_perm + 1.0)
    for i, (r, c) in enumerate(zip(valid_rows, valid_cols)):
        pvalue_raster[int(r), int(c)] = p_values[i]

    elapsed = time.time() - t0
    sig_n = int(np.sum(p_values < 0.05))
    print(
        f"[fairness] LISA 置换检验: {n_perm} 次排列, {n} 像元, "
        f"p<0.05 像元 {sig_n}, 耗时 {elapsed:.1f}s"
    )
    return pvalue_raster

def compute_local_moran_i(accessibility_raster: np.ndarray, transform: Affine, lambda_mix: float, beta: float, alpha: float, d_max: float, route_calculator: Any = None, return_weights: bool = False):
    raster = accessibility_raster.astype(np.float64)
    raster[raster < 0] = np.nan
    valid_mask = ~np.isnan(raster)
    h, w = raster.shape
    valid_rows, valid_cols = np.where(valid_mask)
    n_cells = len(valid_rows)
    lag_empty = np.full((h, w), np.nan, dtype=np.float64)
    if n_cells == 0:
        meta = {"distance_metric": "none", "row_standardized_W": True}
        empty = (np.full((h, w), np.nan), np.full((h, w), np.nan), lag_empty, meta)
        if return_weights:
            return (*empty, None, valid_rows, valid_cols)
        return empty
    valid_vals = raster[valid_mask].astype(np.float64)
    s_sq = float(np.var(valid_vals, ddof=0))
    if np.isnan(s_sq):
        meta = {"distance_metric": "none", "row_standardized_W": True}
        empty = (np.full((h, w), np.nan), np.full((h, w), np.nan), lag_empty, meta)
        if return_weights:
            return (*empty, None, valid_rows, valid_cols)
        return empty
    if s_sq < 1e-15:
        moran_raster = np.full((h, w), np.nan, dtype=np.float64)
        z_raster = np.full((h, w), np.nan, dtype=np.float64)
        lag_raster = np.full((h, w), np.nan, dtype=np.float64)
        for r, c in zip(valid_rows, valid_cols):
            moran_raster[r, c] = 0.0
            z_raster[r, c] = 0.0
            lag_raster[r, c] = 0.0
        meta = {
            "distance_metric": "none",
            "row_standardized_W": True,
            "n_valid_cells": int(n_cells),
            "weight_scheme": "constant_accessibility_field",
            "note": "可达性在有效像元上为常数，局部 Moran 置为 0",
        }
        if return_weights:
            return (moran_raster, z_raster, lag_raster, meta, None, valid_rows, valid_cols)
        return (moran_raster, z_raster, lag_raster, meta)

    W_norm, meta = _build_moran_weight_matrix(
        valid_rows,
        valid_cols,
        transform,
        lambda_mix,
        beta,
        alpha,
        d_max,
        route_calculator,
    )
    if W_norm is None:
        empty = (np.full((h, w), np.nan), np.full((h, w), np.nan), lag_empty, meta)
        if return_weights:
            return (*empty, None, valid_rows, valid_cols)
        return empty

    moran_vals, z_vals, weighted_lag = _local_moran_values_and_lag(valid_vals, W_norm)
    moran_raster = np.full((h, w), np.nan, dtype=np.float64)
    z_raster = np.full((h, w), np.nan, dtype=np.float64)
    lag_raster = np.full((h, w), np.nan, dtype=np.float64)
    for i, (r, c) in enumerate(zip(valid_rows, valid_cols)):
        if i < len(moran_vals):
            moran_raster[r, c] = moran_vals[i]
            z_raster[r, c] = z_vals[i]
            lag_raster[r, c] = weighted_lag[i]
    if return_weights:
        return (moran_raster, z_raster, lag_raster, meta, W_norm, valid_rows, valid_cols)
    return (moran_raster, z_raster, lag_raster, meta)

def classify_lisa_clusters(moran_raster: np.ndarray, lag_raster: np.ndarray) -> np.ndarray:
    h, w = moran_raster.shape
    cluster = np.full((h, w), 0, dtype=np.int32)
    I = moran_raster.astype(np.float64)
    L = lag_raster.astype(np.float64)
    valid = ~np.isnan(I) & ~np.isnan(L)
    sl = np.zeros_like(L, dtype=np.int8)
    si = np.zeros_like(I, dtype=np.int8)
    sl[valid & (L > 0)] = 1
    sl[valid & (L < 0)] = -1
    si[valid & (I > 0)] = 1
    si[valid & (I < 0)] = -1
    cluster[valid & (sl < 0) & (si > 0)] = 1
    cluster[valid & (sl > 0) & (si > 0)] = 4
    cluster[valid & (sl > 0) & (si < 0)] = 2
    cluster[valid & (sl < 0) & (si < 0)] = 3
    rem = valid & (cluster == 0)
    cluster[rem & (sl == 0) & (si > 0)] = 4
    cluster[rem & (sl == 0) & (si < 0)] = 2
    cluster[rem & (si == 0) & (sl < 0)] = 1
    cluster[rem & (si == 0) & (sl > 0)] = 4
    cluster[valid & (cluster == 0)] = 1
    return cluster

def get_raster_bounds(raster: np.ndarray, transform: Affine) -> Dict[str, float]:
    h, w = raster.shape
    return {
        "north": float(transform.f),
        "south": float(transform.f + transform.e * h),
        "west": float(transform.c),
        "east": float(transform.c + transform.a * w),
    }

def compute_quantile_thresholds(raster_data: np.ndarray, num_levels: int = 9) -> List[float]:
    valid = raster_data[~np.isnan(raster_data)]
    if valid.size == 0:
        return [0.0] * num_levels
    percentiles = np.percentile(valid, np.linspace(0, 100, num_levels + 1))
    unique = np.unique(percentiles)
    return [round(float(t), 6) for t in unique[:num_levels]]

LISA_CLUSTER_RGBA_UINT8 = {
    0: (0, 0, 0, 0),
    1: (33, 113, 181, 200),
    2: (146, 197, 222, 200),
    3: (244, 165, 130, 200),
    4: (202, 0, 32, 200),
}

def _safe_flat_value(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(fv):
        return None
    return fv

def raster_to_flat_response(raster: np.ndarray, bounds: Dict[str, float]) -> Dict[str, Any]:
    arr = np.asarray(raster)
    h, w = arr.shape
    flat = arr.flatten()
    values = [_safe_flat_value(v) for v in flat]
    return {
        "width": int(w),
        "height": int(h),
        "values": values,
        "bounds": {
            "north": float(bounds["north"]),
            "south": float(bounds["south"]),
            "west": float(bounds["west"]),
            "east": float(bounds["east"]),
        },
    }

def _build_lisa_rgba_image(cluster_raster: np.ndarray) -> np.ndarray:
    h, w = cluster_raster.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    cluster = np.round(cluster_raster).astype(np.int32)
    for cid, color in LISA_CLUSTER_RGBA_UINT8.items():
        mask = cluster == cid
        if not np.any(mask):
            continue
        rgba[mask, 0] = color[0] / 255.0
        rgba[mask, 1] = color[1] / 255.0
        rgba[mask, 2] = color[2] / 255.0
        rgba[mask, 3] = color[3] / 255.0
    return rgba

def _build_significance_pvalue_rgba(pvalue_raster: np.ndarray) -> np.ndarray:
    """P 值显著性地图：P 越小颜色越深（双侧检验，α=0.05 为参考）。"""
    h, w = pvalue_raster.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    sig_colors = [
        (165, 0, 38),
        (215, 48, 39),
        (244, 109, 67),
        (253, 174, 97),
        (254, 224, 144),
    ]

    for r in range(h):
        for c in range(w):
            p = pvalue_raster[r, c]
            if not np.isfinite(p):
                continue
            p = float(np.clip(p, 0.0, 1.0))
            if p > 0.20:
                rgba[r, c, 3] = max(0.0, 0.12 * (1.0 - (p - 0.20) / 0.80))
                rgba[r, c, :3] = 0.92
                continue
            t = 1.0 - p / 0.20
            idx = min(len(sig_colors) - 1, int(t * len(sig_colors)))
            color = sig_colors[idx]
            alpha = 0.45 + 0.50 * t
            rgba[r, c, 0] = color[0] / 255.0
            rgba[r, c, 1] = color[1] / 255.0
            rgba[r, c, 2] = color[2] / 255.0
            rgba[r, c, 3] = alpha
    return rgba

def _build_moran_fallback_rgba(moran_raster: np.ndarray) -> np.ndarray:
    h, w = moran_raster.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    valid = np.isfinite(moran_raster)
    if not np.any(valid):
        return rgba
    vals = moran_raster[valid]
    vmax = float(max(np.max(np.abs(vals)), 1e-12))
    moran_colors = [
        (33, 113, 181, 210),
        (146, 197, 222, 210),
        (244, 165, 130, 210),
        (202, 0, 32, 210),
    ]

    def pick_color(value: float) -> Tuple[int, int, int, int]:
        if not np.isfinite(value):
            return (0, 0, 0, 0)
        t = max(-1.0, min(1.0, value / vmax))
        if t <= -0.5:
            return moran_colors[0]
        if t < 0:
            return moran_colors[1]
        if t < 0.5:
            return moran_colors[2]
        return moran_colors[3]

    for r in range(h):
        for c in range(w):
            color = pick_color(float(moran_raster[r, c]))
            rgba[r, c, 0] = color[0] / 255.0
            rgba[r, c, 1] = color[1] / 255.0
            rgba[r, c, 2] = color[2] / 255.0
            rgba[r, c, 3] = color[3] / 255.0
    return rgba

def generate_equity_heatmap_image(
    moran_raster: np.ndarray,
    cluster_raster: np.ndarray,
    bounds: Dict[str, float],
    mode: str = "cluster",
    pvalue_raster: Optional[np.ndarray] = None,
) -> Tuple[str, Dict[str, Any]]:
    import base64
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    moran = np.asarray(moran_raster, dtype=np.float64)
    cluster = np.asarray(cluster_raster, dtype=np.int32)
    view_mode = str(mode or "cluster").strip().lower()
    if view_mode not in ("significance", "cluster"):
        view_mode = "cluster"

    if view_mode == "cluster":
        rgba = _build_lisa_rgba_image(cluster)
    elif pvalue_raster is not None:
        rgba = _build_significance_pvalue_rgba(np.asarray(pvalue_raster, dtype=np.float64))
    else:
        rgba = np.zeros((moran.shape[0], moran.shape[1], 4), dtype=np.float32)
    west = float(bounds["west"])
    east = float(bounds["east"])
    south = float(bounds["south"])
    north = float(bounds["north"])
    extent = [west, east, south, north]

    fig = plt.figure(figsize=(10, 8), dpi=150, facecolor="none")
    ax = fig.add_axes([0, 0, 1, 1])
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.set_axis_off()
    ax.imshow(rgba, extent=extent, origin="lower", interpolation="nearest")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0, transparent=True)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)

    meta = {
        "mode": view_mode,
        "bounds": {
            "north": north,
            "south": south,
            "west": west,
            "east": east,
        },
        "width": int(moran.shape[1]),
        "height": int(moran.shape[0]),
    }
    return img_base64, meta

def compute_gini_coefficient(values: np.ndarray) -> float:
    v = values[np.isfinite(values) & (values >= 0)]
    n = len(v)
    if n < 2:
        return 0.0
    sorted_v = np.sort(v)
    cumsum = np.cumsum(sorted_v)
    if cumsum[-1] < 1e-15:
        return 0.0
    lorenz_y = cumsum / cumsum[-1]
    lorenz_x = np.linspace(0, 1, n)
    area = np.trapz(lorenz_y, lorenz_x)
    return float(np.clip(1.0 - 2.0 * area, 0.0, 1.0))

def compute_theil_index(values: np.ndarray) -> Dict[str, float]:
    v = values[np.isfinite(values) & (values > 0)]
    n = len(v)
    if n < 2:
        return {"theil_t": 0.0, "theil_l": 0.0, "theil_t_ge1": 0.0, "cv_squared": 0.0}
    mean_val = float(np.mean(v))
    if mean_val <= 1e-15:
        return {"theil_t": 0.0, "theil_l": 0.0, "theil_t_ge1": 0.0, "cv_squared": 0.0}
    ratios = v / mean_val
    theil_t = float(np.sum(ratios * np.log(ratios)) / n)
    theil_l = float(-np.sum(np.log(ratios)) / n)
    cv_squared = float((np.std(v) / mean_val)**2)
    return {
        "theil_t": max(0.0, theil_t),
        "theil_l": max(0.0, theil_l),
        "theil_t_ge1": max(0.0, theil_t),
        "cv_squared": max(0.0, cv_squared),
    }

def load_city_population_raster(city: str) -> Optional[np.ndarray]:
    city_key = str(city or "南京").strip()
    base_dir = Path(__file__).parent
    candidates: List[Path] = []
    try:
        from transport_modes import get_city_data_config, webgis_base_dir
        cfg = get_city_data_config(city_key)
        pop_file = cfg.get("population_file")
        if pop_file:
            candidates.append(Path(webgis_base_dir()) / pop_file)
        prefix = cfg.get("prefix", "nanjing")
        candidates.append(base_dir / "Network" / city_key / f"{prefix}_population.tif")
    except Exception:
        pass
    legacy_dirs = {
        "南京": ("Nanjin2018", "nanjing"),
        "武汉": ("Wuhan", "wuhan"),
        "北京": ("Beijing", "beijing"),
        "上海": ("Shanghai", "shanghai"),
    }
    folder, prefix = legacy_dirs.get(city_key, legacy_dirs["南京"])
    candidates.append(base_dir / folder / f"{prefix}_population.tif")
    try:
        import rasterio
        for path in candidates:
            if not path.is_file():
                continue
            with rasterio.open(path) as src:
                data = src.read(1).astype(np.float64)
                nodata = src.nodata
                if nodata is not None:
                    data = np.where(data == nodata, np.nan, data)
                data = np.where(data < 0, np.nan, data)
                return data
    except Exception:
        return None
    return None

def _align_raster_to_shape(raster: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    h_t, w_t = target_shape
    arr = np.asarray(raster, dtype=np.float64)
    if arr.shape == (h_t, w_t):
        return arr
    try:
        from PIL import Image
        img = Image.fromarray(arr.astype(np.float32))
        resized = img.resize((w_t, h_t), Image.BILINEAR)
        return np.array(resized, dtype=np.float64)
    except Exception as ex:
        raise RuntimeError(f"栅格对齐失败: {ex}") from ex

def load_city_gdp_raster(city: str) -> Optional[np.ndarray]:
    """加载城市 GDP 栅格。仅加载该城市自身的 GDP 数据，不回退到其他城市。"""
    city_key = str(city or "南京").strip()
    candidates: List[Path] = []
    try:
        from transport_modes import get_city_data_config, get_city_gdp_path, webgis_base_dir
        gdp_path = get_city_gdp_path(city_key, webgis_base_dir())
        if gdp_path:
            candidates.append(Path(gdp_path))
        cfg = get_city_data_config(city_key)
        gdp_file = cfg.get("gdp_file")
        if gdp_file:
            candidates.append(Path(webgis_base_dir()) / gdp_file)
    except Exception:
        pass
    # 仅对城市保留硬编码回退（城市是唯一有 GDP 栅格的城市）
    if city_key == "南京":
        candidates.append(Path(__file__).parent / "Network" / "Nanjing" / "Nan_GDP_2020.tif")
    try:
        import rasterio
        seen = set()
        for path in candidates:
            p = str(path.resolve()) if path.exists() else str(path)
            if p in seen or not path.is_file():
                continue
            seen.add(p)
            with rasterio.open(path) as src:
                data = src.read(1).astype(np.float64)
                nodata = src.nodata
                if nodata is not None and np.isfinite(nodata):
                    data = np.where(data == nodata, np.nan, data)
                data = np.where(data < 0, np.nan, data)
                return data
    except Exception:
        return None
    return None

GWR_NEIGHBOR_K = 40
GWR_MIN_VALID_CELLS = 80

def _gwr_local_coefficients(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> Optional[np.ndarray]:
    """加权最小二乘求局部回归系数。支持可变设计矩阵列数（如含/不含 GDP 变量）。"""
    w = np.asarray(weights, dtype=np.float64)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if np.sum(w) <= 1e-12:
        return None
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw
    n_cols = int(X.shape[1])
    try:
        beta, _, rank, _ = np.linalg.lstsq(Xw, yw, rcond=1e-8)
    except Exception:
        return None
    if rank < n_cols or beta.shape[0] < n_cols:
        return None
    if not np.all(np.isfinite(beta)):
        return None
    return beta

def compute_gwr_beta_rasters(
    accessibility_raster: np.ndarray,
    transform: Affine,
    city: str,
    resolution_deg: float,
    *,
    neighbor_k: int = GWR_NEIGHBOR_K,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    地理加权回归（GWR）：
      若有 GDP 数据:  y = β0 + β1·人均GDP + β2·人口密度 + ε
      若无 GDP 数据:  y = β0 + β1·人口密度 + ε
    y 为可达性得分；返回 β 栅格及元数据。
    当某变量数据缺失时，对应系数栅格为 None，前端对应视图不可用。
    """
    acc = np.asarray(accessibility_raster, dtype=np.float64)
    h, w = acc.shape
    city_str = str(city or "南京")

    pop_raw = load_city_population_raster(city_str)
    gdp_raw = load_city_gdp_raster(city_str)

    if pop_raw is None and gdp_raw is None:
        return {
            "success": False,
            "message": f"未找到城市人口和 GDP 栅格数据: {city}",
            "gdp_available": False,
            "population_available": False,
        }
    if pop_raw is None:
        return {
            "success": False,
            "message": f"未找到城市人口栅格: {city}",
            "gdp_available": gdp_raw is not None,
            "population_available": False,
        }

    gdp_available = gdp_raw is not None
    population_available = True

    pop = _align_raster_to_shape(pop_raw, (h, w))
    resolution_m = float(resolution_deg) * DEGREE_TO_METER
    cell_area_km2 = max((resolution_m / 1000.0) ** 2, 1e-12)
    pop_density = pop / cell_area_km2

    if gdp_available:
        gdp = _align_raster_to_shape(gdp_raw, (h, w))
        gdp_per_capita = np.full_like(gdp, np.nan, dtype=np.float64)
        np.divide(gdp, pop, out=gdp_per_capita, where=pop > 0)
        valid = (
            np.isfinite(acc)
            & (acc >= 0)
            & np.isfinite(gdp_per_capita)
            & (gdp_per_capita > 0)
            & np.isfinite(pop_density)
            & (pop_density > 0)
        )
    else:
        gdp_per_capita = None
        valid = (
            np.isfinite(acc)
            & (acc >= 0)
            & np.isfinite(pop_density)
            & (pop_density > 0)
        )

    rows, cols = np.where(valid)
    n = len(rows)
    if n < GWR_MIN_VALID_CELLS:
        return {
            "success": False,
            "message": f"GWR 有效像元不足 ({n} < {GWR_MIN_VALID_CELLS})",
            "gdp_available": gdp_available,
            "population_available": population_available,
        }

    y = acc[rows, cols].astype(np.float64)
    x_pop = np.log1p(pop_density[rows, cols].astype(np.float64))
    if gdp_available:
        x_gdp = np.log1p(gdp_per_capita[rows, cols].astype(np.float64))

    lons = transform.c + (cols.astype(np.float64) + 0.5) * transform.a
    lats = transform.f + (rows.astype(np.float64) + 0.5) * transform.e
    coords_rad = np.column_stack([np.radians(lats), np.radians(lons)])

    k_query = min(int(neighbor_k) + 1, n)
    tree = BallTree(coords_rad, metric="haversine")
    dist_rad, neighbor_idx = tree.query(coords_rad, k=k_query)
    dist_m = dist_rad * 6371000.0

    beta_gdp = np.full(n, np.nan, dtype=np.float64)
    beta_pop = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        if progress_callback and i % 400 == 0:
            pct = 88.0 + (i / max(n, 1)) * 8.0
            progress_callback(min(96.0, pct), f"正在计算 GWR 局部系数 ({i}/{n})...")

        nbr = neighbor_idx[i]
        d_i = dist_m[i]
        bw = float(d_i[-1]) if len(d_i) > 1 else max(float(d_i[0]), 500.0)
        if not np.isfinite(bw) or bw <= 1.0:
            bw = 500.0

        d_nbr = d_i
        u = d_nbr / bw
        weights = np.where(u < 1.0, (1.0 - u * u) ** 2, 0.0)
        weights = np.where(np.isfinite(weights), weights, 0.0)
        if np.sum(weights) <= 1e-12:
            continue

        if gdp_available:
            X = np.column_stack([
                np.ones(len(nbr), dtype=np.float64),
                x_gdp[nbr],
                x_pop[nbr],
            ])
            beta = _gwr_local_coefficients(X, y[nbr], weights)
            if beta is None:
                continue
            beta_gdp[i] = float(beta[1])
            beta_pop[i] = float(beta[2])
        else:
            X = np.column_stack([
                np.ones(len(nbr), dtype=np.float64),
                x_pop[nbr],
            ])
            beta = _gwr_local_coefficients(X, y[nbr], weights)
            if beta is None:
                continue
            beta_pop[i] = float(beta[1])

    beta_gdp_raster = None
    beta_gdp_raster_np = None
    fitted_info = ""
    if gdp_available:
        beta_gdp_raster_np = np.full((h, w), np.nan, dtype=np.float64)
        beta_gdp_raster_np[rows, cols] = beta_gdp
        beta_gdp_raster = beta_gdp_raster_np

    beta_pop_raster = np.full((h, w), np.nan, dtype=np.float64)
    beta_pop_raster[rows, cols] = beta_pop

    if gdp_available:
        valid_beta = np.isfinite(beta_gdp) & np.isfinite(beta_pop)
        fitted_cells = int(np.sum(valid_beta))
        fitted_info = ""
    else:
        valid_beta = np.isfinite(beta_pop)
        fitted_cells = int(np.sum(valid_beta))

    if fitted_cells < 10:
        return {
            "success": False,
            "message": "GWR 系数估计失败（有效系数过少）",
            "gdp_available": gdp_available,
            "population_available": population_available,
        }

    gwr_meta = {
        "model": "GWR",
        "dependent_var": "accessibility",
        "gdp_available": gdp_available,
        "population_available": population_available,
        "neighbor_k": int(neighbor_k),
        "valid_cells": int(n),
        "fitted_cells": fitted_cells,
        "cell_area_km2": float(cell_area_km2),
    }
    if gdp_available:
        gwr_meta["x1_label"] = "人均GDP (log1p)"
        gwr_meta["x2_label"] = "人口密度 (log1p, 人/km²)"
        gwr_meta["beta_gdp_mean"] = float(np.nanmean(beta_gdp))
        gwr_meta["beta_pop_mean"] = float(np.nanmean(beta_pop))
    else:
        gwr_meta["x1_label"] = "人口密度 (log1p, 人/km²)"
        gwr_meta["beta_pop_mean"] = float(np.nanmean(beta_pop))

    return {
        "success": True,
        "beta_gdp_raster": beta_gdp_raster,
        "beta_pop_density_raster": beta_pop_raster,
        "gdp_available": gdp_available,
        "population_available": population_available,
        "gwr_meta": gwr_meta,
    }

def _build_continuous_rgba_image(values: np.ndarray, *, diverging: bool = True) -> np.ndarray:
    """连续系数栅格 → RGBA 图像（无效值为透明）。"""
    arr = np.asarray(values, dtype=np.float64)
    h, w = arr.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return rgba

    vals = arr[valid]
    if diverging:
        vmax = float(np.percentile(np.abs(vals), 98))
        if vmax <= 1e-12:
            vmax = float(np.max(np.abs(vals))) or 1.0
        norm = np.clip(arr / vmax, -1.0, 1.0)
        pos = norm >= 0
        rgba[pos, 0] = 0.84
        rgba[pos, 1] = np.clip(0.38 + 0.62 * norm[pos], 0, 1)
        rgba[pos, 2] = np.clip(0.20 + 0.30 * (1.0 - norm[pos]), 0, 1)
        neg = ~pos & valid
        rgba[neg, 0] = np.clip(0.20 + 0.60 * (1.0 + norm[neg]), 0, 1)
        rgba[neg, 1] = np.clip(0.45 + 0.40 * (1.0 + norm[neg]), 0, 1)
        rgba[neg, 2] = 0.84
    else:
        vmin = float(np.percentile(vals, 2))
        vmax = float(np.percentile(vals, 98))
        if vmax <= vmin:
            vmax = vmin + 1.0
        norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
        rgba[..., 0] = np.clip(0.15 + 0.85 * norm, 0, 1)
        rgba[..., 1] = np.clip(0.35 + 0.45 * (1.0 - np.abs(norm - 0.5) * 2), 0, 1)
        rgba[..., 2] = np.clip(1.0 - 0.85 * norm, 0, 1)

    rgba[valid, 3] = 0.82
    rgba[~valid, 3] = 0.0
    return rgba

def generate_gwr_heatmap_image(
    coef_raster: np.ndarray,
    bounds: Dict[str, float],
    *,
    title: str = "GWR",
) -> Tuple[str, Dict[str, Any]]:
    import base64
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgba = _build_continuous_rgba_image(coef_raster, diverging=True)
    west = float(bounds["west"])
    east = float(bounds["east"])
    south = float(bounds["south"])
    north = float(bounds["north"])
    extent = [west, east, south, north]

    fig = plt.figure(figsize=(10, 8), dpi=150, facecolor="none")
    ax = fig.add_axes([0, 0, 1, 1])
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.set_axis_off()
    ax.imshow(rgba, extent=extent, origin="lower", interpolation="nearest")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0, transparent=True)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)

    h, w = coef_raster.shape
    return img_base64, {
        "mode": title,
        "bounds": {"north": north, "south": south, "west": west, "east": east},
        "width": int(w),
        "height": int(h),
    }

def compute_population_weighted_fairness(accessibility_raster: np.ndarray, population_raster: np.ndarray, gdf_bounds: Dict[str, float] = None) -> Dict[str, Any]:
    h_acc, w_acc = accessibility_raster.shape
    if population_raster.shape != (h_acc, w_acc):
        try:
            from PIL import Image
            pop_img = Image.fromarray(population_raster.astype(np.float32))
            pop_resized = pop_img.resize((w_acc, h_acc), Image.BILINEAR)
            pop_aligned = np.array(pop_resized, dtype=np.float64)
        except Exception:
            return {
                "success": False,
                "message": "人口栅格对齐失败",
                "pop_weighted_gini": None,
                "pop_weighted_mean": None,
            }
    else:
        pop_aligned = population_raster.astype(np.float64)
    mask = np.isfinite(accessibility_raster) & (accessibility_raster
                                                >= 0) & np.isfinite(pop_aligned) & (pop_aligned > 0)
    acc_valid = accessibility_raster[mask]
    pop_valid = pop_aligned[mask]
    if len(acc_valid) < 2:
        return {
            "success": False,
            "message": "有效像元不足",
            "pop_weighted_gini": None,
            "pop_weighted_mean": None,
        }
    total_pop = float(np.sum(pop_valid))
    pop_weighted_mean = float(np.sum(acc_valid * pop_valid) / total_pop) if total_pop > 0 else 0.0
    sort_idx = np.argsort(acc_valid)
    acc_sorted = acc_valid[sort_idx]
    pop_sorted = pop_valid[sort_idx]
    cum_pop_share = np.cumsum(pop_sorted) / total_pop
    cum_acc_share = np.cumsum(acc_sorted * pop_sorted)
    if cum_acc_share[-1] < 1e-15:
        pop_weighted_gini = 0.0
    else:
        cum_acc_share = cum_acc_share / cum_acc_share[-1]
        area = np.trapz(cum_acc_share, cum_pop_share)
        pop_weighted_gini = float(np.clip(1.0 - 2.0 * area, 0.0, 1.0))
    n_deciles = 10
    decile_boundaries = []
    pop_cum_frac = 0.0
    idx = 0
    for d in range(1, n_deciles):
        target = d / n_deciles
        while idx < len(cum_pop_share) and cum_pop_share[idx] < target:
            idx += 1
        if idx < len(acc_sorted):
            decile_boundaries.append(round(float(acc_sorted[min(idx, len(acc_sorted) - 1)]), 4))
        else:
            decile_boundaries.append(round(float(acc_sorted[-1]), 4))
    return {
        "success": True,
        "pop_weighted_gini": pop_weighted_gini,
        "pop_weighted_mean": round(pop_weighted_mean, 4),
        "unweighted_gini": compute_gini_coefficient(acc_valid),
        "unweighted_mean": round(float(np.mean(acc_valid)), 4),
        "valid_cell_count": int(len(acc_valid)),
        "total_population": round(total_pop, 2),
        "pop_weighted_decile_boundaries": decile_boundaries,
    }

EQUITY_TRAVEL_TIME_DEFAULT = 30

EQUITY_TRANSPORT_MODE_DEFAULT = "bike"

EQUITY_TRAVEL_TIME_MIN = 5

EQUITY_TRAVEL_TIME_MAX = 180

EQUITY_TRAVEL_TIME_STEP = 5

def normalize_equity_travel_params(travel_time_min=None, transport_mode=None) -> Tuple[int, str, float]:
    from transport_modes import speed_ms
    tt = int(float(travel_time_min if travel_time_min is not None else EQUITY_TRAVEL_TIME_DEFAULT))
    tt = round(tt / EQUITY_TRAVEL_TIME_STEP) * EQUITY_TRAVEL_TIME_STEP
    tt = max(EQUITY_TRAVEL_TIME_MIN, min(EQUITY_TRAVEL_TIME_MAX, tt))
    mode = str(transport_mode or EQUITY_TRANSPORT_MODE_DEFAULT).strip() or EQUITY_TRANSPORT_MODE_DEFAULT
    d_max = tt * 60.0 * speed_ms(mode)
    d_max = max(500.0, min(float(d_max), 100000.0))
    return (tt, mode, d_max)

def travel_mode_label_zh(mode: str) -> str:
    from transport_modes import normalize_mode_key
    key = normalize_mode_key(mode)
    labels = {
        "walk": "步行",
        "bike": "骑行",
        "metro": "地铁",
        "bus": "公交",
        "drive": "私家车",
    }
    return labels.get(key, str(mode or ""))

def compute_accessibility_raster_for_equity(poi_category: str, city: str, subcategory: Optional[str], resolution: float, travel_time_min: int, transport_mode: str, progress_callback=None) -> Tuple[Optional[np.ndarray], Optional[Affine], Optional[str]]:
    from AccessibilityAnalysis import read_all_city_data, NanjingAccessibilityCalculator
    from Data_loader import filter_poi_by_subcategory
    from transport_modes import speed_ms
    (
        networks,
        poi_data,
        population_data,
        pop_transform,
        nodes_with_population,
        admin_boundary,
        pop_raster_crs,
    ) = read_all_city_data(city=city, use_cache=True)
    if not poi_data or all((getattr(gdf, "empty", True) for gdf in poi_data.values() if gdf is not None)):
        return (None, None, "POI 数据为空")
    calculator = NanjingAccessibilityCalculator(
        networks,
        poi_data,
        admin_boundary,
        population_data=population_data,
        pop_transform=pop_transform,
        city=city,
        pop_raster_crs=pop_raster_crs,
        nodes_with_population=nodes_with_population,
    )
    subcategory_keyword = str(subcategory or "").strip()
    poi_gdf_filtered = None
    if poi_category:
        poi_gdf = calculator.poi_data.get(poi_category)
        if poi_gdf is not None and (not getattr(poi_gdf, "empty", True)):
            calculator.activities = [poi_category]
            calculator.activity_weights = {poi_category: 1.0}
            if subcategory_keyword:
                poi_gdf_filtered = filter_poi_by_subcategory(poi_gdf, subcategory_keyword, poi_category)
    calculator.transport_mode = transport_mode
    calculator.travel_time_minutes = int(travel_time_min)
    t0_distance_m = float(travel_time_min) * 60.0 * speed_ms(transport_mode)
    raster_data, transform, _stats, _activity_raster = calculator.calculate_composite_accessibility(
        resolution=float(resolution),
        progress_callback=progress_callback,
        subcategory_keyword=subcategory_keyword,
        subcategory_poi_gdf=poi_gdf_filtered,
        subcategory_poi_activity=poi_category,
        t0_distance_m=t0_distance_m,
    )
    return (raster_data, transform, None)

def resolution_to_meters(resolution_degree: float) -> int:
    return int(round(float(resolution_degree) * DEGREE_TO_METER))

def build_accessibility_cache_key(city: str, poi_category: str, subcategory: Optional[str], travel_time_min: int, transport_mode: str, resolution: Optional[float] = None) -> Tuple[str, str, str, int, str, int]:
    res_m = resolution_to_meters(resolution) if resolution is not None and resolution > 0 else 0
    return (
        str(city or "南京").strip(),
        str(poi_category or "work").strip(),
        str(subcategory or "").strip(),
        int(travel_time_min),
        str(transport_mode or EQUITY_TRANSPORT_MODE_DEFAULT).strip(),
        res_m,
    )

def store_accessibility_cache_entry(cache: Dict, city: str, poi_category: str, subcategory: Optional[str], travel_time_min: int, transport_mode: str, raster_data: np.ndarray, transform: Affine, stats: Optional[Dict] = None, activity_raster_data: Optional[Dict] = None, resolution: Optional[float] = None) -> Tuple[str, str, str, int, str, int]:
    from datetime import datetime
    key = build_accessibility_cache_key(city, poi_category, subcategory, travel_time_min, transport_mode, resolution)
    cache[key] = {
        "city": str(city or "南京").strip(),
        "resolution_m": key[5],
        "raster_data": np.asarray(raster_data).copy(),
        "transform": transform,
        "stats": stats.copy() if isinstance(stats, dict) else stats,
        "last_calculated": datetime.now().isoformat(),
        "activity_raster_data": ({k: v.copy() for k, v in activity_raster_data.items()}
                                 if isinstance(activity_raster_data, dict) else None),
    }
    return key

def get_or_compute_accessibility_raster_for_equity(poi_category: str, city: str, subcategory: Optional[str], resolution: float, travel_time_min: int, transport_mode: str, cache: Optional[Dict] = None, progress_callback=None) -> Tuple[Optional[np.ndarray], Optional[Affine], Optional[str], bool]:
    city_name = str(city or "南京").strip()
    cache_key = build_accessibility_cache_key(
        city_name,
        poi_category,
        subcategory,
        travel_time_min,
        transport_mode,
        resolution,
    )
    if cache is not None:
        cached = cache.get(cache_key)
        if cached and cached.get("raster_data") is not None and (cached.get("transform") is not None):
            print(
                f"[公平性分析] 使用内存缓存可达性栅格: city={cache_key[0]}, poi={cache_key[1]}, sub={cache_key[2] or '(全部)'}, time={cache_key[3]}min, mode={cache_key[4]}, resol={cache_key[5]}m"
            )
            return (cached["raster_data"], cached["transform"], None, True)
    print(
        f"[公平性分析] 未命中内存缓存，开始计算可达性栅格 (磁盘目录 resol_{cache_key[5]}): city={cache_key[0]}, poi={cache_key[1]}, sub={cache_key[2] or '(全部)'}, time={cache_key[3]}min, mode={cache_key[4]}"
    )
    raster_data, transform, err = compute_accessibility_raster_for_equity(
        poi_category=poi_category,
        city=city_name,
        subcategory=subcategory,
        resolution=resolution,
        travel_time_min=int(travel_time_min),
        transport_mode=str(transport_mode),
        progress_callback=progress_callback,
    )
    if err:
        return (None, None, err, False)
    if raster_data is None or transform is None:
        return (None, None, "可达性栅格计算失败", False)
    if cache is not None:
        store_accessibility_cache_entry(
            cache,
            city_name,
            poi_category,
            subcategory,
            int(travel_time_min),
            str(transport_mode),
            raster_data,
            transform,
            resolution=resolution,
        )
        print(f"[公平性分析] 可达性栅格已写入内存缓存 (resol_{cache_key[5]}m): {cache_key}")
    return (raster_data, transform, None, False)

def _find_district_name_column(gdf) -> str:
    """在行政区划 GeoDataFrame 中查找区名列。"""
    if 'name' in gdf.columns:
        return 'name'
    # 城市的 shp 第一个字段为中文名称（如 "名称" / "NAME" 等）
    for col in gdf.columns:
        if col.lower() in ('name', 'district', '名称', '区', '区县'):
            return col
    # 回退到第一个非 geometry 列
    for col in gdf.columns:
        if col != 'geometry':
            return col
    return gdf.columns[0]

def compute_district_stats(
    accessibility_raster: np.ndarray,
    transform: Affine,
    city: str,
    cluster_raster: np.ndarray = None,
) -> Optional[Dict[str, Any]]:
    """按行政区划统计可达性，返回各区明细（按均值升序——最差在前）。

    策略：先用 rasterio.features.rasterize(all_touched=True) 做一次性 ID 光栅化；
    若某区得到的像元数不足，退回 shapely Point.contains 逐点判定兜底。
    """
    import geopandas as gpd
    from rasterio.features import rasterize
    from shapely.geometry import Point

    city_str = str(city or "南京").strip()
    base_dir = Path(__file__).parent
    try:
        from transport_modes import get_city_data_config
        cfg = get_city_data_config(city_str)
        boundary_file = cfg.get("boundary_file")
        if not boundary_file:
            return None
        boundary_path = os.path.join(str(base_dir), boundary_file)
    except Exception:
        return None

    if not os.path.isfile(boundary_path):
        return None

    try:
        gdf = gpd.read_file(boundary_path)
    except Exception:
        return None

    if gdf is None or len(gdf) == 0:
        return None

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    elif gdf.crs.to_string() != "EPSG:4326":
        try:
            gdf = gdf.to_crs("EPSG:4326")
        except Exception:
            pass

    name_col = _find_district_name_column(gdf)
    h, w = accessibility_raster.shape
    acc = np.asarray(accessibility_raster, dtype=np.float64)

    shapes = []
    idx_to_key = {}  # rid → (gdf_index, name)
    for idx_val, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        rid = int(idx_val) + 1
        shapes.append((geom, rid))
        name = str(row[name_col]) if name_col in gdf.columns else str(row.iloc[0])
        idx_to_key[rid] = (idx_val, name)

    if not shapes:
        return None

    try:
        id_raster = rasterize(
            shapes,
            out_shape=(h, w),
            transform=transform,
            fill=0,
            dtype=np.int32,
            all_touched=True,
        )
    except Exception:
        return None

    a = float(transform.a)
    c0 = float(transform.c)
    e_val = float(transform.e)
    f0 = float(transform.f)

    def _make_grid_for_bbox(minx, miny, maxx, maxy):
        """返回 (sub_lons, sub_lats, sub_acc, row_min, row_max, col_min, col_max)。"""
        c_first = int(np.floor((minx - c0) / a - 0.5))
        c_last  = int(np.ceil((maxx - c0) / a - 0.5))
        cmi = max(0, c_first - 1)
        cma = min(w, c_last + 2)
        if e_val < 0:
            r_first = int(np.floor((maxy - f0) / e_val - 0.5))
            r_last  = int(np.ceil((miny - f0) / e_val - 0.5))
        else:
            r_first = int(np.floor((miny - f0) / e_val - 0.5))
            r_last  = int(np.ceil((maxy - f0) / e_val - 0.5))
        rmi = max(0, r_first - 1)
        rma = min(h, r_last + 2)
        if rmi >= rma or cmi >= cma:
            return None
        col_1d = np.arange(cmi, cma, dtype=np.float64)
        row_1d = np.arange(rmi, rma, dtype=np.float64)
        slons = c0 + a * (col_1d + 0.5)
        slats = f0 + e_val * (row_1d + 0.5)
        lons2d, lats2d = np.meshgrid(slons, slats)
        return lons2d, lats2d, acc[rmi:rma, cmi:cma], rmi, rma, cmi, cma

    districts = []
    for rid, (gdf_idx, name) in idx_to_key.items():
        row = gdf.loc[gdf_idx]
        geom = row.geometry

        # 3a. 先用 ID 光栅化结果
        mask = id_raster == rid
        n_raster = int(np.sum(mask))
        if n_raster >= 5:
            region_acc = acc[mask]
            valid_acc = region_acc[np.isfinite(region_acc)]
            if len(valid_acc) >= 5:
                stats = {
                    "name": name,
                    "mean_accessibility": float(np.mean(valid_acc)),
                    "median_accessibility": float(np.median(valid_acc)),
                    "cell_count": int(len(valid_acc)),
                }
                if cluster_raster is not None:
                    region_cl = cluster_raster[mask]
                    for label, label_name in CLUSTER_LABELS.items():
                        if label == 0:
                            continue
                        stats[f"cluster_{label_name}"] = int(np.sum(region_cl == label))
                districts.append(stats)
                continue
            # 否则：有 mask 像素但有效值不足 → 尝试 shapely 兜底
        # 否则：光栅化未捕获到足够像素 → shapely 兜底

        # 3b. Shapely 兜底
        grid = _make_grid_for_bbox(*geom.bounds)
        if grid is None:
            continue
        sub_lons, sub_lats, sub_acc, _rmi, _rma, _cmi, _cma = grid
        n_cells = int(sub_lons.size)
        inside = np.zeros(n_cells, dtype=bool)
        batch = 1500
        flat_lons = sub_lons.ravel()
        flat_lats = sub_lats.ravel()
        for start in range(0, n_cells, batch):
            end = min(start + batch, n_cells)
            for i in range(start, end):
                pt = Point(float(flat_lons[i]), float(flat_lats[i]))
                inside[i] = geom.contains(pt)
                if not inside[i]:
                    inside[i] = geom.touches(pt) or geom.intersects(pt)

        if not np.any(inside):
            continue

        region_acc = sub_acc.ravel()[inside]
        valid_acc = region_acc[np.isfinite(region_acc)]
        if len(valid_acc) < 5:
            continue

        stats = {
            "name": name,
            "mean_accessibility": float(np.mean(valid_acc)),
            "median_accessibility": float(np.median(valid_acc)),
            "cell_count": int(len(valid_acc)),
        }
        if cluster_raster is not None:
            sub_cl = cluster_raster[_rmi:_rma, _cmi:_cma]
            region_cl = sub_cl.ravel()[inside]
            for label, label_name in CLUSTER_LABELS.items():
                if label == 0:
                    continue
                stats[f"cluster_{label_name}"] = int(np.sum(region_cl == label))
        districts.append(stats)

    if not districts:
        return None

    districts.sort(key=lambda x: x["mean_accessibility"])

    return {
        "districts": districts,
        "city": city_str,
        "total_districts": len(districts),
    }

def analyze_fairness_from_poi(poi_category: str, city: str = "南京", subcategory: str = None, resolution: float = DEFAULT_RESOLUTION, search_radius_m: float = None, travel_time_min: int = None, transport_mode: str = None, accessibility_raster: np.ndarray = None, accessibility_cache: Optional[Dict] = None, bounds: Tuple[float, float, float, float] = None, spatial_weight_mix_lambda: Optional[float] = None, population_weighted: bool = False, progress_callback: Optional[Any] = None) -> Dict:
    default_poi = "medical"
    weight_params = dict(MORAN_SPATIAL_WEIGHT_PARAMS.get(poi_category, MORAN_SPATIAL_WEIGHT_PARAMS[default_poi]))
    use_travel_params = travel_time_min is not None or transport_mode is not None
    resolved_travel_time = None
    resolved_transport_mode = None
    if use_travel_params:
        resolved_travel_time, resolved_transport_mode, d_max_from_travel = normalize_equity_travel_params(
            travel_time_min, transport_mode
        )
        weight_params["d_max"] = d_max_from_travel
    elif search_radius_m is not None:
        weight_params["d_max"] = float(search_radius_m)
    if spatial_weight_mix_lambda is not None:
        weight_params["lambda_mix"] = float(np.clip(spatial_weight_mix_lambda, 0.0, 1.0))
    d_max = float(weight_params["d_max"])
    lam_w = float(weight_params["lambda_mix"])
    beta_w = float(weight_params["beta"])
    alpha_w = float(weight_params["alpha"])
    accessibility_from_cache = False
    try:
        _report_analysis_progress(progress_callback, 5, "正在准备公平性分析...")
        if accessibility_raster is None:
            if use_travel_params:
                acc_progress = _wrap_sub_progress(progress_callback, 8, 40)
                accessibility_raster, transform, acc_err, accessibility_from_cache = get_or_compute_accessibility_raster_for_equity(
                    poi_category=poi_category,
                    city=str(city or "南京").strip(),
                    subcategory=subcategory,
                    resolution=resolution,
                    travel_time_min=resolved_travel_time,
                    transport_mode=resolved_transport_mode,
                    cache=accessibility_cache,
                    progress_callback=acc_progress,
                )
                if acc_err:
                    return {"success": False, "message": acc_err}
                if accessibility_raster is None or transform is None:
                    return {"success": False, "message": "可达性栅格计算失败"}
                if accessibility_from_cache:
                    _report_analysis_progress(progress_callback, 45, "已使用缓存可达性栅格")
                else:
                    _report_analysis_progress(progress_callback, 45, "可达性栅格计算完成")
            else:
                city_name = str(city or "南京").strip()
                poi_folder = CITY_FOLDER_MAP.get(city_name, "Nanjing")
                base_dir = Path(__file__).parent
                poi_dir = base_dir / "POI" / poi_folder / "Location-based"
                file_path = find_poi_file(poi_dir, poi_category, subcategory)
                if file_path is None or not file_path.exists():
                    return {
                        "success": False,
                        "message": f"未找到 POI 文件: {poi_dir / poi_category}",
                    }
                poi_data = load_poi_npz(str(file_path))
                if poi_data is None:
                    return {
                        "success": False,
                        "message": f"加载 POI 文件失败: {file_path}",
                    }
                _report_analysis_progress(progress_callback, 15, "正在由 POI 生成可达性栅格...")
                accessibility_raster, transform, _bounds_tuple = points_to_raster(
                    poi_data["demand_coords"],
                    poi_data["ai"],
                    resolution=resolution,
                    bounds=bounds,
                )
                _report_analysis_progress(progress_callback, 45, "可达性栅格准备完成")
        else:
            if bounds is None:
                return {
                    "success": False,
                    "message": "传入 accessibility_raster 时必须同时提供 bounds=(min_lon, min_lat, max_lon, max_lat)",
                }
            min_lon, min_lat, max_lon, max_lat = bounds
            h, w = accessibility_raster.shape
            if h < 1 or w < 1:
                return {"success": False, "message": "可达性栅格尺寸无效"}
            transform = Affine(
                (max_lon - min_lon) / w,
                0.0,
                min_lon,
                0.0,
                -(max_lat - min_lat) / h,
                max_lat,
            )
        if accessibility_raster.size == 0:
            return {"success": False, "message": "可达性栅格为空"}
        valid_mask = ~np.isnan(accessibility_raster)
        if not np.any(valid_mask):
            return {"success": False, "message": "可达性栅格无有效值"}
        _report_analysis_progress(progress_callback, 48, "正在计算局部 Moran 指数...")
        route_calc = try_init_route_calculator(str(city or "南京"))
        moran_raster, z_raster, lag_raster, wmeta, W_norm, valid_rows, valid_cols = compute_local_moran_i(
            accessibility_raster,
            transform,
            lambda_mix=lam_w,
            beta=beta_w,
            alpha=alpha_w,
            d_max=d_max,
            route_calculator=route_calc,
            return_weights=True,
        )
        print(f"[fairness] 正在计算 LISA 显著性 P 值（{LISA_PERMUTATION_COUNT} 次随机置换）...")
        pvalue_raster = compute_lisa_pvalue_raster(
            accessibility_raster,
            moran_raster,
            W_norm,
            valid_rows,
            valid_cols,
            n_permutations=LISA_PERMUTATION_COUNT,
            progress_callback=progress_callback,
            progress_base=50.0,
            progress_span=38.0,
        )
        _report_analysis_progress(progress_callback, 90, "正在生成 LISA 聚类与公平性指标...")
        cluster_raster = classify_lisa_clusters(moran_raster, lag_raster)
        acc_valid = accessibility_raster[np.isfinite(accessibility_raster) & (accessibility_raster >= 0)]
        gini_coef = compute_gini_coefficient(acc_valid)
        theil_indices = compute_theil_index(acc_valid)
        gwr_result = compute_gwr_beta_rasters(
            accessibility_raster,
            transform,
            str(city or "南京"),
            float(resolution),
            progress_callback=progress_callback,
        )
        gwr_meta = None
        beta_gdp_raster = None
        beta_pop_raster = None
        gdp_view_available = False
        population_view_available = False
        if gwr_result.get("success"):
            beta_gdp_raster = gwr_result.get("beta_gdp_raster")
            beta_pop_raster = gwr_result.get("beta_pop_density_raster")
            gwr_meta = gwr_result.get("gwr_meta")
            gdp_view_available = bool(gwr_result.get("gdp_available", beta_gdp_raster is not None))
            population_view_available = bool(gwr_result.get("population_available", beta_pop_raster is not None))
        else:
            # 即使 GWR 整体失败，也传递各视图的可用性（用于前后端联动隐藏不可用按钮）
            gdp_view_available = bool(gwr_result.get("gdp_available", False))
            population_view_available = bool(gwr_result.get("population_available", False))
            print(f"[fairness] GWR 跳过: {gwr_result.get('message')}")
        pop_weighted_metrics = None
        if population_weighted:
            pop_raster = load_city_population_raster(str(city or "南京"))
            if pop_raster is None:
                pop_weighted_metrics = {
                    "success": False,
                    "message": f"未找到城市人口栅格: {city}",
                }
            else:
                pop_weighted_metrics = compute_population_weighted_fairness(accessibility_raster, pop_raster)
        moran_vals = moran_raster[~np.isnan(moran_raster)]
        gmean = float(np.mean(moran_vals)) if len(moran_vals) > 0 else 0.0
        if not np.isfinite(gmean):
            gmean = 0.0
        cluster_counts = {}
        for label, name in CLUSTER_LABELS.items():
            if label == 0:
                continue
            cluster_counts[name] = int(np.sum(cluster_raster == label))
        color_thresholds = compute_quantile_thresholds(moran_raster, num_levels=9)
        equity_bounds = get_raster_bounds(accessibility_raster, transform)

        def _finite_stat(arr, fn):
            if not np.any(~np.isnan(arr)):
                return 0.0
            v = float(fn(arr))
            return v if np.isfinite(v) else 0.0

        moran_flat = raster_to_flat_response(moran_raster, equity_bounds)
        cluster_flat = raster_to_flat_response(cluster_raster.astype(np.float64), equity_bounds)
        z_flat = raster_to_flat_response(z_raster, equity_bounds)
        pvalue_flat = raster_to_flat_response(pvalue_raster, equity_bounds)
        gwr_gdp_flat = (
            raster_to_flat_response(beta_gdp_raster, equity_bounds)
            if beta_gdp_raster is not None
            else None
        )
        gwr_pop_flat = (
            raster_to_flat_response(beta_pop_raster, equity_bounds)
            if beta_pop_raster is not None
            else None
        )
        pvalue_valid = pvalue_raster[np.isfinite(pvalue_raster)]
        sig_cells_p005 = int(np.sum(pvalue_valid < 0.05)) if pvalue_valid.size > 0 else 0

        # 行政区划级别的可达性与聚类统计
        district_stats = compute_district_stats(
            accessibility_raster, transform, str(city or "南京"), cluster_raster
        )

        _report_analysis_progress(progress_callback, 98, "正在整理分析结果...")

        return {
            "success": True,
            "poi_category": poi_category,
            "subcategory": subcategory or "",
            "equity_bounds": equity_bounds,
            "raster_data": moran_flat,
            "cluster_raster_data": cluster_flat,
            "z_raster_data": z_flat,
            "pvalue_raster_data": pvalue_flat,
            "gwr_gdp_raster_data": gwr_gdp_flat,
            "gwr_pop_density_raster_data": gwr_pop_flat,
            "_moran_raster_np": moran_raster,
            "_cluster_raster_np": cluster_raster,
            "_z_raster_np": z_raster,
            "_pvalue_raster_np": pvalue_raster,
            "_gwr_beta_gdp_raster_np": beta_gdp_raster,
            "_gwr_beta_pop_raster_np": beta_pop_raster,
            "fairness_metrics": {
                "gini_coefficient": gini_coef,
                "theil_t": theil_indices["theil_t"],
                "theil_l": theil_indices["theil_l"],
                "cv_squared": theil_indices["cv_squared"],
            },
            "moran_stats": {
                "global_moran": gmean,
                "significant_cells": int(np.sum(~np.isnan(moran_raster))),
                "significant_cells_p005": sig_cells_p005,
                "lisa_permutation_n": LISA_PERMUTATION_COUNT,
                "lisa_pvalue_threshold": 0.05,
                "lisa_null_hypothesis": "空间单元属性值与其邻居之间不存在空间关联（完全随机分布）",
                "lisa_pvalue_formula": "双侧检验 P = (|I_perm|≥|I_obs| 的次数 + 1) / (排列次数 + 1)",
                "cluster_counts": cluster_counts,
                "mean_accessibility": _finite_stat(accessibility_raster, np.nanmean),
                "median_accessibility": _finite_stat(accessibility_raster, np.nanmedian),
                "accessibility_variance_s2": _finite_stat(accessibility_raster, np.nanvar),
                "spatial_weight_mix_lambda": lam_w,
                "beta": beta_w,
                "alpha": alpha_w,
                "d_max_m": d_max,
                "travel_time_min": resolved_travel_time,
                "transport_mode": resolved_transport_mode,
                "transport_mode_label": (
                    travel_mode_label_zh(resolved_transport_mode) if resolved_transport_mode else None
                ),
                "row_standardized_W": bool(wmeta.get("row_standardized_W", True)),
                "distance_metric": wmeta.get("distance_metric", "haversine"),
                "moran_weight_meta": wmeta,
                "local_moran_note": "I_i=(A_i-Ā)/S²·Σ_j w̃_ij(A_j-Ā)，w̃ 为行标准化后权重；LISA：LL(L<0,I>0)、LH(L>0,I<0)、HL(L<0,I<0)、HH(L>0,I>0)；w_ij=(λ·exp(-βd_ij)+(1-λ)·d_ij^(-α))·I(d_ij≤d_max)，d_ij 在像元数较少时为路网最短路否则为大圆距离",
            },
            "color_thresholds": color_thresholds,
            "resolution": float(resolution),
            "search_radius_m": d_max,
            "travel_time_min": resolved_travel_time,
            "transport_mode": resolved_transport_mode,
            "transport_mode_label": (
                travel_mode_label_zh(resolved_transport_mode) if resolved_transport_mode else None
            ),
            "spatial_weight_mix_lambda": lam_w,
            "population_weighted": bool(population_weighted),
            "pop_weighted_metrics": pop_weighted_metrics,
            "gwr_meta": gwr_meta,
            "gdp_view_available": gdp_view_available,
            "population_view_available": population_view_available,
            "district_stats": district_stats,
            "accessibility_from_cache": accessibility_from_cache,
            "message": ("分析完成（使用缓存可达性栅格）" if accessibility_from_cache else "分析完成"),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "message": f"分析失败: {str(e)}"}
