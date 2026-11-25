import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import threading
import glob

_POI_CACHE = {}
_POI_CACHE_LOCK = threading.Lock()

_RESIDENTIAL_CACHE = {}
_RESIDENTIAL_CACHE_LOCK = threading.Lock()

_POI_SUBCATEGORY_CACHE = {}

# POI 目录缓存结构版本（shopping/catering 拆分后需失效旧缓存）
_POI_LOAD_CACHE_VERSION = 4

_POI_CATEGORY_COL_CANDIDATES = ('类别', 'category', 'Category', 'type', 'Type')

def _find_poi_category_column(gdf):
    """兼容城市/上海 GeoJSON 的类别字段命名。"""
    for field_name in _POI_CATEGORY_COL_CANDIDATES:
        if field_name in gdf.columns:
            return field_name
    return None

def get_poi_subcategories(poi_dir='Nanjin2018', city='南京', use_cache=True):
    """从POI文件中提取每个大类下的子类别列表，按中类统计"""
    poi_dir = os.path.abspath(poi_dir)
    cache_key = (poi_dir, city, _POI_LOAD_CACHE_VERSION)
    if use_cache:
        if cache_key in _POI_SUBCATEGORY_CACHE:
            return _POI_SUBCATEGORY_CACHE[cache_key]

    subcategory_map = {
        'education': {'label': '教育', 'subcategories': {}},
        'work': {'label': '工作', 'subcategories': {}},
        'medical': {'label': '医疗', 'subcategories': {}},
        'shopping': {'label': '购物', 'subcategories': {}},
        'catering': {'label': '餐饮', 'subcategories': {}},
        'transportation_station': {'label': '交通站点', 'subcategories': {}},
        'recreation': {'label': '休闲娱乐', 'subcategories': {}}
    }

    poi_file_mapping = {
        'education': [
            '南京市_education.geojson', 'nanjing_education_poi.geojson',
            '上海市-education.geojson', 'shanghai_education.geojson',
            '北京市-education.geojson', '北京市_education.geojson',
            '武汉市-education.geojson', '武汉市_education.geojson'
        ],
        'work': [
            '南京市_work_细分.geojson', '南京市_work.geojson', 'nanjing_work_poi.geojson',
            '上海市-work.geojson', 'shanghai_work.geojson',
            '北京市-work.geojson', '北京市_work.geojson',
            '武汉市-work.geojson', '武汉市_work.geojson'
        ],
        'medical': [
            '南京市_medical.geojson',
            '上海市-medical.geojson', 'shanghai_medical.geojson',
            '北京市-medical.geojson', '北京市_medical.geojson',
            '武汉市-medical.geojson', '武汉市_medical.geojson'
        ],
        'shopping': [
            '上海市-shopping.geojson', '北京市-shopping.geojson', '南京市_shopping.geojson',
            'nanjing_shopping_poi.geojson', '*shopping*', 'shanghai-shopping.geojson',
            '南京市_nutrition.geojson',
            '上海市-nutrition.geojson', 'shanghai_nutrition.geojson', '*nutrition*', '*购物*',
            '北京市-nutrition.geojson', '北京市_nutrition.geojson', 'beijing_nutrition.geojson',
            '武汉市-nutrition.geojson', '武汉市_nutrition.geojson', 'wuhan_nutrition.geojson'
        ],
        'catering': [
            '上海市-catering.geojson', '北京市-catering.geojson', '南京市_catering.geojson',
            '*catering*', 'shanghai-catering.geojson'
        ],
        'transportation_station': [
            '南京市_transport_station.geojson',
            '上海市-transport.geojson', 'shanghai_transport.geojson',
            '北京市-transport_station.geojson', '北京市_transport_station.geojson',
            '武汉市-transport_station.geojson', '武汉市_transport_station.geojson'
        ],
        'recreation': [
            '南京市_recreation.geojson', 'nanjing_leisure_poi.geojson',
            '上海市-recreation.geojson', 'shanghai_recreation.geojson',
            '北京市-recreation.geojson', '北京市_recreation.geojson',
            '武汉市-recreation.geojson', '武汉市_recreation.geojson'
        ]
    }

    def parse_category_field(val):
        if not isinstance(val, str):
            return None, None, None
        parts = [p.strip() for p in val.split(';')]
        major = parts[0] if len(parts) > 0 else None
        middle = parts[1] if len(parts) > 1 else None
        minor = parts[2] if len(parts) > 2 else None
        return major, middle, minor

    log_tag = '上海子类别' if city == '上海' else '子类别'

    for category, filenames in poi_file_mapping.items():
        file_path = None
        for fn in filenames:
            if '*' in fn:
                continue
            candidate = os.path.join(poi_dir, fn)
            if os.path.isfile(candidate):
                file_path = candidate
                break
        if file_path is None:
            for fn in filenames:
                if '*' not in fn:
                    continue
                pattern = os.path.join(poi_dir, fn)
                matches = sorted(glob.glob(pattern))
                if matches:
                    file_path = matches[0]
                    break
        if file_path is None:
            all_geojson = glob.glob(os.path.join(poi_dir, '*.geojson'))
            for f in all_geojson:
                fname = os.path.basename(f).lower()
                if category in fname or f'{category}_' in fname:
                    file_path = f
                    break
        if file_path is None:
            print(f"  [{log_tag}] {category}: 文件未找到")
            continue

        try:
            gdf = gpd.read_file(file_path)
            print(f"  [{log_tag}] {category}: 读取 {os.path.basename(file_path)}, 共 {len(gdf)} 条")
            if gdf.empty:
                continue
            geom_col = gdf.geometry
            if geom_col is None or len(geom_col) == 0:
                continue
            geom_types = geom_col.geom_type.unique()
            if any(gt in ('Polygon', 'MultiPolygon') for gt in geom_types):
                gdf['geometry'] = gdf.geometry.centroid

            col = _find_poi_category_column(gdf)
            if col is None:
                print(f"  [{log_tag}] {category}: 无类别字段，可用: {list(gdf.columns)}")
                continue

            if category == 'work':
                middle_counter = {}
                for val in gdf[col].dropna():
                    _, middle, _ = parse_category_field(str(val))
                    if middle:
                        middle_counter[middle] = middle_counter.get(middle, 0) + 1
                subcats = {
                    k: v for k, v in sorted(middle_counter.items(), key=lambda x: -x[1]) if v >= 1000
                }
                subcategory_map['work']['subcategories'] = subcats
                print(f"  [{log_tag}] work: {len(subcats)} 个中类（阈值≥1000）")
                for k, v in list(subcats.items())[:10]:
                    print(f'    {k}: {v}个')
                continue

            middle_counter = {}
            for val in gdf[col].dropna():
                _, middle, _ = parse_category_field(str(val))
                if middle:
                    middle_counter[middle] = middle_counter.get(middle, 0) + 1

            subcategory_map[category]['subcategories'] = {
                k: v for k, v in sorted(middle_counter.items(), key=lambda x: -x[1]) if v >= 1000
            }
            print(
                f"  [{log_tag}] {category}: "
                f"{len(subcategory_map[category]['subcategories'])} 个中类（阈值≥1000）"
            )

        except Exception as e:
            print(f"  警告: {category} 提取子类别失败: {e}")
            import traceback
            traceback.print_exc()

    result = subcategory_map
    # 兼容原 LocationAccessibility 上海 API 的 transport 键名
    if city == '上海':
        ts = result.get('transportation_station')
        if ts and not result.get('transport', {}).get('subcategories'):
            result['transport'] = ts
    _POI_SUBCATEGORY_CACHE[cache_key] = result
    return result

def get_shanghai_poi_subcategories(poi_dir=None, use_cache=True):
    """上海 POI 子类别（统一走 get_poi_subcategories，保留兼容导入）。"""
    if poi_dir is None:
        base = os.path.dirname(os.path.abspath(__file__))
        poi_dir = os.path.join(base, 'POI', 'Shanghai')
    return get_poi_subcategories(poi_dir=poi_dir, city='上海', use_cache=use_cache)

def filter_poi_by_subcategory(poi_gdf, subcategory_keyword, poi_category='work'):
    """根据子类别关键词过滤POI GeoDataFrame，按中类过滤"""
    if poi_gdf is None or poi_gdf.empty:
        return poi_gdf
    col = '类别' if '类别' in poi_gdf.columns else (
        'category' if 'category' in poi_gdf.columns else None
    )
    if col is None:
        return poi_gdf

    def match_work(val):
        if not isinstance(val, str):
            return False
        parts = [p.strip() for p in val.split(';')]
        if len(parts) < 2:
            return False
        return subcategory_keyword in parts[1]

    mask = poi_gdf[col].astype(str).apply(match_work)
    return poi_gdf[mask].copy()

def load_poi_from_geojson(poi_dir='Nanjin2018', use_cache=True):
    poi_dir = os.path.abspath(poi_dir)
    cache_key = (poi_dir, _POI_LOAD_CACHE_VERSION)
    if use_cache:
        with _POI_CACHE_LOCK:
            if cache_key in _POI_CACHE:
                print(f"使用已缓存的POI数据（路径: {poi_dir}）")
                return _POI_CACHE[cache_key]

    print(f"正在从 {poi_dir} 文件夹读取POI数据...")

    if not os.path.exists(poi_dir):
        print(f"  错误: POI目录不存在: {poi_dir}")
        return {cat: gpd.GeoDataFrame() for cat in [
            'work', 'education', 'medical', 'shopping', 'catering', 'recreation', 'transportation_station'
        ]}
    
    # 列出目录中的所有 geojson 文件用于调试
    all_files = os.listdir(poi_dir)
    geojson_files = [f for f in all_files if f.endswith('.geojson')]
    print(f"  目录中的GeoJSON文件: {geojson_files}")

    # 支持的文件名映射（支持中文和英文文件名）
    poi_file_candidates_mapping = {
        'work': [
            '南京市_work_细分.geojson', '南京市_work.geojson', 'nanjing_work_poi.geojson',
            '上海市-work.geojson', 'shanghai_work.geojson', '*work*',
            '北京市-work.geojson', '北京市_work.geojson', 'beijing_work.geojson',
            '武汉市-work.geojson', '武汉市_work.geojson', 'wuhan_work.geojson'
        ],
        'education': [
            '南京市_education.geojson', 'nanjing_education_poi.geojson', '*education*',
            '上海市-education.geojson', 'shanghai_education.geojson', '*学校*',
            '北京市-education.geojson', '北京市_education.geojson', 'beijing_education.geojson',
            '武汉市-education.geojson', '武汉市_education.geojson', 'wuhan_education.geojson'
        ],
        'shopping': [
            '上海市-shopping.geojson', '北京市-shopping.geojson', '南京市_shopping.geojson',
            'nanjing_shopping_poi.geojson', '*shopping*',
            'shanghai-shopping.geojson',
            '南京市_nutrition.geojson',
            '上海市-nutrition.geojson', 'shanghai_nutrition.geojson', '*nutrition*', '*购物*',
            '北京市-nutrition.geojson', '北京市_nutrition.geojson', 'beijing_nutrition.geojson',
            '武汉市-nutrition.geojson', '武汉市_nutrition.geojson', 'wuhan_nutrition.geojson'
        ],
        'catering': [
            '上海市-catering.geojson', '北京市-catering.geojson', '南京市_catering.geojson',
            '*catering*', 'shanghai-catering.geojson'
        ],
        'medical': [
            '南京市_medical.geojson', 'nanjing_errand_poi.geojson', '*medical*',
            '上海市-medical.geojson', 'shanghai_medical.geojson', '*医疗*', '*医院*',
            '北京市-medical.geojson', '北京市_medical.geojson', 'beijing_medical.geojson',
            '武汉市-medical.geojson', '武汉市_medical.geojson', 'wuhan_medical.geojson'
        ],
        'recreation': [
            '南京市_recreation.geojson', '*recreation*', '*leisure*',
            '上海市-recreation.geojson', 'shanghai_recreation.geojson', '*休闲*',
            '北京市-recreation.geojson', '北京市_recreation.geojson', 'beijing_recreation.geojson',
            '武汉市-recreation.geojson', '武汉市_recreation.geojson', 'wuhan_recreation.geojson'
        ],
        'transportation_station': [
            '南京市_transport_station.geojson', '*transport*',
            '上海市-transport.geojson', 'shanghai_transport.geojson', '*交通*',
            '北京市-transport_station.geojson', '北京市_transport_station.geojson', 'beijing_transport.geojson',
            '武汉市-transport_station.geojson', '武汉市_transport_station.geojson', 'wuhan_transport.geojson'
        ]
    }

    poi_data = {}

    for category, candidate_names in poi_file_candidates_mapping.items():
        file_path = None
        for n in candidate_names:
            if '*' in n:
                continue
            candidate = os.path.join(poi_dir, n)
            if os.path.exists(candidate):
                file_path = candidate
                break
        
        # 精确匹配失败，使用通配符
        if file_path is None:
            for n in candidate_names:
                if '*' not in n:
                    continue
                pattern = os.path.join(poi_dir, n.replace('*', '*'))
                matches = sorted(glob.glob(pattern))
                if matches:
                    file_path = matches[0]
                    break
        
        if file_path is None:
            # 最后尝试：直接列出目录中所有geojson文件，模糊匹配类别
            all_geojson = glob.glob(os.path.join(poi_dir, '*.geojson'))
            for f in all_geojson:
                fname = os.path.basename(f).lower()
                if category in fname or f'{category}_' in fname:
                    file_path = f
                    break
        
        if file_path and os.path.exists(file_path):
            filename = os.path.basename(file_path)
            try:
                gdf = gpd.read_file(file_path)
                # 如果geometry是多边形，转换为点（取中心点）
                geom_types = gdf.geometry.geom_type  # 始终返回Series，兼容单类型和多类型
                if geom_types.isin(['Polygon', 'MultiPolygon']).any():
                    gdf['geometry'] = gdf.geometry.centroid
                poi_data[category] = gdf
                print(f"  {category}: {len(gdf)} 个POI (从 {filename})")
            except Exception as e:
                print(f"  错误: 读取 {file_path} 失败: {e}")
                poi_data[category] = gpd.GeoDataFrame()
        else:
            print(f"  警告: 未找到 {category} 类别的POI文件")
            poi_data[category] = gpd.GeoDataFrame()

    print("\n=== POI分类统计 ===")
    for category, gdf_cat in poi_data.items():
        print(f"  {category}: {len(gdf_cat)} 个POI")

    if use_cache:
        with _POI_CACHE_LOCK:
            _POI_CACHE[cache_key] = poi_data
    return poi_data

def load_poi_from_excel(excel_path='POI/POI.xlsx'):
    print(f"正在从 {excel_path} 读取POI数据...")

    df = pd.read_excel(excel_path)

    required_columns = ['经度', '纬度']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"POI Excel文件缺少必需的列: {missing_columns}")

    geometry = [Point(lon, lat) for lon, lat in zip(df['经度'], df['纬度'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

    poi_data = {
        'work': gpd.GeoDataFrame(),
        'medical': gpd.GeoDataFrame(),
        'shopping': gpd.GeoDataFrame(),
        'catering': gpd.GeoDataFrame(),
        'education': gpd.GeoDataFrame()
    }

    category_by_major = {
        'education': ['科教文化'],
        'shopping': ['购物消费'],
        'catering': ['餐饮美食'],
    }

    category_by_minor_keywords = {
        'education': ['学校', '教育', '培训', '大学', '中学', '小学', '幼儿园', '学院', '研究院'],
        'medical': ['医院', '诊所', '医疗', '药店', '卫生', '卫生院', '社区卫生'],
        'shopping': ['市场', '超市', '商店', '便利店', '菜市场'],
        'catering': ['餐饮', '餐厅', '饭店', '小吃', '快餐'],
        'work': ['公司', '企业', '办公', '写字楼', '银行', '金融', '商务', '大厦']
    }

    if '大类' in df.columns:
        print("根据大类分类POI...")
        for category, keywords in category_by_major.items():
            if keywords:
                mask = gdf['大类'].isin(keywords)
                poi_data[category] = gdf[mask].copy()
                print(f"  {category} (大类): {len(poi_data[category])} 个POI")

    if '中类' in df.columns:
        print("根据中类补充分类POI...")

        classified_indices = set()
        for gdf_cat in poi_data.values():
            if not gdf_cat.empty:
                classified_indices.update(gdf_cat.index)

        unclassified_mask = ~gdf.index.isin(classified_indices)
        unclassified_gdf = gdf[unclassified_mask].copy()

        if not unclassified_gdf.empty:
            for category, keywords in category_by_minor_keywords.items():
                if keywords:
                    mask = unclassified_gdf['中类'].str.contains('|'.join(keywords), na=False, case=False)
                    if mask.any():
                        new_pois = unclassified_gdf[mask].copy()
                        if poi_data[category].empty:
                            poi_data[category] = new_pois
                        else:
                            poi_data[category] = gpd.GeoDataFrame(
                                pd.concat([poi_data[category], new_pois], ignore_index=True),
                                crs='EPSG:4326'
                            )
                        print(f"  {category} (中类补充): {len(new_pois)} 个POI")

    print("\n=== POI分类统计 ===")
    for category, gdf_cat in poi_data.items():
        print(f"  {category}: {len(gdf_cat)} 个POI")

    return poi_data

def load_residential_locations_from_excel(excel_path='POI/Residential_location_2.xlsx', use_cache=True):
    excel_path = os.path.abspath(excel_path)
    if use_cache:
        with _RESIDENTIAL_CACHE_LOCK:
            if excel_path in _RESIDENTIAL_CACHE:
                print(f"使用已缓存的居住地点数据（路径: {excel_path}）")
                return _RESIDENTIAL_CACHE[excel_path]

    print(f"正在从 {excel_path} 读取居住地点数据...")

    df = pd.read_excel(excel_path)

    required_columns = ['lon_o', 'lat_o']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"居住地点Excel文件缺少必需的列: {missing_columns}")

    locations = []
    for idx, row in df.iterrows():
        try:
            lon = float(row['lon_o'])
            lat = float(row['lat_o'])
            # 增加坐标有效性检查
            if pd.notna(lon) and pd.notna(lat) and -180 <= lon <= 180 and -90 <= lat <= 90:
                locations.append(Point(lon, lat))
            else:
                print(f"警告: 跳过无效的坐标行 {idx}: 经度={lon}, 纬度={lat}")
        except (ValueError, TypeError) as e:
            print(f"警告: 跳过无效的坐标行 {idx}: {e}")
            continue

    if len(locations) == 0:
        raise ValueError(f"未能从 {excel_path} 中加载任何有效的居住地点数据")

    print(f"  成功加载 {len(locations)} 个居住地点")
    if use_cache:
        with _RESIDENTIAL_CACHE_LOCK:
            _RESIDENTIAL_CACHE[excel_path] = locations
    return locations

if __name__ == '__main__':
    import os

    backend_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(backend_dir)

    print("测试POI数据加载...")
    try:
        poi_data = load_poi_from_excel('POI/POI.xlsx')
        print(f"\nPOI数据加载成功，共{len(poi_data)}类")
    except Exception as e:
        print(f"POI数据加载失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("测试居住地点数据加载...")
    try:
        locations = load_residential_locations_from_excel('POI/Residential_location_2.xlsx')
        print(f"居住地点数据加载成功，共{len(locations)}个地点")
    except Exception as e:
        print(f"居住地点数据加载失败: {e}")
        import traceback
        traceback.print_exc()
