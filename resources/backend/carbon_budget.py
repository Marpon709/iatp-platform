from CalculateDistance import CarbonBudgetCalculator

__all__ = [
    'CarbonBudgetCalculator',
    'map_poi_categories',
    'NANJING_DISTRICTS',
    'CARBON_ALPHA_G_PER_KM',
]

NANJING_DISTRICTS = [
    '全部区域',
    '玄武区',
    '秦淮区',
    '建邺区',
    '鼓楼区',
    '栖霞区',
    '雨花台区',
    '江宁区',
    '浦口区',
    '六合区',
    '溧水区',
    '高淳区'
]

# 各交通方式碳排因子 α（克 CO₂e / 人·公里），用于出行推荐等模块
CARBON_ALPHA_G_PER_KM = {
    'foot': 0.0,
    'walk': 0.0,
    'bike': 0.0,
    'ebike': 26.0,
    'bus': 88.0,
    'subway': 41.0,
    'drive': 182.0,
}

def map_poi_categories(existing_poi_data):
    import geopandas as gpd

    mapped_poi_data = {}

    print(f"\n开始映射POI类别...")
    print(f"原始POI数据键: {list(existing_poi_data.keys())}")

    def is_valid_poi(poi_data):
        if poi_data is None:
            return False
        if isinstance(poi_data, gpd.GeoDataFrame):
            return not poi_data.empty and len(poi_data) > 0
        return False

    if 'work' in existing_poi_data:
        poi = existing_poi_data['work']
        print(f"  work: 类型={type(poi)}, 是否为空={poi.empty if hasattr(poi, 'empty') else 'N/A'}, 长度={len(poi) if hasattr(poi, '__len__') else 'N/A'}")
        if is_valid_poi(poi):
            mapped_poi_data['work'] = poi
            print(f"  [OK] work: {len(poi)} 个POI")
        else:
            print(f"  [FAIL] work: 数据无效")
    else:
        print(f"  [FAIL] work: 未找到")

    if 'medical' in existing_poi_data and is_valid_poi(existing_poi_data['medical']):
        mapped_poi_data['medical'] = existing_poi_data['medical']
        print(f"  [OK] medical: {len(existing_poi_data['medical'])} 个POI")
    elif 'errand' in existing_poi_data:
        poi = existing_poi_data['errand']
        print(f"  errand: 类型={type(poi)}, 是否为空={poi.empty if hasattr(poi, 'empty') else 'N/A'}, 长度={len(poi) if hasattr(poi, '__len__') else 'N/A'}")
        if is_valid_poi(poi):
            mapped_poi_data['medical'] = poi
            print(f"  [OK] medical (使用errand): {len(poi)} 个POI")
        else:
            print(f"  [FAIL] medical: errand数据无效")
    else:
        print(f"  [FAIL] medical: 未找到")

    if 'shopping' in existing_poi_data:
        poi = existing_poi_data['shopping']
        print(f"  shopping: 类型={type(poi)}, 是否为空={poi.empty if hasattr(poi, 'empty') else 'N/A'}, 长度={len(poi) if hasattr(poi, '__len__') else 'N/A'}")
        if is_valid_poi(poi):
            mapped_poi_data['shopping'] = poi
            print(f"  [OK] shopping: {len(poi)} 个POI")
        else:
            print(f"  [FAIL] shopping: 数据无效")
    else:
        print(f"  [FAIL] shopping: 未找到")

    if 'catering' in existing_poi_data:
        poi = existing_poi_data['catering']
        print(f"  catering: 类型={type(poi)}, 是否为空={poi.empty if hasattr(poi, 'empty') else 'N/A'}, 长度={len(poi) if hasattr(poi, '__len__') else 'N/A'}")
        if is_valid_poi(poi):
            mapped_poi_data['catering'] = poi
            print(f"  [OK] catering: {len(poi)} 个POI")
        else:
            print(f"  [FAIL] catering: 数据无效")
    else:
        print(f"  [FAIL] catering: 未找到")

    if 'transportation_station' in existing_poi_data:
        poi = existing_poi_data['transportation_station']
        print(f"  transportation_station: 类型={type(poi)}, 是否为空={poi.empty if hasattr(poi, 'empty') else 'N/A'}, 长度={len(poi) if hasattr(poi, '__len__') else 'N/A'}")
        if is_valid_poi(poi):
            mapped_poi_data['transportation_station'] = poi
            print(f"  [OK] transportation_station: {len(poi)} 个POI")
        else:
            print(f"  [FAIL] transportation_station: 数据无效")
    else:
        print(f"  [FAIL] transportation_station: 未找到")

    if 'education' in existing_poi_data:
        poi = existing_poi_data['education']
        print(f"  education: 类型={type(poi)}, 是否为空={poi.empty if hasattr(poi, 'empty') else 'N/A'}, 长度={len(poi) if hasattr(poi, '__len__') else 'N/A'}")
        if is_valid_poi(poi):
            mapped_poi_data['education'] = poi
            print(f"  [OK] education: {len(poi)} 个POI")
        else:
            print(f"  [FAIL] education: 数据无效")
    else:
        print(f"  [FAIL] education: 未找到")

    if 'recreation' in existing_poi_data:
        poi = existing_poi_data['recreation']
        print(f"  recreation: 类型={type(poi)}, 是否为空={poi.empty if hasattr(poi, 'empty') else 'N/A'}, 长度={len(poi) if hasattr(poi, '__len__') else 'N/A'}")
        if is_valid_poi(poi):
            mapped_poi_data['recreation'] = poi
            print(f"  [OK] recreation: {len(poi)} 个POI")
        else:
            print(f"  [FAIL] recreation: 数据无效")
    elif 'leisure' in existing_poi_data:
        poi = existing_poi_data['leisure']
        if is_valid_poi(poi):
            mapped_poi_data['recreation'] = poi
            print(f"  [OK] recreation (使用leisure): {len(poi)} 个POI")
    elif 'entertainment' in existing_poi_data:
        poi = existing_poi_data['entertainment']
        if is_valid_poi(poi):
            mapped_poi_data['recreation'] = poi
            print(f"  [OK] recreation (使用entertainment): {len(poi)} 个POI")
    elif 'shopping' in existing_poi_data and 'shopping' not in mapped_poi_data:
        poi = existing_poi_data['shopping']
        if is_valid_poi(poi):
            mapped_poi_data['recreation'] = poi
            print(f"  [OK] recreation (使用shopping): {len(poi)} 个POI")
    else:
        print(f"  [WARN] recreation: 未找到，使用空数据")

    print(f"\nPOI类别映射完成: {list(mapped_poi_data.keys())}")
    print(f"映射后POI总数: {sum(len(v) for v in mapped_poi_data.values())}")
    return mapped_poi_data
