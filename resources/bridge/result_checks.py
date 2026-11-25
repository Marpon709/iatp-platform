"""检查已保存结果的基本一致性，不修改数值，也不判断模型是否科学正确。"""
import math


def nonnegative(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def valid_coordinates(value):
    """递归检查 GeoJSON 坐标，叶节点必须是经度、纬度。"""
    if not isinstance(value, list) or not value:
        return False
    if isinstance(value[0], (int, float)):
        return (len(value) >= 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                and math.isfinite(x) for x in value[:2])
                and -180 <= value[0] <= 180 and -90 <= value[1] <= 90)
    return all(valid_coordinates(item) for item in value)


def validate_result(kind, result):
    checks = []

    def add(code, label, passed, message):
        checks.append({'code': code, 'label': label, 'passed': bool(passed), 'message': message})

    if not isinstance(result, dict) or not result.get('success'):
        return {'status': 'skipped', 'checks': [], 'message': '计算未成功，暂不执行结果一致性校验。'}
    if kind == 'route-chain':
        legs = result.get('legs') or []
        for field, total, label in [('carbonG', 'totalCarbonG', '分段碳排与总量'),
                                    ('timeMin', 'totalTravelMin', '分段时间与总量')]:
            values = [leg.get(field) for leg in legs if isinstance(leg, dict)]
            valid = bool(legs) and len(values) == len(legs) and all(nonnegative(v) for v in values) and nonnegative(result.get(total))
            consistent = valid and abs(sum(values) - result[total]) <= .15
            add(total, label, consistent, '各段数值须有限且非负，分段之和与总量相差不超过 0.15（显示舍入容差）。')
        add('coordinates', '路线经纬度范围', bool(legs) and all(
            isinstance(leg, dict) and valid_coordinates(leg.get('coordinates')) for leg in legs),
            '经度应在 -180 至 180、纬度应在 -90 至 90 之间；此检查可发现常见经纬度颠倒。')
    elif kind == 'activity-space':
        areas = result.get('areas') or []
        add('areas', '活动空间几何坐标', (not result.get('hasActivitySpace') and not areas) or
            bool(areas) and all(isinstance(area, dict) and valid_coordinates(area.get('coordinates')) for area in areas),
            '有活动空间时应包含有效经纬度坐标；无可活动空档时允许范围为空。')
        categories = result.get('categories') or []
        add('categories', '设施计数与得分', bool(categories) and all(isinstance(c, dict) and
            nonnegative(c.get('score')) and isinstance(c.get('count'), int) and not isinstance(c.get('count'), bool)
            and c['count'] >= 0 for c in categories), '分类设施数应为非负整数，得分应为有限非负数。')
    else:
        return {'status': 'skipped', 'checks': [], 'message': '本类型暂未配置自动一致性规则。'}
    return {'status': 'passed' if all(c['passed'] for c in checks) else 'failed', 'checks': checks,
            'message': '基础一致性校验不替代真实路线、数据来源与模型方法的验证。'}
