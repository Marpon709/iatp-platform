import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import osmnx as ox  # 第三方开源库，用于城市路网下载、GraphML 读写与最短路径计算
ox.settings.timeout = 30
ox.settings.max_retries = 2
import warnings
import matplotlib.pyplot as plt
from rasterio.transform import from_origin
import rasterio
import os
import threading
from numba import njit

import matplotlib.font_manager as fm

warnings.filterwarnings('ignore')

# G2SFCA 可达性结果统一放大系数（显示/统计用；对 ICB 反比分配无净影响）
ACCESSIBILITY_RESULT_SCALE = 1000.0
# 正值过小的可达性抬升阈值：放大后若 0 < A ≤ 该值，统一取该值
ACCESSIBILITY_MIN_POSITIVE = 0.1

class CarbonBudgetParams:
    """碳预算分析模块的参数配置类"""

    def __init__(self):
        self.travel_time_threshold = 15.0      # 分钟
        self.distance_decay_beta = 1.5         # β系数

        self.poi_weights = {
            'work': 100.0 / 7,
            'medical': 100.0 / 7,
            'shopping': 100.0 / 7,
            'catering': 100.0 / 7,
            'education': 100.0 / 7,
            'recreation': 100.0 / 7,
            'transport_station': 100.0 / 7,
        }

        # 交通方式速度 km/h（与 transport_modes 全项目统一）
        from transport_modes import carbon_budget_mode_speeds
        self.mode_speeds = carbon_budget_mode_speeds()

        self.min_living_time_budget = 60.0    # 分钟
        self.essential_facilities = {
            'work': True,
            'medical': True,
            'shopping': True,
            'catering': True,
            'education': True,
            'recreation': True,
            'transport_station': True,
        }
        self.fallback_mode = 'bus'  # 兜底交通方式

    def normalize_poi_weights(self):
        """归一化POI权重，确保总和为100%"""
        total = sum(self.poi_weights.values())
        if total > 0:
            for k in self.poi_weights:
                self.poi_weights[k] = self.poi_weights[k] / total * 100.0

    def to_dict(self):
        return {
            'travel_time_threshold': self.travel_time_threshold,
            'distance_decay_beta': self.distance_decay_beta,
            'poi_weights': self.poi_weights.copy(),
            'mode_speeds': self.mode_speeds.copy(),
            'min_living_time_budget': self.min_living_time_budget,
            'essential_facilities': self.essential_facilities.copy(),
            'fallback_mode': self.fallback_mode,
        }

    @classmethod
    def from_dict(cls, d):
        params = cls()
        params.travel_time_threshold = d.get('travel_time_threshold', 15.0)
        params.distance_decay_beta = d.get('distance_decay_beta', 1.5)
        params.poi_weights = d.get('poi_weights', params.poi_weights.copy())
        params.mode_speeds = d.get('mode_speeds', params.mode_speeds.copy())
        params.min_living_time_budget = d.get('min_living_time_budget', 60.0)
        params.essential_facilities = d.get('essential_facilities', params.essential_facilities.copy())
        params.fallback_mode = d.get('fallback_mode', 'bus')
        return params

try:
    from PyQt5.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
        QSlider, QLabel, QDoubleSpinBox, QGroupBox, QFormLayout,
        QPushButton, QCheckBox, QComboBox, QTextEdit,
        QScrollArea, QGridLayout, QLineEdit, QRadioButton,
        QButtonGroup, QFrame, QDialogButtonBox
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5 import QtGui
    _PYQT5_AVAILABLE = True
except ImportError:
    _PYQT5_AVAILABLE = False

class AccessibilitySettingsDialog(QDialog if _PYQT5_AVAILABLE else object):
    """
    碳预算分析 - 参数设置对话框
    包含两个子模块：
      1. 可达性计算参数（含四个子模块）
      2. 最低生活标准（含两个子模块）
    """

    def __init__(self, params=None, parent=None):
        if not _PYQT5_AVAILABLE:
            raise RuntimeError("PyQt5 未安装，无法创建设置对话框")
        super().__init__(parent)
        self.params = params if params is not None else CarbonBudgetParams()
        self.setWindowTitle("碳预算分析 - 参数设置")
        self.setMinimumSize(700, 580)
        self._init_ui()

    def _make_slider(self, min_val, max_val, step, value, decimals=0):
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(int(min_val / step))
        slider.setMaximum(int(max_val / step))
        slider.setSingleStep(1)
        slider.setValue(int(value / step))
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(int((max_val - min_val) / step / 5) or 1)
        label = QLabel(f"{value:.{decimals}f}" if decimals else str(int(value)))
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        label.setMinimumWidth(50)
        slider.valueChanged.connect(
            lambda v: label.setText(f"{v * step:.{decimals}f}" if decimals else str(int(v * step)))
        )
        return slider, label

    def _sync_slider_label(self, slider, label, step, decimals=0):
        val = slider.value() * step
        label.setText(f"{val:.{decimals}f}" if decimals else str(int(val)))
        return val

    def _make_percent_slider(self, value):
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(100)
        slider.setValue(int(value))
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(10)
        lbl = QLabel(str(int(value)))
        lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lbl.setMinimumWidth(40)
        slider.valueChanged.connect(lambda v: lbl.setText(str(v)))
        return slider, lbl

    def _labeled_slider_row(self, label_text, min_v, max_v, step, default, decimals=0):
        lbl = QLabel(label_text)
        lbl.setWordWrap(True)
        slider, val_lbl = self._make_slider(min_v, max_v, step, default, decimals)
        return lbl, slider, val_lbl

    def _build_tab1(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)

        grp1 = QGroupBox("① 出行时间阈值")
        g1 = QFormLayout()
        t0_slider, t0_lbl = self._labeled_slider_row("时间阈值（分钟）", 5, 60, 1, self.params.travel_time_threshold)
        self._t0_slider = t0_slider
        self._t0_lbl = t0_lbl
        g1.addRow(t0_slider, t0_lbl)
        grp1.setLayout(g1)

        grp2 = QGroupBox("② 距离衰减系数 (β)")
        g2 = QFormLayout()
        beta_slider, beta_lbl = self._labeled_slider_row("β 系数", 1.0, 2.0, 0.01, self.params.distance_decay_beta, decimals=2)
        self._beta_slider = beta_slider
        self._beta_lbl = beta_lbl
        hint2 = QLabel("反映出行者对距离的敏感程度：β 越大，远距离出行的权重衰减越快")
        hint2.setStyleSheet("color: #555; font-size: 11px;")
        g2.addRow(beta_slider, beta_lbl)
        grp2.layout().addRow(hint2)
        grp2.setLayout(g2)

        grp3 = QGroupBox("③ POI权重配置（%）：")
        g3 = QGridLayout()
        poi_items = ['work', 'medical', 'shopping', 'catering', 'education', 'recreation', 'transport_station']
        poi_labels_cn = [
            '工作 Work', '医疗 Medical', '购物 Shopping', '餐饮 Catering',
            '教育 Education', '休闲 Recreation', '交通站点 Transport'
        ]
        self._poi_sliders = {}
        self._poi_labels = {}
        total = sum(self.params.poi_weights.get(k, 16.67) for k in poi_items)

        for row, (key, cn) in enumerate(zip(poi_items, poi_labels_cn)):
            lbl = QLabel(cn)
            sl, vl = self._make_percent_slider(self.params.poi_weights.get(key, 100.0 / len(poi_items)))
            self._poi_sliders[key] = sl
            self._poi_labels[key] = vl
            g3.addWidget(lbl, row, 0)
            g3.addWidget(sl, row, 1)
            g3.addWidget(vl, row, 2)

        self._poi_norm_lbl = QLabel(f"当前总和: {total:.1f}%  （自动归一化后生效）")
        self._poi_norm_lbl.setStyleSheet("color: #888; font-size: 11px;")
        for sl in self._poi_sliders.values():
            sl.valueChanged.connect(self._update_poi_sum)
        g3.addWidget(self._poi_norm_lbl, len(poi_items), 0, 1, 3)
        grp3.setLayout(g3)

        grp4 = QGroupBox("④ 交通方式速度设定（km/h）")
        g4 = QFormLayout()
        self._speed_boxes = {}
        mode_items = [
            ('drive', 'Drive（开车）'),
            ('bus', 'Bus（公交）'),
            ('bike', 'Bike（自行车）'),
            ('ebike', 'E-bike（电动车）'),
            ('walk', 'Walk（步行）'),
            ('metro', 'Metro（地铁）'),
        ]
        for key, label in mode_items:
            sp = QDoubleSpinBox()
            sp.setRange(0.5, 200.0)
            sp.setSingleStep(1.0)
            sp.setValue(self.params.mode_speeds.get(key, 20.0))
            sp.setSuffix(" km/h")
            g4.addRow(label, sp)
            self._speed_boxes[key] = sp
        grp4.setLayout(g4)

        lay.addWidget(grp1)
        lay.addWidget(grp2)
        lay.addWidget(grp3)
        lay.addWidget(grp4)
        return tab

    def _update_poi_sum(self):
        total = sum(sl.value() for sl in self._poi_sliders.values())
        self._poi_norm_lbl.setText(f"当前总和: {total:.1f}%  （自动归一化后生效）")

    def _build_tab3(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)

        grp1 = QGroupBox("① 基本生活时间预算")
        g1 = QFormLayout()
        self._time_budget_spin = QDoubleSpinBox()
        self._time_budget_spin.setRange(0, 480)
        self._time_budget_spin.setSingleStep(5)
        self._time_budget_spin.setValue(self.params.min_living_time_budget)
        self._time_budget_spin.setSuffix(" 分钟")
        g1.addRow("每日最低出行时间预算:", self._time_budget_spin)
        hint1 = QLabel("满足居民日常生活基本出行需求的最低时间上限（分钟）")
        hint1.setStyleSheet("color: #555; font-size: 11px;")
        g1.addRow(hint1)
        grp1.setLayout(g1)

        grp2 = QGroupBox("② 必需设施清单（七类POI）")
        g2 = QVBoxLayout()
        self._facility_checks = {}
        fac_items = [
            ('work', '工作 Work'),
            ('medical', '医疗 Medical'),
            ('shopping', '购物 Shopping'),
            ('catering', '餐饮 Catering'),
            ('education', '教育 Education'),
            ('recreation', '休闲 Recreation'),
            ('transport_station', '交通站点 Transport'),
        ]
        for key, label in fac_items:
            cb = QCheckBox(label)
            cb.setChecked(self.params.essential_facilities.get(key, True))
            g2.addWidget(cb)
            self._facility_checks[key] = cb
        grp2.setLayout(g2)

        grp3 = QGroupBox("③ 兜底交通方式")
        g3 = QFormLayout()
        self._fallback_combo = QComboBox()
        self._fallback_combo.addItems(['drive', 'bus', 'metro', 'bike', 'ebike'])
        idx = self._fallback_combo.findText(self.params.fallback_mode)
        if idx >= 0:
            self._fallback_combo.setCurrentIndex(idx)
        g3.addRow("当无可用交通方式时使用:", self._fallback_combo)
        grp3.setLayout(g3)

        lay.addWidget(grp1)
        lay.addWidget(grp2)
        lay.addWidget(grp3)
        lay.addStretch()
        return tab

    def _init_ui(self):
        main_lay = QVBoxLayout(self)

        title = QLabel("碳预算分析 - 参数配置")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 6px;")
        main_lay.addWidget(title)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_tab1(), "① 可达性计算参数")
        self._tabs.addTab(self._build_tab3(), "② 最低生活标准")
        main_lay.addWidget(self._tabs)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        main_lay.addWidget(btns)

    def _on_accept(self):
        self._apply_params()
        self.accept()

    def _apply_params(self):
        self.params.travel_time_threshold = self._t0_slider.value() * 1.0
        self.params.distance_decay_beta = self._beta_slider.value() * 0.01
        for key, sl in self._poi_sliders.items():
            self.params.poi_weights[key] = float(sl.value())
        self.params.normalize_poi_weights()
        for key, sp in self._speed_boxes.items():
            self.params.mode_speeds[key] = sp.value()

        self.params.min_living_time_budget = self._time_budget_spin.value()
        for key, cb in self._facility_checks.items():
            self.params.essential_facilities[key] = cb.isChecked()
        self.params.fallback_mode = self._fallback_combo.currentText()

    def get_params(self):
        return self.params

#  度到米的转换系数（城市地区约32度纬度）
_DEGREE_TO_METER = 111000.0  # 1度纬度约等于111公里

def resolution_to_meters(resolution_degree):
    """将分辨率从度转换为米"""
    return int(round(resolution_degree * _DEGREE_TO_METER))

def resolution_to_degrees(resolution_meter):
    """将分辨率从米转换为度"""
    return resolution_meter / _DEGREE_TO_METER

def setup_chinese_font():
    chinese_fonts = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'Arial Unicode MS', 'Microsoft YaHei']

    available_fonts = [f.name for f in fm.fontManager.ttflist]

    font_found = None
    for font_name in chinese_fonts:
        if font_name in available_fonts:
            font_found = font_name
            break

    if font_found:
        plt.rcParams['font.family'] = font_found
        return True
    else:
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        return False

font_setup_success = setup_chinese_font()
if font_setup_success:
    print("中文字体设置成功，当前使用字体：", plt.rcParams['font.family'])
else:
    print("未找到中文字体，但已启用负号显示支持")

class NanjingAccessibilityCalculator:
    def __init__(self, networks, poi_data, admin_boundary, population_data=None, pop_transform=None,
                 carbon_params=None, city='南京', pop_raster_crs=None, nodes_with_population=None):
        """
        可达性计算器
        carbon_params: CarbonBudgetParams 实例，用于可达性计算参数
        city: 城市名称，用于确定缓存目录
        pop_raster_crs: 人口栅格的坐标系，用于坐标转换
        nodes_with_population: 矢量格式的人口数据（GeoDataFrame）
        """
        self.networks = networks
        self.poi_data = poi_data
        self.admin_boundary = admin_boundary
        self.carbon_params = carbon_params if carbon_params is not None else CarbonBudgetParams()
        self.city = city
        self.pop_raster_crs = pop_raster_crs
        self.nodes_with_population = nodes_with_population

        if self.admin_boundary.crs is None:
            self.admin_boundary = self.admin_boundary.set_crs('EPSG:4326', allow_override=True)
            print(f"警告: {city} 行政边界 CRS 未设置，已设置为 EPSG:4326")
        
        # 确保 admin_boundary 是 EPSG:4326
        if self.admin_boundary.crs.to_string() != 'EPSG:4326':
            self.admin_boundary = self.admin_boundary.to_crs(epsg=4326)
            print(f"警告: {city} 行政边界已从 {self.admin_boundary.crs} 转换为 EPSG:4326")

        self.bbox = admin_boundary.total_bounds
        print(f"{city}全市范围 (bbox): minX={self.bbox[0]:.4f}, minY={self.bbox[1]:.4f}, maxX={self.bbox[2]:.4f}, maxY={self.bbox[3]:.4f}")

        self.population_data = population_data
        self.pop_transform = pop_transform
        self.activities = ['work', 'education', 'medical', 'shopping', 'catering', 'transportation_station', 'recreation']
        # Step1 分母 Σ(P_i·G) 使用原始人口 P_i，不再取对数
        self.population_scale_mode = 'raw'

        # G2SFCA 距离阈值 t0（米）：出行时间 × 步行速度（与 transport_modes 一致）
        from transport_modes import speed_ms
        self.t0_distance_m = self.carbon_params.travel_time_threshold * 60.0 * speed_ms('walk')

        self.max_times = self._get_predefined_times('uniform.20')
        self.transport_mode = 'foot'
        self.travel_time_minutes = 20
        self.mode_weights = None
        self.custom_activity_times = None
        self._drive_spatial_index = None
        self._road_walk_cache = {}
        from transport_modes import load_transit_networks, TransitAccessibilityRouter

        try:
            self._transit_networks = load_transit_networks(city)
        except Exception as ex:
            self._transit_networks = {}
            print(f"[Transit] {city} 公共交通线网加载失败: {ex}")
        self._transit = TransitAccessibilityRouter(self._access_walk_distance_m)

        print(f"{city}可达性计算器初始化完成")

    def _get_predefined_times(self, time_preset='uniform.20'):
        """根据预设方案获取时间限制"""
        predefined_times = {
            'uniform.20': {
                'work': {'foot': 20, 'bike': 20, 'drive': 20, 'pt': 20, 'bnr': 20, 'pnr': 20},
                'education': {'foot': 20, 'bike': 20, 'drive': 20, 'pt': 20, 'bnr': 20, 'pnr': 20},
                'medical': {'foot': 20, 'bike': 20, 'drive': 20, 'pt': 20, 'bnr': 20, 'pnr': 20},
                'shopping': {'foot': 20, 'bike': 20, 'drive': 20, 'pt': 20, 'bnr': 20, 'pnr': 20},
                'catering': {'foot': 20, 'bike': 20, 'drive': 20, 'pt': 20, 'bnr': 20, 'pnr': 20},
                'transportation_station': {'foot': 20, 'bike': 20, 'drive': 20, 'pt': 20, 'bnr': 20, 'pnr': 20},
                'recreation': {'foot': 20, 'bike': 20, 'drive': 20, 'pt': 20, 'bnr': 20, 'pnr': 20}
            },
            'uniform.30': {
                'work': {'foot': 30, 'bike': 30, 'drive': 30, 'pt': 30, 'bnr': 30, 'pnr': 30},
                'education': {'foot': 30, 'bike': 30, 'drive': 30, 'pt': 30, 'bnr': 30, 'pnr': 30},
                'medical': {'foot': 30, 'bike': 30, 'drive': 30, 'pt': 30, 'bnr': 30, 'pnr': 30},
                'shopping': {'foot': 30, 'bike': 30, 'drive': 30, 'pt': 30, 'bnr': 30, 'pnr': 30},
                'catering': {'foot': 30, 'bike': 30, 'drive': 30, 'pt': 30, 'bnr': 30, 'pnr': 30},
                'transportation_station': {'foot': 30, 'bike': 30, 'drive': 30, 'pt': 30, 'bnr': 30, 'pnr': 30},
                'recreation': {'foot': 30, 'bike': 30, 'drive': 30, 'pt': 30, 'bnr': 30, 'pnr': 30}
            }
        }
        return predefined_times.get(time_preset, predefined_times['uniform.20'])

    def _build_custom_times(self, activity_minutes):
        """将每个活动的统一分钟数转为按方式的时间限制"""
        all_modes = ['foot', 'bike', 'drive', 'pt', 'bnr', 'pnr']
        
        custom_times = {}
        for activity in self.activities:
            minutes = float(activity_minutes.get(activity, 0)) if isinstance(activity_minutes, dict) else 0.0
            if minutes <= 0:
                # 若无有效输入，沿用当前配置
                custom_times[activity] = self.max_times.get(activity, {}).copy()
                continue
            per_mode_times = {}
            for mode in all_modes:
                per_mode_times[mode] = minutes
            custom_times[activity] = per_mode_times
        return custom_times

    def _normalize_mode_weights(self, mode_weights=None):
        """归一化交通方式权重；无配置时回退为当前单一 transport_mode。"""
        from transport_modes import normalize_mode_key

        raw = mode_weights
        if raw is None:
            raw = getattr(self, 'mode_weights', None)
        merged = {}
        if isinstance(raw, dict) and raw:
            for k, v in raw.items():
                fv = float(v)
                if fv <= 0:
                    continue
                nk = normalize_mode_key(k)
                merged[nk] = merged.get(nk, 0.0) + fv
        if not merged:
            tm = normalize_mode_key(getattr(self, 'transport_mode', 'foot'))
            merged = {tm: 1.0}
        total = sum(merged.values())
        if total <= 0:
            tm = normalize_mode_key(getattr(self, 'transport_mode', 'foot'))
            return {tm: 1.0}
        return {k: v / total for k, v in merged.items()}

    def _activity_travel_minutes(self, activity, default_min=None):
        """活动出行时间（分钟）：优先 custom_activity_times，否则默认阈值。"""
        ct = getattr(self, 'custom_activity_times', None)
        if isinstance(ct, dict) and activity in ct:
            val = ct[activity]
            if isinstance(val, dict):
                nums = [float(v) for v in val.values() if v is not None]
                if nums:
                    return max(5, int(round(max(nums))))
            else:
                return max(5, int(round(float(val))))
        base = default_min
        if base is None:
            base = getattr(self, 'travel_time_minutes', 20)
        return max(5, int(round(float(base or 20))))

    def compute_activity_accessibility_mode_weighted(
        self,
        activity,
        poi_gdf,
        demand_coords,
        demand_pop,
        demand_pop_valid,
        resolution=None,
        mode_weights=None,
        travel_time_min=None,
        subcategory_keyword=None,
        beta=None,
        progress_callback=None,
    ):
        """
        单类活动可达性 A_(i,k) = Σ_m w_m · A_(i,k,m)。
        各 A_(i,k,m) 优先读 Location-based 缓存（POI+子类+t{分钟}+mode），
        缓存键与基于位置模块一致。
        """
        if poi_gdf is None or getattr(poi_gdf, 'empty', True) or len(demand_coords) == 0:
            return np.zeros(len(demand_coords), dtype=np.float64)

        from transport_modes import speed_ms, normalize_mode_key

        weights = self._normalize_mode_weights(mode_weights)
        tt = self._activity_travel_minutes(activity, travel_time_min)
        beta_val = float(beta if beta is not None else self.carbon_params.distance_decay_beta)
        n = len(demand_coords)
        a_ik = np.zeros(n, dtype=np.float64)

        saved_mode = getattr(self, 'transport_mode', 'foot')
        saved_tt = getattr(self, 'travel_time_minutes', 20)

        for mode_key, w_m in weights.items():
            if w_m <= 0:
                continue
            t0_m = tt * 60.0 * speed_ms(mode_key)
            self.transport_mode = mode_key
            self.travel_time_minutes = tt

            ai_m = self._load_cached_activity(
                activity,
                demand_coords,
                subcategory_keyword=subcategory_keyword,
                travel_time_min=tt,
                transport_mode=mode_key,
                resolution=resolution,
            )
            if ai_m is None:
                if callable(progress_callback):
                    progress_callback(
                        0,
                        f'计算 {activity}（{mode_key}，{tt}min，d0≈{int(t0_m)}m）',
                    )
                print(
                    f"  [G2SFCA] {activity} mode={mode_key} t={tt}min d0={int(t0_m)}m "
                    f"POI={len(poi_gdf)}"
                )
                ai_m = self._calculate_activity_g2sfca(
                    demand_coords,
                    demand_pop,
                    demand_pop_valid,
                    poi_gdf,
                    activity,
                    beta=beta_val,
                    t0_distance_m=t0_m,
                    progress_callback=progress_callback,
                )
                self._save_cached_activity(
                    activity,
                    demand_coords,
                    ai_m,
                    subcategory_keyword=subcategory_keyword,
                    travel_time_min=tt,
                    transport_mode=mode_key,
                    resolution=resolution,
                )
            else:
                print(f"  [缓存命中] {activity} mode={mode_key} t={tt}min")
            a_ik += float(w_m) * np.asarray(ai_m, dtype=np.float64)

        self.transport_mode = saved_mode
        self.travel_time_minutes = saved_tt
        # 结果统一放大，再对放大后的正值做下限：0 < A ≤ 0.1 → 0.1
        a_ik = np.asarray(a_ik, dtype=np.float64) * ACCESSIBILITY_RESULT_SCALE
        small_pos = (a_ik > 0) & (a_ik <= ACCESSIBILITY_MIN_POSITIVE)
        a_ik[small_pos] = ACCESSIBILITY_MIN_POSITIVE
        return a_ik

    @staticmethod
    @njit
    def _calculate_haversine_distance(lon1, lat1, lon2, lat2):
        """使用Haversine公式计算两点间的距离（米）"""
        lon1_rad = np.radians(float(lon1))
        lat1_rad = np.radians(float(lat1))
        lon2_rad = np.radians(float(lon2))
        lat2_rad = np.radians(float(lat2))

        # Haversine公式
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371000  # 地球平均半径（米）
        return c * r  # 返回距离（米）

    def _get_supply_value(self, poi_gdf):
        """从POI中提取供给量 S_j，不存在时默认为1"""
        if poi_gdf is None or poi_gdf.empty:
            return np.array([], dtype=float)
        candidates = ['supply', 'capacity', 'beds', 'area', 'weight']
        col = next((c for c in candidates if c in poi_gdf.columns), None)
        if col is None:
            return np.ones(len(poi_gdf), dtype=float)
        values = pd.to_numeric(poi_gdf[col], errors='coerce').fillna(0).to_numpy(dtype=float)
        values[values < 0] = 0.0
        values[values == 0] = 1.0
        return values

    @staticmethod
    def _gaussian_decay(distance, d0, beta=None):
        """
        高斯距离衰减函数 G(d_ij, d0)：
            G = [exp(-1/2 · (d_ij/d0)^2) - exp(-1/2)] / [1 - exp(-1/2)]  (d_ij ≤ d0)
            G = 0  (d_ij > d0)
        beta 保留仅为兼容旧调用，公式中不再使用。
        """
        if d0 is None or d0 <= 0:
            return np.zeros_like(distance, dtype=float)
        distance = np.asarray(distance, dtype=float)
        weights = np.zeros_like(distance, dtype=float)
        mask = distance <= d0
        if np.any(mask):
            e_half = np.exp(-0.5)
            numer = np.exp(-0.5 * (distance[mask] / d0) ** 2) - e_half
            denom = 1.0 - e_half
            if denom != 0:
                weights[mask] = numer / denom
        return weights

    def _extract_population_for_points(self, point_coords):
        """从人口栅格或矢量数据提取每个需求点人口 P_i；缺失时使用1
        返回 (pop_weights, valid_mask)，其中 valid_mask 为 True 表示该点有有效人口
        """
        pop = np.ones(len(point_coords), dtype=float)
        valid = np.zeros(len(point_coords), dtype=bool)
        
        if self.population_data is None and hasattr(self, 'nodes_with_population') and self.nodes_with_population is not None:
            if len(self.nodes_with_population) > 0 and not self.nodes_with_population.empty:
                from shapely.geometry import Point

                pop_col = 'Z' if 'Z' in self.nodes_with_population.columns else self.nodes_with_population.columns[0]
                pop_gdf = self.nodes_with_population[[pop_col, 'geometry']].copy()
                
                demand_gdf = gpd.GeoDataFrame({
                    'point_idx': range(len(point_coords)),
                    'geometry': [Point(x, y) for x, y in point_coords]
                }, crs='EPSG:4326')
                
                joined = gpd.sjoin(demand_gdf, pop_gdf, how='left', predicate='within')
                
                for _, row in joined.iterrows():
                    idx = row['point_idx']
                    val = row.get(pop_col)
                    if val is not None and np.isfinite(val) and val > 0:
                        pop[idx] = val
                        valid[idx] = True
                
                return pop, valid
        
        if self.population_data is None or self.pop_transform is None or len(point_coords) == 0:
            return np.ones(len(point_coords), dtype=float), np.ones(len(point_coords), dtype=bool)
        
        height, width = self.population_data.shape
        
        # 如果栅格不是 EPSG:4326，需要转换坐标
        need_transform = (self.pop_raster_crs is not None and 
                          self.pop_raster_crs.to_string() != 'EPSG:4326')
        
        if need_transform:
            from pyproj import Transformer
            transformer = Transformer.from_crs('EPSG:4326', self.pop_raster_crs.to_string(), always_xy=True)
            projected_coords = []
            for x, y in point_coords:
                px, py = transformer.transform(x, y)
                projected_coords.append((px, py))
            projected_coords = np.array(projected_coords)
        else:
            projected_coords = point_coords
        
        for idx, (x, y) in enumerate(projected_coords):
            try:
                r, c = rasterio.transform.rowcol(self.pop_transform, x, y)
                if 0 <= r < height and 0 <= c < width:
                    val = float(self.population_data[r, c])
                    if np.isfinite(val) and val > 0:
                        pop[idx] = val
                        valid[idx] = True
            except Exception:
                continue
        if self.population_scale_mode == 'log1600':
            clipped = np.clip(pop, a_min=1.0, a_max=None)
            pop = np.log(clipped) / np.log(1600.0)
        elif self.population_scale_mode == 'log10':
            clipped = np.clip(pop, a_min=1.0, a_max=None)
            pop = np.log(clipped) / np.log(10.0)
        else:
            # raw：直接使用人口栅格原始值作为 P_i
            pop = np.asarray(pop, dtype=np.float64)

        # 避免极小 P_i 使 G2SFCA 分母 Σ(P·G) 趋近于 0，进而 R_j=S/denom 爆炸
        pop = np.maximum(np.asarray(pop, dtype=np.float64), 0.1)

        # 粗栅格人口 TIF 下大量需求点未命中像元：仍应在行政边界内参与供需平衡
        unfixed = ~valid
        if np.any(unfixed) and self.admin_boundary is not None and not self.admin_boundary.empty:
            med = float(np.median(pop[valid])) if np.any(valid) else 1.0
            if not np.isfinite(med) or med < 0.1:
                med = 1.0
            try:
                from shapely.geometry import Point

                ix = np.flatnonzero(unfixed)
                pts = gpd.GeoDataFrame(
                    {'_oi': ix},
                    geometry=[Point(float(point_coords[i, 0]), float(point_coords[i, 1])) for i in ix],
                    crs='EPSG:4326',
                )
                adm = self.admin_boundary
                joined = gpd.sjoin(pts, adm[['geometry']], how='left', predicate='within')
                for _, row in joined.iterrows():
                    if pd.notna(row.get('index_right')):
                        oi = int(row['_oi'])
                        pop[oi] = med
                        valid[oi] = True
            except Exception:
                pass

        return pop, valid

    def _ensure_drive_spatial_index(self):
        if self._drive_spatial_index is not None:
            return self._drive_spatial_index
        G = self.networks.get('drive')
        if G is None:
            return None
        from scipy.spatial import cKDTree
        nodes_data = []
        for node, data in G.nodes(data=True):
            x_val = y_val = None
            if 'x' in data and 'y' in data:
                x_val, y_val = float(data['x']), float(data['y'])
            elif 'lon' in data and 'lat' in data:
                x_val, y_val = float(data['lon']), float(data['lat'])
            elif 'lng' in data and 'lat' in data:
                x_val, y_val = float(data['lng']), float(data['lat'])
            if x_val is not None and y_val is not None:
                nodes_data.append((x_val, y_val, node))
        if not nodes_data:
            return None
        node_coords = np.array([[d[0], d[1]] for d in nodes_data], dtype=float)
        node_ids = [d[2] for d in nodes_data]
        node_idx_map = {node_ids[i]: i for i in range(len(node_ids))}
        self._drive_spatial_index = {
            'G': G,
            'node_coords': node_coords,
            'node_ids': node_ids,
            'node_idx_map': node_idx_map,
            'kd': cKDTree(node_coords),
        }
        return self._drive_spatial_index

    def _road_walk_distance_m(self, lon1, lat1, lon2, lat2):
        """道路网步行接驳距离（米）。"""
        key = (round(float(lon1), 5), round(float(lat1), 5), round(float(lon2), 5), round(float(lat2), 5))
        cached = self._road_walk_cache.get(key)
        if cached is not None:
            return cached

        import networkx as nx
        idx = self._ensure_drive_spatial_index()
        if idx is None:
            dist = self._calculate_haversine_distance(lon1, lat1, lon2, lat2) * 1.25
        else:
            G = idx['G']
            node_coords = idx['node_coords']
            node_ids = idx['node_ids']
            kd = idx['kd']
            oi = kd.query([[lon1, lat1]])[1][0]
            di = kd.query([[lon2, lat2]])[1][0]
            on, dn = node_ids[oi], node_ids[di]
            access_o = self._calculate_haversine_distance(
                lon1, lat1, node_coords[oi, 0], node_coords[oi, 1]
            )
            access_d = self._calculate_haversine_distance(
                lon2, lat2, node_coords[di, 0], node_coords[di, 1]
            )
            if on == dn:
                dist = float(access_o + access_d)
            else:
                try:
                    internal = nx.shortest_path_length(G, on, dn, weight='length')
                    dist = float(access_o + internal + access_d)
                except Exception:
                    dist = self._calculate_haversine_distance(lon1, lat1, lon2, lat2) * 1.25
        self._road_walk_cache[key] = dist
        return dist

    def _straight_walk_distance_m(self, lon1, lat1, lon2, lat2):
        """公交接驳：步行速度下按直线几何距离（米）。"""
        key = (round(float(lon1), 5), round(float(lat1), 5), round(float(lon2), 5), round(float(lat2), 5))
        cached = self._road_walk_cache.get(key)
        if cached is not None:
            return cached
        dist = float(self._calculate_haversine_distance(lon1, lat1, lon2, lat2))
        self._road_walk_cache[key] = dist
        return dist

    def _access_walk_distance_m(self, lon1, lat1, lon2, lat2, mode=None):
        from transport_modes import normalize_mode_key

        m = normalize_mode_key(mode if mode is not None else getattr(self, 'transport_mode', 'foot'))
        if m == 'bus':
            return self._straight_walk_distance_m(lon1, lat1, lon2, lat2)
        if m == 'metro':
            return self._road_walk_distance_m(lon1, lat1, lon2, lat2)
        return self._straight_walk_distance_m(lon1, lat1, lon2, lat2)

    def _metro_snap_threshold_m(self):
        """地铁接驳容许距离（米）= 出行时间 × 步行 5km/h。"""
        from transport_modes import speed_ms

        minutes = float(
            getattr(self, 'travel_time_minutes', None)
            or getattr(self.carbon_params, 'travel_time_threshold', 20)
        )
        return minutes * 60.0 * speed_ms('walk')

    def _nearest_stop_access_distance_m(self, lon, lat, tn, mode='metro'):
        """点到线网最近站点的接驳距离（米）。"""
        stops = tn.nearest_stops(lon, lat, k=1, max_dist_m=50000.0)
        if not stops:
            return float('inf')
        _, slon, slat, _ = stops[0]
        return self._access_walk_distance_m(lon, lat, slon, slat, mode=mode)

    def _point_beyond_metro_snap(self, lon, lat, tn):
        return self._nearest_stop_access_distance_m(lon, lat, tn, 'metro') > self._metro_snap_threshold_m()

    def _build_metro_walk_mask_batch(self, demand_coords, tn, threshold_m, progress_fn=None):
        """
        批量判定格点是否超出地铁接驳范围（直线距离 × BallTree，避免逐格路网最短路）。
        直线距离已大于阈值时，道路距离必然更大，可安全标为远站步行格。
        """
        from sklearn.neighbors import BallTree

        n = len(demand_coords)
        if n == 0:
            return np.zeros(0, dtype=bool)
        stop_lonlat = getattr(tn, 'stop_lonlat', None) or {}
        if not stop_lonlat:
            return np.ones(n, dtype=bool)

        if progress_fn:
            progress_fn(0, n, '正在批量匹配最近地铁站…')

        stop_arr = np.array(list(stop_lonlat.values()), dtype=float)
        stop_ll = np.radians(np.column_stack([stop_arr[:, 1], stop_arr[:, 0]]))
        demand_ll = np.radians(np.column_stack([demand_coords[:, 1], demand_coords[:, 0]]))
        tree = BallTree(stop_ll, metric='haversine')
        dist_rad, _ = tree.query(demand_ll, k=1)
        straight_m = dist_rad.reshape(-1) * 6371000.0

        if progress_fn:
            progress_fn(n, n, '接驳距离判定完成')

        return straight_m > float(threshold_m)

    def _pair_distance_for_accessibility_m(self, lon1, lat1, lon2, lat2):
        """按当前交通方式计算两点间可达性用距离（米）。"""
        from transport_modes import normalize_mode_key
        from Get_Network import multimodal_transit_distance_m

        mode = normalize_mode_key(getattr(self, 'transport_mode', 'foot'))
        if mode == 'metro':
            tn = self._transit_networks.get('metro')
            if tn is not None and self._point_beyond_metro_snap(lon1, lat1, tn):
                return float('inf')
            if tn is not None:
                d = multimodal_transit_distance_m(
                    lon1, lat1, lon2, lat2, 'metro', tn,
                    road_walk_distance_fn=self._road_walk_distance_m,
                )
                if d is not None and np.isfinite(d) and d > 0:
                    return float(d)
            return float('inf')
        if mode == 'bus':
            tn = self._transit_networks.get('bus')
            if tn is not None:
                d = multimodal_transit_distance_m(
                    lon1, lat1, lon2, lat2, 'bus', tn,
                    road_walk_distance_fn=self._straight_walk_distance_m,
                )
                if d is not None and np.isfinite(d) and d > 0:
                    return float(d)
        if mode in ('bike', 'ebike', 'drive'):
            idx = self._ensure_drive_spatial_index()
            if idx is not None:
                return self._road_walk_distance_m(lon1, lat1, lon2, lat2)
        return self._calculate_haversine_distance(lon1, lat1, lon2, lat2)

    def _build_drive_g2sfca_context(self, demand_coords, demand_pop_valid, facility_coords):
        """构建驾车路网 G2SFCA 上下文（供地铁远站步行回退复用）。"""
        import heapq
        from scipy.spatial import cKDTree

        G = self.networks.get('drive')
        if G is None:
            return None

        nodes_data = []
        for node, data in G.nodes(data=True):
            x_val = y_val = None
            if 'x' in data and 'y' in data:
                x_val, y_val = float(data['x']), float(data['y'])
            elif 'lon' in data and 'lat' in data:
                x_val, y_val = float(data['lon']), float(data['lat'])
            elif 'lng' in data and 'lat' in data:
                x_val, y_val = float(data['lng']), float(data['lat'])
            if x_val is not None and y_val is not None:
                nodes_data.append((x_val, y_val, node))

        if not nodes_data:
            return None

        node_coords = np.array([[d[0], d[1]] for d in nodes_data], dtype=float)
        node_ids = [d[2] for d in nodes_data]
        n_nodes = len(node_ids)
        node_idx_map = {node_ids[i]: i for i in range(n_nodes)}
        node_kd = cKDTree(node_coords)

        demand_len = len(demand_coords)
        n_facilities = len(facility_coords)
        demand_node_idx = node_kd.query(demand_coords)[1]
        facility_node_idx = node_kd.query(facility_coords)[1]

        demand_to_node_dist = np.array([
            self._calculate_haversine_distance(
                demand_coords[i, 0], demand_coords[i, 1],
                node_coords[demand_node_idx[i], 0], node_coords[demand_node_idx[i], 1],
            ) for i in range(demand_len)
        ], dtype=float)
        facility_to_node_dist = np.array([
            self._calculate_haversine_distance(
                facility_coords[j, 0], facility_coords[j, 1],
                node_coords[facility_node_idx[j], 0], node_coords[facility_node_idx[j], 1],
            ) for j in range(n_facilities)
        ], dtype=float)

        adj = [[] for _ in range(n_nodes)]
        for u, v, data in G.edges(data=True):
            w = data.get('length') or data.get('length_m') or data.get('dist') or data.get('distance')
            if w is not None:
                w = float(w)
            else:
                ui, vi = node_idx_map[u], node_idx_map[v]
                w = self._calculate_haversine_distance(
                    node_coords[ui, 0], node_coords[ui, 1],
                    node_coords[vi, 0], node_coords[vi, 1],
                )
            ui, vi = node_idx_map[u], node_idx_map[v]
            adj[ui].append((vi, w))
            adj[vi].append((ui, w))

        demand_point_at_node = [[] for _ in range(n_nodes)]
        for di in range(demand_len):
            if demand_pop_valid[di]:
                demand_point_at_node[demand_node_idx[di]].append(di)
        facility_at_node = [[] for _ in range(n_nodes)]
        for fi in range(n_facilities):
            facility_at_node[facility_node_idx[fi]].append(fi)

        return {
            'adj': adj,
            'n_nodes': n_nodes,
            'demand_len': demand_len,
            'n_facilities': n_facilities,
            'demand_node_idx': demand_node_idx,
            'facility_node_idx': facility_node_idx,
            'demand_to_node_dist': demand_to_node_dist,
            'facility_to_node_dist': facility_to_node_dist,
            'demand_point_at_node': demand_point_at_node,
            'facility_at_node': facility_at_node,
            'heapq': heapq,
        }

    def _road_g2sfca_ai_one_demand(self, i, ctx, d0_walk, rj, beta):
        heapq = ctx['heapq']
        adj = ctx['adj']
        demand_node_idx = ctx['demand_node_idx']
        facility_at_node = ctx['facility_at_node']
        demand_to_node_dist = ctx['demand_to_node_dist']
        facility_to_node_dist = ctx['facility_to_node_dist']
        n_nodes = ctx['n_nodes']
        n_facilities = ctx['n_facilities']

        initial_dist = demand_to_node_dist[i]
        visited_n = np.zeros(n_nodes, dtype=bool)
        visited_f = np.zeros(n_facilities, dtype=bool)
        ai_val = 0.0
        heap = [(initial_dist, demand_node_idx[i])]
        while heap:
            d, node = heapq.heappop(heap)
            if d > d0_walk or visited_n[node]:
                continue
            visited_n[node] = True
            for fi in facility_at_node[node]:
                if not visited_f[fi]:
                    total = d + facility_to_node_dist[fi]
                    if total <= d0_walk and rj[fi] > 0:
                        g = self._gaussian_decay(np.array([total]), d0_walk, beta=beta)[0]
                        ai_val += rj[fi] * g
                        visited_f[fi] = True
            for nxt, w in adj[node]:
                nd = d + w
                if nd < d0_walk and not visited_n[nxt]:
                    heapq.heappush(heap, (nd, nxt))
        return ai_val

    def _metro_walk_facility_denom(
        self, j, drive_ctx, walk_mask, demand_pop, demand_pop_valid, d0_walk, beta
    ):
        """远站格点：单设施道路网步行侧 Step1 分母 Σ P_i·G(d_ij)。"""
        heapq = drive_ctx['heapq']
        adj = drive_ctx['adj']
        demand_len = drive_ctx['demand_len']
        demand_point_at_node = drive_ctx['demand_point_at_node']
        demand_to_node_dist = drive_ctx['demand_to_node_dist']
        visited_d = np.zeros(demand_len, dtype=bool)
        heap = [(0.0, drive_ctx['facility_node_idx'][j])]
        total = 0.0
        while heap:
            d, node = heapq.heappop(heap)
            if d > d0_walk:
                break
            for di in demand_point_at_node[node]:
                if walk_mask[di] and not visited_d[di] and demand_pop_valid[di]:
                    dist = d + demand_to_node_dist[di]
                    if dist <= d0_walk:
                        visited_d[di] = True
                        g = self._gaussian_decay(np.array([dist]), d0_walk, beta=beta)[0]
                        total += float(demand_pop[di]) * g
        return total

    def _metro_walk_accessibility_haversine(
        self, i, demand_coords, facility_coords, facility_tree, r_rad_walk, rj, d0_walk, beta
    ):
        """远站格点无路网时：Haversine + BallTree 汇总 Step2 可达性。"""
        dx, dy = demand_coords[i, 0], demand_coords[i, 1]
        cand = facility_tree.query_radius(
            np.array([[np.radians(dy), np.radians(dx)]]), r=r_rad_walk, return_distance=False
        )[0]
        ai_val = 0.0
        for j in cand:
            if rj[j] <= 0:
                continue
            d_ij = self._calculate_haversine_distance(
                dx, dy, facility_coords[j, 0], facility_coords[j, 1]
            )
            if d_ij > d0_walk:
                continue
            g_ij = self._gaussian_decay(np.array([d_ij]), d0_walk, beta=beta)[0]
            ai_val += rj[j] * g_ij
        return ai_val

    def _calculate_activity_g2sfca_metro_with_walk_fallback(
        self, demand_coords, demand_pop, demand_pop_valid, poi_gdf, activity, beta=1.5, t0_distance_m=None,
        progress_callback=None,
    ):
        """地铁 G2SFCA：仅对能在出行时间×步行速度内接驳最近地铁站的格点计算；其余格点可达性为 0。"""
        from sklearn.neighbors import BallTree

        def _report(pct_in_stage, status):
            if callable(progress_callback):
                try:
                    progress_callback(int(pct_in_stage), status)
                except Exception:
                    pass

        self._road_walk_cache.clear()

        d0_metro = float(t0_distance_m if t0_distance_m is not None else self.t0_distance_m)
        d0_walk = float(self._metro_snap_threshold_m())
        k_stops = 3
        snap_m = 2500.0

        tn = self._transit_networks.get('metro')
        if tn is None:
            return np.zeros(len(demand_coords), dtype=float)

        demand_len = len(demand_coords)
        if demand_len == 0:
            return np.zeros(0, dtype=float)

        _report(0, f'地铁模式：正在分析 {demand_len} 个格点接驳情况（批量）…')

        def _mask_progress(done, total, msg):
            pct = int(2 * done / max(total, 1))
            _report(pct, f'{msg}（{done}/{total}）')

        walk_mask = self._build_metro_walk_mask_batch(
            demand_coords, tn, d0_walk, progress_fn=_mask_progress
        )

        n_walk = int(np.sum(walk_mask))
        n_metro = demand_len - n_walk
        metro_indices = np.flatnonzero(~walk_mask)
        print(
            f"  [地铁] 接驳阈值={d0_walk:.0f}m（出行时间×步行5km/h），"
            f"不可接驳 {n_walk} 格（可达性=0），可接驳 {n_metro} 格"
        )
        _report(
            5,
            f'地铁接驳分析完成：可接驳 {n_metro} 格，不可接驳 {n_walk} 格（阈值 {d0_walk:.0f}m）',
        )

        ai = np.zeros(demand_len, dtype=float)
        if n_metro == 0:
            _report(100, '地铁可达性计算完成（无格点可接驳地铁站）')
            return ai

        facility_coords = np.array([[p.x, p.y] for p in poi_gdf.geometry], dtype=float)
        supply_values = self._get_supply_value(poi_gdf)
        n_facilities = len(facility_coords)

        # 栅格/设施接驳预计算：地铁使用直线步行距离（与公交一致），避免上万次路网最短路
        straight_walk = self._straight_walk_distance_m
        saved_access_walk = self._transit._access_walk
        self._transit._access_walk = straight_walk
        try:
            _report(10, f'预计算 {n_metro} 个地铁格点接驳站点…')
            print(f"  [地铁线网] 预计算地铁格点接驳（{n_metro} 格，直线接驳）...")
            demand_access = [[] for _ in range(demand_len)]
            metro_coords = demand_coords[metro_indices]

            def _demand_snap_progress(i, n):
                pct = 10 + int(12 * i / max(n, 1))
                _report(pct, f'预计算地铁格点接驳 {i}/{n}')

            metro_access_sub = self._transit.precompute_grid_access(
                metro_coords, tn, k_stops=k_stops, snap_m=snap_m,
                progress_every=500, progress_fn=_demand_snap_progress,
            )
            for k, i in enumerate(metro_indices):
                demand_access[i] = metro_access_sub[k]

            _report(22, f'预计算 {n_facilities} 个设施接驳站点…')
            print(f"  [地铁线网] 预计算设施接驳（直线接驳）...")

            def _fac_snap_progress(i, n):
                pct = 22 + int(3 * i / max(n, 1))
                _report(pct, f'预计算设施接驳 {i}/{n}')

            facility_access = self._transit.precompute_grid_access(
                facility_coords, tn, k_stops=k_stops, snap_m=snap_m,
                progress_every=500, progress_fn=_fac_snap_progress,
            )
        finally:
            self._transit._access_walk = saved_access_walk

        metro_demand_access = [
            demand_access[i] if not walk_mask[i] else [] for i in range(demand_len)
        ]
        stop_index_metro = self._transit.build_stop_to_demand_index(metro_demand_access)

        # Step 1: R_j（仅统计可接驳地铁站的格点）
        _report(25, '正在计算设施服务比率（地铁线网）...')
        print(f"  Step 1: 计算 {n_facilities} 个设施的服务比率（地铁线网）...")
        facility_fields = self._transit.facility_fields_by_access(
            facility_access, tn, d0_metro, progress_every=500
        )
        rj = np.zeros(n_facilities, dtype=float)
        for j in range(n_facilities):
            if j % 500 == 0:
                print(f"    设施服务比率: {j}/{n_facilities}")
                if n_facilities > 0:
                    pct = 25 + int(35 * j / n_facilities)
                    _report(pct, f'地铁 Step1 设施服务比率 {j}/{n_facilities}')
            denom = self._transit.service_ratio_denom_from_facility_field(
                facility_fields[j],
                stop_index_metro,
                demand_pop,
                demand_pop_valid,
                d0_metro,
                beta=beta,
            )
            if denom > 0:
                rj[j] = supply_values[j] / max(denom, 1e-3)

        # Step 2: A_i（地铁线网；不可接驳格点保持 0）
        _report(60, f'地铁 Step2：计算 {demand_len} 个需求点可达性…')
        print(f"  Step 2: 计算 {demand_len} 个需求点可达性...")
        metro_coords = demand_coords[metro_indices]
        metro_access_sub = [demand_access[i] for i in metro_indices]
        print(f"  [地铁线网] 需求侧线网扩散（{n_metro} 格，聚类）...")
        metro_fields_sub = self._transit.build_demand_fields_clustered(
            metro_coords, metro_access_sub, tn, d0_metro, geo_round=2
        )
        demand_fields = [{} for _ in range(demand_len)]
        for k, i in enumerate(metro_indices):
            demand_fields[i] = metro_fields_sub[k]
        stop_to_fac = self._transit.build_stop_to_facility_index(facility_access, rj=rj)
        ai_metro = self._transit.summarize_accessibility_ai(
            demand_fields, stop_to_fac, rj, d0_metro, beta=beta, progress_every=2000
        )
        ai[~walk_mask] = ai_metro[~walk_mask]

        _report(90, '正在复核地铁步行接驳可达性…')
        for wi, i in enumerate(metro_indices):
            if wi % 2000 == 0 and n_metro > 0:
                pct = 90 + int(8 * wi / n_metro)
                _report(pct, f'地铁接驳复核 {wi}/{n_metro}')
            if self._point_beyond_metro_snap(demand_coords[i, 0], demand_coords[i, 1], tn):
                ai[i] = 0.0

        _report(100, '地铁可达性计算完成')
        return ai

    def _calculate_activity_g2sfca_multimodal(
        self, demand_coords, demand_pop, demand_pop_valid, poi_gdf, activity, beta=1.5, t0_distance_m=None,
        progress_callback=None,
    ):
        """
        公交 G2SFCA（P0 加速）：
        - Step1：设施侧去重 Dijkstra + 站点倒排；
        - Step2：需求侧聚类线网扩散 + 倒排汇总。
        地铁：仅对能在「出行时间 × 步行速度」内接驳最近地铁站的格点计算 G2SFCA，其余格点可达性为 0。
        """
        if poi_gdf is None or poi_gdf.empty or len(demand_coords) == 0:
            return np.zeros(len(demand_coords), dtype=float)

        from transport_modes import normalize_mode_key

        mode = normalize_mode_key(getattr(self, 'transport_mode', 'foot'))
        if mode == 'metro':
            return self._calculate_activity_g2sfca_metro_with_walk_fallback(
                demand_coords, demand_pop, demand_pop_valid, poi_gdf, activity, beta=beta, t0_distance_m=t0_distance_m,
                progress_callback=progress_callback,
            )

        self._road_walk_cache.clear()
        d0 = float(t0_distance_m if t0_distance_m is not None else self.t0_distance_m)
        tn = self._transit_networks.get(mode)
        if tn is None:
            return self._calculate_activity_g2sfca_haversine(
                demand_coords, demand_pop, demand_pop_valid, poi_gdf, beta=beta, t0_distance_m=d0
            )

        facility_coords = np.array([[p.x, p.y] for p in poi_gdf.geometry], dtype=float)
        supply_values = self._get_supply_value(poi_gdf)
        demand_len = len(demand_coords)
        n_facilities = len(facility_coords)

        k_stops = 3
        snap_m = 2500.0

        # 轻量预计算：栅格/设施 → 最近站点接驳（无 Dijkstra）
        print(f"  [线网] 预计算 {demand_len} 个需求点接驳站点...")
        demand_access = self._transit.precompute_grid_access(
            demand_coords, tn, k_stops=k_stops, snap_m=snap_m
        )
        print(f"  [线网] 预计算 {n_facilities} 个设施接驳站点...")
        facility_access = self._transit.precompute_grid_access(
            facility_coords, tn, k_stops=k_stops, snap_m=snap_m
        )

        # Step 1: R_j（接驳源去重 Dijkstra + 站点倒排，避免每设施大球 query_radius）
        print(f"  Step 1: 计算 {n_facilities} 个设施的服务比率（设施侧线网扩散）...")
        stop_index = self._transit.build_stop_to_demand_index(demand_access)
        facility_fields = self._transit.facility_fields_by_access(
            facility_access, tn, d0, progress_every=500
        )
        rj = np.zeros(n_facilities, dtype=float)
        for j in range(n_facilities):
            if j % 500 == 0:
                print(f"    设施服务比率: {j}/{n_facilities}")
            denom = self._transit.service_ratio_denom_from_facility_field(
                facility_fields[j],
                stop_index,
                demand_pop,
                demand_pop_valid,
                d0,
                beta=beta,
            )
            if denom > 0:
                rj[j] = supply_values[j] / max(denom, 1e-3)

        # Step 2: A_i（聚类需求侧 Dijkstra + 站点倒排汇总，避免每格对全部 POI 做 query_radius）
        print(f"  Step 2: 计算 {demand_len} 个需求点的可达性（聚类需求侧线网扩散）...")
        demand_fields = self._transit.build_demand_fields_clustered(
            demand_coords, demand_access, tn, d0, geo_round=2
        )
        stop_to_fac = self._transit.build_stop_to_facility_index(facility_access, rj=rj)
        ai = self._transit.summarize_accessibility_ai(
            demand_fields, stop_to_fac, rj, d0, beta=beta, progress_every=2000
        )

        return ai

    def _calculate_activity_g2sfca(self, demand_coords, demand_pop, demand_pop_valid, poi_gdf, activity, beta=1.5, t0_distance_m=None, progress_callback=None):
        if poi_gdf is None or poi_gdf.empty or len(demand_coords) == 0:
            return np.zeros(len(demand_coords), dtype=float)

        from transport_modes import normalize_mode_key

        d0 = t0_distance_m if t0_distance_m is not None else self.t0_distance_m
        mode = normalize_mode_key(getattr(self, 'transport_mode', 'foot'))
        if mode in ('bus', 'metro') and self._transit_networks.get(mode) is not None:
            return self._calculate_activity_g2sfca_multimodal(
                demand_coords, demand_pop, demand_pop_valid, poi_gdf, activity, beta=beta, t0_distance_m=d0,
                progress_callback=progress_callback,
            )

        import networkx as nx
        import heapq
        from scipy.spatial import cKDTree

        G = self.networks.get('drive')
        if G is None:
            print(f"[G2SFCA] 路网中无drive，回退到Haversine方法")
            return self._calculate_activity_g2sfca_haversine(
                demand_coords, demand_pop, demand_pop_valid, poi_gdf, beta=beta, t0_distance_m=d0
            )

        # 尝试获取节点坐标，支持不同的属性名称（Shape文件用 x/y，GraphML用 lon/lat）
        nodes_data = []
        for node, data in G.nodes(data=True):
            x_val, y_val = None, None
            if 'x' in data and 'y' in data:
                x_val, y_val = float(data['x']), float(data['y'])
            elif 'lon' in data and 'lat' in data:
                x_val, y_val = float(data['lon']), float(data['lat'])
            elif 'lng' in data and 'lat' in data:
                x_val, y_val = float(data['lng']), float(data['lat'])
            if x_val is not None and y_val is not None:
                nodes_data.append((x_val, y_val, node))
        if not nodes_data:
            print(f"[G2SFCA] 路网节点缺少坐标属性，回退到Haversine方法")
            return self._calculate_activity_g2sfca_haversine(
                demand_coords, demand_pop, demand_pop_valid, poi_gdf, beta=beta, t0_distance_m=d0
            )

        node_coords = np.array([[d[0], d[1]] for d in nodes_data])
        node_ids = [d[2] for d in nodes_data]
        n_nodes = len(node_ids)
        node_idx_map = {node_ids[i]: i for i in range(n_nodes)}

        # 使用 cKDTree 找最近节点，避免生成 n_nodes x demand_len 的巨型矩阵
        node_kd = cKDTree(node_coords)

        facility_coords = np.array([[p.x, p.y] for p in poi_gdf.geometry], dtype=float)
        facility_kd = cKDTree(facility_coords)
        supply_values = self._get_supply_value(poi_gdf)
        demand_len = len(demand_coords)
        n_facilities = len(facility_coords)

        demand_node_idx = node_kd.query(demand_coords)[1]
        facility_node_idx = node_kd.query(facility_coords)[1]

        # 预计算首尾段 Haversine 距离
        demand_to_node_dist = np.array([
            self._calculate_haversine_distance(
                demand_coords[i, 0], demand_coords[i, 1],
                node_coords[demand_node_idx[i], 0], node_coords[demand_node_idx[i], 1]
            ) for i in range(demand_len)
        ])
        facility_to_node_dist = np.array([
            self._calculate_haversine_distance(
                facility_coords[j, 0], facility_coords[j, 1],
                node_coords[facility_node_idx[j], 0], node_coords[facility_node_idx[j], 1]
            ) for j in range(n_facilities)
        ])

        adj = [[] for _ in range(n_nodes)]
        for u, v, data in G.edges(data=True):
            # 获取边的长度属性（单位：米）
            w = data.get('length')
            if w is None:
                w = data.get('length_m')
            if w is None:
                w = data.get('dist')
            if w is None:
                w = data.get('distance')
            if w is not None:
                w = float(w)
            else:
                # 如果边没有长度属性，根据节点坐标计算（使用 Haversine 公式）
                ui = node_idx_map[u]
                vi = node_idx_map[v]
                x1, y1 = node_coords[ui]
                x2, y2 = node_coords[vi]
                from math import radians, sin, cos, sqrt, atan2
                R = 6371000
                lat1, lat2 = radians(y1), radians(y2)
                dlat = radians(y2 - y1)
                dlon = radians(x2 - x1)
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                c = 2 * atan2(sqrt(a), sqrt(1-a))
                w = R * c
            ui, vi = node_idx_map[u], node_idx_map[v]
            adj[ui].append((vi, w))
            adj[vi].append((ui, w))

        demand_point_at_node = [[] for _ in range(n_nodes)]
        for di in range(demand_len):
            if demand_pop_valid[di]:
                demand_point_at_node[demand_node_idx[di]].append(di)
        facility_at_node = [[] for _ in range(n_nodes)]
        for fi in range(n_facilities):
            facility_at_node[facility_node_idx[fi]].append(fi)

        valid_demand_indices = np.where(demand_pop_valid)[0]
        valid_demand_len = len(valid_demand_indices)

        # Step 1: 计算 R_j
        print(f"  Step 1: 计算 {n_facilities} 个设施的服务比率...")
        rj = np.zeros(n_facilities, dtype=float)
        for j in range(n_facilities):
            if j % 500 == 0:
                print(f"    设施进度: {j}/{n_facilities}")
            visited_d = np.zeros(demand_len, dtype=bool)
            heap = [(0.0, facility_node_idx[j])]
            while heap:
                d, node = heapq.heappop(heap)
                if d > d0:
                    break
                for di in demand_point_at_node[node]:
                    if not visited_d[di]:
                        total = d + demand_to_node_dist[di]
                        if total <= d0:
                            visited_d[di] = True
            # 只对有效人口的需求点计算高斯衰减 R_j
            valid_visited = visited_d & demand_pop_valid
            if not np.any(valid_visited):
                continue
            g = self._gaussian_decay(demand_to_node_dist[valid_visited], d0, beta=beta)
            denom = float(np.sum(demand_pop[valid_visited] * g))
            if denom > 0:
                # 防止分母过小导致 R_j 爆炸（与人口权重下限配合）
                rj[j] = supply_values[j] / max(denom, 1e-3)

        # Step 2: 计算 A_i（所有需求点）
        print(f"  Step 2: 计算 {demand_len} 个需求点的可达性...")
        ai = np.zeros(demand_len, dtype=float)

        for i in range(demand_len):
            if i % 1000 == 0:
                print(f"    需求点进度: {i}/{demand_len}")

            # 修复：初始化距离应该加上需求点到最近路网节点的距离
            initial_dist = demand_to_node_dist[i]

            visited_n = np.zeros(n_nodes, dtype=bool)
            visited_f = np.zeros(n_facilities, dtype=bool)

            # 使用初始距离作为起点（需求点到路网的距离）
            heap = [(initial_dist, demand_node_idx[i])]
            while heap:
                d, node = heapq.heappop(heap)
                if d > d0 or visited_n[node]:
                    continue
                visited_n[node] = True
                for fi in facility_at_node[node]:
                    if not visited_f[fi]:
                        total = d + facility_to_node_dist[fi]
                        if total <= d0 and rj[fi] > 0:
                            g = self._gaussian_decay(np.array([total]), d0, beta=beta)[0]
                            ai[i] += rj[fi] * g
                            visited_f[fi] = True
                for nxt, w in adj[node]:
                    nd = d + w
                    if nd < d0 and not visited_n[nxt]:
                        heapq.heappush(heap, (nd, nxt))

        return ai

    def _calculate_activity_g2sfca_haversine(self, demand_coords, demand_pop, demand_pop_valid, poi_gdf, beta=1.5, t0_distance_m=None):
        """回退方法：使用 Haversine 直线距离计算 G2SFCA"""
        if poi_gdf is None or poi_gdf.empty or len(demand_coords) == 0:
            print(f"[Haversine] POI为空或需求点为空，POI数量={len(poi_gdf) if poi_gdf is not None else 'None'}, 需求点数量={len(demand_coords)}")
            return np.zeros(len(demand_coords), dtype=float)

        d0 = t0_distance_m if t0_distance_m is not None else self.t0_distance_m
        print(f"[Haversine] d0={d0}m, beta={beta}, demand_valid={np.sum(demand_pop_valid)}/{len(demand_pop_valid)}")

        facility_coords = np.array([[p.x, p.y] for p in poi_gdf.geometry], dtype=float)
        print(f"[Haversine] 设施坐标范围: X=[{facility_coords[:,0].min():.4f}, {facility_coords[:,0].max():.4f}], Y=[{facility_coords[:,1].min():.4f}, {facility_coords[:,1].max():.4f}]")
        print(f"[Haversine] 需求点坐标范围: X=[{demand_coords[:,0].min():.4f}, {demand_coords[:,0].max():.4f}], Y=[{demand_coords[:,1].min():.4f}, {demand_coords[:,1].max():.4f}]")
        
        supply_values = self._get_supply_value(poi_gdf)
        demand_len = len(demand_coords)

        # 第一步：只对有效人口的需求点计算
        rj = np.zeros(len(facility_coords), dtype=float)
        for j, (fx, fy) in enumerate(facility_coords):
            dists = np.array([
                self._calculate_haversine_distance(fx, fy, dx, dy)
                for dx, dy in demand_coords
            ], dtype=float)
            within_mask = dists <= d0
            valid_within = within_mask & demand_pop_valid
            if not np.any(valid_within):
                continue
            g_ij = self._gaussian_decay(dists[valid_within], d0, beta=beta)
            denom = float(np.sum(demand_pop[valid_within] * g_ij))
            if denom > 0:
                rj[j] = supply_values[j] / max(denom, 1e-3)

        # 第二步：所有需求点
        ai = np.zeros(demand_len, dtype=float)
        for j, (fx, fy) in enumerate(facility_coords):
            if rj[j] <= 0:
                continue
            dists = np.array([
                self._calculate_haversine_distance(dx, dy, fx, fy)
                for dx, dy in demand_coords
            ], dtype=float)
            within_mask = dists <= d0
            g_ij = self._gaussian_decay(dists[within_mask], d0, beta=beta)
            ai[within_mask] += rj[j] * g_ij

        print(f"[Haversine] 计算完成，非零值: {np.sum(ai > 0)}/{demand_len}")
        return ai

    def calculate_composite_accessibility(self, resolution=0.004, num_workers=None,
                                         progress_callback=None,
                                         subcategory_keyword=None,
                                         subcategory_poi_gdf=None,
                                         subcategory_poi_activity=None,
                                         t0_distance_m=None):
        """基于高斯衰减网格化2SFCA的综合可达性计算

        Args:
            resolution: 栅格分辨率（度）
            num_workers: 并行线程数（已废弃，保留兼容）
            progress_callback: 进度回调函数
            subcategory_keyword: 子类别关键词，用于过滤POI数据
            subcategory_poi_gdf: 若已在前端过滤，则传入已过滤的POI GeoDataFrame
            subcategory_poi_activity: 标记过滤后的POI GeoDataFrame 属于哪个活动，
                           只有当遍历到该 activity 时才使用过滤后的POI
            t0_distance_m: 距离阈值（米），由出行时间×交通方式速度计算得出；
                           若为None则使用默认值（基于carbon_params计算）
        """
        def report_progress(progress, status):
            if callable(progress_callback):
                try:
                    progress_callback(int(progress), status)
                except Exception:
                    pass

        print("开始计算 G2SFCA 综合可达性栅格...")
        report_progress(40, "正在生成需求网格")
        x_coords = np.arange(self.bbox[0], self.bbox[2], resolution)
        y_coords = np.arange(self.bbox[1], self.bbox[3], resolution)
        raster_data = np.full((len(y_coords), len(x_coords)), np.nan, dtype=float)

        demand_cells = []
        for j, y in enumerate(y_coords):
            for i, x in enumerate(x_coords):
                point = Point(x, y)
                if self.admin_boundary.contains(point).any():
                    demand_cells.append((i, j, x, y))

        if not demand_cells:
            stats = {'mean': np.nan, 'median': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan, 'cells_with_data': 0, 'activity_stats': {}}
            print(f"[可达性计算] 警告: 边界内无需求点! admin_boundary.crs={getattr(self.admin_boundary, 'crs', None)}, 点数量={len(self.admin_boundary)}")
            return raster_data, None, stats, {}

        transform = from_origin(self.bbox[0], self.bbox[3], resolution, resolution)
        
        demand_coords = np.array([[x, y] for _, _, x, y in demand_cells], dtype=float)
        report_progress(45, "正在提取人口数据")
        demand_pop, demand_pop_valid = self._extract_population_for_points(demand_coords)

        if not np.any(demand_pop_valid):
            report_progress(80, "无有效人口数据")
            stats = {'mean': np.nan, 'median': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan, 'cells_with_data': 0, 'activity_stats': {}}
            return raster_data, transform, stats, {}

        total_access = np.zeros(len(demand_cells), dtype=float)
        total_weight = 0.0
        valid_activities = []
        # 构建 POI 权重映射：支持 carbon_params 中的6类 POI vs 计算器内部5类
        poi_weight_map = {
            'work': self.carbon_params.poi_weights.get('work', 100.0 / 7),
            'education': self.carbon_params.poi_weights.get('education', 100.0 / 7),
            'medical': self.carbon_params.poi_weights.get('medical', 100.0 / 7),
            'shopping': self.carbon_params.poi_weights.get('shopping', 100.0 / 7),
            'catering': self.carbon_params.poi_weights.get('catering', 100.0 / 7),
            'transportation_station': self.carbon_params.poi_weights.get('transport_station', 100.0 / 7),
            'recreation': self.carbon_params.poi_weights.get('recreation', 100.0 / 7),
        }
        beta = self.carbon_params.distance_decay_beta
        for activity in self.activities:
            poi_gdf = self.poi_data.get(activity)
            if poi_gdf is None or poi_gdf.empty:
                continue
            # 只有当遍历到的 activity 与过滤 POI 的 activity 匹配时，才使用过滤后的 POI
            if (subcategory_poi_gdf is not None and not subcategory_poi_gdf.empty
                    and activity == subcategory_poi_activity):
                poi_gdf = subcategory_poi_gdf
            valid_activities.append((activity, poi_gdf, poi_weight_map.get(activity, 0)))

        if not valid_activities:
            report_progress(80, "未找到可用POI，正在返回空结果")
            stats = {'mean': np.nan, 'median': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan, 'cells_with_data': 0, 'activity_stats': {}}
            return raster_data, transform, stats, {}

        # 收集每个活动类型的可达性数组用于统计
        activity_ai_arrays = {}

        print(f"[可达性计算] valid_activities: {[(a[0], len(a[1])) for a in valid_activities]}")
        
        mode_weights_norm = self._normalize_mode_weights()
        print(f"[可达性计算] 交通方式加权: {mode_weights_norm}")

        for idx, (activity, poi_gdf, w_activity) in enumerate(valid_activities, start=1):
            start = 45
            end = 80
            n_act = max(len(valid_activities), 1)
            stage_start = start + (end - start) * (idx - 1) / n_act
            stage_end = start + (end - start) * idx / n_act

            def make_activity_progress(stage_lo, stage_hi):
                def _activity_progress(pct_in_stage, status):
                    mapped = int(stage_lo + (stage_hi - stage_lo) * max(0, min(100, pct_in_stage)) / 100)
                    report_progress(mapped, status)
                return _activity_progress

            activity_progress = make_activity_progress(stage_start, stage_end)

            report_progress(stage_start, f"正在计算 {activity} 可达性（多交通方式加权）")
            ai = self.compute_activity_accessibility_mode_weighted(
                activity,
                poi_gdf,
                demand_coords,
                demand_pop,
                demand_pop_valid,
                resolution=resolution,
                mode_weights=mode_weights_norm,
                travel_time_min=getattr(self, 'travel_time_minutes', None),
                subcategory_keyword=subcategory_keyword,
                beta=beta,
                progress_callback=activity_progress,
            )
            print(f"[可达性计算] {activity} 完成，非零值: {np.sum(ai > 0)}/{len(ai)}")

            total_access += ai * w_activity
            total_weight += w_activity
            activity_ai_arrays[activity] = ai

        if total_weight > 0:
            total_access = total_access / total_weight

        # 计算每个活动类型的可达性统计
        rows, cols = raster_data.shape
        activity_stats = {}
        activity_raster_data = {}  # 每个活动类型的 2D 栅格数组
        for activity, ai_vals in activity_ai_arrays.items():
            valid_ai = ai_vals[ai_vals > 0]
            if len(valid_ai) > 0:
                activity_stats[activity] = {
                    'mean': float(np.mean(valid_ai)),
                    'median': float(np.median(valid_ai)),
                    'max': float(np.max(valid_ai)),
                    'min': float(np.min(valid_ai)),
                    'count': int(len(valid_ai)),
                }
            # 构建该活动类型的 2D 栅格
            act_raster = np.full((rows, cols), np.nan, dtype=float)
            for idx, (i, j, _, _) in enumerate(demand_cells):
                act_raster[j, i] = ai_vals[idx]
            activity_raster_data[activity] = act_raster

        report_progress(80, "正在回填栅格结果")
        for idx, (i, j, _, _) in enumerate(demand_cells):
            raster_data[j, i] = total_access[idx]

        valid_data = raster_data[~np.isnan(raster_data)]
        stats = {
            'mean': float(np.mean(valid_data)) if len(valid_data) else np.nan,
            'median': float(np.median(valid_data)) if len(valid_data) else np.nan,
            'std': float(np.std(valid_data)) if len(valid_data) else np.nan,
            'min': float(np.min(valid_data)) if len(valid_data) else np.nan,
            'max': float(np.max(valid_data)) if len(valid_data) else np.nan,
            'cells_with_data': int(len(valid_data)),
            'activity_stats': activity_stats,
        }
        transform = from_origin(self.bbox[0], self.bbox[3], resolution, resolution)
        print("G2SFCA 计算完成")
        return raster_data, transform, stats, activity_raster_data

    @staticmethod
    def _cache_token(text):
        import re
        s = str(text or '').strip()
        if not s:
            return 'all'
        return re.sub(r'[^\w\u4e00-\u9fff\-]+', '_', s)[:80] or 'all'

    def _get_cache_dir(self, resolution=None):
        """基于位置缓存目录：POI/{城市}/Location-based/resol_{分辨率米}/"""
        base = os.path.dirname(os.path.abspath(__file__))
        city_folder_map = {
            '上海': 'Shanghai',
            '南京': 'Nanjing',
            '武汉': 'Wuhan',
            '北京': 'Beijing',
        }
        city_folder = city_folder_map.get(self.city, 'Nanjing')
        cache_dir = os.path.join(base, 'POI', city_folder, 'Location-based')
        if resolution is not None:
            resolution_m = int(round(resolution_to_meters(resolution)))
            cache_dir = os.path.join(cache_dir, f'resol_{resolution_m}')
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def _get_cache_path(
        self,
        activity,
        subcategory_keyword=None,
        travel_time_min=None,
        transport_mode=None,
        resolution=None,
    ):
        """
        缓存键：POI 类型 + 子类别 + 出行时间(分钟) + 交通方式；
        按分辨率存入 resol_{米} 子目录。
        文件名示例：work_高等院校_t30_metro.npz
        """
        from transport_modes import normalize_mode_key
        mode_tag = normalize_mode_key(transport_mode or getattr(self, 'transport_mode', 'foot'))
        tt = int(travel_time_min if travel_time_min is not None else getattr(self, 'travel_time_minutes', 20))
        sub_tag = self._cache_token(subcategory_keyword)
        filename = f'{activity}_{sub_tag}_t{tt}_{mode_tag}.npz'
        return os.path.join(self._get_cache_dir(resolution), filename)

    def _load_cached_activity(
        self,
        activity,
        demand_coords,
        subcategory_keyword=None,
        travel_time_min=None,
        transport_mode=None,
        resolution=None,
    ):
        """尝试从磁盘加载缓存（键：POI类型+子类别+出行时间+交通方式；并校验网格分辨率）。"""
        path = self._get_cache_path(
            activity, subcategory_keyword, travel_time_min, transport_mode, resolution
        )
        if not os.path.exists(path):
            return None
        try:
            data = np.load(path, allow_pickle=True)
            cached_coords = data['demand_coords']
            if cached_coords.shape != demand_coords.shape:
                return None
            if not np.allclose(cached_coords, demand_coords, rtol=1e-5):
                return None
            ai = data['ai']
            if len(ai) != len(demand_coords):
                return None
            sub_kw = f' [{subcategory_keyword}]' if subcategory_keyword else ''
            tt = int(travel_time_min if travel_time_min is not None else getattr(self, 'travel_time_minutes', 20))
            tm = transport_mode or getattr(self, 'transport_mode', 'foot')
            print(f"  [缓存命中] {activity}{sub_kw} (t={tt}min, mode={tm}) → {os.path.basename(path)}")
            return ai
        except Exception as e:
            print(f"  缓存加载失败: {e}")
            return None

    def _save_cached_activity(
        self,
        activity,
        demand_coords,
        ai,
        subcategory_keyword=None,
        travel_time_min=None,
        transport_mode=None,
        resolution=None,
    ):
        """保存活动可达性（键：POI类型+子类别+出行时间+交通方式）。"""
        path = self._get_cache_path(
            activity, subcategory_keyword, travel_time_min, transport_mode, resolution
        )
        try:
            payload = {
                'demand_coords': demand_coords.astype(np.float64),
                'ai': ai.astype(np.float32),
                'activity': activity,
                'subcategory': str(subcategory_keyword or ''),
                'travel_time_min': int(travel_time_min if travel_time_min is not None else getattr(self, 'travel_time_minutes', 20)),
                'transport_mode': str(transport_mode or getattr(self, 'transport_mode', 'foot')),
            }
            if resolution is not None:
                payload['resolution'] = np.float64(resolution)
                payload['resolution_m'] = np.float64(resolution_to_meters(resolution))
            np.savez_compressed(path, **payload)
            sub_kw = f' [{subcategory_keyword}]' if subcategory_keyword else ''
            print(f"  [已缓存] {activity}{sub_kw} → {path}")
        except Exception as e:
            print(f"  缓存保存失败: {e}")

    def save_raster(self, raster_data, transform, output_path, crs='EPSG:4326'):
        """保存栅格数据"""
        with rasterio.open(output_path, 'w',
                           driver='GTiff',
                           height=raster_data.shape[0],
                           width=raster_data.shape[1],
                           count=1,
                           dtype=raster_data.dtype,
                           crs=crs,
                           transform=transform,
                           nodata=np.nan) as dst:
            dst.write(raster_data, 1)
        print(f"栅格数据已保存至: {output_path}")

def visualize_raster(raster_data, transform, stats=None, cmap='viridis_r', title='综合可达性分布', output_path=None, ax=None):
    """可视化栅格数据"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 12), dpi=150)
    else:
        fig = ax.figure

    left = transform[2]  # 左上角x坐标
    right = transform[2] + transform[0] * raster_data.shape[1]  # 右下角x坐标
    top = transform[5]  # 左上角y坐标
    bottom = transform[5] + transform[4] * raster_data.shape[0]  # 右下角y坐标
    extent = [left, right, bottom, top]  # 注意顺序：[左, 右, 下, 上]

    # 计算更合适的色标范围（使用数据的实际范围）
    valid_data = raster_data[~np.isnan(raster_data)]
    vmin = np.percentile(valid_data, 5) if len(valid_data) > 0 else 0  # 去除极端值
    vmax = np.percentile(valid_data, 95) if len(valid_data) > 0 else 1 

    # 使用反转的viridis色彩方案，使高值显示为暖色
    # origin='lower'更符合GIS数据的显示方式
    im = ax.imshow(raster_data, extent=extent, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('可达性得分', fontsize=14)

    if stats:
        if "mean" in stats and "median" in stats:
            title += f'\n均值: {stats["mean"]:.3f}, 中位数: {stats["median"]:.3f}'
        title += f'\n范围: {stats["min"]:.3f} - {stats["max"]:.3f}'

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('经度', fontsize=14)
    ax.set_ylabel('纬度', fontsize=14)

    ax.grid(True, linestyle='--', alpha=0.3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # 如果是独立调用，则显示图形

    plt.tight_layout()
    plt.show()
    #plt.close()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存至 {output_path}")

    return fig

# 兼容旧 import 路径（数据加载已迁至 transport_modes）
from transport_modes import read_all_city_data, read_all_nanjing_data, _CITY_DATA_CACHE  # noqa: E402,F401

def main():
    """主函数"""
    print("=== 南京综合可达性分析 ===")

    networks, poi_data, population_data, pop_transform, nodes_with_population, admin_boundary, _ = read_all_nanjing_data()

    # 读取碳预算参数（可视化界面入口）
    if _PYQT5_AVAILABLE:
        default_params = CarbonBudgetParams()
        dlg = AccessibilitySettingsDialog(params=default_params)
        if dlg.exec_() == dlg.Accepted:
            carbon_params = dlg.get_params()
            print(f"碳预算参数已设置: 出行时间阈值={carbon_params.travel_time_threshold}min, "
                  f"β={carbon_params.distance_decay_beta}, "
                  f"POI权重={carbon_params.poi_weights}, "
                  f"速度={carbon_params.mode_speeds}, "
                  f"生活时间预算={carbon_params.min_living_time_budget}min, "
                  f"必需设施={carbon_params.essential_facilities}, "
                  f"兜底交通={carbon_params.fallback_mode}")
        else:
            carbon_params = default_params
    else:
        carbon_params = CarbonBudgetParams()
        print("PyQt5 不可用，使用默认碳预算参数")

    calculator = NanjingAccessibilityCalculator(networks, poi_data, admin_boundary,
                                                  carbon_params=carbon_params)

    raster_data, transform, stats = calculator.calculate_composite_accessibility(resolution=0.02)

    print(f"综合可达性统计: {stats}")

    calculator.save_raster(raster_data, transform, 'nanjing_composite_accessibility.tif')

    visualize_raster(raster_data, transform, stats,
                     cmap='RdYlGn_r',  # 红色-黄色-绿色（反向，红色表示可达性差）
                     title='南京综合可达性分布')

    print("=== 分析完成 ===")

    return calculator, raster_data, transform

if __name__ == '__main__':
    calculator, test_raster, test_transform = main()
