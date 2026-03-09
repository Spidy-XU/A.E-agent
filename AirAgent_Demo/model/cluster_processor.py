"""
cluster_processor.py
输出 5 类图表：

  图表1  各城市污染等级特征雷达图   → {region}_污染等级雷达图.png   （每城市一张，按等级分子图）
  图表2  各城市污染特征综合分析图   → {region}_污染特征综合图.png   （每城市一张，双饼+堆积柱）
  图表3  城市污染分类因子贡献饼图   → {region_col}_污染6因子占比饼图.png   （3×3总图）
  图表4  城市污染分类特征雷达图     → {region_col}_污染类型雷达特征图.png   （3×3总图）
  图表5  城市污染物-污染类型箱图    → {region_col}污染物-污染类型箱图.png   （6宫格总图）

对外接口：
  generate_cluster_all(df, region_col) → dict   ← app.py 主要调用入口
  generate_cluster_radar(df, region_name)→ str  ← 旧接口兼容保留
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import os
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from model.basic_function import font, get_resource_path

# ═══════════════════════════════════════════════════════════════════════════════
# 全局配置
# ═══════════════════════════════════════════════════════════════════════════════

RADAR_POLLUTANTS = ['PM2.5', 'PMc', 'CO', 'NO2', 'SO2']       # 雷达图5因子
PIE_POLLUTANTS   = ['PM2.5', 'PMc', 'CO', 'NO2', 'SO2', 'O3'] # 饼图6因子
N_CLUSTERS = 8

# AQI 等级 → 颜色（与气象行业常用色系一致）
_AQI_LEVEL_COLORS = {
    '优':     '#00e400',
    '良':     '#ffff00',
    '轻度污染': '#ff7e00',
    '中度污染': '#ff0000',
    '重度污染': '#8f3f97',
    '严重污染': '#7e0023',
}

_PIE_COLORS = [
    "#496EA9", '#DD8452', "#52A966", "#C4464A",
    "#7D6FAC", "#9B7B5F", "#D783BF", "#8F8A8A",
]
_BOX_COLORS = [
    '#3498db', '#2ecc71', '#e74c3c', '#f39c12',
    '#9b59b6', '#1abc9c', '#e67e22', '#34495e',
]
_BAR_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def _save_dir(region_col: str, sub: str = "") -> str:
    base = get_resource_path(f"mapout/{region_col}污染分类图")
    path = os.path.join(str(base), sub) if sub else str(base)
    os.makedirs(path, exist_ok=True)
    return path


def _ts() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


# ── AQI 折算（GB 3095—2012，用于内部等级打标，后续可修改） ───────────────────────────────
_IAQI_BP = {
    'PM2.5': ([0, 35, 75, 115, 150, 250, 350, 500],
              [0, 50, 100, 150, 200, 300, 400, 500]),
    'PM10':  ([0, 50, 150, 250, 350, 420, 500, 600],
              [0, 50, 100, 150, 200, 300, 400, 500]),
    'SO2':   ([0, 50, 150, 475, 800, 1600, 2100, 2620],
              [0, 50, 100, 150, 200, 300,  400,  500]),
    'NO2':   ([0, 40, 80, 180, 280, 565, 750, 940],
              [0, 50, 100, 150, 200, 300, 400, 500]),
    'CO':    ([0, 2, 4, 14, 24, 36, 48, 60],
              [0, 50, 100, 150, 200, 300, 400, 500]),
    'O3':    ([0, 100, 160, 215, 265, 800],
              [0, 50,  100, 150, 200, 300]),
}

def _iaqi(pollutant: str, conc: float) -> float:
    cp_list, iaqi_list = _IAQI_BP[pollutant]
    if conc <= 0:
        return 0.0
    for i in range(len(cp_list) - 1):
        if cp_list[i] <= conc <= cp_list[i + 1]:
            cp_lo, cp_hi = cp_list[i], cp_list[i + 1]
            ia_lo, ia_hi = iaqi_list[i], iaqi_list[i + 1]
            return (ia_hi - ia_lo) / (cp_hi - cp_lo) * (conc - cp_lo) + ia_lo
    return float(iaqi_list[-1])


def _aqi_grade(aqi: float) -> str:
    if aqi <= 50:   return '优'
    if aqi <= 100:  return '良'
    if aqi <= 150:  return '轻度污染'
    if aqi <= 200:  return '中度污染'
    if aqi <= 300:  return '重度污染'
    return '严重污染'


def _calc_daily_aqi_grade(df: pd.DataFrame, region_col: str) -> pd.DataFrame:
    """
    从小时级原始浓度数据计算每个区域每天的 AQI 等级。
    返回 DataFrame，列为 [region_col, '日期', '等级']。
    """
    df = df.copy()
    df['时间'] = pd.to_datetime(df['时间'])
    df['_date'] = df['时间'].dt.date

    pollutants_needed = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    valid_p = [p for p in pollutants_needed if p in df.columns]

    records = []
    for region in df[region_col].unique():
        rdf = df[df[region_col] == region].copy()
        daily = rdf.groupby('_date')[valid_p].mean()
        for date, row in daily.iterrows():
            iaqi_vals = []
            for p in valid_p:
                if pd.notna(row[p]):
                    iaqi_vals.append(_iaqi(p, row[p]))
            aqi = max(iaqi_vals) if iaqi_vals else 0
            records.append({region_col: region, '日期': date.strftime('%m-%d'),
                             '等级': _aqi_grade(aqi)})
    return pd.DataFrame(records)


# ── 污染源识别 ─────────────────

def _identify_pollution_source(center_vals: np.ndarray) -> str:
    """基于标准化聚类中心推断污染源类型"""
    pm25, pmc, co, no2, so2 = center_vals[:5]
    if pm25 > 0.8 and no2 > 0.5 and so2 > 0.3:
        return "偏燃煤型（燃煤+热力）"
    if pm25 > 0.8 and so2 > 0.5 and co > 0.3:
        return "偏燃煤型（纯燃煤）"
    if pm25 > 0.6 and no2 > 0.6 and co > 0.4:
        return "偏机动车型（高浓度尾气）"
    if pmc > 0.8 and no2 > 0.5:
        return "偏综合型（PMc+NO2）"
    if pmc > 0.6 and pm25 < 0.2:
        return "偏沙尘型（纯沙尘）"
    if pm25 > 0.5 and no2 > 0.3 and so2 > 0.2 and co > 0.2:
        return "偏综合型（混合污染）"
    if max(pm25, pmc, co, no2, so2) > 0.3:
        return "偏综合型（中浓度混合）"
    return "偏综合型（低浓度背景）"


def _identify_source_normalized(vals: np.ndarray) -> str:
    """
    基于已完成 z-score 标准化的5因子均值向量推断污染源类型
    （用于等级雷达图和饼图）
    """
    pm25, pmc, co, no2, so2 = vals[:5]
    # 全部低于 -0.3：清洁
    if max(pm25, pmc, co, no2, so2) < -0.3:
        return "显著清洁型"
    if pm25 > 0.5 and so2 > 0.3 and co > 0.3:
        return "偏燃煤型"
    if pm25 > 0.5 and no2 > 0.3 and co > 0.2:
        return "偏机动车型"
    if pmc > 0.5 and pm25 < 0:
        return "偏扬尘型"
    if pm25 > 0.3 and no2 > 0.2 and so2 > 0.1:
        return "偏综合型"
    return "偏综合型"


# ═══════════════════════════════════════════════════════════════════════════════
# 基础绘图组件
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_radar(ax, values, labels, title):
    """极坐标填充雷达图（5因子固定，绿色背景）"""
    plot_labels = list(labels)[:5]
    plot_values = np.array(values[:5], dtype=float)
    n_vars = len(plot_labels)

    angles        = np.linspace(0, 2 * np.pi, n_vars, endpoint=False)
    angles_closed = np.concatenate([angles, [angles[0]]])
    vals_closed   = np.concatenate([plot_values, [plot_values[0]]])

    val_max = float(plot_values.max())
    val_min = float(plot_values.min())

    bg = mpatches.Circle((0, 0), val_max + 2, facecolor='#e8f5e9', alpha=0.9, zorder=0)
    ax.add_patch(bg)

    ax.plot(angles_closed, vals_closed, 'o-',
            linewidth=1.5, color='#1f77b4', alpha=0.8, zorder=3, markersize=3)
    ax.fill(angles_closed, vals_closed, color='#1f77b4', alpha=0.4, zorder=2)

    ax.set_xticks(angles)
    ax.set_xticklabels(plot_labels, fontsize=12)
    ax.set_yticklabels([])
    ax.grid(True, axis='y', alpha=0.8, linestyle='--', zorder=1)
    ax.grid(True, axis='x', alpha=0.8, linestyle='-',  zorder=1)
    ax.set_title(title, fontsize=15, pad=15)
    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_linewidth(1.2)
    ax.spines['polar'].set_color('#000000')
    ax.spines['polar'].set_alpha(0.8)
    ax.set_ylim(min(0, val_min - 0.2), val_max + 0.2)
    if ax.legend_:
        ax.legend_.remove()


def _plot_pie(ax, data, labels, title, is_percent=True):
    """带白色旋转百分比的饼图"""
    data   = np.array(data, dtype=float)
    mask   = (data > 0) & (~np.isnan(data))
    vdata  = data[mask]
    vlabels = [labels[i] for i in range(len(labels)) if mask[i]]

    if len(vdata) == 0:
        ax.text(0.5, 0.5, "无有效数据", ha='center', va='center', fontsize=12)
        ax.set_title(title, fontsize=15, pad=10)
        return

    colors = (_PIE_COLORS * ((len(vdata) // len(_PIE_COLORS)) + 1))[:len(vdata)]

    wedges, texts, autotexts = ax.pie(
        vdata, labels=vlabels,
        autopct='%1.1f%%' if is_percent else '%1.0f',
        textprops={'fontsize': 12},
        startangle=90,
        colors=colors,
        explode=[0.05] * len(vdata),
        labeldistance=1.1
    )
    ax.set_title(title, fontsize=15, pad=10)

    for autotext, wedge in zip(autotexts, wedges):
        autotext.set_fontsize(12)
        autotext.set_color('white')
        ang = (wedge.theta2 + wedge.theta1) / 2
        if 90 <= ang <= 270:
            ang += 180
        autotext.set_rotation_mode("anchor")
        autotext.set_rotation(ang)
        autotext.set_horizontalalignment('center')
        autotext.set_verticalalignment('center')


# ═══════════════════════════════════════════════════════════════════════════════
# 核心：数据预处理 + KMeans 聚类
# ═══════════════════════════════════════════════════════════════════════════════

def _prepare_and_cluster(df: pd.DataFrame, region_col: str):
    """
    完整复现 julei.py fenlei_plot 数据处理段：
      1. 按区域→小时均值→日质控（≥20小时）
      2. PMc = PM10-PM2.5 ; 各因子/限值 ; PM2.5×1.5
      3. KMeans(8) 聚类 + 污染源命名
    返回 (data, centers_df, pollution_names)
    """
    raw_pollutants = ['PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3']
    divide_coeffs  = [35,      70,     4,    40,    60,    160]
    coeff_map      = dict(zip(raw_pollutants, divide_coeffs))

    df = df.copy()
    df['时间'] = pd.to_datetime(df['时间'])

    frames = []
    for region in df[region_col].unique():
        rdf = df[df[region_col] == region].copy()
        rdf = rdf.groupby('时间')[raw_pollutants].mean()
        rdf['日期'] = rdf.index.date
        valid_dates = rdf.groupby('日期').size()
        valid_dates = valid_dates[valid_dates >= 20].index
        filtered = rdf[rdf['日期'].isin(valid_dates)].drop(columns=['日期'])
        filtered[region_col] = region
        filtered['日期'] = filtered.index
        frames.append(filtered)

    data = pd.concat(frames, ignore_index=True)
    data.reset_index(drop=True, inplace=True)
    data = data.dropna(subset=raw_pollutants)
    data = data[~(data[raw_pollutants] == 0).all(axis=1)]

    data['PM10'] = data['PM10'] - data['PM2.5']
    for col, coeff in coeff_map.items():
        data[col] = data[col] / coeff
    data.rename(columns={'PM10': 'PMc'}, inplace=True)
    data['PM2.5'] = data['PM2.5'] * 1.5

    if len(data) < N_CLUSTERS * 2:
        return None, None, None

    X = data[RADAR_POLLUTANTS].copy()
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans   = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    data['聚类标签'] = kmeans.fit_predict(X_scaled)

    centers_scaled   = kmeans.cluster_centers_
    cluster_map      = {}
    pollution_names  = []
    for i in range(N_CLUSTERS):
        name = _identify_pollution_source(centers_scaled[i])
        cluster_map[i] = name
        pollution_names.append(name)

    centers_df = pd.DataFrame(centers_scaled, columns=RADAR_POLLUTANTS,
                               index=pollution_names)
    data['污染类型'] = data['聚类标签'].map(cluster_map)

    return data, centers_df, pollution_names


# ═══════════════════════════════════════════════════════════════════════════════
# 图表 4：城市污染分类特征雷达图（3×3，8雷达+1占比饼）
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_type_radar_grid(data, centers_df, pollution_names,
                          region_col, save_dir) -> str:
    """对应 julei.py fenlei_plot 可视化1"""
    font()
    type_counts = data['污染类型'].value_counts()

    fig = plt.figure(figsize=(15, 12), dpi=100)
    fig.suptitle(f'{region_col}污染分类特征雷达图', fontsize=20, y=0.98)

    for i in range(8):
        ax = fig.add_subplot(3, 3, i + 1, projection='polar')
        if i < len(pollution_names):
            _plot_radar(ax, centers_df.iloc[i].values, RADAR_POLLUTANTS, pollution_names[i])
        else:
            ax.axis('off')

    ax9 = fig.add_subplot(3, 3, 9)
    _plot_pie(ax9, type_counts.values, list(type_counts.index), '各类型污染时间占比统计')

    plt.tight_layout()
    path = os.path.join(save_dir, f"{region_col}_污染类型雷达特征图.png")
    plt.savefig(path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# 图表 3：城市污染分类因子贡献饼图（3×3，8因子饼+1占比饼）
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_factor_pie_grid(data, pollution_names, region_col, save_dir) -> str:
    """对应 julei.py fenlei_plot 可视化2"""
    font()
    type_counts    = data['污染类型'].value_counts()
    type_mean_6fac = data.groupby('污染类型')[PIE_POLLUTANTS].mean()

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), dpi=100)
    fig.suptitle(f'{region_col}污染分类因子综合指数贡献饼图', fontsize=20, y=0.98)
    axes = axes.flatten()

    for i in range(8):
        if i < len(pollution_names):
            pname = pollution_names[i]
            if pname in type_mean_6fac.index:
                _plot_pie(axes[i], type_mean_6fac.loc[pname].values,
                          PIE_POLLUTANTS, f'{pname}\n因子贡献比')
            else:
                axes[i].text(0.5, 0.5, "无数据", ha='center', va='center', fontsize=10)
                axes[i].set_title(pname, fontsize=9)
        else:
            axes[i].axis('off')

    _plot_pie(axes[8], type_counts.values, list(type_counts.index), '各类型污染时间占比')

    plt.tight_layout()
    path = os.path.join(save_dir, f"{region_col}_污染6因子占比饼图.png")
    plt.savefig(path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# 图表 5：城市污染物-污染类型箱线图（6宫格）
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_box_grid(data, region_col, save_dir) -> str:
    """对应 julei.py juleibox_plot"""
    font()
    pollutant_cols = ['SO2', 'NO2', 'CO', 'O3', 'PMc', 'PM2.5']
    data = data.copy()
    data["污染类型"] = data["污染类型"].astype("category")

    unique_types  = sorted(data["污染类型"].unique())
    n_types       = len(unique_types)
    type_seq_map  = {t: i + 1 for i, t in enumerate(unique_types)}
    type_name_map = {i + 1: str(t) for i, t in enumerate(unique_types)}
    palette       = (_BOX_COLORS * 2)[:n_types]

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.patch.set_facecolor('#f8f9fa')

    for idx, pollutant in enumerate(pollutant_cols):
        ax = axes[idx // 2, idx % 2]
        ax.set_facecolor('#ffffff')

        box_data, box_labels = [], []
        for t in unique_types:
            vals = data[data["污染类型"] == t][pollutant].dropna().values
            if len(vals) > 0:
                box_data.append(vals)
                box_labels.append(str(type_seq_map[t]))

        if box_data:
            bp = ax.boxplot(box_data, patch_artist=True, labels=box_labels)
            for patch, color in zip(bp['boxes'], palette[:len(box_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            plt.setp(bp['whiskers'], color='#2c3e50', linewidth=1.5)
            plt.setp(bp['caps'],     color='#2c3e50', linewidth=1.5)
            plt.setp(bp['medians'],  color='#e74c3c', linewidth=2.5)
            plt.setp(bp['fliers'],   marker='o', color='#f39c12', alpha=0.7, markersize=4)

        ax.set_title(f'{pollutant}标准化指数分布', fontsize=18,
                     color=_BOX_COLORS[idx % len(_BOX_COLORS)], pad=15)
        ax.set_ylabel(f'{pollutant}标准化指数', fontsize=12, color='#34495e')
        ax.set_xticklabels(box_labels, rotation=0, fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#bdc3c7')
        ax.spines['bottom'].set_color('#bdc3c7')
        ax.grid(True, alpha=0.5, linestyle='--', color='#ecf0f1')
        ax.set_axisbelow(True)
        ax.tick_params(axis='x', labelsize=11, colors='#2c3e50')
        ax.tick_params(axis='y', labelsize=10, colors='#2c3e50')

    fig.suptitle(f'{region_col}各污染类型污染物标准化指数分布',
                 fontsize=25, color='#2c3e50', y=0.98)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=palette[i], alpha=0.7,
                       label=f'{type_seq_map[t]} - {type_name_map[type_seq_map[t]]}')
        for i, t in enumerate(unique_types)
    ]
    n_col = max(1, (n_types + 1) // 2)
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.05),
               fontsize=11, ncol=n_col, frameon=True, fancybox=True)

    plt.tight_layout(rect=[0, 0.08, 1, 0.98])

    path = os.path.join(save_dir, f"{region_col}污染物-污染类型箱图.png")
    plt.savefig(path, dpi=400, facecolor='#f8f9fa', edgecolor='none')
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# 图表 2：各城市污染特征综合分析图（双饼 + 小时级堆积柱）
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_region_composite(data, region_col, sub_dir) -> list:
    """对应 julei.py juleibar_plot，每个子区域一张图"""
    font()
    bar_pie_pollutants = ['PM2.5', 'PMc', 'NO2', 'SO2', 'O3', 'CO']
    bar_color_map = dict(zip(bar_pie_pollutants, _BAR_COLORS))
    paths = []

    for region in data[region_col].unique():
        rdf = data[data[region_col] == region].copy()
        rdf['日期'] = pd.to_datetime(rdf['日期'], errors='coerce')
        rdf = rdf.set_index('日期')
        avail_p = [p for p in bar_pie_pollutants if p in rdf.columns]
        rdf = rdf[avail_p + ['污染类型']].dropna()
        if len(rdf) == 0:
            continue

        # 饼图1：原始顺序污染物指数贡献
        pie1_data   = rdf[avail_p].mean().values
        pie1_labels = avail_p

        # 饼图2：污染类型时间占比（大小间隔排序）
        type_counts   = rdf['污染类型'].value_counts()
        s_labels      = type_counts.index.tolist()
        s_vals        = type_counts.values.tolist()
        inter_l, inter_v = [], []
        l, r, turn = 0, len(s_labels) - 1, True
        while l <= r:
            if turn:
                inter_l.append(s_labels[l]); inter_v.append(s_vals[l]); l += 1
            else:
                inter_l.append(s_labels[r]); inter_v.append(s_vals[r]); r -= 1
            turn = not turn

        # 柱状图：按均值贡献降序排列
        mean_vals       = rdf[avail_p].mean().sort_values(ascending=False)
        sorted_p        = mean_vals.index.tolist()
        sorted_colors   = [bar_color_map.get(p, '#999999') for p in sorted_p]
        hour_data       = rdf[sorted_p]
        x_vals          = np.arange(len(hour_data))
        x_dates         = hour_data.index

        # 均匀取 ≤10 刻度
        n_total = len(x_dates)
        if n_total <= 10:
            tick_pos    = list(range(n_total))
            tick_labels = [d.strftime('%Y-%m-%d') for d in x_dates]
        else:
            step     = max(1, n_total // 10)
            tick_pos = list(range(0, n_total, step))[:10]
            if tick_pos[-1] != n_total - 1:
                tick_pos.append(n_total - 1)
            tick_pos    = tick_pos[:10]
            tick_labels = [x_dates[i].strftime('%Y-%m-%d') for i in tick_pos]

        gs  = gridspec.GridSpec(3, 4, height_ratios=[1, 1, 2], width_ratios=[1, 1, 1, 1])
        fig = plt.figure(figsize=(20, 8), constrained_layout=False)

        ax_pie1 = fig.add_subplot(gs[0:2, 0:2])
        ax_pie2 = fig.add_subplot(gs[0:2, 2:4])
        ax_bar  = fig.add_subplot(gs[2, :])

        _plot_pie(ax_pie1, pie1_data,  pie1_labels, '污染物指数贡献占比')
        _plot_pie(ax_pie2, inter_v,    inter_l,     '污染类型时间占比')

        bottom = np.zeros(len(hour_data))
        for i, p in enumerate(sorted_p):
            vals = hour_data[p].values
            ax_bar.bar(x_vals, vals, bottom=bottom, label=p,
                       color=sorted_colors[i], alpha=0.8, width=0.8)
            bottom += vals

        ax_bar.set_xticks(tick_pos)
        ax_bar.set_xticklabels(tick_labels, fontsize=10, rotation=0)
        ax_bar.set_title('污染物标准化指数小时级分布', fontsize=16, pad=15)
        ax_bar.set_xlabel('日期', fontsize=14)
        ax_bar.set_ylabel('标准化指数', fontsize=14)
        ax_bar.tick_params(axis='y', labelsize=10)
        ax_bar.legend(loc='upper right', fontsize=10, ncol=6)
        ax_bar.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        ax_bar.set_xlim(-0.5, len(x_vals) - 0.5)

        fig.suptitle(f'{region} 污染特征综合分析', fontsize=20, y=0.98)

        safe = str(region).replace('/', '_').replace('\\', '_')
        path = os.path.join(sub_dir, f"{safe}_污染特征综合图.png")
        plt.savefig(path, dpi=400, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        paths.append(path)

    return paths


# ═══════════════════════════════════════════════════════════════════════════════
# 图表 1：各城市污染等级特征雷达图（每城市一张，按等级分子图）
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_grade_radar(df_raw, region_col, grade_df=None, sub_dir=None) -> list:
    """
    对应 julei.py youjiang_plot。
    AQI 等级直接从小时级浓度数据折算，无需外部 grade_df，
    避免日期格式不匹配导致 merge 失败的 KeyError。
    """
    font()
    raw_pollutants = ['PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3']
    divide_coeffs  = [35,      70,     4,    40,    60,    160]
    coeff_map      = dict(zip(raw_pollutants, divide_coeffs))
    final_cols     = ['PM2.5', 'PMc', 'CO', 'NO2', 'SO2']

    df = df_raw.copy()
    df['时间'] = pd.to_datetime(df['时间'])

    # ── 直接从浓度数据按日折算 AQI 等级 ──────────────────────────────────────
    valid_p = [p for p in ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3'] if p in df.columns]
    df['_date'] = df['时间'].dt.date

    daily_mean = df.groupby([region_col, '_date'])[valid_p].mean().reset_index()

    def _row_grade(row):
        iaqi_vals = [_iaqi(p, row[p]) for p in valid_p if pd.notna(row.get(p))]
        aqi = max(iaqi_vals) if iaqi_vals else 0
        return _aqi_grade(aqi)

    daily_mean['等级'] = daily_mean.apply(_row_grade, axis=1)
    grade_map = {(r[region_col], r['_date']): r['等级']
                 for _, r in daily_mean.iterrows()}

    df['等级'] = df.apply(lambda r: grade_map.get((r[region_col], r['_date']), np.nan), axis=1)
    df = df.drop(columns=['_date']).dropna(subset=['等级'])

    # 数据预处理
    frames = []
    for region in df[region_col].unique():
        rdf = df[df[region_col] == region].copy()
        rdf = rdf.groupby(['时间', '等级'])[raw_pollutants].mean().reset_index()
        rdf.set_index('时间', inplace=True)
        rdf['日期'] = rdf.index.date
        valid_dates = rdf.groupby('日期').size()
        valid_dates = valid_dates[valid_dates >= 20].index
        filtered = rdf[rdf['日期'].isin(valid_dates)].drop(columns=['日期'])
        filtered[region_col] = region
        frames.append(filtered)

    if not frames:
        return []
    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=raw_pollutants)
    data = data[~(data[raw_pollutants] == 0).all(axis=1)]
    data['PM10'] = data['PM10'] - data['PM2.5']
    for col, coeff in coeff_map.items():
        data[col] = data[col] / coeff
    data.rename(columns={'PM10': 'PMc'}, inplace=True)
    data['PM2.5'] = data['PM2.5'] * 1.5

    # z-score 标准化
    sddf = data.copy()
    scaler = StandardScaler()
    sddf[final_cols] = scaler.fit_transform(sddf[final_cols])

    paths = []
    for region in sddf[region_col].unique():
        qdf = sddf[sddf[region_col] == region]
        if qdf.empty:
            continue

        unique_levels = qdf['等级'].unique()
        n_levels = len(unique_levels)
        if n_levels == 0:
            continue

        # 动态布局
        if n_levels <= 4:
            n_rows, n_cols = 2, 2
        elif n_levels <= 6:
            n_rows, n_cols = 2, 3
        else:
            n_rows = (n_levels + 2) // 3
            n_cols = 3

        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(4 * n_cols, 4 * n_rows),
                                  subplot_kw=dict(projection='polar'))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, level in enumerate(unique_levels):
            level_data = qdf[qdf['等级'] == level]
            if level_data.empty:
                continue
            mean_vals = level_data[final_cols].mean().values
            leixing   = _identify_source_normalized(mean_vals)
            _plot_radar(axes[idx], mean_vals, final_cols,
                        f"{level}等级\n{leixing}")

        for idx in range(n_levels, len(axes)):
            fig.delaxes(axes[idx])

        fig.suptitle(f"{region} 各污染等级污染特征雷达图", fontsize=20)
        plt.tight_layout()
        plt.subplots_adjust(top=0.82)

        safe = str(region).replace('/', '_').replace('\\', '_')
        path = os.path.join(sub_dir, f"{safe}_污染等级雷达图.png")
        plt.savefig(path, dpi=400, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        paths.append(path)

    return paths


# ═══════════════════════════════════════════════════════════════════════════════
# 对外主接口
# ═══════════════════════════════════════════════════════════════════════════════

def generate_cluster_all(df: pd.DataFrame, region_col: str) -> dict:
    """
    一次性生成全部 5 类聚类分析图表，返回路径字典：
      {
        'grade_radar':  list[str],  # 图表1：各城市污染等级特征雷达图
        'composite':    list[str],  # 图表2：各城市污染特征综合分析图
        'factor_pie':   str,        # 图表3：城市污染分类因子贡献饼图(3×3)
        'type_radar':   str,        # 图表4：城市污染分类特征雷达图(3×3)
        'box':          str,        # 图表5：城市污染物-污染类型箱图
      }
    """
    font()
    data, centers_df, pollution_names = _prepare_and_cluster(df, region_col)

    if data is None:
        return {k: ([] if k in ('grade_radar', 'composite') else None)
                for k in ('grade_radar', 'composite', 'factor_pie', 'type_radar', 'box')}

    save_dir = _save_dir(region_col)
    sub_dir  = _save_dir(region_col, region_col)

    # 图表 3、4（总图）
    type_radar_path = _plot_type_radar_grid(
        data, centers_df, pollution_names, region_col, save_dir)
    factor_pie_path = _plot_factor_pie_grid(
        data, pollution_names, region_col, save_dir)

    # 图表 5（总图）
    box_path = _plot_box_grid(data, region_col, save_dir)

    # 图表 2（各子区域）
    composite_paths = _plot_region_composite(data, region_col, sub_dir)

    # 图表 1（各子区域等级雷达）：直接从原始数据折算等级
    grade_paths = _plot_grade_radar(df, region_col, sub_dir=sub_dir)

    return {
        'grade_radar':  grade_paths,
        'composite':    composite_paths,
        'factor_pie':   factor_pie_path,
        'type_radar':   type_radar_path,
        'box':          box_path,
    }


# ─── 旧接口兼容（保留签名，生成单区域类型雷达图） ────────────────────────────

def generate_cluster_radar(df: pd.DataFrame, region_name: str,
                           n_clusters: int = N_CLUSTERS) -> str:
    """旧接口保留，建议升级至 generate_cluster_all。"""
    font()
    raw_p  = ['PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3']
    coeffs = [35, 70, 4, 40, 60, 160]
    data   = df[raw_p].dropna().copy()
    if len(data) < n_clusters * 2:
        return None

    for col, c in zip(raw_p, coeffs):
        data[col] = data[col] / c
    data['PM10'] = (data['PM10'] - data['PM2.5']).clip(lower=0)
    data.rename(columns={'PM10': 'PMc'}, inplace=True)
    data['PM2.5'] = data['PM2.5'] * 1.5

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(data[RADAR_POLLUTANTS])
    kmeans   = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels   = kmeans.fit_predict(X_scaled)

    centers    = pd.DataFrame(kmeans.cluster_centers_, columns=RADAR_POLLUTANTS)
    pol_names  = [_identify_pollution_source(centers.iloc[i].values) for i in range(n_clusters)]
    type_cnts  = pd.Series(labels).map(dict(enumerate(pol_names))).value_counts()

    fig = plt.figure(figsize=(15, 12), dpi=100)
    fig.suptitle(f'【{region_name}】大气污染溯源与特征聚类雷达图', fontsize=18, y=0.98)
    for i in range(8):
        ax = fig.add_subplot(3, 3, i + 1, projection='polar')
        if i < len(pol_names):
            _plot_radar(ax, centers.iloc[i].values, RADAR_POLLUTANTS, pol_names[i])
        else:
            ax.axis('off')
    ax9 = fig.add_subplot(3, 3, 9)
    _plot_pie(ax9, type_cnts.values, list(type_cnts.index), '各类型时间占比')

    plt.tight_layout()
    save_dir = _save_dir("Cluster_Agent")
    path = os.path.join(save_dir, f"Cluster_{region_name}_{_ts()}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path

# ═══════════════════════════════════════════════════════════════════════════════
# 公开工具接口：为外部模块（如 anomaly_processor）提供污染类型打标
# ═══════════════════════════════════════════════════════════════════════════════

def label_pollution_types(df: pd.DataFrame, region_col: str) -> pd.DataFrame:
    """
    对原始小时级 df 运行 KMeans 聚类，为每条记录打上「污染类型」标签。
    返回新增了「污染类型」列的 DataFrame（与输入行数相同，无法打标的行填「未知类型」）。

    供 anomaly_processor.py 调用，实现 AQI 柱状图按污染类型配色。

    处理逻辑：
      1. 调用 _prepare_and_cluster 得到带污染类型的小时聚合数据
      2. 以 (region_col, datetime_hour) 为 key 建立映射字典
      3. 将映射回写到原始 df 每一行（以小时为精度匹配）
    """
    df_out = df.copy()
    df_out['时间'] = pd.to_datetime(df_out['时间'], errors='coerce')
    df_out['_hour'] = df_out['时间'].dt.floor('H')

    try:
        clustered, _, _ = _prepare_and_cluster(df, region_col)
    except Exception:
        clustered = None

    if clustered is None:
        df_out['污染类型'] = '未知类型'
        return df_out.drop(columns=['_hour'])

    # 建立 (region, datetime_hour) → 污染类型 的映射
    clustered['_hour'] = pd.to_datetime(clustered['日期']).dt.floor('H')
    type_map = {
        (row[region_col], row['_hour']): row['污染类型']
        for _, row in clustered[[region_col, '_hour', '污染类型']].iterrows()
    }

    df_out['污染类型'] = df_out.apply(
        lambda r: type_map.get((r[region_col], r['_hour']), '未知类型'), axis=1
    )
    return df_out.drop(columns=['_hour'])