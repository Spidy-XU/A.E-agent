import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from model.basic_function import font, get_resource_path

# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from model.basic_function import font, get_resource_path

# ==========================================
# 智能污染类型分类器
# 逻辑来源：basic_function.identify_pollution_source / identify_pollution_source_average_normalized
# 输入：各污染物原始浓度均值（SO2/NO2/O3/PMc/PM2.5 单位 μg/m³，CO 单位 mg/m³）
# ==========================================
def classify_pollution_type(means: pd.Series) -> str:
    """
    基于各污染物平均浓度与国家标准限值之比，判定污染来源类型。
    输入：means —— 包含 SO2, NO2, CO, O3, PMc, PM2.5 的原始浓度均值 Series
    逻辑参考：identify_pollution_source_average_normalized / identify_pollution_source
    """
    # ---------- 标准限值 (μg/m³，CO 单位 mg/m³) ----------
    limits = {
        'SO2':  60,
        'NO2':  40,
        'CO':    4,
        'O3':  160,
        'PMc':  30,
        'PM2.5': 30,
    }

    # ---------- 计算各指标超标比（ratio > 1 表示超标） ----------
    def r(key):
        val = means.get(key, np.nan)
        if pd.isna(val):
            return 0.0
        return float(val) / limits[key]

    so2  = r('SO2')
    no2  = r('NO2')
    co   = r('CO')
    o3   = r('O3')
    pmc  = r('PMc')
    pm25 = r('PM2.5')

    all_ratios   = [so2, no2, co, o3, pmc, pm25]
    max_ratio    = max(all_ratios)
    mean_ratio   = float(np.mean(all_ratios))
    no2_co_mean  = (no2 + co) / 2

    # ===== 第1优先级：整体清洁 =====
    if max_ratio < 0.4:
        return "清洁背景"
    if max_ratio < 0.7 and mean_ratio < 0.5:
        return "轻度清洁型"

    # ===== 第2优先级：O3 光化学污染 =====
    if o3 > 1.0 and o3 >= max(so2, no2, co, pmc, pm25) * 1.3:
        if no2 >= 0.6:
            return "光化学烟雾型（NOx-O3）"
        return "典型光化学污染型"

    # ===== 第3优先级：单一主导污染源 =====

    # 燃煤型（SO2 显著主导）
    if so2 > 1.0 and so2 >= max(no2, co, pmc, pm25) * 1.5:
        if no2 >= 0.6 or co >= 0.6:
            return "典型燃煤工业型"
        return "典型燃煤型"

    if so2 >= 0.7 and so2 >= max(no2, co, pmc, pm25) * 1.3:
        if no2 >= 0.5:
            return "燃煤工业复合型"
        return "燃煤型"

    # 机动车型（NO2 + CO 联合主导）
    if no2 >= 0.8 and co >= 0.6:
        if no2_co_mean >= max(so2, pmc, pm25) * 1.4:
            if no2_co_mean > 1.0:
                return "高浓度机动车型"
            return "机动车型"

    # 沙尘 / 扬尘型（PMc 极端主导）
    if pmc > 1.0 and pmc >= pm25 * 1.8:
        return "沙尘型"

    if pmc >= 0.8 and pmc >= pm25 * 1.5:
        if no2 >= 0.5 or co >= 0.5:
            return "交通扬尘型"
        return "扬尘型"

    # 二次污染型（PM2.5 偏高，一次污染源低）
    if pm25 > 1.0 and max(co, no2, so2) < 0.6 and pmc < 0.6:
        return "典型二次污染型"

    if pm25 >= 0.7 and max(co, no2, so2) < 0.5:
        return "二次污染型"

    # 高 CO 燃烧型
    if co >= 0.8 and co >= max(pm25, pmc, no2, so2) * 1.4:
        if no2 >= 0.5:
            return "交通燃烧复合型"
        return "燃烧过程型"

    # ===== 第4优先级：双因子 / 复合组合 =====

    # 工业复合型（SO2 + NO2 共同偏高）
    if so2 >= 0.6 and no2 >= 0.6:
        if abs(so2 - no2) <= 0.2:
            return "典型工业复合型"
        if so2 > no2:
            return "工业燃煤偏重型"
        return "工业NOx偏重型"

    # 工业交通混合（三者均有贡献）
    if so2 >= 0.5 and no2 >= 0.5 and co >= 0.4:
        return "工业交通混合型"

    # 交通相关混合（NO2 主导，兼有 CO 或 PM2.5）
    if no2 >= 0.6 and (co >= 0.4 or pm25 >= 0.5):
        return "交通相关混合型"

    # 燃煤相关混合（SO2 主导，兼有颗粒物）
    if so2 >= 0.5 and (no2 >= 0.4 or pm25 >= 0.5):
        return "燃煤相关混合型"

    # 颗粒物复合污染（PM2.5 + PMc 均偏高）
    if pm25 >= 0.5 and pmc >= 0.5 and max(co, no2, so2) < 0.5:
        return "颗粒物复合污染"

    # ===== 第5优先级：轻度 / 兜底 =====

    # 统计哪些因子显著偏高（超标率 ≥ 0.5）
    high = []
    if pm25 >= 0.5:  high.append("PM2.5")
    if pmc  >= 0.5:  high.append("PMc")
    if co   >= 0.5:  high.append("CO")
    if no2  >= 0.5:  high.append("NO2")
    if so2  >= 0.5:  high.append("SO2")
    if o3   >= 0.5:  high.append("O3")

    if len(high) == 1:
        factor = high[0]
        ratio_val = r(factor)
        label = "偏高" if ratio_val >= 0.7 else "轻微影响"
        return f"轻度{factor}{label}"

    if len(high) == 2:
        return f"{high[0]}+{high[1]}混合型"

    if len(high) >= 3:
        if mean_ratio >= 0.7:
            return "轻度多因子混合污染"
        return "轻微多因子混合"

    if mean_ratio >= 0.5:
        return "轻度污染混合型"

    return "接近背景水平"


# ==========================================
# 图1：单区域特征雷达图 (PM10 替换为 PMc，增加分类标签)浓度图回答"相对于其他城市，哪种污染物偏高/偏低？特征图回答"污染物有没有超标？超了多少倍？"
# ==========================================
def generate_pollution_radar(df: pd.DataFrame, region_name: str, level: str = "分析区域") -> str:
    font()
    df = df.copy()

    # ---------- 数值转换 ----------
    req_cols = ['SO2', 'NO2', 'CO', 'O3', 'PM10', 'PM2.5']
    for p in req_cols:
        if p in df.columns:
            df[p] = pd.to_numeric(df[p], errors='coerce')

    if 'PM10' in df.columns and 'PM2.5' in df.columns:
        df['PMc'] = (df['PM10'] - df['PM2.5']).clip(lower=0)
    else:
        return None

    # ---------- 污染物顺序 & 国标限值 ----------
    pollutants = ['SO2', 'NO2', 'CO', 'O3', 'PMc', 'PM2.5']
    limits = {'SO2': 60, 'NO2': 40, 'CO': 4, 'O3': 160, 'PMc': 35, 'PM2.5': 35}

    means = df[pollutants].mean()
    if means.isna().all():
        return None

    # ---------- 污染类型分类 ----------
    pollution_type = classify_pollution_type(means)

    # ---------- 归一化为超标比值（ratio = 浓度 / 国标限值） ----------
    N = len(pollutants)
    values = [float(means[p]) / limits[p] if pd.notna(means.get(p)) else 0.0 for p in pollutants]
    values_closed = values + values[:1]

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    # ---------- 配色方案（与 feature_analysis 保持一致） ----------
    color_palette = {
        'primary':    ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'],
        'background': ['#F8F9FA', '#E9ECEF', '#DEE2E6'],
        'grid':       ['#ADB5BD', '#CED4DA', '#E9ECEF'],
        'limit_line': '#E63946',   # 国标限值线颜色
    }

    # ---------- 坐标范围：至少覆盖 [0, 1.2]，若有超标则自动扩展 ----------
    fill_min = 0.0
    fill_max = max(max(values) * 1.15, 1.2)

    # ---------- 创建图形 ----------
    fig = plt.figure(figsize=(10, 10), facecolor='white')
    ax = fig.add_subplot(111, polar=True, facecolor=color_palette['background'][1])

    # 1. 渐变同心圆底色（5 层）
    for lv in range(1, 6):
        lv_val    = fill_min + (fill_max - fill_min) * lv / 5
        lv_angles = angles_closed
        lv_values = [lv_val] * len(lv_angles)
        alpha_lv  = 0.03 + 0.03 * lv
        color_lv  = plt.cm.Blues(0.1 + 0.1 * lv)
        ax.fill(lv_angles, lv_values, color=color_lv, alpha=alpha_lv,
                edgecolor=color_palette['grid'][0], linewidth=0.5)

    # 2. 扇形底色 + 径向虚线
    for i in range(N):
        angle      = angles[i]
        next_angle = angles[(i + 1) % N]
        ax.fill_between([angle, next_angle], fill_min, fill_max,
                        color=color_palette['background'][2], alpha=0.1)
        ax.plot([angle, angle], [fill_min, fill_max],
                color=color_palette['grid'][0], alpha=0.3, linewidth=0.8, linestyle='--')

    # 3. 国标限值参考圈（ratio = 1.0）
    limit_ring = [1.0] * len(angles_closed)
    ax.plot(angles_closed, limit_ring,
            color=color_palette['limit_line'], linestyle='--', linewidth=2.0,
            label='国标限值 (ratio=1.0)', zorder=3)

    # 4. 数据折线（主图形）
    ax.plot(angles_closed, values_closed,
            color=color_palette['primary'][0], linewidth=3.5,
            marker='o', markersize=8,
            markerfacecolor='white', markeredgewidth=2,
            label=region_name, zorder=4)

    # 多层填充（内向外渐变透明度）
    for alpha_mult in [1.0, 0.7, 0.4]:
        ax.fill(angles_closed, values_closed,
                color=color_palette['primary'][1],
                alpha=0.15 * alpha_mult, edgecolor='none')

    # 5. 坐标轴标签 & 旋转（与 feature_analysis 一致）
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(pollutants, fontsize=12, fontweight='600',
                       color='#2D3748', fontfamily='Microsoft YaHei')

    for label, ang in zip(ax.get_xticklabels(), angles):
        if ang in [0, np.pi]:
            label.set_horizontalalignment('center')
        elif 0 < ang < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')
        label.set_rotation(ang * 180 / np.pi - 90)

    # 6. 径向刻度
    ytick_vals = np.linspace(fill_min, fill_max, 6)
    ax.set_yticks(ytick_vals)
    ax.set_yticklabels([f'{v:.2f}' for v in ytick_vals],
                       fontsize=9, color='#4A5568')
    ax.tick_params(axis='y', labelsize=9, colors='#718096', pad=10)
    ax.set_ylim(fill_min, fill_max)

    # 7. 网格线美化
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8,
            color=color_palette['grid'][0])

    # 8. 各顶点数值标签（带白底圆角框）
    for i, (ang, val) in enumerate(zip(angles, values)):
        y_pos = val + (fill_max - fill_min) * 0.04
        ax.text(ang, y_pos, f'{val:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='600',
                color=color_palette['primary'][0],
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          alpha=0.85, edgecolor=color_palette['primary'][0],
                          linewidth=0.8))

    # 9. 分拆标题：主标题（suptitle）+ 副标题（ax.set_title）
    plt.suptitle(f'【{region_name}】污染特征雷达图',
                 fontsize=18, fontweight='bold', color='#1A202C',
                 y=0.98, fontfamily='Microsoft YaHei')
    ax.set_title(f'\n污染诊断：{pollution_type} | 分析级别：{level}',
                 fontsize=13, color='#4A5568', pad=20,
                 fontfamily='Microsoft YaHei')

    # 10. 图例
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.15),
                       frameon=True, framealpha=0.95, fontsize=11,
                       facecolor='white', edgecolor=color_palette['grid'][0],
                       borderpad=1, labelspacing=1)
    legend.get_frame().set_linewidth(1.5)

    # 11. 底部来源标注
    fig.text(0.5, 0.02, '数据来源：环境监测数据 | 制图：XIANEMC1011',
             fontsize=10, color='#718096', style='italic',
             ha='center', fontfamily='Microsoft YaHei')

    # 12. 外边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(color_palette['grid'][0])
        spine.set_linewidth(1.5)

    # ---------- 保存 ----------
    save_dir = get_resource_path(f"mapout/Feature_Agent/{level}")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir,
                             f"Radar_{region_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
    plt.savefig(file_path, dpi=400, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.5)
    plt.close()
    return file_path

# ==========================================
# 图2：单区域基础散点图 (PM2.5 vs PMc) - 颜色改为 O3
# ==========================================
def generate_feature_scatter(df: pd.DataFrame, region_name: str, level: str = "分析区域") -> str:
    font()
    df = df.copy()
    req_cols = ['PM10', 'PM2.5', 'O3']
    for c in req_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
    if 'PM10' in df.columns and 'PM2.5' in df.columns:
        df['PMc'] = (df['PM10'] - df['PM2.5']).clip(lower=0)
    else: return None

    # 防止表格里缺少 O3 导致报错，设个默认值
    if 'O3' not in df.columns: df['O3'] = 50

    df = df.dropna(subset=['PM2.5', 'PMc', 'O3'])
    if df.empty: return None

    fig, ax = plt.subplots(figsize=(6, 5))
    size = df['PM2.5'] * 0.5 + 20 
    
    # 🚨 核心修改：圆点颜色 c 映射为 O3
    scatter = ax.scatter(df['PMc'], df['PM2.5'], s=size, c=df['O3'], 
                         cmap='coolwarm', alpha=0.75, edgecolors='w', linewidth=0.5)
    
    lims = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.2, zorder=0, label='1:1 比例线')

    ax.set_xlabel('粗颗粒物浓度 (PMc)', fontsize=10)
    ax.set_ylabel('细颗粒物浓度 (PM2.5)', fontsize=10)
    ax.set_title(f'【{region_name}】颗粒物关联\n(颜色: O3浓度)', fontsize=11, pad=10, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.colorbar(scatter, label='O3 浓度')
    
    save_dir = get_resource_path(f"mapout/Feature_Agent/{level}")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"Scatter_{region_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()
    return file_path

# ==========================================
# 图3：全局各市四象限分布比对图 (颜色严格表示O3)
# ==========================================
def generate_global_quadrant_scatter(df: pd.DataFrame, region_col: str, level: str = "分析区域") -> str:
    font()
    df = df.copy()
    req_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'O3']
    for col in req_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    region_means = df.groupby(region_col)[req_cols].mean().dropna()
    if region_means.empty: return None

    region_means['PMc'] = (region_means['PM10'] - region_means['PM2.5']).clip(lower=0)
    region_means['NO2'] = region_means['NO2'].replace(0, np.nan)
    region_means['SN_ratio'] = (region_means['SO2'] / region_means['NO2']).fillna(0)
    
    min_sn, max_sn = region_means['SN_ratio'].min(), region_means['SN_ratio'].max()
    if max_sn > min_sn:
        sizes = 100 + 500 * (region_means['SN_ratio'] - min_sn) / (max_sn - min_sn)
    else:
        sizes = np.array([200] * len(region_means))

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 🚨 核心要求：颜色 c 严格映射为 O3，圆点代表各个城市
    scatter = ax.scatter(region_means['PM2.5'], region_means['PMc'], s=sizes, c=region_means['O3'], 
                         cmap='coolwarm', alpha=0.85, edgecolors='white', linewidth=1.2)

    x_center = region_means['PM2.5'].mean()
    y_center = region_means['PMc'].mean()

    ax.axvline(x_center, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(y_center, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

    for region, row in region_means.iterrows():
        ax.text(row['PM2.5'], row['PMc'] + (y_center*0.02), region, 
                fontsize=11, ha='center', va='bottom', fontweight='bold', color='black')

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    pad_x = (x_max - x_min) * 0.05
    pad_y = (y_max - y_min) * 0.05

    ax.text(x_max - pad_x, y_max - pad_y, '第一象限\n(高细/高粗)', ha='right', va='top', fontsize=12, color='gray', fontweight='bold', alpha=0.5)
    ax.text(x_min + pad_x, y_max - pad_y, '第二象限\n(低细/高粗)', ha='left', va='top', fontsize=12, color='gray', fontweight='bold', alpha=0.5)
    ax.text(x_min + pad_x, y_min + pad_y, '第三象限\n(低细/低粗)', ha='left', va='bottom', fontsize=12, color='gray', fontweight='bold', alpha=0.5)
    ax.text(x_max - pad_x, y_min + pad_y, '第四象限\n(高细/低粗)', ha='right', va='bottom', fontsize=12, color='gray', fontweight='bold', alpha=0.5)

    ax.set_xlabel('平均 PM2.5 浓度 (细颗粒物)', fontsize=13, fontweight='bold')
    ax.set_ylabel('平均 PMc 浓度 (粗颗粒物 = PM10 - PM2.5)', fontsize=13, fontweight='bold')
    ax.set_title(f'各【{level}】PM2.5 与 PMc 四象限分布比对图\n(点大小: S/N比值反映煤烟/机动车倾向 | 颜色: O3浓度)', fontsize=15, pad=15, fontweight='bold')

    cbar = plt.colorbar(scatter)
    cbar.set_label('平均 O3 浓度', fontsize=12, fontweight='bold')

    save_dir = get_resource_path(f"mapout/Feature_Agent/{level}")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"Quadrant_Global_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()
    return file_path