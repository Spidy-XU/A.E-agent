"""
anomaly_processor.py
输出两类图表：

  图表1  污染物日距平 + AQI 小时时间序列   → {region}_距平_AQI分析图.png   （每城市一张）
  图表2  24小时污染物距平 + AQI 时间序列   → {region}_距平_AQI分析图24.png  （每城市一张）

对外接口：
  generate_anomaly_all(df, region_col)           → dict   ← app.py 推荐调用入口
  generate_anomaly_plot(df, region_col, region)  → str    ← 旧接口兼容保留
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from model.basic_function import font, get_resource_path

# 延迟导入，避免循环依赖；运行时才真正调用
def _try_label_pollution_types(df, region_col):
    """尝试从 cluster_processor 获取 KMeans 污染类型，失败则返回「未知类型」"""
    try:
        from model.cluster_processor import label_pollution_types
        return label_pollution_types(df, region_col)
    except Exception:
        df = df.copy()
        df['污染类型'] = '未知类型'
        return df

# ===============================================================
# 全局配置
# ===============================================================

# 使用 PMc（=PM10-PM2.5）而非 PM10
_POLLUTANT_COLS = ['SO2', 'NO2', 'CO', 'O3', 'PMc', 'PM2.5']

# 距平配色：正=偏高（暖红），负=偏低（青绿）
_POS_COLOR = '#FF6B6B'
_NEG_COLOR = '#4ECDC4'

# 标准限值（与 cluster_processor 一致）：用于 AQI 归一化求和
_STANDARD_LIMITS = {'SO2': 60, 'NO2': 40, 'CO': 4, 'O3': 160, 'PMc': 70, 'PM2.5': 35}


# ===============================================================
# 内部工具
# ===============================================================

def _save_dir(region_col: str) -> str:
    path = get_resource_path(f"mapout/{region_col}污染距平分析")
    os.makedirs(str(path), exist_ok=True)
    return str(path)


def _build_color_map(unique_types) -> dict:
    """
    全局统一配色映射（juping.py 1.2 节逻辑）：
    TABLEAU_COLORS(10种) + CSS4_COLORS 间隔取色，
    确保跨区域同一污染类型颜色固定一致。
    """
    base_colors   = list(mcolors.TABLEAU_COLORS.values())
    extend_colors = list(mcolors.CSS4_COLORS.values())[::5]
    all_colors    = base_colors + extend_colors
    return dict(zip(unique_types, all_colors[:len(unique_types)]))


def _preprocess(df: pd.DataFrame, region_col: str) -> pd.DataFrame:
    """
    统一预处理：
      1. 解析时间，floor 到小时，生成 '日期' 列（datetime）
      2. PMc = (PM10 - PM2.5).clip(0)
      3. 补全缺失污染物列为 0
      4. 若无 '污染类型' 列置为 '未知类型'
    """
    df = df.copy()
    df['时间'] = pd.to_datetime(df['时间'], errors='coerce')
    df['日期'] = df['时间'].dt.floor('H')

    if 'PMc' not in df.columns:
        if 'PM10' in df.columns and 'PM2.5' in df.columns:
            df['PMc'] = (df['PM10'] - df['PM2.5']).clip(lower=0)
        elif 'PM10' in df.columns:
            df['PMc'] = df['PM10'].clip(lower=0)
        else:
            df['PMc'] = 0.0

    for p in _POLLUTANT_COLS:
        if p not in df.columns:
            df[p] = 0.0

    # 调用 KMeans 聚类（cluster_processor）为每条记录打上污染类型标签
    # 若聚类模块不可用或数据不足，自动降级为「未知类型」
    df = _try_label_pollution_types(df, region_col)

    return df


# ===============================================================
# 图表1：污染物日距平 + AQI 小时时间序列
# ===============================================================

def _plot_daily_anomaly(df: pd.DataFrame, region_col: str,
                        color_map: dict, save_dir: str) -> list:
    """
    布局：4行×2列
      行0-2：6 个污染物日距平柱状图（前3行填满2列）
      行3  ：AQI 小时时间序列通栏图
    """
    font()

    # 全域日均值（距平基准） 
    daily_avg = df.groupby(df['日期'].dt.date)[_POLLUTANT_COLS].mean()
    daily_avg.index = pd.to_datetime(daily_avg.index)

    paths = []
    for region in df[region_col].unique():
        rdf = df[df[region_col] == region].copy()
        if len(rdf) == 0:
            continue

        # 子区域日均值 & 距平
        region_daily = rdf.groupby(rdf['日期'].dt.date)[_POLLUTANT_COLS].mean()
        region_daily.index = pd.to_datetime(region_daily.index)
        aligned = region_daily.reindex(daily_avg.index, fill_value=np.nan)
        anom = (aligned[_POLLUTANT_COLS] - daily_avg[_POLLUTANT_COLS]).reset_index()
        anom.rename(columns={'index': '日期'}, inplace=True)

        # AQI 小时数据 
        # 先按标准限值归一化再求和
        rdf_aqi = rdf.copy()
        for p, lim in _STANDARD_LIMITS.items():
            if p in rdf_aqi.columns:
                rdf_aqi[p] = rdf_aqi[p] / lim
        rdf_aqi['AQI'] = rdf_aqi[_POLLUTANT_COLS].sum(axis=1)
        # 按小时聚合（同一小时多条记录取均值），消除多站点重复行导致的条形图间隔
        aqi_h = (rdf_aqi.groupby('日期')
                        .agg(AQI=('AQI', 'mean'), 污染类型=('污染类型', 'first'))
                        .reset_index()
                        .sort_values('日期'))
        # 补全完整小时序列，消除缺测时段的空隙
        full_hours = pd.date_range(aqi_h['日期'].min(), aqi_h['日期'].max(), freq='H')
        aqi_h = aqi_h.set_index('日期').reindex(full_hours).reset_index()
        aqi_h.rename(columns={'index': '日期'}, inplace=True)
        aqi_h['AQI'] = aqi_h['AQI'].fillna(0)
        aqi_h['污染类型'] = aqi_h['污染类型'].fillna('未知类型')
        aqi_h['color'] = aqi_h['污染类型'].map(color_map).fillna('#999999')

        # 画布 
        fig = plt.figure(figsize=(18, 12), dpi=100)
        fig.patch.set_facecolor('white')
        gs  = plt.GridSpec(4, 2, height_ratios=[1, 1, 1, 1.2], hspace=0.45, wspace=0.3)

        # 6个距平子图 
        for idx, pollutant in enumerate(_POLLUTANT_COLS):
            ax     = fig.add_subplot(gs[idx // 2, idx % 2])
            values = anom[pollutant].values
            dates  = anom['日期']
            colors = [_POS_COLOR if v >= 0 else _NEG_COLOR for v in values]

            ax.bar(range(len(values)), values, color=colors, alpha=0.7, width=0.6)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax.set_title(f'{pollutant} 距平日均值', fontsize=12, fontweight='bold')
            ax.set_xlabel('日期', fontsize=10)
            ax.set_ylabel('距平值', fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if len(dates) > 0:
                step  = max(1, len(dates) // 5)
                ticks = range(0, len(dates), step)
                ax.set_xticks(ticks)
                ax.set_xticklabels(
                    [dates.iloc[i].strftime('%m-%d') for i in ticks],
                    rotation=0, fontsize=9)

        # AQI 通栏 
        ax_aqi = fig.add_subplot(gs[3, :])
        ax_aqi.bar(range(len(aqi_h)), aqi_h['AQI'],
                   color=aqi_h['color'], alpha=0.8, width=0.8)
        ax_aqi.set_title('AQI小时总和时间序列', fontsize=12, fontweight='bold')
        ax_aqi.set_xlabel('时间', fontsize=10)
        ax_aqi.set_ylabel('AQI数值（总和）', fontsize=10)
        ax_aqi.grid(axis='y', alpha=0.3)
        ax_aqi.spines['top'].set_visible(False)
        ax_aqi.spines['right'].set_visible(False)

        if len(aqi_h) > 0:
            step_h = max(1, len(aqi_h) // 10)
            ticks  = range(0, len(aqi_h), step_h)
            ax_aqi.set_xticks(ticks)
            ax_aqi.set_xticklabels(
                [aqi_h['日期'].iloc[i].strftime('%m-%d %H:%M') for i in ticks],
                rotation=0, fontsize=9)

        # 图例 
        current_types   = aqi_h['污染类型'].unique()
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1,
                          color=color_map.get(t, '#999999'), label=t, alpha=0.8)
            for t in current_types
        ]
        ax_aqi.legend(handles=legend_elements, loc='upper right',
                      fontsize=9, ncol=min(4, len(legend_elements)))

        # 总标题 & 保存 
        fig.suptitle(f'{region} 污染物距平与AQI时间序列分析',
                     fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        safe = str(region).replace('/', '_').replace('\\', '_')
        path = os.path.join(save_dir, f"{safe}_距平_AQI分析图.png")
        plt.savefig(path, dpi=400, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        paths.append(path)

    return paths


# ===============================================================
# 图表2：24小时距平 + AQI 小时时间序列
# ===============================================================

def _plot_24h_anomaly(df: pd.DataFrame, region_col: str,
                      color_map: dict, save_dir: str) -> list:
    """
    布局同图表1，前3行距平 X 轴为小时（0-23），
    第4行仍为完整日期时间序列 AQI。
    """
    font()

    # 全域24小时均值（距平基准） 
    hourly_avg = df.groupby(df['日期'].dt.hour)[_POLLUTANT_COLS].mean()

    paths = []
    for region in df[region_col].unique():
        rdf = df[df[region_col] == region].copy()
        if len(rdf) == 0:
            continue

        # 24h 均值 & 距平 
        region_hourly = rdf.groupby(rdf['日期'].dt.hour)[_POLLUTANT_COLS].mean()
        aligned       = region_hourly.reindex(hourly_avg.index, fill_value=np.nan)
        anom          = aligned[_POLLUTANT_COLS] - hourly_avg[_POLLUTANT_COLS]

        # AQI 小时数据 
        rdf_aqi = rdf.copy()
        for p, lim in _STANDARD_LIMITS.items():
            if p in rdf_aqi.columns:
                rdf_aqi[p] = rdf_aqi[p] / lim
        rdf_aqi['AQI'] = rdf_aqi[_POLLUTANT_COLS].sum(axis=1)
        aqi_h = (rdf_aqi.groupby('日期')
                        .agg(AQI=('AQI', 'mean'), 污染类型=('污染类型', 'first'))
                        .reset_index()
                        .sort_values('日期'))
        full_hours = pd.date_range(aqi_h['日期'].min(), aqi_h['日期'].max(), freq='H')
        aqi_h = aqi_h.set_index('日期').reindex(full_hours).reset_index()
        aqi_h.rename(columns={'index': '日期'}, inplace=True)
        aqi_h['AQI'] = aqi_h['AQI'].fillna(0)
        aqi_h['污染类型'] = aqi_h['污染类型'].fillna('未知类型')
        aqi_h['color'] = aqi_h['污染类型'].map(color_map).fillna('#999999')

        # 画布 
        fig = plt.figure(figsize=(18, 12), dpi=100)
        fig.patch.set_facecolor('white')
        gs  = plt.GridSpec(4, 2, height_ratios=[1, 1, 1, 1.2], hspace=0.45, wspace=0.3)

        # 6个24h距平子图 
        for idx, pollutant in enumerate(_POLLUTANT_COLS):
            ax     = fig.add_subplot(gs[idx // 2, idx % 2])
            values = anom[pollutant].values
            hours  = anom.index          # 0-23
            colors = [_POS_COLOR if v >= 0 else _NEG_COLOR for v in values]

            ax.bar(range(len(values)), values, color=colors, alpha=0.7, width=0.6)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax.set_title(f'{pollutant} 距平小时均值', fontsize=12, fontweight='bold')
            ax.set_xlabel('小时（0-23）', fontsize=10)
            ax.set_ylabel('距平值', fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if len(hours) > 0:
                step  = max(1, len(hours) // 6)
                ticks = range(0, len(hours), step)
                ax.set_xticks(ticks)
                ax.set_xticklabels([hours[i] for i in ticks], rotation=0, fontsize=9)

        # AQI 通栏 
        ax_aqi = fig.add_subplot(gs[3, :])
        ax_aqi.bar(range(len(aqi_h)), aqi_h['AQI'],
                   color=aqi_h['color'], alpha=0.8, width=0.8)
        ax_aqi.set_title('AQI小时时间序列', fontsize=12, fontweight='bold')
        ax_aqi.set_xlabel('时间', fontsize=10)
        ax_aqi.set_ylabel('AQI数值（总和）', fontsize=10)
        ax_aqi.grid(axis='y', alpha=0.3)
        ax_aqi.spines['top'].set_visible(False)
        ax_aqi.spines['right'].set_visible(False)

        if len(aqi_h) > 0:
            step_h = max(1, len(aqi_h) // 10)
            ticks  = range(0, len(aqi_h), step_h)
            ax_aqi.set_xticks(ticks)
            ax_aqi.set_xticklabels(
                [aqi_h['日期'].iloc[i].strftime('%m-%d %H:%M') for i in ticks],
                rotation=0, fontsize=9)

        # 图例 
        current_types   = aqi_h['污染类型'].unique()
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1,
                          color=color_map.get(t, '#999999'), label=t, alpha=0.8)
            for t in current_types
        ]
        ax_aqi.legend(handles=legend_elements, loc='upper right',
                      fontsize=9, ncol=min(4, len(legend_elements)))

        # 总标题 & 保存 
        fig.suptitle(f'{region} 24小时污染物距平与AQI时间序列分析',
                     fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        safe = str(region).replace('/', '_').replace('\\', '_')
        path = os.path.join(save_dir, f"{safe}_距平_AQI分析图24.png")
        plt.savefig(path, dpi=400, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        paths.append(path)

    return paths


# ===============================================================
# 对外主接口
# ===============================================================

def generate_anomaly_all(df: pd.DataFrame, region_col: str) -> dict:
    """
    一次性生成两类距平图，返回路径字典：
      {
        'daily': list[str],   # 各区域日距平 + AQI 时间序列图
        'h24':   list[str],   # 各区域24小时距平 + AQI 时间序列图
      }

    app.py 调用示例：
        from model.anomaly_processor import generate_anomaly_all
        results = generate_anomaly_all(df, analysis_level)
    """
    font()
    df = _preprocess(df, region_col)

    # 全局统一配色
    color_map = _build_color_map(df['污染类型'].unique())
    save_dir  = _save_dir(region_col)

    return {
        'daily': _plot_daily_anomaly(df, region_col, color_map, save_dir),
        'h24':   _plot_24h_anomaly(df, region_col, color_map, save_dir),
    }


# ── 旧接口兼容 ────────────────────────────────────────────────────

def generate_anomaly_plot(df: pd.DataFrame, region_col: str,
                          target_region: str) -> str:
    """
    旧接口保留，生成单个区域的日距平图并返回路径。
    建议升级至 generate_anomaly_all。
    """
    font()
    df        = _preprocess(df, region_col)
    color_map = _build_color_map(df['污染类型'].unique())
    save_dir  = _save_dir(region_col)
    paths     = _plot_daily_anomaly(df, region_col, color_map, save_dir)

    safe   = str(target_region).replace('/', '_').replace('\\', '_')
    target = os.path.join(save_dir, f"{safe}_距平_AQI分析图.png")
    return target if os.path.exists(target) else (paths[0] if paths else None)