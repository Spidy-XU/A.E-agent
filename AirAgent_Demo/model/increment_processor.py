"""
increment_processor.py

  图1（子图1）每日增量百分比变化线形图（日中值，O3 右Y轴）
  图1（子图2）24小时日内平均增量变化趋势（O3 右Y轴）
  图1（子图3）日平均增量变化热图（逐行污染物归一化，数值标注）

  每个区域生成一张 PNG → {region}_污染增量分析.png

对外接口：
  generate_increment_all(df, region_col)       → list[str]  ← app.py 推荐调用
  generate_increment_heatmap(df, region_col)   → str        ← 旧接口兼容保留（返回第一张图路径）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from model.basic_function import font, get_resource_path

# ═══════════════════════════════════════════════════════════════════════════════
# 全局配置
# ═══════════════════════════════════════════════════════════════════════════════

_PIE_POLLUTANTS = ['PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3']

_COLORS = {
    'PM2.5': '#E74C3C',
    'PM10':  '#F39C12',
    'CO':    '#F1C40F',
    'NO2':   '#2ECC71',
    'SO2':   '#3498DB',
    'O3':    '#9B59B6',
}


# ═══════════════════════════════════════════════════════════════════════════════
# 核心算法：小时增长百分比
# ═══════════════════════════════════════════════════════════════════════════════

def _calc_hourly_growth_pct(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    计算每小时相对最近有效前一小时的增长百分比。
    特殊规则：第一行数据的增量比例用第二行数据填充。
    df 须以时间为索引，含「原始有效标记」bool 列。
    """
    growth_pct = pd.DataFrame(index=df.index, columns=[f'{c}(%)' for c in cols])

    for idx in df.index:
        prev_rows        = df.loc[:idx].iloc[:-1]
        valid_prev_rows  = prev_rows[prev_rows['原始有效标记']] if not prev_rows.empty else prev_rows

        if valid_prev_rows.empty:
            for c in cols:
                growth_pct.loc[idx, f'{c}(%)'] = np.nan
        else:
            last_idx   = valid_prev_rows.index[-1]
            last_vals  = df.loc[last_idx, cols]
            curr_vals  = df.loc[idx, cols]
            for c in cols:
                if pd.isna(curr_vals[c]) or pd.isna(last_vals[c]):
                    growth_pct.loc[idx, f'{c}(%)'] = np.nan
                elif last_vals[c] == 0:
                    growth_pct.loc[idx, f'{c}(%)'] = np.nan
                else:
                    pct = (curr_vals[c] - last_vals[c]) / last_vals[c] * 100
                    growth_pct.loc[idx, f'{c}(%)'] = round(pct, 2)

    result = pd.concat([df[cols], growth_pct], axis=1)
    result = result[df['原始有效标记']].reset_index(drop=False)

    # 第一行用第二行填充
    if len(result) >= 2:
        for c in [f'{col}(%)' for col in cols]:
            result.loc[0, c] = result.loc[1, c]

    result = result.set_index('index')
    result.index.name = '时间'
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 保存目录工具
# ═══════════════════════════════════════════════════════════════════════════════

def _save_dir(region_col: str) -> str:
    clean = region_col.replace('/', '_').replace('\\', '_')
    path  = get_resource_path(f"mapout/{clean}污染增量分析")
    os.makedirs(str(path), exist_ok=True)
    return str(path)


def _clean(name: str) -> str:
    for ch in r'/\:*?"<>|':
        name = name.replace(ch, '_')
    return name


# ═══════════════════════════════════════════════════════════════════════════════
# 核心绘图函数
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_one_region(region: str, plot_data: pd.DataFrame,
                     heatmap_daily_mean: pd.DataFrame,
                     save_dir: str) -> str:
    """
    绘制单个区域的三合一增量分析图，返回保存路径。
    plot_data       : 含 PM2.5(%) PM10(%) ... 等列，时间索引
    heatmap_daily_mean : 按日 resample('D').mean() 的结果
    """
    font()

    plot_cols = [f'{p}(%)' for p in _PIE_POLLUTANTS]
    o3_col    = 'O3(%)'
    non_o3    = [c for c in plot_cols if c != o3_col]

    # ── 日中值数据（子图1 X 轴） ─────────────────────────────────────────────
    daily_mean_df = plot_data.groupby(plot_data.index.date).median().reset_index()
    daily_mean_df.rename(columns={'index': 'date'}, inplace=True)
    daily_dates   = daily_mean_df['date']
    n_dates       = len(daily_dates)
    x_pos         = np.arange(n_dates)

    # ── 24h 均值数据（子图2 X 轴） ───────────────────────────────────────────
    plot_data_h        = plot_data.copy()
    plot_data_h['小时'] = plot_data_h.index.hour
    hourly_daily_mean  = plot_data_h.groupby('小时')[plot_cols].mean()

    # ── 热图数据（子图3） ────────────────────────────────────────────────────
    heatmap_daily_dates_fmt = heatmap_daily_mean.index.strftime('%m-%d')
    valid_pollutants        = [c.replace('(%)', '') for c in plot_cols
                               if c in heatmap_daily_mean.columns]
    hm_aligned = heatmap_daily_mean[[c for c in plot_cols
                                     if c in heatmap_daily_mean.columns]]
    n_poll     = len(valid_pollutants)
    n_d        = len(heatmap_daily_dates_fmt)

    hm_orig = hm_aligned.T.values.flatten()
    target  = n_poll * n_d
    if len(hm_orig) > target:
        hm_orig = hm_orig[:target]
    elif len(hm_orig) < target:
        hm_orig = np.concatenate([hm_orig, np.full(target - len(hm_orig), np.nan)])
    hm_orig = hm_orig.reshape(n_poll, n_d)

    # 逐行归一化到 [-1, 1]
    hm_norm = np.zeros_like(hm_orig, dtype=float)
    for i in range(n_poll):
        row = hm_orig[i, :]
        if np.all(np.isnan(row)):
            hm_norm[i, :] = np.nan
            continue
        abs_max = np.nanmax(np.abs(row))
        hm_norm[i, :] = 0 if abs_max == 0 else row / abs_max

    # ── 画布 ─────────────────────────────────────────────────────────────────
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(16, 18), sharex=False,
        gridspec_kw={'hspace': 0.3}
    )
    fig.suptitle(f'{region} 污染物小时/日增量变化分析',
                 fontsize=20, fontweight='bold', y=0.98)

    # ── 子图1：每日增量百分比线形图 ──────────────────────────────────────────
    ax1_right = ax1.twinx()

    for col in non_o3:
        p_name = col.replace('(%)', '')
        if col not in daily_mean_df.columns:
            continue
        vals = pd.to_numeric(daily_mean_df[col], errors='coerce').fillna(0)
        ax1.plot(x_pos, vals, color=_COLORS[p_name], linewidth=1.2,
                 marker='o', markersize=4, label=f'{p_name} 日中值')

    if o3_col in daily_mean_df.columns:
        o3_vals = pd.to_numeric(daily_mean_df[o3_col], errors='coerce').fillna(0)
        ax1_right.plot(x_pos, o3_vals, color=_COLORS['O3'], linewidth=1.2,
                       marker='s', markersize=4, linestyle='--', label='O3 日中值')

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.8,
                label='不增不减基准线')
    ax1.set_title('每日增量百分比变化线形图', fontsize=16, pad=10)
    ax1.set_ylabel('增量百分比（%）', fontsize=12)
    ax1.set_xlabel('日期', fontsize=12)

    tick_step = 1 if n_dates <= 10 else 2
    ax1.set_xticks(x_pos[::tick_step])
    ax1.set_xticklabels([d.strftime('%m-%d') for d in daily_dates[::tick_step]],
                        rotation=0, ha='center')
    ax1.grid(axis='y', alpha=0.3)
    ax1_right.set_ylabel('O3 增量百分比（%）', fontsize=12, color=_COLORS['O3'])
    ax1_right.tick_params(axis='y', labelcolor=_COLORS['O3'])

    # 合并图例去重
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax1_right.get_legend_handles_labels()
    seen, ul, ulb = set(), [], []
    for h, l in zip(l1 + l2, lb1 + lb2):
        if l not in seen:
            seen.add(l); ul.append(h); ulb.append(l)
    ax1.legend(ul, ulb, loc='upper right', ncol=2, fontsize=9)

    # ── 子图2：24小时日内平均增量趋势 ────────────────────────────────────────
    ax2_right = ax2.twinx()

    for col in non_o3:
        p_name = col.replace('(%)', '')
        ax2.plot(hourly_daily_mean.index, hourly_daily_mean[col],
                 label=p_name, color=_COLORS[p_name],
                 linewidth=2, marker='o', markersize=4)

    ax2_right.plot(hourly_daily_mean.index, hourly_daily_mean[o3_col],
                   label='O3', color=_COLORS['O3'],
                   linewidth=2.2, marker='s', markersize=4, linestyle='--')

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.8,
                label='不增不减基准线')
    ax2.set_title('24小时日内平均增量变化趋势', fontsize=16, pad=10)
    ax2.set_ylabel('增量百分比（%）', fontsize=12)
    ax2.set_xlabel('小时', fontsize=12)
    ax2.set_xticks(range(0, 24))
    ax2.set_xticklabels([f'{h}:00' for h in range(0, 24)], fontsize=10)
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)
    ax2_right.set_ylabel('O3 增量百分比（%）', fontsize=12, color=_COLORS['O3'])
    ax2_right.tick_params(axis='y', labelcolor=_COLORS['O3'])

    l1, lb1 = ax2.get_legend_handles_labels()
    l2, lb2 = ax2_right.get_legend_handles_labels()
    ax2.legend(l1 + l2, lb1 + lb2, loc='upper right', ncol=3)

    # ── 子图3：日平均增量热图 ─────────────────────────────────────────────────
    im = ax3.imshow(hm_norm, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax3.set_yticks(range(n_poll))
    ax3.set_yticklabels(valid_pollutants, fontsize=12)
    ax3.set_xticks(range(n_d))
    ax3.set_xticklabels(heatmap_daily_dates_fmt, rotation=0, ha='center', fontsize=10)
    ax3.set_title('日平均增量变化（每行污染物独立色阶）', fontsize=16, pad=10)

    # 数值标注（日期少于20时）
    if n_d < 20:
        for i in range(n_poll):
            for j in range(n_d):
                if not np.isnan(hm_orig[i, j]):
                    ax3.text(j, i, f'{int(round(hm_orig[i, j]))}',
                             ha='center', va='center',
                             color='white', fontsize=8, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax3, orientation='horizontal',
                        shrink=0.6, pad=0.1, location='bottom')
    cbar.set_label('相对增量强度（-1=最小，0=中间，1=最大）',
                   fontsize=12, loc='center', labelpad=5)
    cbar.ax.tick_params(labelsize=10)

    # ── 保存 ─────────────────────────────────────────────────────────────────
    plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.08,
                        hspace=0.4, wspace=0.2)
    path = os.path.join(save_dir, f'{_clean(region)}_污染增量分析.png')
    plt.savefig(path, dpi=400, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# 对外主接口
# ═══════════════════════════════════════════════════════════════════════════════

def generate_increment_all(df: pd.DataFrame, region_col: str) -> list:
    """
    为所有区域生成污染物小时/日增量变化分析图，返回路径列表。

    app.py 调用示例：
        from model.increment_processor import generate_increment_all
        paths = generate_increment_all(df, analysis_level)
    """
    font()
    df = df.copy()
    df['时间'] = pd.to_datetime(df['时间'], errors='coerce')
    df['时间'] = df['时间'].dt.floor('H')

    valid_cols = [p for p in _PIE_POLLUTANTS if p in df.columns]
    if not valid_cols:
        return []

    # 补全缺失污染物
    for p in _PIE_POLLUTANTS:
        if p not in df.columns:
            df[p] = np.nan

    # 按区域+时间分组取均值
    df = df.groupby([region_col, '时间'])[_PIE_POLLUTANTS].mean().reset_index()

    save_dir = _save_dir(region_col)
    paths    = []

    for region in df[region_col].unique():
        qdf = df[df[region_col] == region].copy().set_index('时间')[_PIE_POLLUTANTS]

        # 补全完整小时序列
        full_idx = pd.date_range(start=qdf.index.min(), end=qdf.index.max(), freq='H')
        qdf      = qdf.reindex(full_idx)
        qdf['原始有效标记'] = qdf[_PIE_POLLUTANTS].notna().all(axis=1)

        # 计算小时增长百分比
        result = _calc_hourly_growth_pct(qdf, _PIE_POLLUTANTS)
        plot_cols = [f'{p}(%)' for p in _PIE_POLLUTANTS]
        plot_data = result[plot_cols].copy()
        plot_data.index = pd.to_datetime(plot_data.index)

        # 热图数据源（日均值）
        heatmap_daily_mean = plot_data.resample('D').mean().fillna(0).astype(np.float64)

        try:
            path = _plot_one_region(region, plot_data, heatmap_daily_mean, save_dir)
            paths.append(path)
        except Exception as e:
            print(f"[increment_processor] 绘图失败 {region}: {e}")

    return paths


# ── 旧接口兼容：保留函数签名，返回单张图路径 ─────────────────────────────────

def generate_increment_heatmap(df: pd.DataFrame, region_col: str) -> str:
    """
    旧接口保留。内部调用 generate_increment_all，返回第一张图路径。
    建议升级至 generate_increment_all 以获取所有区域图表。
    """
    paths = generate_increment_all(df, region_col)
    return paths[0] if paths else None