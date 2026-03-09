import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from datetime import datetime
from model.basic_function import font, get_resource_path

# ═══════════════════════════════════════════════════════════════════════════════
# 全局配置：颜色 & 标签
# ═══════════════════════════════════════════════════════════════════════════════

_STACK_COLS_ORDER = ['SO2', 'NO2', 'CO', 'O3', 'PMc', 'PM2.5']
_COLORS           = ['#FF6B9D', '#6A8EAE', '#57CC99', '#FFD166', '#9A7AA0', '#4ECDC4']
_COLOR_MAP        = dict(zip(_STACK_COLS_ORDER, _COLORS))
_LABEL_MAP        = {
    'SO2':   r'$SO_2$',
    'NO2':   r'$NO_2$',
    'CO':    r'$CO$',
    'O3':    r'$O_3$',
    'PMc':   r'$PMc$',
    'PM2.5': r'$PM_{2.5}$',
}

# 折线图专用多色系（支持 30+ 区域）
_LINE_COLORS = [
    '#FF4500', '#FF8C00', '#FFD700', '#9ACD32', '#32CD32', '#00FF7F',
    '#00CED1', '#1E90FF', '#9370DB', '#FF69B4', '#FF1493', '#8B4513',
    '#A9A9A9', '#000080', '#800080', '#008080', '#F0E68C', '#ADD8E6',
    '#FFB6C1', '#FFA07A', '#20B2AA', "#DD9DDD", '#B0C4DE', '#DC143C',
    '#87CEEB', '#98FB98', '#9932CC', '#7FFF00', '#FF6347', '#00FFFF',
]


# ═══════════════════════════════════════════════════════════════════════════════
# 内部工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def _save_dir() -> str:
    path = get_resource_path("mapout/Ranking_Agent")
    os.makedirs(path, exist_ok=True)
    return path


def _ts() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def _custom_stats(series: pd.Series) -> float:
    """
    跨天日均值序列汇总（与 zhonghepaiming.py 保持一致）：
      - O3  → 90 百分位（有效天数 ≥ 10，否则取最大值）
      - CO  → 95 百分位（有效天数 ≥ 10，否则取最大值）
      - 其余 → 算术均值
    """
    valid_data = series.dropna()
    valid_count = len(valid_data)
    pct = {'O3': 90, 'CO': 95}.get(series.name, None)
    if pct is not None:
        return np.nanpercentile(series, pct) if valid_count >= 10 else (
            valid_data.max() if valid_count > 0 else np.nan)
    return valid_data.mean() if valid_count > 0 else np.nan


def _gen_line_colors(n: int) -> list:
    """动态生成 n 种折线颜色，超出基础列表时调整亮度避免重复感。"""
    colors = []
    for i in range(n):
        base = _LINE_COLORS[i % len(_LINE_COLORS)]
        if i >= len(_LINE_COLORS):
            rgba = mcolors.to_rgba(base)
            b = 0.7 + (i % 5) * 0.075
            colors.append((min(1, rgba[0]*b), min(1, rgba[1]*b), min(1, rgba[2]*b), rgba[3]))
        else:
            colors.append(base)
    return colors


def _compute_index_data(df: pd.DataFrame, region_col: str) -> pd.DataFrame:
    """
    核心计算函数（供三张图复用）：
      1. 按区域 → 小时均值聚合
      2. 日质控：有效小时数 ≥ 20 的天才保留
      3. 日聚合：O3 用 8h 滚动均值→日最大；其余→日均值
      4. 跨天汇总：O3取90%位、CO取95%位、其余取均值
      5. 综合指数折算：
           PMc  = (PM10 - PM2.5).clip(0) / 70
           PM2.5 = PM2.5/70 + PM2.5/35
           NO2  = NO2/40, SO2 = SO2/60, CO = CO/4, O3 = O3/160
      6. 计算综合指数 = 各分项之和，按升序排列
    返回 DataFrame（index = 区域名，列 = 分项指数 + 综合指数）。
    """
    df = df.copy()
    df['时间'] = pd.to_datetime(df['时间'])

    raw_pollutants = ['SO2', 'NO2', 'CO', 'O3', 'PM10', 'PM2.5']
    valid_cols = [p for p in raw_pollutants if p in df.columns]
    if not valid_cols:
        return pd.DataFrame()

    records = []
    for region in df[region_col].unique():
        rdf = df[df[region_col] == region].groupby('时间')[valid_cols].mean()

        # 质控：过滤有效小时数不足 20 的天
        rdf['_d'] = rdf.index.date
        valid_dates = rdf.groupby('_d').size()
        valid_dates = valid_dates[valid_dates >= 20].index
        filtered = rdf[rdf['_d'].isin(valid_dates)].drop(columns=['_d'])
        if filtered.empty:
            filtered = rdf.drop(columns=['_d'])

        # 日聚合
        daily = pd.DataFrame(index=pd.DatetimeIndex(filtered.index.date).unique(),
                             columns=filtered.columns)
        for col in filtered.columns:
            if col == 'O3':
                rm = filtered[col].rolling(window='8h', min_periods=1, center=False).mean()
                daily[col] = rm.resample('D').max()
            else:
                daily[col] = filtered[col].resample('D').mean()

        stats = daily[valid_cols].apply(_custom_stats).round(4)
        stats.name = region
        records.append(stats)

    data = pd.DataFrame(records)
    data.index.name = region_col
    data = data.fillna(0)

    # 综合指数折算
    if 'PM10' in data.columns and 'PM2.5' in data.columns:
        data['PMc'] = (data['PM10'] - data['PM2.5']).clip(lower=0) / 70
    elif 'PM10' in data.columns:
        data['PMc'] = data['PM10'] / 70

    if 'PM2.5' in data.columns:
        data['PM2.5'] = data['PM2.5'] / 70 + data['PM2.5'] / 35  # 双重折算
    if 'NO2' in data.columns:
        data['NO2'] = data['NO2'] / 40
    if 'SO2' in data.columns:
        data['SO2'] = data['SO2'] / 60
    if 'CO'  in data.columns:
        data['CO']  = data['CO']  / 4
    if 'O3'  in data.columns:
        data['O3']  = data['O3']  / 160

    data.drop(columns=[c for c in ['PM10'] if c in data.columns], inplace=True)

    stack_cols = [c for c in _STACK_COLS_ORDER if c in data.columns]
    data['综合指数'] = data[stack_cols].sum(axis=1)
    data = data.sort_values('综合指数', ascending=True)   # 升序；横图大的在顶部
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# 图表 1：综合指数堆积横向排名条形图
# ═══════════════════════════════════════════════════════════════════════════════

def generate_ranking_bar(df: pd.DataFrame, region_col: str) -> str:
    """
    绘制各区域综合指数堆积横向排名条形图。
    堆叠顺序按各污染物总贡献降序排列（贡献最大的居底层）。
    返回保存路径。
    """
    font()
    data = _compute_index_data(df, region_col)
    if data.empty:
        return None

    stack_cols  = [c for c in _STACK_COLS_ORDER if c in data.columns]
    stack_order = data[stack_cols].sum().sort_values(ascending=False).index.tolist()

    fig, ax = plt.subplots(figsize=(12, max(6, len(data) * 0.55)))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#F8F9FA')

    bottom = np.zeros(len(data))
    for pollutant in stack_order:
        ax.barh(
            data.index, data[pollutant], left=bottom,
            color=_COLOR_MAP[pollutant],
            label=_LABEL_MAP.get(pollutant, pollutant),
            alpha=0.85, edgecolor='white', linewidth=0.5
        )
        bottom += data[pollutant].values

    for idx, val in enumerate(data['综合指数']):
        ax.text(val + 0.02, idx, f'{val:.4f}', va='center', fontweight='bold', fontsize=10)

    ax.set_xlabel('综合评价指数（数值越大，污染越严重）',
                  fontsize=12, fontweight='bold', color='#2C3E50')
    ax.set_title(f'各{region_col}空气质量综合指数排名及构成',
                 fontsize=16, pad=20, fontweight='bold', color='#2C3E50')

    legend = ax.legend(title='污染因子', bbox_to_anchor=(1.05, 1), loc='upper left',
                       fontsize=10, framealpha=0.9, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.4, color='lightgray')
    plt.tight_layout()

    path = os.path.join(_save_dir(), f"Ranking_Bar_{_ts()}.png")
    plt.savefig(path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# 图表 2：各区域污染物24小时浓度变化趋势图（6宫格折线图）
# ═══════════════════════════════════════════════════════════════════════════════

def generate_hourly_trend(df: pd.DataFrame, region_col: str) -> str:
    """
    绘制 SO2 / NO2 / CO / O3 / PM10 / PM2.5 六个污染物各区域
    24 小时平均浓度折线图（3行×2列子图），图例置于底部居中。
    返回保存路径。
    """
    font()
    df = df.copy()
    df['时间'] = pd.to_datetime(df['时间'])

    pollutant_cols = ['SO2', 'NO2', 'CO', 'O3', 'PM10', 'PM2.5']
    valid_cols = [p for p in pollutant_cols if p in df.columns]
    if not valid_cols:
        return None

    # 按区域 → 小时 聚合均值
    frames = []
    for region in df[region_col].unique():
        rdf = df[df[region_col] == region].groupby('时间')[valid_cols].mean()
        rdf['小时'] = rdf.index.hour
        rdf = rdf.groupby('小时')[valid_cols].mean()
        rdf[region_col] = region
        rdf['小时'] = rdf.index
        frames.append(rdf)

    combined = pd.concat(frames, ignore_index=True)
    pivot = combined.pivot_table(index='小时', columns=region_col,
                                 values=valid_cols, aggfunc='mean')

    regions     = list(pivot.columns.get_level_values(1).unique())
    color_dict  = dict(zip(regions, _gen_line_colors(len(regions))))

    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    fig.patch.set_facecolor('white')
    axes = axes.flatten()
    lines, labels = [], []

    for i, pollutant in enumerate(valid_cols):
        ax = axes[i]
        ax.set_facecolor('#F8F9FA')
        poll_data = pivot[pollutant]

        for region in poll_data.columns:
            line, = ax.plot(
                poll_data.index, poll_data[region],
                color=color_dict[region], linewidth=2.2, alpha=0.9, label=region
            )
            if i == 0:          # 只在第一个子图收集图例句柄
                lines.append(line)
                labels.append(region)

        ax.set_title(f'{pollutant} 各{region_col}小时浓度变化',
                     fontsize=14, fontweight='bold', pad=12, color='#2C3E50')
        ax.set_xlabel('小时', fontsize=11, color='#2C3E50')
        ax.set_ylabel('浓度值', fontsize=11, color='#2C3E50')
        ax.set_xticks(range(0, 24, 2))
        ax.set_xlim(0, 23)
        ax.grid(True, alpha=0.5, linestyle='--', color='lightgray')
        ax.tick_params(axis='both', labelsize=10)
        for spine in ax.spines.values():
            spine.set_color('lightgray')

    fig.legend(lines, labels,
               loc='lower center', bbox_to_anchor=(0.5, 0.01),
               fontsize=10, ncol=min(8, len(labels)),
               columnspacing=0.8, handlelength=2,
               frameon=True, fancybox=True, shadow=True)

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    path = os.path.join(_save_dir(), f"Hourly_Trend_{_ts()}.png")
    plt.savefig(path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# 图表 3：各区域综合指数饼状图（每区域单独一张，含排名标注）
# ═══════════════════════════════════════════════════════════════════════════════

def generate_pie_charts(df: pd.DataFrame, region_col: str) -> list:
    """
    为每个区域绘制综合指数构成饼状图，标注排名和指数和。
    排名规则：污染越重 = 排名越靠前（rank 1 = 污染最重）。
    图片按 Pie_rank01_xxx.png 格式命名，返回路径列表。
    """
    font()
    data = _compute_index_data(df, region_col)
    if data.empty:
        return []

    stack_cols  = [c for c in _STACK_COLS_ORDER if c in data.columns]
    pie_colors  = [_COLOR_MAP[c] for c in stack_cols]
    pie_labels  = [_LABEL_MAP.get(c, c) for c in stack_cols]
    total_n     = len(data)
    saved_paths = []

    # data 已按综合指数升序排列；index 0 = 污染最轻 = 排名最后
    for rank_from_bottom, (region, row) in enumerate(data.iterrows()):
        rank   = total_n - rank_from_bottom          # 排名 1 = 污染最重
        values = row[stack_cols].values.astype(float)
        total  = float(row['综合指数'])

        fig, ax = plt.subplots(figsize=(9, 9))
        fig.patch.set_facecolor('white')

        wedges, texts, autotexts = ax.pie(
            values,
            labels=pie_labels,
            colors=pie_colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 14, 'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.8},
            shadow=True,
            pctdistance=0.78,
        )
        for at in autotexts:
            at.set_color('white')
            at.set_weight('bold')
            at.set_fontsize(11)
        for t in texts:
            t.set_fontsize(12)
            t.set_fontweight('bold')

        ax.set_title(
            f'{region}\n\n综合指数排名：第{rank}名  |  指数和：{total:.4f}',
            fontsize=16, fontweight='bold', pad=30, color='#2C3E50'
        )
        ax.legend(wedges, pie_labels, title='污染物',
                  loc='lower center', bbox_to_anchor=(0.5, -0.15),
                  fontsize=11, ncol=3, frameon=True, fancybox=True)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.18)

        safe = str(region).replace('/', '_').replace('\\', '_')
        path = os.path.join(_save_dir(), f"Pie_rank{rank:02d}_{safe}_{_ts()}.png")
        plt.savefig(path, dpi=300, facecolor='white', bbox_inches='tight', edgecolor='none')
        plt.close()
        saved_paths.append(path)

    return saved_paths


# ═══════════════════════════════════════════════════════════════════════════════
# 对外统一入口：一次性生成全部图表
# ═══════════════════════════════════════════════════════════════════════════════

def generate_all(df: pd.DataFrame, region_col: str) -> dict:
    """
    一次性生成全部三类图表，返回路径字典：
      {
        'ranking_bar':  str,        # 综合指数堆积横向排名图
        'hourly_trend': str,        # 6宫格污染物小时浓度折线图
        'pie_charts':   list[str],  # 各区域综合指数饼状图（按排名命名）
      }
    调用示例：
        results = generate_all(df, '城市')
        print(results['ranking_bar'])
        print(results['hourly_trend'])
        for p in results['pie_charts']:
            print(p)
    """
    font()
    return {
        'ranking_bar':  generate_ranking_bar(df, region_col),
        'hourly_trend': generate_hourly_trend(df, region_col),
        'pie_charts':   generate_pie_charts(df, region_col),
    }