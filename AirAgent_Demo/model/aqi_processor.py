import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

# 导入你基础环境里已有的函数
from model.basic_function import font, get_resource_path
from model.IAQI import IAQI_CO, IAQI_NO2, IAQI_SO2, IAQI_PM2_5, IAQI_PM10, IAQI_O3, youliang

def calculate_daily_aqi(df: pd.DataFrame, region_col: str) -> pd.DataFrame:
    """
    Agent专用工具：计算区域每日AQI及首要污染物
    """
    # 1. 基础清理与时间索引
    df = df.copy()
    df['时间'] = pd.to_datetime(df['时间'])
    df = df.set_index('时间')
    
    pollutants = ['SO2', 'NO2', 'CO', 'O3', 'PM10', 'PM2.5']
    
    # 2. 计算每日均值与 O3 8小时滑动最大值
    def process_region(group):
        # 统计每日有效小时数 (不全为空即为有效)
        valid_hours = group[pollutants].notna().any(axis=1).resample('D').count()
        valid_days = valid_hours[valid_hours >= 20].index
        
        if len(valid_days) == 0:
            return pd.DataFrame()
            
        # O3: 8小时滑动平均的最大值
        o3_8h_max = group['O3'].rolling('8h', min_periods=1).mean().resample('D').max()
        # 其他: 24小时均值
        other_means = group[['SO2', 'NO2', 'CO', 'PM10', 'PM2.5']].resample('D').mean()
        
        daily_res = pd.concat([other_means, o3_8h_max], axis=1)
        daily_res = daily_res.loc[daily_res.index.isin(valid_days)]
        return daily_res

    # 应用向量化分组计算
    daily_df = df.groupby(region_col).apply(process_region).reset_index()
    if daily_df.empty:
        return pd.DataFrame()

    # 自动兼容 Pandas reset_index 后的列名
    if '时间' in daily_df.columns:
        daily_df = daily_df.rename(columns={'时间': '日期'})
    elif 'level_1' in daily_df.columns:
        daily_df = daily_df.rename(columns={'level_1': '日期'})

    # 3. 映射 IAQI 分指数 (向量化提速)
    daily_df['IAQI_CO'] = daily_df['CO'].apply(lambda x: IAQI_CO(x) if pd.notna(x) else np.nan)
    daily_df['IAQI_NO2'] = daily_df['NO2'].apply(lambda x: IAQI_NO2(x) if pd.notna(x) else np.nan)
    daily_df['IAQI_SO2'] = daily_df['SO2'].apply(lambda x: IAQI_SO2(x) if pd.notna(x) else np.nan)
    daily_df['IAQI_PM2.5'] = daily_df['PM2.5'].apply(lambda x: IAQI_PM2_5(x) if pd.notna(x) else np.nan)
    daily_df['IAQI_PM10'] = daily_df['PM10'].apply(lambda x: IAQI_PM10(x) if pd.notna(x) else np.nan)
    daily_df['IAQI_O3'] = daily_df['O3'].apply(lambda x: IAQI_O3(x) if pd.notna(x) else np.nan)

    # 4. 计算最终 AQI 与 首要污染物
    iaqi_cols = ['IAQI_CO', 'IAQI_NO2', 'IAQI_SO2', 'IAQI_PM2.5', 'IAQI_PM10', 'IAQI_O3']
    
    # 提取最大值作为AQI
    daily_df['AQI'] = daily_df[iaqi_cols].max(axis=1)
    
    # 获取最大值对应的列名，并去掉 'IAQI_' 前缀作为首要污染物
    def get_primary(row):
        if pd.isna(row['AQI']): return '无'
        idx = row[iaqi_cols].astype(float).idxmax()
        return idx.replace('IAQI_', '') if pd.notna(idx) else '无'
        
    daily_df['首要污染物'] = daily_df.apply(get_primary, axis=1)
    
    # 获取颜色和污染等级
    daily_df['color'] = daily_df['AQI'].apply(lambda x: youliang(x)[0] if pd.notna(x) else '#FFFFFF')
    daily_df['等级'] = daily_df['AQI'].apply(lambda x: youliang(x)[1] if pd.notna(x) else '无数据')

    return daily_df

def generate_aqi_calendar_heatmap(daily_df: pd.DataFrame, region_col: str) -> str:
    """
    生成AQI日历热力图，返回生成的图片绝对路径
    """
    if daily_df.empty:
        raise ValueError("传入的数据为空，无法绘图。")

    font() # 初始化中文字体
    
    # 整理绘图矩阵
    daily_df['日期_str'] = daily_df['日期'].dt.strftime('%m-%d')
    
    # 使用 pivot 建立区域-日期的二维表
    aqi_matrix = daily_df.pivot(index=region_col, columns='日期_str', values='AQI')
    color_matrix = daily_df.pivot(index=region_col, columns='日期_str', values='color').fillna('#FFFFFF')
    level_matrix = daily_df.pivot(index=region_col, columns='日期_str', values='等级').fillna('无数据')
    primary_matrix = daily_df.pivot(index=region_col, columns='日期_str', values='首要污染物').fillna('无')

    regions = aqi_matrix.index.tolist()
    dates = aqi_matrix.columns.tolist()
    
    # 动态画布大小
    width = max(10, len(dates) * 0.8)
    height = max(6, len(regions) * 1.2)
    fig, ax = plt.subplots(figsize=(width, height))
    
    # 隐藏坐标轴边框
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 绘制色块与文本
    for i, region in enumerate(regions):
        for j, date_str in enumerate(dates):
            color = color_matrix.iat[i, j]
            aqi_val = aqi_matrix.iat[i, j]
            level = level_matrix.iat[i, j]
            primary = primary_matrix.iat[i, j]
            
            # 画方块
            rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='white', linewidth=2)
            ax.add_patch(rect)
            
            # 填文字
            if pd.notna(aqi_val):
                text_color = 'white' if color in ['#00E400', '#8F3F97', '#7E0023', '#FF0000'] else 'black'
                content = f"{int(aqi_val)}\n{level}\n{primary}"
                ax.text(j + 0.5, i + 0.5, content, ha='center', va='center', color=text_color, fontsize=9, fontweight='bold')
            else:
                ax.text(j + 0.5, i + 0.5, "无数据", ha='center', va='center', color='#999999', fontsize=8)

    # 坐标轴设置
    ax.set_xlim(0, len(dates))
    ax.set_ylim(0, len(regions))
    ax.invert_yaxis() # 让第一个区域在最上面
    
    ax.set_xticks(np.arange(len(dates)) + 0.5)
    ax.set_xticklabels(dates, rotation=45, ha='right', fontsize=10)
    
    ax.set_yticks(np.arange(len(regions)) + 0.5)
    ax.set_yticklabels(regions, fontsize=11, fontweight='bold')
    
    ax.set_title(f'区域 AQI 及首要污染物演变日历图', fontsize=18, pad=20, fontweight='bold')
    
    # 保存并返回路径
    save_dir = get_resource_path("mapout/AQI_Agent")
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(save_dir, f"AQI_Heatmap_{timestamp}.png")
    
    plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return file_path