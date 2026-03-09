import numpy as np
import pandas as pd
import math

# 使用字典统一管理各类污染物的国家标准折点 (Breakpoints)
# 格式: {污染物: (浓度折点列表, 对应的IAQI折点列表)}
AQI_STANDARDS = {
    'CO': ([0, 2, 4, 14, 24, 36, 48], [0, 50, 100, 150, 200, 300, 500]),
    'NO2': ([0, 40, 80, 180, 280, 565, 940], [0, 50, 100, 150, 200, 300, 500]),
    'SO2': ([0, 50, 150, 475, 800, 1600, 2620], [0, 50, 100, 150, 200, 300, 500]),
    'PM2.5': ([0, 35, 75, 115, 150, 250, 500], [0, 50, 100, 150, 200, 300, 500]),
    'PM10': ([0, 50, 150, 250, 350, 420, 600], [0, 50, 100, 150, 200, 300, 500]),
    'O3': ([0, 100, 160, 215, 265, 800, 1200], [0, 50, 100, 150, 200, 300, 500]) # 整合了8h和1h的高值
}

def _calculate_iaqi(val: float, pollutant: str) -> float:
    """通用 IAQI 计算引擎"""
    if pd.isna(val) or not isinstance(val, (int, float)) or math.isinf(val) or val < 0:
        return np.nan
    bp_c, bp_i = AQI_STANDARDS[pollutant]
    val = min(val, bp_c[-1]) # 限制在最大折点内
    iaqi = np.interp(val, bp_c, bp_i)
    return math.ceil(iaqi)

# 确保 aqi_processor.py 能够无缝调用
def IAQI_CO(x): return _calculate_iaqi(x, 'CO')
def IAQI_NO2(x): return _calculate_iaqi(x, 'NO2')
def IAQI_SO2(x): return _calculate_iaqi(x, 'SO2')
def IAQI_PM2_5(x): return _calculate_iaqi(x, 'PM2.5')
def IAQI_PM10(x): return _calculate_iaqi(x, 'PM10')
def IAQI_O3(x): return _calculate_iaqi(x, 'O3')

def youliang(aqi: float) -> tuple:
    """根据 AQI 返回 (十六进制颜色代码, 污染级别)"""
    if pd.isna(aqi) or aqi < 0:
        return ('#FFFFFF', '无数据')
    
    levels = [
        (50, '#00E400', '优'),
        (100, '#FFFF00', '良'),
        (150, '#FF7E00', '轻度污染'),
        (200, '#FF0000', '中度污染'),
        (300, '#8F3F97', '重度污染'),
        (float('inf'), '#7E0023', '严重污染')
    ]
    
    for threshold, color, label in levels:
        if aqi <= threshold:
            return (color, label)
    return ('#7E0023', '严重污染')