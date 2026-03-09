import pandas as pd
import re

def clean_env_data(df: pd.DataFrame, time_col='时间') -> pd.DataFrame:
    """智能清洗环境数据：处理中文时间、剔除机器故障导致的极值异常"""
    df = df.copy()
    
    # 1. 强力清洗中文时间字符串
    def clean_time_str(t):
        if pd.isna(t): return t
        t = str(t)
        t = re.sub(r'[年月日时分秒]', '-', t)
        t = re.sub(r'星期[一二三四五六日天]', '', t)
        t = re.sub(r'\s+', ' ', t).strip()
        t = re.sub(r'-+', '-', t)
        return t.rstrip('-')

    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col].apply(clean_time_str), errors='coerce')

    # 2. 物理极值过滤
    limits = {'SO2': 2000, 'NO2': 1500, 'CO': 100, 'O3': 1200, 'PM10': 3000, 'PM2.5': 2000}
    
    # 动态识别当前表格中实际存在的污染物列 (避免 KeyError)
    existing_pollutants = []
    for col in ['SO2', 'NO2', 'CO', 'O3', 'PM10', 'PM2.5']:
        if col in df.columns:
            existing_pollutants.append(col)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            max_limit = limits.get(col, 5000)
            df.loc[df[col] > max_limit, col] = pd.NA
            df.loc[df[col] < 0, col] = 0 # 浓度不可能为负数

    # 3. 剔除全为空值的无效行 (只针对存在的列进行过滤)
    if existing_pollutants:
        df = df.dropna(subset=existing_pollutants, how='all')
    
    return df