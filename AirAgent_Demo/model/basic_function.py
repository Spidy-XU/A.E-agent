import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. 核心路径管理 
# ==========================================
def get_root_path() -> str:
    """获取程序根目录，兼容 PyInstaller 打包环境"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_resource_path(relative_path: str) -> Path:
    """获取资源绝对路径"""
    return Path(get_root_path()) / relative_path

def get_model_path() -> Path:
    """获取 model 文件夹绝对路径"""
    return Path(get_root_path()) / "model"

def check_and_create_folder(folder_path):
    """安全创建文件夹目录"""
    if folder_path:
        os.makedirs(str(folder_path), exist_ok=True)

# ==========================================
# 2. 全局画图设置 
# ==========================================
def font():
    """初始化 matplotlib 全局中文字体设置"""
    # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 3. 数据与文件读写强化
# ==========================================
def open_excel_by_version(file_path, hd=0) -> pd.DataFrame:
    """自动兼容并健壮读取 CSV 或 Excel 数据"""
    file_str = str(file_path)
    if not os.path.exists(file_str):
        raise FileNotFoundError(f"文件不存在: {file_str}")
        
    if file_str.lower().endswith('.csv'):
        try:
            return pd.read_csv(file_str, header=hd, encoding='utf-8')
        except UnicodeDecodeError:
            return pd.read_csv(file_str, header=hd, encoding='gbk')
    else:
        return pd.read_excel(file_str, header=hd, engine='openpyxl')

def save(result, save_path):
    """统一下游数据保存接口"""
    file_path = str(save_path)
    check_and_create_folder(os.path.dirname(file_path))
    
    # 强制转为 DataFrame 确保安全保存
    df = result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)
    df.to_excel(file_path, index=False, engine='openpyxl')
    print(f"数据已安全存至: {file_path}")

# ==========================================
# 4. 时间格式智能清洗
# ==========================================
def clean_chinese_time(time_str) -> str:
    """极致清洗夹杂中文字符的时间字符串"""
    if pd.isna(time_str):
        return time_str
    
    s = str(time_str)
    # 替换中文时间单位为横杠
    s = re.sub(r'[年月日时分秒]', '-', s)
    # 剔除星期几的干扰
    s = re.sub(r'星期[一二三四五六日天]', '', s)
    # 收敛多余空格与横杠
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'-+', '-', s)
    return s.rstrip('-')

def parse_custom_chinese_datetime(df: pd.DataFrame, time_col='时间', verbose=True) -> pd.DataFrame:
    """自动识别并解析异常时间列格式"""
    res_df = df.copy()
    if time_col not in res_df.columns:
        if verbose:
            print(f"提示: 数据中未发现【{time_col}】列。")
        return res_df

    # 剔除隐形空格后走清洗管道
    cleaned_series = res_df[time_col].astype(str).str.replace(' ', '').str.strip().apply(clean_chinese_time)
    res_df[time_col] = pd.to_datetime(cleaned_series, errors='coerce')
    return res_df

def get_month_week(date_val) -> int:
    """计算某个日期属于当月的第几周"""
    if pd.isna(date_val):
        return np.nan
    dt = pd.to_datetime(date_val)
    first_day = dt.replace(day=1)
    # 日期偏移计算逻辑
    return (dt.day + first_day.weekday()) // 7 + 1

# ==========================================
# 5. 数学与标准化计算
# ==========================================
def min_max_normalize(series: pd.Series) -> pd.Series:
    """最小-最大归一化 (0-1区间)"""
    range_val = series.max() - series.min()
    if range_val == 0:
        return pd.Series(0, index=series.index)
    return (series - series.min()) / range_val

def z_score_normalize(series: pd.Series) -> pd.Series:
    """Z-Score 标准化 (均值为0，方差为1)"""
    std_val = series.std()
    if std_val == 0:
        return pd.Series(0, index=series.index)
    return (series - series.mean()) / std_val

# ==========================================
# 6. 老旧业务兼容 
# ==========================================
def identify_pollution_source(c: dict) -> str:
    """
    老版本的硬编码溯源规则 (已在 cluster_processor 中升级为动态引擎)
    此处仅为兼容未重写的旧模块保持函数签名存在。
    """
    if c.get('PM10', 0) > 0.8 and c.get('PM2.5', 0) < 0.3:
        return "扬尘源"
    elif c.get('SO2', 0) > 0.8:
        return "燃煤源"
    elif c.get('NO2', 0) > 0.8:
        return "机动车源"
    elif c.get('O3', 0) > 1.0:
        return "二次转化源"
    return "复合型污染源"

if __name__ == '__main__':
    print("环境底座 basic_function.py 初始化完毕。")
