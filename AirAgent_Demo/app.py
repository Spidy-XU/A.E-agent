import streamlit as st
import pandas as pd
import os
import sys
import json
import re
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from model.data_cleaner import clean_env_data
    from model.anomaly_processor import generate_anomaly_plot, generate_anomaly_all
    from model.aqi_processor import calculate_daily_aqi, generate_aqi_calendar_heatmap
    from model.cluster_processor import generate_cluster_radar, generate_cluster_all
    from model.increment_processor import generate_increment_heatmap, generate_increment_all
    from model.ranking_processor import generate_all
    from model.feature_processor import generate_feature_scatter, generate_pollution_radar, generate_global_quadrant_scatter
except ImportError as e:
    st.error(f"导入算法模块失败，请检查 model 文件夹。错误: {e}")

st.set_page_config(page_title="大气环境分析 AI 助手", page_icon="🌍", layout="wide")
st.title("🌍 大气污染智能体 (全能分析版)")

TEMP_DIR = "temp_data"
os.makedirs(TEMP_DIR, exist_ok=True)

with st.sidebar:
    st.header("🔑 AI 大脑配置")
    api_key = st.text_input("请输入硅基流动 API Key (sk-...)", type="password")
    st.divider()
    st.header("📂 数据输入")
    uploaded_file = st.file_uploader("请上传空气质量数据 (Excel/CSV)", type=["xlsx", "xls", "csv"])
    if uploaded_file:
        file_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("数据上传成功！")
        
    st.divider()
    st.header("💡 快捷指令参考")
    st.info(
        "💡 **默认按【市级】分析，需要看细分数据请带上“区县”或“站点”字眼**\n\n"
        "1. `画 AQI 趋势热力图`\n"
        "2. `分析各市的颗粒物多维特征`\n"
        "3. `分析各区县的颗粒物特征`\n"
        "4. `污染溯源聚类分析`\n"
        "5. `距平分析`\n"
        "5. `增量分析`\n"
        "6. `根据上传数据，进行综合指数排名`"
    )

client = None
if api_key:
    client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")

def llm_map_columns(original_columns):
    prompt = f"找出代表以下9个字段的实际列名：['时间', '城市', '区县', 'SO2', 'NO2', 'CO', 'O3', 'PM10', 'PM2.5']。表头为：{original_columns}。输出 JSON。如表内只有站点没有区县，可将站点名填入区县。"
    try:
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3.2", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        content = response.choices[0].message.content
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1:
            return json.loads(content[start:end+1])
        return {}
    except:
        return {}

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "你好！请上传数据。系统默认进行市级数据分析，如需分析区县站点可指令我“分析区县数据”。"}]

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if "images" in msg and len(msg["images"]) > 0:
            # 判断是否为当前对话的最后一条消息
            is_latest = (i == len(st.session_state.messages) - 1)
            
            if is_latest:
                # 最新生成的图表，直接大图展示
                for img in msg["images"]: 
                    st.image(img, use_container_width=True)
            else:
                # 历史图表，放入折叠面板中，并使用三列布局作为“缩略图”
                with st.expander(f"🖼️ 查看历史生成的 {len(msg['images'])} 张图表", expanded=False):
                    cols = st.columns(3)
                    for idx, img in enumerate(msg["images"]):
                        cols[idx % 3].image(img, use_container_width=True)

if prompt := st.chat_input("请输入需求..."):
    with st.chat_message("user"): st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        is_aqi_req = any(k in prompt.upper() for k in ["AQI", "趋势", "日历", "热力图"])
        is_feature_req = any(k in prompt.upper() for k in ["特征", "颗粒物", "PM", "散点", "多维"])
        is_cluster_req = any(k in prompt.upper() for k in ["溯源", "聚类", "雷达图", "来源"])
        is_anomaly_req = any(k in prompt.upper() for k in ["距平", "拖后腿", "平均与差值"])
        is_increment_req = any(k in prompt.upper() for k in ["增量", "爆发", "突变"])
        is_ranking_req = any(k in prompt.upper() for k in ["排名", "综合指数", "成绩单", "最差"])
        
        needs_analysis = is_aqi_req or is_feature_req or is_cluster_req or is_anomaly_req or is_increment_req or is_ranking_req
        
        if needs_analysis and api_key and uploaded_file:
            with st.spinner("AI 正在解析层级结构与核心数据..."):
                try:
                    df = pd.read_csv(file_path, encoding='utf-8') if file_path.endswith('.csv') else pd.read_excel(file_path, engine='openpyxl')
                    
                    # 1. AI 智能映射
                    mapping = llm_map_columns(df.columns.tolist())
                    rename_dict = {v: k for k, v in mapping.items() if v}
                    df.rename(columns=rename_dict, inplace=True)
                    
                    # 2. 暴力安全兜底：强制抓取时间与区域
                    if '时间' not in df.columns:
                        for col in df.columns:
                            if any(k in str(col) for k in ['时间', '日期']): df.rename(columns={col: '时间'}, inplace=True); break
                    
                    # 3. 暴力安全兜底：强制抓取污染物(无视括号和前后缀)
                    for p in ['SO2', 'NO2', 'CO', 'O3', 'PM10', 'PM2.5']:
                        if p not in df.columns:
                            for col in df.columns:
                                col_upper = str(col).upper()
                                if p == 'PM2.5' and ('PM2.5' in col_upper or 'PM25' in col_upper):
                                    df.rename(columns={col: p}, inplace=True); break
                                elif p in col_upper and 'COLOR' not in col_upper and 'AQI' not in col_upper:
                                    df.rename(columns={col: p}, inplace=True); break
                    
                    # 4. 判断分析级别
                    if "区县" in prompt or "站点" in prompt or "详细" in prompt:
                        analysis_level = "区县"
                    else:
                        analysis_level = "城市" 
                    
                    if analysis_level not in df.columns:
                        available_levels = [col for col in ['城市', '区县', '站点名称', '分析区域', '位置'] if col in df.columns]
                        if available_levels:
                            analysis_level = available_levels[0]
                        else:
                            st.error("❌ 数据解析失败，表格中既找不到城市列，也找不到区县列。")
                            analysis_level = None

                    if analysis_level and '时间' in df.columns:
                        st.info(f"📍 正在按【{analysis_level}】级别为您进行数据聚合与分析...")
                        
                        # 核心清洗与绘图路由
                        df = clean_env_data(df, time_col='时间')
                        generated_images = []
                        unique_regions = df[analysis_level].dropna().unique()
                        
                        if is_aqi_req:
                            daily_df = calculate_daily_aqi(df, analysis_level)
                            if not daily_df.empty:
                                img = generate_aqi_calendar_heatmap(daily_df, analysis_level)
                                st.image(img, use_container_width=True)
                                generated_images.append(img)
                            msg_content = f"✅ 【{analysis_level}级】AQI 趋势日历已生成。"
                                
                        elif is_feature_req:
                            st.write(f"正在执行横向特征比对分析（所有图片将分类保存在 Feature_Agent/{analysis_level} 目录下）...")
                            
                            st.markdown(f"### 🌐 1. 全局污染特征四象限分布 ({analysis_level}级)")
                            
                            # 全局各市四象限分布大图
                            img_quad = generate_global_quadrant_scatter(df, analysis_level, level=analysis_level)
                            if img_quad:
                                st.image(img_quad, use_container_width=True)
                                generated_images.append(img_quad)
                                
                            st.divider()
                            st.markdown(f"### 📍 2. 各【{analysis_level}】污染特征专属雷达图")
                            
                            # 为每个城市单独画雷达图（一行排三个）浓度图回答"相对于其他城市，哪种污染物偏高/偏低？特征图回答"污染物有没有超标？超了多少倍？"
                            cols_radar = st.columns(3)
                            for idx, region in enumerate(unique_regions):
                                img_radar = generate_pollution_radar(df[df[analysis_level] == region], region, level=analysis_level)
                                if img_radar:
                                    with cols_radar[idx % 3]: 
                                        st.image(img_radar, use_container_width=True)
                                    generated_images.append(img_radar)

                            # 新增：颗粒物散点图
                            st.divider()
                            st.markdown(f"### 🔵 3. 各【{analysis_level}】颗粒物关联散点图")
                            cols_scatter = st.columns(3)
                            for idx, region in enumerate(unique_regions):
                                img_scatter = generate_feature_scatter(df[df[analysis_level] == region], region, level=analysis_level)
                                if img_scatter:
                                    with cols_scatter[idx % 3]:
                                        st.image(img_scatter, use_container_width=True)
                                    generated_images.append(img_scatter)

                            msg_content = f"✅ 【{analysis_level}级】四象限 + 雷达图 + 颗粒物散点图生成完毕！"
                            
                        elif is_cluster_req:
                            with st.spinner("正在执行 KMeans 聚类分析，数据量较大请稍候..."):
                                results = generate_cluster_all(df, analysis_level)

                            # 图表4：城市污染分类特征雷达图（3×3总图）
                            if results['type_radar']:
                                st.markdown(f"### 🕸️ 1. {analysis_level}污染分类特征雷达图")
                                st.image(results['type_radar'], use_container_width=True)
                                generated_images.append(results['type_radar'])

                            # 图表3：城市污染分类因子贡献饼图（3×3总图）
                            if results['factor_pie']:
                                st.markdown(f"### 🥧 2. {analysis_level}污染分类因子综合指数贡献饼图")
                                st.image(results['factor_pie'], use_container_width=True)
                                generated_images.append(results['factor_pie'])

                            # 图表5：城市污染物-污染类型箱图
                            if results['box']:
                                st.markdown(f"### 📦 3. {analysis_level}各污染类型污染物分布箱线图")
                                st.image(results['box'], use_container_width=True)
                                generated_images.append(results['box'])

                            # 图表2：各城市污染特征综合分析图（双饼+堆积柱，两列并排）
                            if results['composite']:
                                st.markdown(f"### 📊 4. 各{analysis_level}污染特征综合分析图")
                                cols = st.columns(2)
                                for i, path in enumerate(results['composite']):
                                    cols[i % 2].image(path, use_container_width=True)
                                generated_images.extend(results['composite'])

                            # 图表1：各城市污染等级特征雷达图（两列并排）
                            if results['grade_radar']:
                                st.markdown(f"### 🎯 5. 各{analysis_level}污染等级特征雷达图")
                                cols = st.columns(2)
                                for i, path in enumerate(results['grade_radar']):
                                    cols[i % 2].image(path, use_container_width=True)
                                generated_images.extend(results['grade_radar'])

                            msg_content = (
                                f"✅ 【{analysis_level}级】聚类分析完成，已生成：\n"
                                f"分类特征雷达图、因子贡献饼图、污染类型箱线图、"
                                f"各{analysis_level}综合分析图、各{analysis_level}等级特征雷达图。"
                            )

                        elif is_anomaly_req:
                            with st.spinner("正在计算污染物距平，请稍候..."):
                                anomaly_results = generate_anomaly_all(df, analysis_level)

                            # 图表1：日距平 + AQI 时间序列（两列并排）
                            if anomaly_results["daily"]:
                                st.markdown(f"### 📉 1. 各{analysis_level}污染物日距平与AQI时间序列")
                                cols = st.columns(2)
                                for i, path in enumerate(anomaly_results["daily"]):
                                    cols[i % 2].image(path, use_container_width=True)
                                generated_images.extend(anomaly_results["daily"])

                            # 图表2：24小时距平 + AQI 时间序列（两列并排）
                            if anomaly_results["h24"]:
                                st.markdown(f"### 🕐 2. 各{analysis_level} 24小时污染物距平与AQI时间序列")
                                cols = st.columns(2)
                                for i, path in enumerate(anomaly_results["h24"]):
                                    cols[i % 2].image(path, use_container_width=True)
                                generated_images.extend(anomaly_results["h24"])

                            msg_content = (
                                f"✅ 【{analysis_level}级】距平分析完成，已生成："
                                f"各{analysis_level}日距平图 × {len(anomaly_results['daily'])} 张、"
                                f"24小时距平图 × {len(anomaly_results['h24'])} 张。"
                            )
                            
                        elif is_increment_req:
                            with st.spinner("正在计算污染物增量变化，请稍候..."):
                                increment_paths = generate_increment_all(df, analysis_level)

                            if increment_paths:
                                st.markdown(f"### 📈 各{analysis_level}污染物小时/日增量变化分析")
                                cols_inc = st.columns(2)
                                for i, path in enumerate(increment_paths):
                                    cols_inc[i % 2].image(path, use_container_width=True)
                                generated_images.extend(increment_paths)
                            msg_content = (
                                f"✅ 【{analysis_level}级】增量分析完成，"
                                f"已生成各{analysis_level}污染物小时/日增量变化图 × {len(increment_paths)} 张。"
                            )
                            
                        elif is_ranking_req:
                            results = generate_all(df, analysis_level)

                            # 1. 堆积排名条形图
                            if results['ranking_bar']:
                                st.image(results['ranking_bar'], use_container_width=True)
                                generated_images.append(results['ranking_bar'])

                            # 2. 24小时浓度趋势图
                            if results['hourly_trend']:
                                st.image(results['hourly_trend'], use_container_width=True)
                                generated_images.append(results['hourly_trend'])

                            # 3. 各区域饼状图（逐张展示）
                            if results['pie_charts']:
                                cols = st.columns(2)  # 两列并排展示
                                for i, pie_path in enumerate(results['pie_charts']):
                                    cols[i % 2].image(pie_path, use_container_width=True)
                                generated_images.extend(results['pie_charts'])

                            msg_content = f"✅ 【{analysis_level}级】综合评价指数排名、小时趋势及各{analysis_level}饼图已生成。"

                        if generated_images:
                            st.success(msg_content)
                            st.session_state.messages.append({"role": "assistant", "content": msg_content, "images": generated_images})
                        else:
                            st.error("分析失败，有效数据不足以生成图表。")
                            
                except Exception as e:
                    st.error(f"处理数据时出错: {e}")
                    
        elif needs_analysis and not api_key:
            st.warning("请先在侧边栏输入 API Key。")
        elif needs_analysis and not uploaded_file:
            st.warning("请先在侧边栏上传数据表格。")
        else:
            # 移除 st.spinner，直接在聊天框打字
            chat_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages if "images" not in m]
            try:
                # 1. 在 API 请求中加入 stream=True
                stream_response = client.chat.completions.create(
                    model="deepseek-ai/DeepSeek-V3.2",
                    messages=[{"role": "system", "content": "你是一个专业的大气环境分析智能体，请用专业且友好的语气回答用户问题。"}] + chat_history,
                    stream=True  # 开启流式传输
                )
                
                # 2. 解析流式数据块的生成器函数
                def generate_stream():
                    for chunk in stream_response:
                        if chunk.choices[0].delta.content is not None:
                            yield chunk.choices[0].delta.content

                # 3. 使用 st.write_stream 实现打字机效果
                reply = st.write_stream(generate_stream())
                
                # 4. 将完整回复存入历史
                st.session_state.messages.append({"role": "assistant", "content": reply})
                
            except Exception as e:
                st.error(f"调用 API 失败：{e}")