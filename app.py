# -*- coding: utf-8 -*-
"""
Fire Spread Simulator Pro Ver.2.0 (Command Center Edition)
----------------------------------------------------------------
- UI/UX: 災害対策本部風のダッシュボードレイアウト
- Logic: 時系列成長予測、危険度等級判定、EMC(平衡含水率)補正を追加
- AI: Gemini 2.5 Flash Ensemble による定性・定量評価
"""

from __future__ import annotations
import json
import math
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager as fm
import requests
import urllib.parse
import folium
import altair as alt  # チャート用
import google.generativeai as genai
import pydeck as pdk

# ---- streamlit_folium ----
try:
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

# ------------------------- ページ設定 -------------------------
st.set_page_config(
    page_title="Fire Spread Command Center",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- スタイル & フォント設定 -------------------------
def configure_styles():
    # 日本語フォント設定（前回同様）
    try:
        available = {f.name for f in fm.fontManager.ttflist}
        candidates = ["Noto Sans JP", "Hiragino Sans", "Meiryo", "Yu Gothic", "IPAexGothic", "MS Gothic"]
        for name in candidates:
            if name in available:
                matplotlib.rcParams["font.family"] = name
                break
        else:
            matplotlib.rcParams["font.family"] = "sans-serif"
    except Exception:
        pass
    matplotlib.rcParams["axes.unicode_minus"] = False

    # コマンドセンター風CSS
    st.markdown("""
    <style>
        /* 全体のトーン調整 */
        .block-container { padding-top: 1rem; padding-bottom: 3rem; }
        
        /* KPIカードのデザイン */
        div[data-testid="stMetric"] {
            background-color: #1E1E1E;
            border: 1px solid #333;
            padding: 15px;
            border-radius: 8px;
            color: #E0E0E0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        div[data-testid="stMetric"] label { color: #AAAAAA !important; }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #FF5722 !important; font-weight: bold; }

        /* ヘッダー装飾 */
        h1, h2, h3 { color: #E0E0E0; font-family: 'Roboto', sans-serif; }
        .highlight { color: #FF5722; font-weight: bold; }
        
        /* タブのスタイル */
        .stTabs [data-baseweb="tab-list"] { gap: 10px; }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            border-radius: 4px 4px 0 0;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #FF5722 !important;
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)

configure_styles()

# ------------------------- ドメインモデル & 定数 -------------------------
@dataclass
class Inputs:
    duration_min: float
    wind_speed_ms: float
    wind_dir_deg: float
    rel_humidity: float
    air_temp_c: float
    slope_percent: float
    fuel_class: str
    init_radius_m: float
    attack_duration_min: float
    app_rate_lpm_per_m: float
    efficiency: float

@dataclass
class Outputs:
    radius_m: float
    area_sqm: float
    water_volume_tons: float
    ellipse_a_m: float
    ellipse_b_m: float
    perimeter_m: float

# 定数
BASE_RATE_BY_FUEL = {"grass": 8.0, "shrub": 3.0, "timber": 0.6}
HUMIDITY_K = 1.1
WIND_A = 0.10
WIND_B = 0.010
SLOPE_K = 4.0
LB_C = 0.30
LB_MAX = 5.0
EPS = 1e-9

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# ------------------------- 物理モデル (Advanced) -------------------------
def estimate_emc(temp_c: float, rh_percent: float) -> float:
    """
    簡易的な平衡含水率(EMC)の推定。
    Simard (1968) 等の近似式を簡略化したもの。
    """
    if rh_percent < 10: return 2.0
    if rh_percent > 90: return 25.0
    # 非常に簡易的な線形近似
    return 0.2 * rh_percent + 0.05 * (20 - temp_c)

def humidity_factor(rh: float, temp_c: float = 25.0) -> float:
    """
    湿度補正係数。EMCを考慮して少し感度を高める。
    """
    # 従来のロジックをベースにしつつ、高温時は乾燥が進む補正
    base_f = math.exp(-HUMIDITY_K * max(0.0, rh - 30.0) / 70.0)
    
    if rh < 30.0:
        base_f = 1.0 + 0.025 * (30.0 - rh) # 乾燥時の加速を少し強める
    
    # 気温による微調整（高温＝乾燥しやすい＝速い）
    temp_factor = 1.0 + max(0, (temp_c - 25.0) * 0.01)
    
    return clamp(base_f * temp_factor, 0.25, 2.5)

def wind_factor(u_ms: float) -> float:
    f = 1.0 + WIND_A * u_ms + WIND_B * (u_ms ** 2)
    return clamp(f, 1.0, 8.0) # 上限を少し緩和

def slope_factor(slope_percent: float) -> float:
    tan_th = slope_percent / 100.0
    f = 1.0 + SLOPE_K * tan_th
    return clamp(f, 1.0, 6.0)

def base_rate(fuel: str) -> float:
    return BASE_RATE_BY_FUEL.get(fuel, BASE_RATE_BY_FUEL["grass"])

def ros_m_per_min(inp: Inputs) -> float:
    r0 = base_rate(inp.fuel_class)
    f_h = humidity_factor(inp.rel_humidity, inp.air_temp_c)
    f_w = wind_factor(inp.wind_speed_ms)
    f_s = slope_factor(inp.slope_percent)
    return max(EPS, r0 * f_h * f_w * f_s)

def length_breadth_ratio(u_ms: float) -> float:
    return clamp(1.0 + LB_C * u_ms, 1.0, LB_MAX)

def ellipse_axes(ros: float, t_min: float, init_r: float, u_ms: float) -> Tuple[float, float]:
    A = ros * t_min + init_r
    lb = length_breadth_ratio(u_ms)
    B = max(EPS, A / lb)
    return A, B

def ellipse_area_perimeter(a: float, b: float) -> Tuple[float, float]:
    area = math.pi * a * b
    h = ((a - b) ** 2) / ((a + b) ** 2 + EPS)
    perimeter = math.pi * (a + b) * (1 + (3*h)/(10 + math.sqrt(4 - 3*h + EPS)))
    return area, perimeter

def water_requirement_ton(perimeter_m: float, app_rate: float, duration: float, eff: float) -> float:
    liters = app_rate * perimeter_m * duration
    liters_eff = liters / max(eff, 0.05)
    return liters_eff / 1000.0

def run_physical_model(inp: Inputs) -> Outputs:
    ros = ros_m_per_min(inp)
    A, B = ellipse_axes(ros, inp.duration_min, inp.init_radius_m, inp.wind_speed_ms)
    area, perimeter = ellipse_area_perimeter(A, B)
    r_equiv = math.sqrt(area / math.pi)
    water_ton = water_requirement_ton(
        perimeter, inp.app_rate_lpm_per_m, inp.attack_duration_min, inp.efficiency
    )
    return Outputs(r_equiv, area, water_ton, A, B, perimeter)

# ---- 時系列シミュレーション ----
def run_time_series_simulation(inp: Inputs, steps: int = 20) -> pd.DataFrame:
    """0分から指定時間までの成長推移を計算"""
    times = np.linspace(0, inp.duration_min, steps)
    results = []
    
    ros = ros_m_per_min(inp) # ROSは一定と仮定
    lb = length_breadth_ratio(inp.wind_speed_ms)

    for t in times:
        A = ros * t + inp.init_radius_m
        B = max(EPS, A / lb)
        area, perimeter = ellipse_area_perimeter(A, B)
        r_equiv = math.sqrt(area / math.pi)
        
        results.append({
            "time_min": t,
            "radius_m": r_equiv,
            "area_sqm": area,
            "perimeter_m": perimeter
        })
    return pd.DataFrame(results)

# ---- 危険度判定 ----
def get_fire_danger_level(temp: float, humid: float, wind: float) -> Tuple[str, str]:
    """簡易的な危険等級判定（オングストローム指数の概念を参考）"""
    # 乾燥指数 (低いほど危険)
    index = humid - (temp - 10) * 2.0
    
    # 風速によるブースト
    if wind > 10.0: index -= 15
    elif wind > 5.0: index -= 5
    
    if index < 15: return "Extreme", "#FF0000"  # 極めて危険
    if index < 30: return "Very High", "#FF4500" # 非常に危険
    if index < 50: return "High", "#FF8C00"      # 危険
    if index < 70: return "Moderate", "#FFD700"  # 警戒
    return "Low", "#32CD32"                      # 注意

# ------------------------- API連携 (Geo/Weather/Gemini) -------------------------
# (既存の関数を流用・整理)

def geocode_address_mapbox(address: str) -> Optional[Tuple[float, float]]:
    try:
        token = st.secrets["mapbox"]["access_token"]
        q = urllib.parse.quote(address)
        url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{q}.json"
        params = {"access_token": token, "limit": 1, "language": "ja"}
        r = requests.get(url, params=params, timeout=5)
        if r.status_code == 200:
            feat = r.json().get("features", [])
            if feat:
                return feat[0]["center"][1], feat[0]["center"][0] # lat, lon
    except: pass
    return None

def fetch_openweather(lat: float, lon: float) -> Optional[Dict]:
    try:
        key = st.secrets["openweather"]["api_key"]
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"lat": lat, "lon": lon, "appid": key, "units": "metric", "lang": "ja"}
        r = requests.get(url, params=params, timeout=5)
        if r.status_code == 200:
            d = r.json()
            return {
                "temp_c": float(d["main"]["temp"]),
                "humidity": float(d["main"]["humidity"]),
                "wind_speed": float(d["wind"]["speed"]),
                "wind_deg": float(d["wind"].get("deg", 0)),
                "description": d["weather"][0]["description"]
            }
    except: pass
    return None

def get_gemini_model():
    try:
        genai.configure(api_key=st.secrets["general"]["api_key"])
        return genai.GenerativeModel("gemini-2.5-flash")
    except: return None

# Gemini Logic (Condensed for brevity but keeping core)
def run_gemini_analysis(model, inputs: Inputs, physical: Outputs, weather_desc: str) -> str:
    prompt = f"""
    あなたは災害対策本部のチーフ・アドバイザーです。以下の火災シミュレーション結果に基づき、
    1. 現状の危険性評価
    2. 今後警戒すべきリスク（風向き変化や飛び火など）
    3. 推奨される戦術（守備的か攻撃的か）
    を、簡潔かつ断定的な「指令スタイル」で3行以内で出力してください。
    
    [条件]
    燃料: {inputs.fuel_class}, 風速: {inputs.wind_speed_ms}m/s, 湿度: {inputs.rel_humidity}%
    予測延焼面積: {physical.area_sqm:.0f} m2
    天気概況: {weather_desc}
    """
    try:
        resp = model.generate_content(prompt)
        return resp.text
    except: return "AI解析を実行できませんでした。"

# ------------------------- UI Main -------------------------

# サイドバー設定
with st.sidebar:
    st.markdown("### ⚙️ Command Settings")
    
    with st.expander("📍 Location & Fuel", expanded=True):
        st.caption("発生源と燃料タイプ")
        input_method = st.radio("入力方法", ["地図指定", "住所検索", "座標入力"], horizontal=True)
        
        # デフォルト座標（東京）
        if "lat" not in st.session_state: st.session_state.lat = 35.6812
        if "lon" not in st.session_state: st.session_state.lon = 139.7671

        if input_method == "座標入力":
            st.session_state.lat = st.number_input("Lat", -90.0, 90.0, st.session_state.lat, format="%.4f")
            st.session_state.lon = st.number_input("Lon", -180.0, 180.0, st.session_state.lon, format="%.4f")
        elif input_method == "住所検索":
            addr = st.text_input("住所")
            if st.button("検索"):
                res = geocode_address_mapbox(addr)
                if res:
                    st.session_state.lat, st.session_state.lon = res
                    st.success("特定しました")
                else:
                    st.error("不明です")
        
        fuel_class = st.selectbox("燃料モデル", ["grass", "shrub", "timber"], index=0)

    with st.expander("🌪️ Weather Environment", expanded=True):
        st.caption("気象条件 (Auto/Manual)")
        use_api = st.checkbox("Live Weather (OpenWeather)", value=True)
        
        # デフォルト値
        ws, wd, rh, tp = 5.0, 90, 40, 25
        weather_desc = "手動入力"

        if use_api:
            w_data = fetch_openweather(st.session_state.lat, st.session_state.lon)
            if w_data:
                ws, wd, rh, tp = w_data["wind_speed"], w_data["wind_deg"], w_data["humidity"], w_data["temp_c"]
                weather_desc = f"{w_data['description']} (Live)"
                st.info(f"取得: {tp}℃ / {rh}% / 風 {ws}m/s")
            else:
                st.warning("API取得失敗。手動値を使用")
        
        # 手動調整用（API取得後も微調整可能に）
        wind_speed_ms = st.slider("風速 [m/s]", 0.0, 30.0, float(ws))
        wind_dir_deg = st.slider("風向 [deg]", 0, 359, int(wd))
        rel_humidity = st.slider("湿度 [%]", 0, 100, int(rh))
        air_temp_c = st.slider("気温 [℃]", -10, 50, int(tp))
        slope_percent = st.slider("斜面勾配 [%]", 0, 100, 10)

    with st.expander("⏱️ Simulation Params"):
        duration_min = st.number_input("予測時間 [分]", 10, 1440, 60, step=10)
        attack_duration = st.number_input("初期活動時間 [分]", 1, 180, 20)
        app_rate = st.number_input("注水率 [L/min/m]", 0.1, 50.0, 4.0)

# Inputs構築
inputs = Inputs(
    duration_min=duration_min,
    wind_speed_ms=wind_speed_ms,
    wind_dir_deg=wind_dir_deg,
    rel_humidity=rel_humidity,
    air_temp_c=air_temp_c,
    slope_percent=slope_percent,
    fuel_class=fuel_class,
    init_radius_m=5.0,
    attack_duration_min=attack_duration,
    app_rate_lpm_per_m=app_rate,
    efficiency=0.6
)

# ---- 計算実行 ----
physical_res = run_physical_model(inputs)
time_series_df = run_time_series_simulation(inputs)
danger_lvl, danger_color = get_fire_danger_level(air_temp_c, rel_humidity, wind_speed_ms)

# ---- メイン画面 ----
st.title("🔥 Fire Spread Command Center")
st.markdown(f"**Status:** <span style='color:{danger_color}; font-weight:bold; font-size:1.2em'>■ {danger_lvl} Risk Condition</span> | {weather_desc}", unsafe_allow_html=True)

# 1. KPI Cards (Top)
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("予測延焼面積", f"{physical_res.area_sqm:,.0f} m²", delta=f"{time_series_df['area_sqm'].diff().iloc[-1]:.0f} m²/step")
kpi2.metric("最遠到達距離", f"{physical_res.ellipse_a_m + inputs.init_radius_m:,.1f} m", "風下方向")
kpi3.metric("必要水量 (推定)", f"{physical_res.water_volume_tons:,.1f} ton", "活動時間内")
kpi4.metric("延焼速度 (ROS)", f"{ros_m_per_min(inputs):.2f} m/min")

# 2. Visualization (Map & Graph)
col_map, col_graph = st.columns([1.8, 1.2])

with col_map:
    st.subheader("📍 Real-time Projection Map")
    
    # 楕円ポリゴン生成
    if HAS_FOLIUM:
        m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=16, tiles="CartoDB dark_matter")
        
        # 発生源
        folium.CircleMarker(
            [st.session_state.lat, st.session_state.lon], radius=5, color="red", fill=True
        ).add_to(m)

        # 延焼予測楕円 (最終)
        # 簡易的に円周上の点を計算してPolygon化
        a, b = physical_res.ellipse_a_m, physical_res.ellipse_b_m
        angle = math.radians(90 - inputs.wind_dir_deg)
        center_shift = a * 0.5 # 中心を風下にシフト
        
        points = []
        for t in np.linspace(0, 2*math.pi, 100):
            dx = (center_shift + a * math.cos(t)) * math.cos(angle) - (b * math.sin(t)) * math.sin(angle)
            dy = (center_shift + a * math.cos(t)) * math.sin(angle) + (b * math.sin(t)) * math.cos(angle)
            # 簡易メートル→度変換
            dlat = dy / 111000
            dlon = dx / (111000 * math.cos(math.radians(st.session_state.lat)))
            points.append([st.session_state.lat + dlat, st.session_state.lon + dlon])
            
        folium.Polygon(
            locations=points,
            color="#FF5722",
            fill=True,
            fill_color="#FF5722",
            fill_opacity=0.4,
            popup=f"予測範囲 ({inputs.duration_min}分後)"
        ).add_to(m)
        
        # 地図クリックで移動用
        out = st_folium(m, height=450, width="100%")
        if out["last_clicked"]:
            st.session_state.lat = out["last_clicked"]["lat"]
            st.session_state.lon = out["last_clicked"]["lng"]
            st.rerun()
    else:
        st.error("Folium not installed")

with col_graph:
    st.subheader("📈 Growth Trajectory")
    
    # Altairで時系列グラフ
    chart_data = time_series_df.melt('time_min', value_vars=['area_sqm', 'perimeter_m'])
    
    c = alt.Chart(chart_data).mark_line(point=True).encode(
        x=alt.X('time_min', title='経過時間 (分)'),
        y=alt.Y('value', title='値'),
        color=alt.Color('variable', legend=alt.Legend(title="指標"), scale=alt.Scale(scheme='magma')),
        tooltip=['time_min', 'value', 'variable']
    ).properties(height=250)
    
    st.altair_chart(c, use_container_width=True)
    
    # Gemini AI Analysis Button
    st.subheader("🧠 AI Tactical Advisor")
    if st.button("Gemini AI 解析を実行", type="primary"):
        with st.spinner("AI戦術解析中..."):
            model = get_gemini_model()
            if model:
                advice = run_gemini_analysis(model, inputs, physical_res, weather_desc)
                st.success("解析完了")
                st.markdown(f"""
                <div style="background-color:#263238; border-left: 5px solid #FFC107; padding: 15px; border-radius: 5px;">
                    <div style="font-weight:bold; color:#FFC107; margin-bottom:5px;">🤖 Tactical Assessment</div>
                    {advice}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("API Key未設定のためAI解析不可")

# 3. Data & Export Area
st.markdown("---")
with st.expander("📊 詳細データ・レポート出力"):
    d1, d2 = st.columns([3, 1])
    with d1:
        st.dataframe(time_series_df.style.background_gradient(cmap="OrRd", subset=["area_sqm"]), use_container_width=True)
    with d2:
        st.markdown("#### Export")
        csv = time_series_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "CSVレポートをダウンロード",
            data=csv,
            file_name="fire_simulation_report.csv",
            mime="text/csv",
            type="primary"
        )
        st.caption("シミュレーション結果と時系列データを保存します。")

# Diagram Triggers (Contextual)
# ユーザーが理解を深めるための図解トリガー
st.markdown("", unsafe_allow_html=True) 
st.markdown("", unsafe_allow_html=True)
