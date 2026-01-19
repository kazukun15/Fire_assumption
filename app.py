# -*- coding: utf-8 -*-
"""
Fire Spread Simulator Pro Ver.3.0 (Command Center Edition)
----------------------------------------------------------------
- UI/UX: Cyberpunk/Tactical Dashboard Style
- Logic: FARSITEベースの数理モデル (維持)
- AI: Tactical Advisor Integration
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import requests
import urllib.parse
import altair as alt
import google.generativeai as genai

# ---- ライブラリの存在チェック ----
try:
    from streamlit_folium import st_folium
    import folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

# ------------------------- ページ設定 (最初に行う必要があります) -------------------------
st.set_page_config(
    page_title="FIRE COMMAND CENTER",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- UI UX デザインシステム (CSS Injection) -------------------------
def inject_custom_css():
    st.markdown("""
    <style>
        /* 全体の背景とフォント */
        .stApp {
            background-color: #0E1117;
            font-family: 'Roboto Mono', monospace;
        }
        
        /* タイトルまわり */
        h1, h2, h3 {
            color: #E0E0E0;
            font-weight: 600;
            letter-spacing: 0.1em;
        }

        /* サイドバー */
        section[data-testid="stSidebar"] {
            background-color: #161B22;
            border-right: 1px solid #30363D;
        }

        /* メトリックカード (HUD風) */
        div[data-testid="stMetric"] {
            background-color: #21262D;
            border: 1px solid #30363D;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            transition: transform 0.2s;
        }
        div[data-testid="stMetric"]:hover {
            border-color: #FF5722;
            transform: translateY(-2px);
        }
        div[data-testid="stMetric"] label {
            color: #8B949E !important;
            font-size: 0.8rem !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #F0F6FC !important;
            font-size: 1.8rem !important;
            font-weight: 700;
            text-shadow: 0 0 10px rgba(255, 87, 34, 0.3);
        }

        /* 危険度バッジ */
        .danger-badge {
            padding: 5px 10px;
            border-radius: 4px;
            font-weight: bold;
            color: #fff;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* タブのスタイル */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #0E1117;
        }
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            background-color: #21262D;
            border-radius: 4px;
            border: 1px solid #30363D;
            color: #8B949E;
        }
        .stTabs [aria-selected="true"] {
            background-color: #FF5722 !important;
            color: white !important;
            border-color: #FF5722 !important;
        }

        /* ボタン */
        .stButton button {
            background-color: #238636;
            color: white;
            border: none;
            font-weight: bold;
            transition: all 0.3s;
        }
        .stButton button:hover {
            background-color: #2EA043;
            box-shadow: 0 0 15px rgba(46, 160, 67, 0.5);
        }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

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

# ------------------------- 物理モデル (Logic) -------------------------
# ロジックは信頼性を担保するため、元のコードを維持しています

def humidity_factor(rh: float, temp_c: float = 25.0) -> float:
    base_f = math.exp(-HUMIDITY_K * max(0.0, rh - 30.0) / 70.0)
    if rh < 30.0:
        base_f = 1.0 + 0.025 * (30.0 - rh)
    temp_factor = 1.0 + max(0, (temp_c - 25.0) * 0.01)
    return clamp(base_f * temp_factor, 0.25, 2.5)

def wind_factor(u_ms: float) -> float:
    f = 1.0 + WIND_A * u_ms + WIND_B * (u_ms ** 2)
    return clamp(f, 1.0, 8.0)

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

def run_time_series_simulation(inp: Inputs, steps: int = 20) -> pd.DataFrame:
    times = np.linspace(0, inp.duration_min, steps)
    results = []
    ros = ros_m_per_min(inp)
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

def get_fire_danger_level(temp: float, humid: float, wind: float) -> Tuple[str, str]:
    # 乾燥指数 (近似)
    index = humid - (temp - 10) * 2.0
    if wind > 10.0: index -= 15
    elif wind > 5.0: index -= 5
    
    if index < 15: return "EXTREME", "#FF0033"
    if index < 30: return "VERY HIGH", "#FF6600"
    if index < 50: return "HIGH", "#FF9900"
    if index < 70: return "MODERATE", "#FFCC00"
    return "LOW", "#00CC66"

# ------------------------- API連携 (Robust Wrappers) -------------------------

def geocode_address_mapbox(address: str) -> Optional[Tuple[float, float]]:
    # mapboxの設定がない場合はスキップ
    if "mapbox" not in st.secrets: return None
    try:
        token = st.secrets["mapbox"]["access_token"]
        q = urllib.parse.quote(address)
        url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{q}.json"
        params = {"access_token": token, "limit": 1, "language": "ja"}
        r = requests.get(url, params=params, timeout=3)
        if r.status_code == 200:
            feat = r.json().get("features", [])
            if feat:
                return feat[0]["center"][1], feat[0]["center"][0]
    except: pass
    return None

def fetch_openweather(lat: float, lon: float) -> Optional[Dict]:
    if "openweather" not in st.secrets: return None
    try:
        key = st.secrets["openweather"]["api_key"]
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"lat": lat, "lon": lon, "appid": key, "units": "metric", "lang": "ja"}
        r = requests.get(url, params=params, timeout=3)
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
    if "general" not in st.secrets or "api_key" not in st.secrets["general"]:
        return None
    try:
        genai.configure(api_key=st.secrets["general"]["api_key"])
        return genai.GenerativeModel("gemini-2.5-flash")
    except: return None

def run_gemini_analysis(model, inputs: Inputs, physical: Outputs, weather_desc: str) -> str:
    prompt = f"""
    You are a tactical fire advisor. Provide a brief, mission-critical assessment.
    Language: Japanese.
    
    [Situation]
    Fuel: {inputs.fuel_class}
    Wind: {inputs.wind_speed_ms}m/s (Dir: {inputs.wind_dir_deg})
    Humidity: {inputs.rel_humidity}%
    Predicted Area: {physical.area_sqm:.0f} m2
    Weather: {weather_desc}
    
    [Output Format]
    1. Threat Level Assessment (1 sentence)
    2. Critical Risk Factor (Bullet point)
    3. Recommended Action (Direct command style)
    """
    try:
        resp = model.generate_content(prompt)
        return resp.text
    except: return "COMMUNICATION ERROR: AI Advisor Unreachable."

# ------------------------- メインアプリケーション -------------------------

# サイドバー：コントロールパネル
with st.sidebar:
    st.markdown("### 🎛️ CONTROL PANEL")
    
    with st.expander("📍 TARGET LOCATION", expanded=True):
        input_method = st.radio("", ["Coordinates", "Address Search"], horizontal=True, label_visibility="collapsed")
        
        # デフォルト座標（東京）
        if "lat" not in st.session_state: st.session_state.lat = 35.6812
        if "lon" not in st.session_state: st.session_state.lon = 139.7671

        if input_method == "Coordinates":
            c1, c2 = st.columns(2)
            with c1:
                st.session_state.lat = st.number_input("Lat", -90.0, 90.0, st.session_state.lat, format="%.4f")
            with c2:
                st.session_state.lon = st.number_input("Lon", -180.0, 180.0, st.session_state.lon, format="%.4f")
        else:
            addr = st.text_input("Address", placeholder="例: 東京都千代田区...")
            if st.button("LOCATE TARGET"):
                res = geocode_address_mapbox(addr)
                if res:
                    st.session_state.lat, st.session_state.lon = res
                    st.success("Target Locked.")
                else:
                    st.error("Target Not Found.")
        
        st.markdown("---")
        fuel_class = st.selectbox("FUEL MODEL", ["grass", "shrub", "timber"], index=0, format_func=lambda x: x.upper())

    with st.expander("🌪️ ENVIRONMENT & WEATHER", expanded=True):
        use_api = st.checkbox("LIVE DATA LINK", value=True)
        
        ws, wd, rh, tp = 5.0, 90, 40, 25
        weather_desc = "MANUAL INPUT"

        if use_api:
            w_data = fetch_openweather(st.session_state.lat, st.session_state.lon)
            if w_data:
                ws, wd, rh, tp = w_data["wind_speed"], w_data["wind_deg"], w_data["humidity"], w_data["temp_c"]
                weather_desc = f"{w_data['description'].upper()} (LIVE)"
                st.caption(f"📡 LINK ESTABLISHED: {tp}℃ / {rh}%")
            else:
                st.caption("⚠️ LINK OFFLINE: Using Manual Data")
        
        # 手動オーバーライド
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            wind_speed_ms = st.number_input("WIND (m/s)", 0.0, 30.0, float(ws))
            rel_humidity = st.number_input("HUMIDITY (%)", 0, 100, int(rh))
        with col_w2:
            wind_dir_deg = st.number_input("DIR (deg)", 0, 359, int(wd))
            air_temp_c = st.number_input("TEMP (℃)", -10, 50, int(tp))
        
        slope_percent = st.slider("SLOPE GRADE (%)", 0, 100, 10)

    with st.expander("⏱️ TACTICAL PARAMS"):
        duration_min = st.slider("PREDICTION WINDOW (min)", 10, 240, 60, step=10)
        attack_duration = st.slider("OPS DURATION (min)", 10, 180, 60)
        app_rate = st.number_input("WATER RATE (L/min/m)", 0.1, 50.0, 4.0)

# Inputs構築 & 計算
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

physical_res = run_physical_model(inputs)
time_series_df = run_time_series_simulation(inputs)
danger_lvl, danger_color = get_fire_danger_level(air_temp_c, rel_humidity, wind_speed_ms)

# ========================== MAIN DASHBOARD ==========================

# 1. HEADER AREA
c_head1, c_head2 = st.columns([3, 1])
with c_head1:
    st.title("🔥 FIRE SPREAD COMMAND")
    st.caption(f"SYSTEM STATUS: ONLINE | LOCATION: {st.session_state.lat:.4f}, {st.session_state.lon:.4f}")
with c_head2:
    st.markdown(f"""
    <div style="text-align:right; margin-top:10px;">
        <span class="danger-badge" style="background-color:{danger_color};">RISK: {danger_lvl}</span>
        <div style="font-size:0.8em; color:#8B949E; margin-top:5px;">{weather_desc}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# 2. HUD METRICS (KPIs)
k1, k2, k3, k4 = st.columns(4)
k1.metric("PREDICTED AREA", f"{physical_res.area_sqm:,.0f} m²", "TOTAL SPREAD")
k2.metric("HEAD DISTANCE", f"{physical_res.ellipse_a_m + inputs.init_radius_m:,.0f} m", "FROM ORIGIN")
k3.metric("RATE OF SPREAD", f"{ros_m_per_min(inputs):.1f} m/min", "FORWARD VELOCITY")
k4.metric("REQ. WATER", f"{physical_res.water_volume_tons:,.1f} ton", "ESTIMATED")

# 3. STRATEGIC MAP & ANALYSIS
st.markdown("### 🗺️ TACTICAL VIEW")
col_map, col_info = st.columns([2, 1])

with col_map:
    if HAS_FOLIUM:
        # 暗めの地図タイルを使用してCommand Center感を演出
        m = folium.Map(
            location=[st.session_state.lat, st.session_state.lon], 
            zoom_start=16, 
            tiles="CartoDB dark_matter"
        )
        
        # 発生源
        folium.CircleMarker(
            [st.session_state.lat, st.session_state.lon], 
            radius=6, color="#FF5722", fill=True, fill_opacity=1.0, 
            popup="IGNITION POINT"
        ).add_to(m)

        # 延焼楕円ポリゴン生成
        a, b = physical_res.ellipse_a_m, physical_res.ellipse_b_m
        angle_rad = math.radians(90 - inputs.wind_dir_deg) # 北基準時計回りを数学座標に変換
        
        # 楕円の中心は風下にシフトする (頂点が発火点にあると仮定する簡易モデル)
        # 実際は発火点から風下に伸びる形になるよう補正
        center_dist = a  # 長半径分ずらすと後端が原点に来る
        
        points = []
        for t in np.linspace(0, 2*math.pi, 72):
            # 楕円の媒介変数表示 (中心原点)
            local_x = a * math.cos(t) + a # 原点を端にするためにシフト
            local_y = b * math.sin(t)
            
            # 回転行列
            rot_x = local_x * math.cos(angle_rad) - local_y * math.sin(angle_rad)
            rot_y = local_x * math.sin(angle_rad) + local_y * math.cos(angle_rad)
            
            # メートル -> 緯度経度 (簡易概算: 緯度1度=111km, 経度は緯度依存)
            dlat = rot_y / 111111.0
            dlon = rot_x / (111111.0 * math.cos(math.radians(st.session_state.lat)))
            
            points.append([st.session_state.lat + dlat, st.session_state.lon + dlon])

        folium.Polygon(
            locations=points,
            color="#FF3333",
            weight=2,
            fill=True,
            fill_color="#FF5722",
            fill_opacity=0.3,
            popup=f"Projection: {inputs.duration_min}min"
        ).add_to(m)
        
        # 風向き矢印 (簡易表示としてラインで描画)
        wind_end_lat = st.session_state.lat + (0.002 * math.sin(angle_rad))
        wind_end_lon = st.session_state.lon + (0.002 * math.cos(angle_rad) / math.cos(math.radians(st.session_state.lat)))
        folium.PolyLine(
            locations=[[st.session_state.lat, st.session_state.lon], [wind_end_lat, wind_end_lon]],
            color="#00BCD4", weight=3, opacity=0.6, tooltip="Wind Direction"
        ).add_to(m)

        st_folium(m, height=500, width="100%")
    else:
        st.error("SYSTEM ERROR: Map Module 'Folium' Not Found.")

with col_info:
    # タブで情報切り替え
    tab1, tab2 = st.tabs(["📈 TRENDS", "🤖 AI ADVISOR"])
    
    with tab1:
        st.markdown("###### GROWTH PROJECTION")
        # Altairチャート (ダークテーマ対応)
        chart_data = time_series_df.melt('time_min', value_vars=['area_sqm', 'perimeter_m'])
        
        base = alt.Chart(chart_data).encode(x=alt.X('time_min', title='Time (min)'))
        
        line = base.mark_line(point=True).encode(
            y=alt.Y('value', title='Value'),
            color=alt.Color('variable', scale=alt.Scale(range=['#FF5722', '#00BCD4'])),
            tooltip=['time_min', 'value', 'variable']
        )
        
        st.altair_chart(line.interactive(), use_container_width=True)
        
        st.info(f"Efficiency Factor: {inputs.efficiency*100:.0f}% applied.")

    with tab2:
        st.markdown("###### TACTICAL ANALYSIS")
        if st.button("REQUEST AI BRIEFING", type="primary", use_container_width=True):
            with st.spinner("ESTABLISHING SECURE LINK..."):
                model = get_gemini_model()
                if model:
                    advice = run_gemini_analysis(model, inputs, physical_res, weather_desc)
                    st.markdown(f"""
                    <div style="background-color:#0D1117; border: 1px solid #FFC107; padding: 15px; border-radius: 5px; margin-top:10px;">
                        <div style="color:#FFC107; font-weight:bold; font-size:0.9em; margin-bottom:10px;">⚡ INCOMING TRANSMISSION</div>
                        <div style="color:#E6EDF3; font-family:'Roboto Mono'; font-size:0.9em; line-height:1.6;">
                            {advice.replace(chr(10), '<br>')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("API KEY MISSING: Check secrets.toml")
        else:
            st.markdown("""
            <div style="text-align:center; color:#555; padding: 20px;">
                WAITING FOR COMMAND...
            </div>
            """, unsafe_allow_html=True)

# 4. FOOTER / DOWNLOAD
st.markdown("---")
with st.expander("📂 EXPORT MISSION DATA"):
    st.dataframe(time_series_df, use_container_width=True)
    csv = time_series_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "DOWNLOAD CSV LOG",
        data=csv,
        file_name="mission_log.csv",
        mime="text/csv"
    )
