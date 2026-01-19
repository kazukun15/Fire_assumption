# -*- coding: utf-8 -*-
"""
Fire Spread Command Center Ver.3.2 (Full Feature Edition)
----------------------------------------------------------------
- UI: Command Center Style (Dark Mode / HUD)
- Feature: Address Search (Mapbox) & Weather API (OpenWeather) fully integrated
- Safety: Robust error handling for API values
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

# ---- ライブラリチェック ----
try:
    from streamlit_folium import st_folium
    import folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

# ------------------------- 1. システム起動設定 -------------------------
st.set_page_config(
    page_title="FIRE COMMAND CENTER",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- 2. UIデザイン (CSS) -------------------------
def inject_custom_css():
    st.markdown("""
    <style>
        /* 全体設定: ダークテーマ */
        .stApp {
            background-color: #0E1117;
            font-family: 'Roboto', sans-serif;
        }
        h1, h2, h3 { color: #E6E6E6 !important; letter-spacing: 0.05em; }
        
        /* サイドバー */
        section[data-testid="stSidebar"] {
            background-color: #161B22;
            border-right: 1px solid #30363D;
        }

        /* HUDメトリクスカード */
        div[data-testid="stMetric"] {
            background-color: #21262D;
            border: 1px solid #30363D;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.5);
            transition: 0.3s;
        }
        div[data-testid="stMetric"]:hover {
            border-color: #FF5722;
            transform: translateY(-2px);
        }
        div[data-testid="stMetric"] label { color: #8B949E !important; }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #F0F6FC !important;
            font-family: 'Courier New', monospace;
            font-weight: bold;
        }

        /* バッジスタイル */
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 4px;
            font-weight: bold;
            color: #fff;
            letter-spacing: 0.1em;
        }
        
        /* AI通信コンソール */
        .ai-console {
            background-color: #0D1117;
            border-left: 4px solid #FFC107;
            padding: 15px;
            margin-top: 10px;
            border-radius: 0 4px 4px 0;
            font-family: 'Courier New', monospace;
            color: #C9D1D9;
        }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ------------------------- 3. ロジック & 定数 -------------------------
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

BASE_RATE_BY_FUEL = {"grass": 8.0, "shrub": 3.0, "timber": 0.6}
HUMIDITY_K = 1.1
WIND_A = 0.10
WIND_B = 0.010
SLOPE_K = 4.0
LB_C = 0.30
LB_MAX = 5.0
EPS = 1e-9

def clamp(x, lo, hi): return max(lo, min(hi, x))

def ros_m_per_min(inp: Inputs) -> float:
    # 湿度補正
    base_f = math.exp(-HUMIDITY_K * max(0.0, inp.rel_humidity - 30.0) / 70.0)
    if inp.rel_humidity < 30.0: base_f = 1.0 + 0.025 * (30.0 - inp.rel_humidity)
    f_h = clamp(base_f * (1.0 + max(0, (inp.air_temp_c - 25.0) * 0.01)), 0.25, 2.5)
    
    # 風速補正
    f_w = clamp(1.0 + WIND_A * inp.wind_speed_ms + WIND_B * (inp.wind_speed_ms ** 2), 1.0, 8.0)
    
    # 傾斜補正
    f_s = clamp(1.0 + SLOPE_K * (inp.slope_percent / 100.0), 1.0, 6.0)
    
    r0 = BASE_RATE_BY_FUEL.get(inp.fuel_class, 1.0)
    return max(EPS, r0 * f_h * f_w * f_s)

def run_physical_model(inp: Inputs) -> Outputs:
    ros = ros_m_per_min(inp)
    lb = clamp(1.0 + LB_C * inp.wind_speed_ms, 1.0, LB_MAX)
    
    A = ros * inp.duration_min + inp.init_radius_m
    B = max(EPS, A / lb)
    
    area = math.pi * A * B
    h = ((A - B) ** 2) / ((A + B) ** 2 + EPS)
    perimeter = math.pi * (A + B) * (1 + (3*h)/(10 + math.sqrt(4 - 3*h + EPS)))
    
    r_equiv = math.sqrt(area / math.pi)
    
    liters = inp.app_rate_lpm_per_m * perimeter * inp.attack_duration_min
    water_ton = (liters / max(inp.efficiency, 0.05)) / 1000.0
    
    return Outputs(r_equiv, area, water_ton, A, B, perimeter)

def run_time_series_simulation(inp: Inputs, steps: int = 20) -> pd.DataFrame:
    times = np.linspace(0, inp.duration_min, steps)
    results = []
    ros = ros_m_per_min(inp)
    lb = clamp(1.0 + LB_C * inp.wind_speed_ms, 1.0, LB_MAX)

    for t in times:
        A = ros * t + inp.init_radius_m
        B = max(EPS, A / lb)
        area = math.pi * A * B
        results.append({"Time (min)": t, "Area (m2)": area})
    return pd.DataFrame(results)

def get_risk_level(temp, humid, wind):
    idx = humid - (temp - 10) * 2.0
    if wind > 10.0: idx -= 15
    elif wind > 5.0: idx -= 5
    
    if idx < 15: return "EXTREME", "#FF0033"
    if idx < 30: return "VERY HIGH", "#FF6600"
    if idx < 50: return "HIGH", "#FF9900"
    if idx < 70: return "MODERATE", "#FFCC00"
    return "LOW", "#00CC66"

# ------------------------- 4. 外部API連携機能 -------------------------

# 【重要】住所検索機能 (Mapbox)
def geocode_address_mapbox(address: str) -> Optional[Tuple[float, float]]:
    if "mapbox" not in st.secrets:
        return None
    try:
        token = st.secrets["mapbox"]["access_token"]
        q = urllib.parse.quote(address)
        url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{q}.json"
        params = {"access_token": token, "limit": 1, "language": "ja"}
        r = requests.get(url, params=params, timeout=3)
        if r.status_code == 200:
            feat = r.json().get("features", [])
            if feat:
                return feat[0]["center"][1], feat[0]["center"][0] # lat, lon
    except: pass
    return None

# 気象情報取得 (OpenWeather)
def fetch_weather(lat, lon) -> Optional[Dict]:
    if "openweather" not in st.secrets: return None
    try:
        key = st.secrets["openweather"]["api_key"]
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"lat": lat, "lon": lon, "appid": key, "units": "metric"}
        r = requests.get(url, params=params, timeout=3)
        if r.status_code == 200:
            d = r.json()
            return {
                "temp": float(d["main"]["temp"]),
                "humid": float(d["main"]["humidity"]),
                "wind": float(d["wind"]["speed"]),
                "deg": float(d["wind"].get("deg", 0)),
                "desc": d["weather"][0]["description"]
            }
    except: pass
    return None

# Gemini AI
def get_gemini_response(prompt):
    if "general" in st.secrets and "api_key" in st.secrets["general"]:
        try:
            genai.configure(api_key=st.secrets["general"]["api_key"])
            model = genai.GenerativeModel("gemini-2.5-flash")
            return model.generate_content(prompt).text
        except: pass
    return "AI SYSTEM OFFLINE: Unable to establish link."

# ------------------------- 5. メイン画面 UI -------------------------

# サイドバー設定
with st.sidebar:
    st.markdown("### 🎛️ SYSTEM CONTROL")
    
    # --- ロケーション設定 (住所検索復活) ---
    with st.expander("📍 TARGET LOCATION", expanded=True):
        # 入力モード切替
        loc_mode = st.radio("Input Mode", ["Coordinates", "Address Search"], label_visibility="collapsed")
        
        if "lat" not in st.session_state: st.session_state.lat = 35.6812
        if "lon" not in st.session_state: st.session_state.lon = 139.7671

        if loc_mode == "Address Search":
            addr_input = st.text_input("Search Target", placeholder="Ex: Tokyo Tower...")
            if st.button("LOCATE TARGET"):
                if "mapbox" in st.secrets:
                    res = geocode_address_mapbox(addr_input)
                    if res:
                        st.session_state.lat, st.session_state.lon = res
                        st.success("TARGET LOCKED.")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("TARGET NOT FOUND.")
                else:
                    st.warning("Mapbox API Key missing in secrets.toml")
        else:
            c1, c2 = st.columns(2)
            with c1: st.session_state.lat = st.number_input("Lat", -90.0, 90.0, st.session_state.lat, format="%.4f")
            with c2: st.session_state.lon = st.number_input("Lon", -180.0, 180.0, st.session_state.lon, format="%.4f")

        fuel_type = st.selectbox("FUEL MODEL", ["grass", "shrub", "timber"], index=0)

    # --- 気象設定 (エラー対策済み) ---
    with st.expander("🌪️ WEATHER CONDITIONS", expanded=True):
        use_api = st.checkbox("LIVE DATA LINK", value=True)
        
        # デフォルト値
        ws, wd, rh, tp = 5.0, 0, 40, 25
        w_desc = "MANUAL"

        if use_api:
            w_data = fetch_weather(st.session_state.lat, st.session_state.lon)
            if w_data:
                ws, wd, rh, tp = w_data["wind"], w_data["deg"], w_data["humid"], w_data["temp"]
                w_desc = f"{w_data['desc'].upper()} (LIVE)"
                st.caption(f"📡 LINK ESTABLISHED: {tp}℃ / {rh}%")
            else:
                st.caption("⚠️ LINK FAILED: Manual Mode")

        # 安全な値を計算 (360度問題を回避)
        safe_wd = int(wd) % 360
        safe_rh = clamp(int(rh), 0, 100)

        wind_speed = st.slider("WIND SPEED (m/s)", 0.0, 30.0, float(ws))
        wind_dir = st.slider("WIND DIR (deg)", 0, 359, safe_wd) # Fix applied
        humidity = st.slider("HUMIDITY (%)", 0, 100, safe_rh)
        temp = st.slider("TEMP (℃)", -10, 50, int(tp))
        slope = st.slider("SLOPE (%)", 0, 100, 10)

    with st.expander("⏱️ SIMULATION PARAMS"):
        duration = st.slider("PREDICTION TIME (min)", 10, 180, 60, step=10)
        app_rate = st.number_input("WATER RATE (L/min/m)", 0.1, 50.0, 4.0)

# 計算実行
inp = Inputs(duration, wind_speed, wind_dir, humidity, temp, slope, fuel_type, 5.0, 60, app_rate, 0.6)
out = run_physical_model(inp)
df_res = run_time_series_simulation(inp)
risk_lvl, risk_col = get_risk_level(temp, humidity, wind_speed)

# ===== DASHBOARD VIEW =====

# Header
c1, c2 = st.columns([3, 1])
with c1:
    st.title("🔥 FIRE COMMAND CENTER")
    st.caption(f"LOC: {st.session_state.lat:.4f}, {st.session_state.lon:.4f} | FUEL: {fuel_type.upper()}")
with c2:
    st.markdown(f"""
    <div style="text-align:right; margin-top:10px;">
        <span class="status-badge" style="background-color:{risk_col};">{risk_lvl}</span>
        <div style="color:#888; font-size:0.8em;">{w_desc}</div>
    </div>
    """, unsafe_allow_html=True)

# HUD
k1, k2, k3, k4 = st.columns(4)
k1.metric("PREDICTED AREA", f"{out.area_sqm:,.0f} m²", "TOTAL")
k2.metric("HEAD DISTANCE", f"{out.ellipse_a_m + 5.0:,.0f} m", "FROM ORIGIN")
k3.metric("SPREAD RATE", f"{ros_m_per_min(inp):.1f} m/min", "VELOCITY")
k4.metric("WATER REQ", f"{out.water_volume_tons:,.1f} ton", "ESTIMATED")

st.markdown("---")

# Map & Graph
m_col, g_col = st.columns([1.8, 1.2])

with m_col:
    st.subheader("🗺️ TACTICAL MAP")
    if HAS_FOLIUM:
        m = folium.Map([st.session_state.lat, st.session_state.lon], zoom_start=15, tiles="CartoDB dark_matter")
        
        # Origin
        folium.CircleMarker(
            [st.session_state.lat, st.session_state.lon], radius=5, color="#FF5722", fill=True, fill_opacity=1
        ).add_to(m)

        # Ellipse Calculation for Map
        a, b = out.ellipse_a_m, out.ellipse_b_m
        pts = []
        angle_rad = math.radians(wind_dir - 180) # Wind Direction vs Spread Direction
        center_lat, center_lon = st.session_state.lat, st.session_state.lon
        
        for t in np.linspace(0, 2*math.pi, 60):
            # Shift center so origin is at one focus/edge approx
            dx = a + a * math.cos(t)
            dy = b * math.sin(t)
            
            # Rotate
            rx = dx * math.sin(angle_rad) - dy * math.cos(angle_rad)
            ry = dx * math.cos(angle_rad) + dy * math.sin(angle_rad)
            
            dlat = ry / 111111.0
            dlon = rx / (111111.0 * math.cos(math.radians(center_lat)))
            pts.append([center_lat - dlat, center_lon - dlon])
            
        folium.Polygon(pts, color="#FF3333", fill=True, fill_color="#FF5722", fill_opacity=0.3).add_to(m)
        st_folium(m, height=450, width="100%")
    else:
        st.error("Folium module missing.")

with g_col:
    tab1, tab2 = st.tabs(["📈 TRENDS", "🤖 AI ADVISOR"])
    with tab1:
        c = alt.Chart(df_res).mark_area(
            line={'color':'#FF5722'},
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='#FF5722', offset=0), alt.GradientStop(color='rgba(255,87,34,0.1)', offset=1)],
                x1=1, x2=1, y1=1, y2=0
            )
        ).encode(x='Time (min)', y='Area (m2)', tooltip=['Time (min)', 'Area (m2)']).properties(height=250)
        st.altair_chart(c, use_container_width=True)
        
    with tab2:
        if st.button("INITIATE AI ANALYSIS", type="primary"):
            with st.spinner("PROCESSING TACTICAL DATA..."):
                prompt = f"Fire Situation: Fuel {fuel_type}, Wind {wind_speed}m/s, Humid {humidity}%. Est Area {out.area_sqm:.0f}m2. Provide tactical advice in Japanese (Bullet points)."
                res_txt = get_gemini_response(prompt)
                st.markdown(f"""
                <div class="ai-console">
                    <strong>⚡ INCOMING TRANSMISSION</strong><br><br>
                    {res_txt.replace(chr(10), '<br>')}
                </div>""", unsafe_allow_html=True)

# Export
st.markdown("---")
with st.expander("📂 MISSION DATA EXPORT"):
    st.dataframe(df_res, use_container_width=True)
    st.download_button("DOWNLOAD CSV LOG", df_res.to_csv(index=False).encode('utf-8'), "mission_log.csv", "text/csv")
