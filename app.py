# -*- coding: utf-8 -*-
"""
Fire Spread Command Center Ver.3.1 (Bug Fix Edition)
----------------------------------------------------------------
- UI/UX: 災害対策本部・司令塔ダッシュボード (ダークモード)
- Fix: 気象APIからの異常値（360度など）によるクラッシュを防止
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

# ---- 外部ライブラリのチェック ----
try:
    from streamlit_folium import st_folium
    import folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

# ------------------------- 1. ページ設定 (最優先) -------------------------
st.set_page_config(
    page_title="FIRE COMMAND CENTER",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- 2. デザインシステム (CSS) -------------------------
def inject_custom_css():
    st.markdown("""
    <style>
        /* ベースカラー: 漆黒に近いダークグレー */
        .stApp {
            background-color: #0E1117;
            font-family: 'Roboto', 'Noto Sans JP', sans-serif;
        }
        
        h1, h2, h3 {
            color: #E6E6E6 !important;
            letter-spacing: 0.05em;
        }

        /* サイドバー */
        section[data-testid="stSidebar"] {
            background-color: #161B22;
            border-right: 1px solid #30363D;
        }

        /* 数値カード (HUD風) */
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
        }
        div[data-testid="stMetric"] label {
            color: #8B949E !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #F0F6FC !important;
            font-family: 'Courier New', monospace;
            font-weight: bold;
        }

        /* 危険度バッジ */
        .danger-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 4px;
            font-weight: bold;
            color: #fff;
            letter-spacing: 0.1em;
            box-shadow: 0 0 10px rgba(255,0,0,0.3);
        }
        
        /* AIアドバイスエリア */
        .ai-console {
            background-color: #0D1117;
            border-left: 4px solid #FFC107;
            padding: 15px;
            margin-top: 10px;
            border-radius: 0 4px 4px 0;
            font-family: 'Courier New', monospace;
            color: #C9D1D9;
        }

        /* ボタン */
        .stButton button {
            width: 100%;
            border-radius: 4px;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ------------------------- 3. データ構造 & 定数 -------------------------
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

# 燃焼物理パラメータ
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

# ------------------------- 4. ロジック (物理モデル) -------------------------

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

def ros_m_per_min(inp: Inputs) -> float:
    r0 = BASE_RATE_BY_FUEL.get(inp.fuel_class, 1.0)
    f_h = humidity_factor(inp.rel_humidity, inp.air_temp_c)
    f_w = wind_factor(inp.wind_speed_ms)
    f_s = slope_factor(inp.slope_percent)
    return max(EPS, r0 * f_h * f_w * f_s)

def length_breadth_ratio(u_ms: float) -> float:
    return clamp(1.0 + LB_C * u_ms, 1.0, LB_MAX)

def ellipse_area_perimeter(a: float, b: float) -> Tuple[float, float]:
    area = math.pi * a * b
    h = ((a - b) ** 2) / ((a + b) ** 2 + EPS)
    perimeter = math.pi * (a + b) * (1 + (3*h)/(10 + math.sqrt(4 - 3*h + EPS)))
    return area, perimeter

def run_physical_model(inp: Inputs) -> Outputs:
    ros = ros_m_per_min(inp)
    lb = length_breadth_ratio(inp.wind_speed_ms)
    A = ros * inp.duration_min + inp.init_radius_m
    B = max(EPS, A / lb)
    
    area, perimeter = ellipse_area_perimeter(A, B)
    r_equiv = math.sqrt(area / math.pi)
    
    liters = inp.app_rate_lpm_per_m * perimeter * inp.attack_duration_min
    liters_eff = liters / max(inp.efficiency, 0.05)
    water_ton = liters_eff / 1000.0
    
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
        results.append({
            "経過時間(分)": t,
            "延焼面積(m2)": area,
            "周囲長(m)": perimeter
        })
    return pd.DataFrame(results)

def get_fire_danger_level(temp: float, humid: float, wind: float) -> Tuple[str, str]:
    index = humid - (temp - 10) * 2.0
    if wind > 10.0: index -= 15
    elif wind > 5.0: index -= 5
    
    if index < 15: return "EXTREME (極めて危険)", "#FF0033"
    if index < 30: return "VERY HIGH (非常に危険)", "#FF6600"
    if index < 50: return "HIGH (危険)", "#FF9900"
    if index < 70: return "MODERATE (警戒)", "#FFCC00"
    return "LOW (注意)", "#00CC66"

# ------------------------- 5. 外部連携 (API) -------------------------
def fetch_openweather(lat: float, lon: float) -> Optional[Dict]:
    """天候データを取得する。失敗時はNoneを返す"""
    # secretsに設定がない場合はNoneを返す
    if "openweather" not in st.secrets:
        return None
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
    if "general" in st.secrets and "api_key" in st.secrets["general"]:
        try:
            genai.configure(api_key=st.secrets["general"]["api_key"])
            return genai.GenerativeModel("gemini-2.5-flash")
        except: pass
    return None

def run_gemini_analysis(model, inputs: Inputs, physical: Outputs, weather_desc: str) -> str:
    prompt = f"""
    あなたは災害対策本部の戦術アドバイザーです。以下の状況に基づき、現場指揮官へ簡潔に指示を出してください。
    
    [状況]
    燃料: {inputs.fuel_class}, 風速: {inputs.wind_speed_ms}m/s, 湿度: {inputs.rel_humidity}%
    予測延焼面積: {physical.area_sqm:.0f} m2
    天候: {weather_desc}
    
    [出力フォーマット]
    1. 現状のリスク評価 (一言で)
    2. 最大の懸念事項 (箇条書き1点)
    3. 推奨アクション (命令口調で)
    """
    try:
        resp = model.generate_content(prompt)
        return resp.text
    except: return "通信エラー: AIアドバイザーに接続できません。"

# ------------------------- 6. UI構築 (メイン) -------------------------

# サイドバー: 設定パネル
with st.sidebar:
    st.markdown("### 🎛️ 作戦コントロールパネル")
    
    with st.expander("📍 ターゲット設定 (Location)", expanded=True):
        if "lat" not in st.session_state: st.session_state.lat = 35.6812
        if "lon" not in st.session_state: st.session_state.lon = 139.7671
        
        c1, c2 = st.columns(2)
        with c1:
            st.session_state.lat = st.number_input("緯度", -90.0, 90.0, st.session_state.lat, format="%.4f")
        with c2:
            st.session_state.lon = st.number_input("経度", -180.0, 180.0, st.session_state.lon, format="%.4f")
            
        fuel_class = st.selectbox("燃料モデル", ["grass (草原)", "shrub (低木)", "timber (森林)"], index=0)
        fuel_key = fuel_class.split()[0]

    with st.expander("🌪️ 環境・気象条件 (Weather)", expanded=True):
        # API利用チェック
        use_api = st.checkbox("LIVE気象データ連携", value=True)
        
        # デフォルト値 (API失敗時用)
        ws, wd, rh, tp = 5.0, 180, 40, 25
        weather_desc_str = "MANUAL INPUT"

        if use_api:
            w_data = fetch_openweather(st.session_state.lat, st.session_state.lon)
            if w_data:
                ws, wd, rh, tp = w_data["wind_speed"], w_data["wind_deg"], w_data["humidity"], w_data["temp_c"]
                weather_desc_str = f"{w_data['description']} (LIVE)"
                st.info(f"📡 データ受信: {tp}℃ / 風{ws}m/s")
            else:
                st.caption("⚠️ API接続不可: 手動設定を使用")
        
        # -----【修正箇所】安全なデフォルト値を計算 -----
        # 風向が360度で来る可能性があるため、360で割った余り(0)にするか、359以下に制限する
        safe_wd = int(wd) % 360  # 360度(北)を0度に変換
        safe_rh = min(100, max(0, int(rh))) # 湿度を0-100に制限
        
        # マニュアル設定 (初期値に安全な値を渡す)
        wind_speed_ms = st.slider("風速 (m/s)", 0.0, 30.0, float(ws))
        wind_dir_deg = st.slider("風向 (度: 北=0)", 0, 359, safe_wd) # ここでエラー回避
        rel_humidity = st.slider("湿度 (%)", 0, 100, safe_rh)
        air_temp_c = st.slider("気温 (℃)", -10, 45, int(tp))
        slope_percent = st.slider("斜面勾配 (%)", 0, 100, 10)

    with st.expander("⏱️ シミュレーション設定"):
        duration_min = st.slider("予測時間 (分後)", 10, 180, 60, step=10)
        app_rate = st.number_input("放水率 (L/min/m)", 0.1, 50.0, 4.0)

# Inputs作成 & 計算実行
inputs = Inputs(
    duration_min=duration_min,
    wind_speed_ms=wind_speed_ms,
    wind_dir_deg=wind_dir_deg,
    rel_humidity=rel_humidity,
    air_temp_c=air_temp_c,
    slope_percent=slope_percent,
    fuel_class=fuel_key,
    init_radius_m=5.0,
    attack_duration_min=60,
    app_rate_lpm_per_m=app_rate,
    efficiency=0.6
)

res = run_physical_model(inputs)
df_ts = run_time_series_simulation(inputs)
danger_lvl, danger_color = get_fire_danger_level(air_temp_c, rel_humidity, wind_speed_ms)

# ===== メイン画面レイアウト =====

# 1. ヘッダー情報
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("🔥 FIRE SPREAD COMMAND")
    st.caption(f"LOCATION: {st.session_state.lat:.4f}, {st.session_state.lon:.4f} | FUEL: {inputs.fuel_class.upper()}")
with col_h2:
    st.markdown(f"""
    <div style="text-align:right; margin-top: 15px;">
        <span class="danger-badge" style="background-color:{danger_color};">{danger_lvl}</span>
        <div style="font-size:0.8em; color:#888;">{weather_desc_str}</div>
    </div>
    """, unsafe_allow_html=True)

# 2. HUD (Head-Up Display) メトリクス
k1, k2, k3, k4 = st.columns(4)
k1.metric("予測延焼面積", f"{res.area_sqm:,.0f} m²", "TOTAL SPREAD")
k2.metric("最前線距離", f"{res.ellipse_a_m + inputs.init_radius_m:,.0f} m", "FROM ORIGIN")
k3.metric("延焼速度 (ROS)", f"{ros_m_per_min(inputs):.1f} m/min", "VELOCITY")
k4.metric("推定必要水量", f"{res.water_volume_tons:,.1f} ton", "WATER REQ")

st.markdown("---")

# 3. 戦術マップと詳細分析
col_map, col_detail = st.columns([1.8, 1.2])

with col_map:
    st.subheader("🗺️ TACTICAL MAP")
    
    if HAS_FOLIUM:
        m = folium.Map(
            location=[st.session_state.lat, st.session_state.lon], 
            zoom_start=15, 
            tiles="CartoDB dark_matter"
        )
        
        folium.CircleMarker(
            [st.session_state.lat, st.session_state.lon], 
            radius=6, color="#FF5722", fill=True, fill_opacity=1.0, popup="発火点"
        ).add_to(m)

        # 延焼予測エリア (楕円)
        a, b = res.ellipse_a_m, res.ellipse_b_m
        points = []
        center_lat = st.session_state.lat
        center_lon = st.session_state.lon
        
        # 楕円の頂点を発火点に合わせるシフト量
        shift_dist = a 
        
        for t in np.linspace(0, 2*math.pi, 60):
            dx = shift_dist + a * math.cos(t)
            dy = b * math.sin(t)
            
            # 風向に合わせて回転
            # wind_dir_degは風が吹いてくる方向(From)。延焼は風下(To)へ。
            # 数学的な回転角に変換
            flow_angle = math.radians(inputs.wind_dir_deg - 180)
            
            rot_x = dx * math.sin(flow_angle) - dy * math.cos(flow_angle)
            rot_y = dx * math.cos(flow_angle) + dy * math.sin(flow_angle)

            # 簡易メートル->緯度経度変換
            dlat = rot_y / 111111.0
            dlon = rot_x / (111111.0 * math.cos(math.radians(center_lat)))
            
            points.append([center_lat - dlat, center_lon - dlon])

        folium.Polygon(
            locations=points,
            color="#FF3333", weight=2,
            fill=True, fill_color="#FF5722", fill_opacity=0.4,
            popup=f"予測範囲 ({inputs.duration_min}分後)"
        ).add_to(m)

        st_folium(m, height=450, width="100%")
        
    else:
        st.error("地図ライブラリ(folium)が見つかりません。")

with col_detail:
    tab1, tab2 = st.tabs(["📈 成長予測グラフ", "🤖 AI戦術参謀"])
    
    with tab1:
        chart_data = df_ts.melt('経過時間(分)', value_vars=['延焼面積(m2)'])
        
        c = alt.Chart(chart_data).mark_area(
            line={'color':'#FF5722'},
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='#FF5722', offset=0),
                       alt.GradientStop(color='rgba(255, 87, 34, 0.1)', offset=1)],
                x1=1, x2=1, y1=1, y2=0
            )
        ).encode(
            x='経過時間(分)',
            y='value',
            tooltip=['経過時間(分)', 'value']
        ).properties(height=250)
        
        st.altair_chart(c, use_container_width=True)

    with tab2:
        st.markdown("###### TACTICAL ADVISOR")
        if st.button("AI解析を実行 (Analysis)", type="primary"):
            model = get_gemini_model()
            if model:
                with st.spinner("AI参謀が戦術を立案中..."):
                    advice = run_gemini_analysis(model, inputs, res, weather_desc_str)
                    st.markdown(f"""
                    <div class="ai-console">
                        <strong>⚡ INCOMING TRANSMISSION</strong><br><br>
                        {advice.replace(chr(10), '<br>')}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("APIキー未設定: デモモード")
                st.markdown("""
                <div class="ai-console">
                    <strong>⚡ DEMO TRANSMISSION</strong><br>
                    1. リスク評価: HIGH (警戒レベル)<br>
                    2. 懸念事項: 風向の変化による市街地への延焼<br>
                    3. アクション: 東側側面の防御線を優先構築せよ
                </div>
                """, unsafe_allow_html=True)

# 4. データエクスポート
st.markdown("---")
with st.expander("📂 作戦ログの出力 (Export Data)"):
    st.dataframe(df_ts, use_container_width=True)
    csv = df_ts.to_csv(index=False).encode('utf-8')
    st.download_button(
        "CSV形式でダウンロード",
        data=csv,
        file_name="fire_simulation_log.csv",
        mime="text/csv"
    )
