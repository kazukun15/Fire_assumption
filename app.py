# -*- coding: utf-8 -*-
"""
Fire Spread Command Center Ver.3.3 (Japanese Edition)
----------------------------------------------------------------
- コンセプト: 災害対策本部・司令塔ダッシュボード (完全日本語化)
- 機能: 住所検索(Mapbox), 気象取得(OpenWeather), AI戦術(Gemini)
- 安全性: API数値の安全処理実装済み
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
    page_title="災害対策本部コマンドセンター",
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
            font-family: 'Noto Sans JP', 'Hiragino Sans', 'Meiryo', sans-serif;
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
        div[data-testid="stMetric"] label { color: #8B949E !important; font-size: 0.8rem !important; }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #F0F6FC !important;
            font-family: 'Courier New', monospace;
            font-weight: bold;
        }

        /* ステータスバッジ */
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 4px;
            font-weight: bold;
            color: #fff;
            letter-spacing: 0.1em;
            font-size: 0.9em;
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
            font-size: 0.9em;
            line-height: 1.6;
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

# 燃料モデル定数
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
    # ラマヌジャンの近似式
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
        results.append({"経過時間(分)": t, "延焼面積(m2)": area})
    return pd.DataFrame(results)

def get_risk_level(temp, humid, wind):
    idx = humid - (temp - 10) * 2.0
    if wind > 10.0: idx -= 15
    elif wind > 5.0: idx -= 5
    
    if idx < 15: return "EXTREME (極めて危険)", "#FF0033"
    if idx < 30: return "VERY HIGH (非常に危険)", "#FF6600"
    if idx < 50: return "HIGH (危険)", "#FF9900"
    if idx < 70: return "MODERATE (警戒)", "#FFCC00"
    return "LOW (注意)", "#00CC66"

# ------------------------- 4. 外部API連携機能 -------------------------

# 【住所検索機能】 Mapbox API
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

# 【気象情報】 OpenWeather API
def fetch_weather(lat, lon) -> Optional[Dict]:
    if "openweather" not in st.secrets: return None
    try:
        key = st.secrets["openweather"]["api_key"]
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"lat": lat, "lon": lon, "appid": key, "units": "metric", "lang": "ja"}
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

# 【AI参謀】 Gemini API
def get_gemini_response(prompt):
    if "general" in st.secrets and "api_key" in st.secrets["general"]:
        try:
            genai.configure(api_key=st.secrets["general"]["api_key"])
            model = genai.GenerativeModel("gemini-2.5-flash")
            return model.generate_content(prompt).text
        except: pass
    return "AIシステム応答なし: 安全な通信を確立できませんでした。"

# ------------------------- 5. メイン画面 UI -------------------------

# サイドバー設定
with st.sidebar:
    st.markdown("### 🎛️ 作戦コントロールパネル")
    
    # --- ロケーション設定 (住所検索あり) ---
    with st.expander("📍 対象地域設定 (Location)", expanded=True):
        # 入力モード切替
        loc_mode = st.radio("入力モード", ["座標入力", "住所検索"], horizontal=True, label_visibility="collapsed")
        
        if "lat" not in st.session_state: st.session_state.lat = 35.6812
        if "lon" not in st.session_state: st.session_state.lon = 139.7671

        if loc_mode == "住所検索":
            addr_input = st.text_input("住所を入力", placeholder="例: 東京都千代田区...")
            if st.button("対象地域を特定"):
                if "mapbox" in st.secrets:
                    res = geocode_address_mapbox(addr_input)
                    if res:
                        st.session_state.lat, st.session_state.lon = res
                        st.success("ターゲット補足完了")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("住所が見つかりませんでした")
                else:
                    st.warning("Mapbox APIキーが設定されていません")
        else:
            c1, c2 = st.columns(2)
            with c1: st.session_state.lat = st.number_input("緯度", -90.0, 90.0, st.session_state.lat, format="%.4f")
            with c2: st.session_state.lon = st.number_input("経度", -180.0, 180.0, st.session_state.lon, format="%.4f")

        fuel_labels = {"grass": "草原 (Grass)", "shrub": "低木 (Shrub)", "timber": "森林 (Timber)"}
        fuel_type_label = st.selectbox("燃料モデル", list(fuel_labels.values()), index=0)
        fuel_type = [k for k, v in fuel_labels.items() if v == fuel_type_label][0]

    # --- 気象設定 (エラー対策済み) ---
    with st.expander("🌪️ 気象条件設定 (Weather)", expanded=True):
        use_api = st.checkbox("LIVE気象データ連携", value=True)
        
        # デフォルト値
        ws, wd, rh, tp = 5.0, 0, 40, 25
        w_desc = "手動入力モード"

        if use_api:
            w_data = fetch_weather(st.session_state.lat, st.session_state.lon)
            if w_data:
                ws, wd, rh, tp = w_data["wind"], w_data["deg"], w_data["humid"], w_data["temp"]
                w_desc = f"{w_data['desc']} (LIVE)"
                st.caption(f"📡 データ受信中: {tp}℃ / {rh}%")
            else:
                st.caption("⚠️ 接続失敗: 手動入力を使用")

        # 安全な値を計算 (360度問題等を回避)
        safe_wd = int(wd) % 360
        safe_rh = clamp(int(rh), 0, 100)

        wind_speed = st.slider("風速 (m/s)", 0.0, 30.0, float(ws))
        wind_dir = st.slider("風向 (度: 北=0)", 0, 359, safe_wd) 
        humidity = st.slider("湿度 (%)", 0, 100, safe_rh)
        temp = st.slider("気温 (℃)", -10, 50, int(tp))
        slope = st.slider("斜面勾配 (%)", 0, 100, 10)

    with st.expander("⏱️ シミュレーション設定"):
        duration = st.slider("予測時間 (分後)", 10, 180, 60, step=10)
        app_rate = st.number_input("放水率 (L/min/m)", 0.1, 50.0, 4.0)

# 計算実行
inp = Inputs(duration, wind_speed, wind_dir, humidity, temp, slope, fuel_type, 5.0, 60, app_rate, 0.6)
out = run_physical_model(inp)
df_res = run_time_series_simulation(inp)
risk_lvl, risk_col = get_risk_level(temp, humidity, wind_speed)

# ===== DASHBOARD VIEW (メイン画面) =====

# ヘッダー
c1, c2 = st.columns([3, 1])
with c1:
    st.title("🔥 災害対策本部コマンドセンター")
    st.caption(f"作戦地域: {st.session_state.lat:.4f}, {st.session_state.lon:.4f} | 燃料タイプ: {fuel_labels[fuel_type]}")
with c2:
    st.markdown(f"""
    <div style="text-align:right; margin-top:10px;">
        <span class="status-badge" style="background-color:{risk_col};">RISK: {risk_lvl}</span>
        <div style="color:#888; font-size:0.8em; margin-top:5px;">{w_desc}</div>
    </div>
    """, unsafe_allow_html=True)

# HUD (ヘッドアップディスプレイ)
k1, k2, k3, k4 = st.columns(4)
k1.metric("予測延焼面積", f"{out.area_sqm:,.0f} m²", "TOTAL SPREAD")
k2.metric("最前線到達距離", f"{out.ellipse_a_m + 5.0:,.0f} m", "FROM ORIGIN")
k3.metric("延焼速度 (ROS)", f"{ros_m_per_min(inp):.1f} m/min", "VELOCITY")
k4.metric("推定必要水量", f"{out.water_volume_tons:,.1f} ton", "WATER REQ")

st.markdown("---")

# 地図とグラフエリア
m_col, g_col = st.columns([1.8, 1.2])

with m_col:
    st.subheader("🗺️ 戦術マップ (Tactical Map)")
    if HAS_FOLIUM:
        m = folium.Map([st.session_state.lat, st.session_state.lon], zoom_start=15, tiles="CartoDB dark_matter")
        
        # 発生源マーカー
        folium.CircleMarker(
            [st.session_state.lat, st.session_state.lon], radius=5, color="#FF5722", fill=True, fill_opacity=1, popup="発火点"
        ).add_to(m)

        # 延焼楕円の描画計算
        a, b = out.ellipse_a_m, out.ellipse_b_m
        pts = []
        angle_rad = math.radians(wind_dir - 180) # 風向(来る方向)から拡散方向へ変換
        center_lat, center_lon = st.session_state.lat, st.session_state.lon
        
        for t in np.linspace(0, 2*math.pi, 60):
            # 焦点を発火点に寄せるためのシフト
            dx = a + a * math.cos(t)
            dy = b * math.sin(t)
            
            # 回転行列
            rx = dx * math.sin(angle_rad) - dy * math.cos(angle_rad)
            ry = dx * math.cos(angle_rad) + dy * math.sin(angle_rad)
            
            # 簡易座標変換
            dlat = ry / 111111.0
            dlon = rx / (111111.0 * math.cos(math.radians(center_lat)))
            pts.append([center_lat - dlat, center_lon - dlon])
            
        folium.Polygon(pts, color="#FF3333", fill=True, fill_color="#FF5722", fill_opacity=0.3, popup="予測延焼範囲").add_to(m)
        st_folium(m, height=450, width="100%")
    else:
        st.error("地図モジュール (Folium) が読み込めません。")

with g_col:
    tab1, tab2 = st.tabs(["📈 拡大推移", "🤖 AI戦術参謀"])
    with tab1:
        c = alt.Chart(df_res).mark_area(
            line={'color':'#FF5722'},
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='#FF5722', offset=0), alt.GradientStop(color='rgba(255,87,34,0.1)', offset=1)],
                x1=1, x2=1, y1=1, y2=0
            )
        ).encode(
            x=alt.X('経過時間(分)', title='経過時間 (min)'),
            y=alt.Y('延焼面積(m2)', title='延焼面積 (m²)'),
            tooltip=['経過時間(分)', '延焼面積(m2)']
        ).properties(height=250)
        st.altair_chart(c, use_container_width=True)
        
    with tab2:
        if st.button("AI解析を実行 (Analysis)", type="primary"):
            with st.spinner("戦術データを解析中..."):
                prompt = f"""
                あなたは災害対策本部の戦術アドバイザーです。以下の火災状況に基づき、現場指揮官へ日本語で簡潔に指示を出してください。
                [状況]
                燃料: {fuel_labels[fuel_type]}
                風速: {wind_speed}m/s
                湿度: {humidity}%
                予測延焼面積: {out.area_sqm:.0f} m2
                天候: {w_desc}
                
                [出力形式]
                1. 現状のリスク評価 (一言で)
                2. 最大の懸念事項 (箇条書き1点)
                3. 推奨アクション (断定的な命令口調で)
                """
                res_txt = get_gemini_response(prompt)
                st.markdown(f"""
                <div class="ai-console">
                    <strong>⚡ 受信メッセージ (INCOMING)</strong><br><br>
                    {res_txt.replace(chr(10), '<br>')}
                </div>""", unsafe_allow_html=True)

# エクスポート
st.markdown("---")
with st.expander("📂 作戦ログ出力 (Data Export)"):
    st.dataframe(df_res, use_container_width=True)
    st.download_button("CSVログをダウンロード", df_res.to_csv(index=False).encode('utf-8'), "mission_log.csv", "text/csv")
