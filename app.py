# app.py  —  火災延焼範囲予測くん
import os
import math
import json
import re
import time
from typing import Optional, Dict, Tuple, List

import requests
import streamlit as st
import folium
from streamlit_folium import st_folium
import pydeck as pdk

# ============================
# ページ設定
# ============================
st.set_page_config(page_title="火災延焼範囲予測くん", layout="wide")

# ============================
# Secrets / APIキー取得
# ============================
MODEL_NAME = "gemini-2.0-flash-001"
API_KEY = None
try:
    API_KEY = st.secrets["general"]["api_key"]
except Exception:
    API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

# Mapbox（任意）：あれば使う。なければPydeckはトークン不要の軽量スタイルを使う
MAPBOX_TOKEN = None
try:
    MAPBOX_TOKEN = st.secrets.get("mapbox", {}).get("access_token")
except Exception:
    MAPBOX_TOKEN = os.environ.get("MAPBOX_API_KEY") or os.environ.get("MAPBOX_TOKEN")

if MAPBOX_TOKEN:
    pdk.settings.mapbox_api_key = MAPBOX_TOKEN
    MAP_STYLE = "mapbox://styles/mapbox/dark-v10"
else:
    MAP_STYLE = "light"   # ✅ トークン不要（3Dが表示されない問題を解消）

# ============================
# JSON抽出（Geminiの```json ...```にも対応）
# ============================
def extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text)
    if not m:
        m = re.search(r"(\{[\s\S]*\})", text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None

# ============================
# Open-Meteoから気象取得
# ============================
@st.cache_data(show_spinner=False)
def get_weather(lat: float, lon: float) -> Optional[Dict[str, float]]:
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&current_weather=true&hourly=relativehumidity_2m,precipitation&timezone=auto"
        )
        resp = requests.get(url, timeout=20)
        data = resp.json()
        cur = data.get("current_weather", {}) or {}
        res = {
            "temperature": cur.get("temperature"),
            "windspeed": cur.get("windspeed"),
            "winddirection": cur.get("winddirection"),
            "weathercode": cur.get("weathercode"),
        }
        t = cur.get("time")
        hourly = data.get("hourly", {}) or {}
        times = hourly.get("time", []) or []
        if t in times:
            i = times.index(t)
            if i < len(hourly.get("relativehumidity_2m", [])):
                res["humidity"] = hourly["relativehumidity_2m"][i]
            if i < len(hourly.get("precipitation", [])):
                res["precipitation"] = hourly["precipitation"][i]
        return res
    except Exception:
        return None

# ============================
# Gemini呼び出し（堅牢パース）
# ============================
def gemini_generate(prompt: str):
    """戻り値: (parsed_json_or_text, raw_json) / APIキー未設定時は (None, None)"""
    if not API_KEY:
        return None, None
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
        headers = {"Content-Type": "application/json"}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        r = requests.post(url, headers=headers, json=payload, timeout=45)
        raw = r.json() if r.content else None
        if r.status_code != 200 or not raw:
            return None, raw
        text = ""
        cands = raw.get("candidates", [])
        if cands:
            if "output" in cands[0]:
                text = cands[0].get("output", "") or ""
            else:
                parts = cands[0].get("content", {}).get("parts", [])
                for p in parts:
                    text += p.get("text", "") or ""
        parsed = extract_json(text)
        return (parsed if parsed is not None else text), raw
    except Exception:
        return None, None

# ============================
# フォールバック推定（Gemini失敗/未設定でも必ず動く）
# ============================
def fallback_predict(wx: Dict[str, float], fuel_label: str, hours: float) -> dict:
    v0 = {"森林": 0.30, "草地": 0.60, "都市部": 0.20}.get(fuel_label, 0.40)  # m/s基準
    wind = float(wx.get("windspeed") or 0)
    rh = float(wx.get("humidity") or 60)
    precip = float(wx.get("precipitation") or 0)
    wind_factor = 1.0 + 0.12 * wind
    humidity_factor = max(0.6, 1.0 - 0.003 * max(0.0, rh - 30.0))
    precip_factor = max(0.5, 1.0 / (1.0 + precip))
    v_eff = v0 * wind_factor * humidity_factor * precip_factor
    radius_m = max(40.0, v_eff * hours * 3600.0)
    area_sqm = 0.5 * math.pi * radius_m * radius_m  # 風向±90°の半円を想定
    water_tons = area_sqm * 0.01  # 10L/㎡
    return {"radius_m": radius_m, "area_sqm": area_sqm, "water_volume_tons": water_tons}

# ============================
# 扇形ポリゴン生成（Folium / Pydeck）
# ============================
def sector_for_folium(lat: float, lon: float, radius_m: float, wind_dir_deg: float, steps: int = 64):
    """Folium用: [(lat, lon), ...]"""
    coords = []
    start, end = wind_dir_deg - 90.0, wind_dir_deg + 90.0
    for i in range(steps + 1):
        ang = math.radians(start + (end - start) * i / steps)
        north_m = radius_m * math.cos(ang)
        east_m = radius_m * math.sin(ang)
        dlat = north_m / 111000.0
        dlon = east_m / 111000.0
        coords.append((lat + dlat, lon + dlon))
    return coords

def sector_for_deck(lat: float, lon: float, radius_m: float, wind_dir_deg: float, steps: int = 64):
    """Pydeck用: [[lon, lat], ...]（クローズドリング）"""
    ring = []
    start, end = wind_dir_deg - 90.0, wind_dir_deg + 90.0
    for i in range(steps + 1):
        ang = math.radians(start + (end - start) * i / steps)
        north_m = radius_m * math.cos(ang)
        east_m = radius_m * math.sin(ang)
        dlat = north_m / 111000.0
        dlon = east_m / 111000.0
        ring.append([lon + dlon, lat + dlat])
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring

# ============================
# セッション初期化
# ============================
if "points" not in st.session_state:
    st.session_state.points: List[Tuple[float, float]] = []
if "weather" not in st.session_state:
    st.session_state.weather: Optional[Dict[str, float]] = None
if "last_pred" not in st.session_state:
    st.session_state.last_pred: Optional[Dict] = None
if "last_polys_2d" not in st.session_state:
    st.session_state.last_polys_2d: Optional[List[List[Tuple[float, float]]]] = None
if "last_polys_3d" not in st.session_state:
    st.session_state.last_polys_3d: Optional[List[dict]] = None

# ============================
# UI — サイドバー（テキスト入力）
# ============================
st.sidebar.header("発生地点と条件設定")
with st.sidebar.form("point_form"):
    st.caption("Googleマップの座標（例: 34.246099951898415, 133.20578422112848）をそのまま貼り付け可。")
    coord_text = st.text_input("緯度,経度（カンマ区切り）", "34.257586,133.204356")
    add = st.form_submit_button("発生地点を追加（テキスト）")
    if add:
        try:
            lat_in, lon_in = [float(x.strip().strip("()")) for x in coord_text.split(",")]
            st.session_state.points.append((lat_in, lon_in))
            st.sidebar.success(f"地点を追加: ({lat_in:.6f}, {lon_in:.6f})")
        except Exception:
            st.sidebar.error("形式が不正です。例: 34.246099951898415, 133.20578422112848")

if st.sidebar.button("登録地点を全消去"):
    st.session_state.points = []
    st.session_state.weather = None
    st.session_state.last_pred = None
    st.session_state.last_polys_2d = None
    st.session_state.last_polys_3d = None
    st.sidebar.info("削除しました")

fuel_opts = {"森林（高燃料)": "森林", "草地（中燃料)": "草地", "都市部（低燃料)": "都市部"}
sel_fuel = st.sidebar.selectbox("燃料特性", list(fuel_opts.keys()))
fuel_type = fuel_opts[sel_fuel]

# ============================
# UI — メイン：タイトル
# ============================
st.title("火災延焼範囲予測くん")

# ============================
# UI — 地図の中心で位置決め（追加）
# ============================
with st.expander("🧭 地図の中心で発火地点を追加（パンして → 追加）", expanded=False):
    # 位置決め用の小さめマップ（2D）
    pick_center = st.session_state.points[-1] if st.session_state.points else (34.257586, 133.204356)
    m_pick = folium.Map(location=[pick_center[0], pick_center[1]], zoom_start=13, tiles="OpenStreetMap")
    # 見やすいよう基準マーカー（現在の中心）を一つ置く（あくまで目安）
    folium.Marker([pick_center[0], pick_center[1]],
                  icon=folium.Icon(color="blue", icon="crosshairs", prefix="fa"),
                  tooltip="この位置からパンして中心を合わせてください").add_to(m_pick)
    ret = st_folium(m_pick, width=700, height=420, key="picker_map")
    center_lat = ret.get("center", {}).get("lat", pick_center[0]) if isinstance(ret, dict) else pick_center[0]
    center_lng = ret.get("center", {}).get("lng", pick_center[1]) if isinstance(ret, dict) else pick_center[1]
    cols = st.columns(3)
    cols[0].metric("現在の中心 緯度", f"{center_lat:.6f}")
    cols[1].metric("現在の中心 経度", f"{center_lng:.6f}")
    if cols[2].button("中心を発火地点として追加", use_container_width=True):
        st.session_state.points.append((center_lat, center_lng))
        st.success(f"中心を追加しました: ({center_lat:.6f}, {center_lng:.6f})")

# ============================
# 気象データ取得ボタン
# ============================
if st.button("🌤 気象データ取得（Open-Meteo）"):
    if st.session_state.points:
        lat0, lon0 = st.session_state.points[0]
        st.session_state.weather = get_weather(lat0, lon0)
        if st.session_state.weather:
            st.success("気象データを取得しました")
        else:
            st.error("気象データ取得に失敗しました")
    else:
        st.warning("発火地点を追加してください")

# ============================
# 表示モード（2D/3D）とタブ
# ============================
mode = st.radio("表示モード", ["2D 地図", "3D 表示"], horizontal=True)
tabs = st.tabs(["時間", "日", "週", "月"])

def run_sim(duration_h: float):
    if not st.session_state.points:
        st.warning("発火地点を追加してください")
        return
    if not st.session_state.weather:
        st.warning("気象データを取得してください")
        return

    lat0, lon0 = st.session_state.points[0]
    wx = st.session_state.weather

    # --- 数値推定（Gemini → 失敗時フォールバック） ---
    pred = None
    raw = None
    if API_KEY:
        prompt = (
            "あなたは火災拡大シミュレーションの専門家です。\n"
            f"発生地点: 緯度 {lat0}, 経度 {lon0}\n"
            f"気象: 風速 {wx.get('windspeed','不明')} m/s, 風向 {wx.get('winddirection','不明')} 度, "
            f"温度 {wx.get('temperature','不明')} ℃, 湿度 {wx.get('humidity','不明')} %, 降水 {wx.get('precipitation','不明')} mm/h\n"
            f"時間: {duration_h} 時間, 燃料: {fuel_type}, 地形 10度傾斜, 標高150m, 植生 松林と草地が混在\n"
            "出力は純粋なJSONのみ。"
            '{"radius_m": <float>, "area_sqm": <float>, "water_volume_tons": <float>}'
        )
        predicted, raw = gemini_generate(prompt)
        if isinstance(predicted, dict) and {"radius_m","area_sqm","water_volume_tons"} <= set(predicted.keys()):
            pred = predicted
    if pred is None:
        pred = fallback_predict(wx, fuel_type, duration_h)
        if raw:
            with st.expander("Gemini 生レスポンス（参考）"):
                st.json(raw)

    st.session_state.last_pred = pred

    # --- 結果の可視化データ（半径を段階的に拡大） ---
    rad = float(pred.get("radius_m", 0.0))
    wd = float(wx.get("winddirection") or 0.0)
    steps = 12
    radii = [rad * (i+1)/steps for i in range(steps)]

    # 2D用（lat,lon）／3D用（[lon,lat]）
    polys_2d = [sector_for_folium(lat0, lon0, r, wd) for r in radii]
    polys_3d = [
        {"coordinates": sector_for_deck(lat0, lon0, r, wd),
         "radius": r,
         "color": [min(255, int(120 + (r/rad)*135)) if rad > 0 else 180, 60, 40, 120 + int(100*(r/rad)) if rad > 0 else 160]}
        for r in radii
    ]

    st.session_state.last_polys_2d = polys_2d
    st.session_state.last_polys_3d = polys_3d

    # --- 数値表示＆要約 ---
    st.subheader("数値結果")
    c1, c2, c3 = st.columns(3)
    c1.metric("半径 (m)", f"{pred.get('radius_m',0):,.0f}")
    c2.metric("面積 (m²)", f"{pred.get('area_sqm',0):,.0f}")
    c3.metric("必要放水量 (トン)", f"{pred.get('water_volume_tons',0):,.1f}")

    if API_KEY:
        sum_prompt = (
            "次の火災シミュレーション結果JSONを、専門用語を避けて短く日本語で説明してください。\n"
            f"{json.dumps(pred, ensure_ascii=False)}"
        )
        summary, _ = gemini_generate(sum_prompt)
        if isinstance(summary, dict):
            summary = json.dumps(summary, ensure_ascii=False)
        if summary:
            st.subheader("Gemini要約")
            st.write(summary)

# ---- タブごとの処理 ----
with tabs[0]:
    hours = st.slider("時間 (1〜24)", 1, 24, 3)
    if st.button("シミュレーション実行（時間）"):
        run_sim(float(hours))

with tabs[1]:
    days = st.slider("日数 (1〜30)", 1, 30, 3)
    if st.button("シミュレーション実行（日）"):
        run_sim(float(days) * 24.0)

with tabs[2]:
    weeks = st.slider("週数 (1〜52)", 1, 52, 1)
    if st.button("シミュレーション実行（週）"):
        run_sim(float(weeks) * 7.0 * 24.0)

with tabs[3]:
    months = st.slider("月数 (1〜12)", 1, 12, 1)
    if st.button("シミュレーション実行（月）"):
        run_sim(float(months) * 30.0 * 24.0)

# ============================
# 地図表示（常時表示 & 最終結果を保持）
# ============================
center = st.session_state.points[-1] if st.session_state.points else (34.257586, 133.204356)

if mode == "2D 地図":
    m = folium.Map(location=[center[0], center[1]], zoom_start=13, tiles="OpenStreetMap")

    # 発火点マーカー
    for (latp, lonp) in st.session_state.points:
        folium.Marker((latp, lonp),
                      icon=folium.Icon(color="red"),
                      tooltip=f"発火地点 ({latp:.6f}, {lonp:.6f})").add_to(m)

    # 最後の結果があれば扇形を段階的に表示
    if st.session_state.last_polys_2d:
        for poly in st.session_state.last_polys_2d:
            folium.Polygon(poly, color="red", fill=True, fill_opacity=0.30).add_to(m)

    st_folium(m, width=950, height=620, key="main_2d")

else:
    # 3D（Pydeck） — Mapboxトークンが無くても MAP_STYLE="light" で確実に表示
    layers = []
    # 発火点
    if st.session_state.points:
        pts = [{"lon": p[1], "lat": p[0]} for p in st.session_state.points]
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=pts,
                get_position='[lon, lat]',
                get_radius=80,
                get_fill_color='[255,0,0]',
            )
        )
    # 扇形（段階的に拡大したリングを重ねる）
    if st.session_state.last_polys_3d:
        data = [{"polygon": d["coordinates"], "color": d["color"], "elev": max(20.0, d["radius"]*0.15)} for d in st.session_state.last_polys_3d]
        layers.append(
            pdk.Layer(
                "PolygonLayer",
                data=data,
                get_polygon="polygon",
                get_fill_color="color",
                get_elevation="elev",
                extruded=True,
                stroked=False,
                pickable=True,
            )
        )

    view_state = pdk.ViewState(latitude=center[0], longitude=center[1], zoom=12.5, pitch=45)
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=MAP_STYLE,    # ✅ トークン無しでも表示されるスタイルを使用
        tooltip={"text": "高度: {elev} m"}
    )
    st.pydeck_chart(deck, use_container_width=True, key="main_3d")
