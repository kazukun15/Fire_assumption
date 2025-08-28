# app.py
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
st.set_page_config(page_title="火災拡大シミュレーション", layout="wide")

# ============================
# Secrets / APIキー取得
# ============================
MODEL_NAME = "gemini-2.0-flash-001"
API_KEY = None
try:
    API_KEY = st.secrets["general"]["api_key"]
except Exception:
    API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

# Mapbox（任意）。設定があれば衛星/ダークを使用、なければ軽量スタイル
MAPBOX_TOKEN = None
try:
    MAPBOX_TOKEN = st.secrets.get("mapbox", {}).get("access_token")
except Exception:
    MAPBOX_TOKEN = os.environ.get("MAPBOX_API_KEY") or os.environ.get("MAPBOX_TOKEN")

if MAPBOX_TOKEN:
    pdk.settings.mapbox_api_key = MAPBOX_TOKEN
    MAP_STYLE = "mapbox://styles/mapbox/dark-v10"
else:
    MAP_STYLE = "light"  # トークン不要のデフォルト

# ============================
# JSON抽出ユーティリティ（Geminiが```jsonで返す場合にも対応）
# ============================
def extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    # ✅ 正しい生の正規表現（ダブルエスケープを排除）
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
# Gemini API呼び出し（堅牢パース）
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
            # 両系統に対応
            if "output" in cands[0]:
                text = cands[0].get("output", "") or ""
            else:
                parts = cands[0].get("content", {}).get("parts", [])
                for p in parts:
                    text += p.get("text", "") or ""
        # JSON抽出（失敗ならtextのまま返す）
        parsed = extract_json(text)
        return (parsed if parsed is not None else text), raw
    except Exception:
        return None, None

# ============================
# フォールバック推定（Gemini失敗/未設定時でも動く簡易物理モデル）
# ============================
def fallback_predict(wx: Dict[str, float], fuel_label: str, hours: float) -> dict:
    # 燃料別の基準拡大速度（m/s）：簡易モデル
    v0 = {"森林": 0.30, "草地": 0.60, "都市部": 0.20}.get(fuel_label, 0.40)
    wind = float(wx.get("windspeed") or 0)
    rh = float(wx.get("humidity") or 60)
    precip = float(wx.get("precipitation") or 0)

    wind_factor = 1.0 + 0.12 * wind               # 風で前進増速
    humidity_factor = max(0.6, 1.0 - 0.003 * max(0.0, rh - 30.0))  # 湿度で抑制
    precip_factor = max(0.5, 1.0 / (1.0 + precip))                  # 降水で抑制

    v_eff = v0 * wind_factor * humidity_factor * precip_factor
    radius_m = max(40.0, v_eff * hours * 3600.0)  # 経過時間h→秒換算
    area_sqm = 0.5 * math.pi * radius_m * radius_m  # 風向±90°の扇形を想定して1/2πr^2
    water_tons = area_sqm * 0.01  # 10L/m^2 → 0.01トン/m^2

    return {"radius_m": radius_m, "area_sqm": area_sqm, "water_volume_tons": water_tons}

# ============================
# 扇形ポリゴン生成（Folium用/Deck.gl用）
# ============================
def sector_latlon_for_folium(lat: float, lon: float, radius_m: float, wind_dir_deg: float, steps: int = 64):
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

def sector_lonlat_for_deck(lat: float, lon: float, radius_m: float, wind_dir_deg: float, steps: int = 64):
    """Deck.gl用: [[lon, lat], ...]（クローズドリング）"""
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
# 3Dアニメーション・フレーム生成
# ============================
def build_frames(lat: float, lon: float, final_radius: float, wind_dir: float, duration_h: float, windspeed: float,
                 n_frames: int = 30) -> List[dict]:
    n_frames = max(6, min(120, n_frames))
    frames = []
    # 成長曲線（やや加速）: frac=(i/N)^gamma、風速でgamma微調整
    gamma = max(0.6, min(1.2, 1.0 - 0.02 * (windspeed - 5.0)))
    for i in range(n_frames):
        t_frac = (i + 1) / n_frames
        frac = pow(t_frac, gamma)
        r = max(20.0, final_radius * frac)
        ring = sector_lonlat_for_deck(lat, lon, r, wind_dir)
        # グラデーション: 新しいほど赤強め&不透明、古いほど橙で半透明
        g = int(40 + 160 * (1.0 - frac))
        a = int(60 + 180 * frac)
        r_ch = min(255, int(200 + 5 * windspeed))
        color = [r_ch, g, 40, max(60, min(255, a))]
        elev = float(max(20.0, windspeed * 40.0 * frac))  # 火勢の強さの高さ表現
        frames.append({
            "polygon": ring,
            "rgba": color,
            "elev": elev,
            "t_idx": i,
            "t_hours": duration_h * t_frac,
            "radius_m": r,
        })
    return frames

# ============================
# セッション状態
# ============================
if "points" not in st.session_state:
    st.session_state.points: List[Tuple[float, float]] = []
if "weather" not in st.session_state:
    st.session_state.weather: Optional[Dict[str, float]] = None
if "last_pred" not in st.session_state:
    st.session_state.last_pred: Optional[Dict] = None
if "frames" not in st.session_state:
    st.session_state.frames: Optional[List[dict]] = None
if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0
if "animating" not in st.session_state:
    st.session_state.animating = False

# ============================
# UIサイドバー（Googleマップ座標コピペ対応）
# ============================
st.sidebar.header("発生地点と条件設定")
with st.sidebar.form("point_form"):
    st.caption("Googleマップの座標（例: 34.246099951898415, 133.20578422112848）をそのまま貼り付けできます。")
    coord_text = st.text_input("緯度,経度（カンマ区切り）", "34.257586,133.204356")
    add = st.form_submit_button("発生地点を追加")
    if add:
        try:
            lat_in, lon_in = [float(x.strip().strip("()")) for x in coord_text.split(",")]
            st.session_state.points.append((lat_in, lon_in))
            st.sidebar.success(f"地点 ({lat_in},{lon_in}) を追加")
        except Exception:
            st.sidebar.error("緯度経度の形式が不正です。例: 34.246099951898415, 133.20578422112848")

if st.sidebar.button("登録地点を消去"):
    st.session_state.points = []
    st.session_state.weather = None
    st.session_state.last_pred = None
    st.session_state.frames = None
    st.sidebar.info("削除しました")

fuel_opts = {"森林（高燃料)": "森林", "草地（中燃料)": "草地", "都市部（低燃料)": "都市部"}
sel_fuel = st.sidebar.selectbox("燃料特性", list(fuel_opts.keys()))
fuel_type = fuel_opts[sel_fuel]

if st.sidebar.button("気象データ取得"):
    if st.session_state.points:
        lat0, lon0 = st.session_state.points[0]
        st.session_state.weather = get_weather(lat0, lon0)
        if st.session_state.weather:
            st.sidebar.success("気象データを取得しました")
        else:
            st.sidebar.error("気象データ取得に失敗しました")
    else:
        st.sidebar.warning("地点を追加してください")

# ============================
# メインビュー
# ============================
st.title("火災拡大シミュレーション（Gemini要約＋3Dアニメーション）")

# 表示モード
mode = st.radio("表示モード", ["2D 地図", "3D 表示"], horizontal=True)

# タブ（時間/日/週/月）
tabs = st.tabs(["時間", "日", "週", "月"])

def run_sim(duration_h: float):
    if not st.session_state.points:
        st.warning("発生地点を追加してください")
        return
    if not st.session_state.weather:
        st.warning("気象データを取得してください")
        return

    lat0, lon0 = st.session_state.points[0]
    wx = st.session_state.weather

    # ---- Gemini で数値推定（JSON想定） ----
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

    # ---- フォールバック ----
    if pred is None:
        pred = fallback_predict(wx, fuel_type, duration_h)
        if raw:  # 参考として生レスポンスを折りたたみ表示
            with st.expander("Gemini 生レスポンス（参考）"):
                st.json(raw)

    # ---- 数値表示 ----
    st.subheader("数値結果")
    c1, c2, c3 = st.columns(3)
    c1.metric("半径 (m)", f"{pred.get('radius_m',0):,.0f}")
    c2.metric("面積 (m²)", f"{pred.get('area_sqm',0):,.0f}")
    c3.metric("必要放水量 (トン)", f"{pred.get('water_volume_tons',0):,.1f}")

    # ---- 要約（APIキーがあれば）----
    if API_KEY:
        sum_prompt = (
            "次の火災シミュレーション結果JSONを、専門用語を避けて短く日本語で説明してください。\n"
            f"{json.dumps(pred, ensure_ascii=False)}"
        )
        summary, _ = gemini_generate(sum_prompt)
        if isinstance(summary, dict):
            # 稀にJSONで返る場合に備えて文字列化
            summary = json.dumps(summary, ensure_ascii=False)
        if summary:
            st.subheader("Gemini要約")
            st.write(summary)

    st.session_state.last_pred = pred

    # ---- 3Dアニメーション・フレーム準備 ----
    wind_dir = float(wx.get("winddirection") or 0)
    windspeed = float(wx.get("windspeed") or 0)
    frames = build_frames(lat0, lon0, float(pred.get("radius_m", 0.0)), wind_dir, duration_h, windspeed, n_frames=30)
    st.session_state.frames = frames
    st.session_state.frame_idx = min(st.session_state.frame_idx, len(frames) - 1)

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
# 2D/3D 表示（マップは常に表示）
# ============================
center = st.session_state.points[0] if st.session_state.points else (34.257586, 133.204356)

if mode == "2D 地図":
    # 常時ベースマップ
    m = folium.Map(location=[center[0], center[1]], zoom_start=13, tiles="OpenStreetMap")

    # 発火点マーカー
    for (latp, lonp) in st.session_state.points:
        folium.Marker((latp, lonp), icon=folium.Icon(color="red"), tooltip=f"地点 ({latp:.6f}, {lonp:.6f})").add_to(m)

    # 直近推定があれば扇形ポリゴン（最終フレームのみ）
    if st.session_state.last_pred and st.session_state.weather and st.session_state.points:
        lat0, lon0 = st.session_state.points[0]
        wd = float(st.session_state.weather.get("winddirection") or 0)
        rad = float(st.session_state.last_pred.get("radius_m", 0.0))
        if rad > 0:
            poly_latlon = sector_latlon_for_folium(lat0, lon0, rad, wd)
            folium.Polygon(
                poly_latlon, color="red", fill=True, fill_opacity=0.35, tooltip=f"半径 {rad:.0f} m"
            ).add_to(m)

    st_folium(m, width=900, height=600)

else:
    # 3D: 再生/停止・スライダーUI + 描画
    st.subheader("3Dアニメーション（段階的拡大）")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("▶ 再生", use_container_width=True):
            st.session_state.animating = True
    with c2:
        if st.button("⏸ 停止", use_container_width=True):
            st.session_state.animating = False
    with c3:
        if st.session_state.frames:
            n = len(st.session_state.frames)
            st.session_state.frame_idx = st.slider(
                "フレーム（時間進行）", 0, n - 1, st.session_state.frame_idx, key="frame_slider",
                help="スライダーでも任意時刻に移動できます"
            )
        else:
            st.info("シミュレーションを実行すると、ここにアニメーションが表示されます。")

    placeholder = st.empty()

    def render_frame(idx: int):
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
        # 累積扇形（0〜idx）を重ねて“拡大”を表現
        if st.session_state.frames:
            show = st.session_state.frames[: idx + 1]
            data = [
                {
                    "polygon": f["polygon"],
                    "rgba": f["rgba"],
                    "elev": f["elev"],
                    "t": f["t_hours"],
                    "r": f["radius_m"],
                }
                for f in show
            ]
            layers.append(
                pdk.Layer(
                    "PolygonLayer",
                    data=data,
                    get_polygon="polygon",
                    get_fill_color="rgba",
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
            map_style=MAP_STYLE,
            tooltip={"text": "半径: {r} m\n時刻: {t} h"}
        )
        placeholder.pydeck_chart(deck, use_container_width=True)

    # 1フレーム描画
    if st.session_state.frames:
        render_frame(st.session_state.frame_idx)
        # 再生中は自動で次フレームへ（Streamlitの簡易アニメ）
        if st.session_state.animating:
            time.sleep(0.4)
            st.session_state.frame_idx = (st.session_state.frame_idx + 1) % len(st.session_state.frames)
            st.experimental_rerun()
    else:
        # データがない時でもベース表示
        render_frame(0)
