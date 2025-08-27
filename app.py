import os
import math
import json
import re
from typing import Optional, Dict, Tuple

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
    API_KEY = os.environ.get("GOOGLE_API_KEY")

# ============================
# JSON抽出ユーティリティ
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
# Googleマップ座標文字列パーサ
# ============================
LATLON_REGEX = re.compile(r"\s*\(?\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)?\s*")

def parse_latlon_text(text: str) -> Optional[Tuple[float, float]]:
    """"34.246099951898415, 133.20578422112848" などの文字列から (lat, lon) を抽出。"""
    if not text:
        return None
    m = LATLON_REGEX.match(text.strip())
    if not m:
        return None
    try:
        lat = float(m.group(1))
        lon = float(m.group(2))
        if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
            return lat, lon
    except Exception:
        return None
    return None

# ============================
# Open-Meteoから気象取得
# ============================
@st.cache_data(show_spinner=False)
def get_weather(lat: float, lon: float) -> Optional[Dict[str, float]]:
    try:
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
               f"&current_weather=true&hourly=relativehumidity_2m,precipitation&timezone=auto")
        resp = requests.get(url, timeout=20)
        data = resp.json()
        cur = data.get("current_weather", {})
        res = {
            "temperature": cur.get("temperature"),
            "windspeed": cur.get("windspeed"),
            "winddirection": cur.get("winddirection"),
            "weathercode": cur.get("weathercode"),
        }
        t = cur.get("time")
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
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
# Gemini API呼び出し
# ============================
def gemini_generate(prompt: str) -> Tuple[Optional[dict], Optional[dict]]:
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
        headers = {"Content-Type": "application/json"}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        raw = r.json() if r.content else None
        if r.status_code != 200 or not raw:
            return None, raw
        text = ""
        cands = raw.get("candidates", [])
        if cands:
            if "output" in cands[0]:
                text = cands[0].get("output", "")
            else:
                parts = cands[0].get("content", {}).get("parts", [])
                for p in parts:
                    text += p.get("text", "")
        parsed = extract_json(text)
        return parsed, raw
    except Exception:
        return None, None

# ============================
# 扇形ポリゴン生成
# ============================
def create_sector_polygon(lat: float, lon: float, radius_m: float, wind_dir_deg: float, steps: int = 48):
    coords = [(lat, lon)]
    start, end = wind_dir_deg - 90.0, wind_dir_deg + 90.0
    for i in range(steps + 1):
        ang = math.radians(start + (end - start) * i / steps)
        north_m = radius_m * math.cos(ang)
        east_m = radius_m * math.sin(ang)
        dlat = north_m / 111000.0
        dlon = east_m / 111000.0
        coords.append((lat + dlat, lon + dlon))
    return coords

# ============================
# UIサイドバー（Googleマップ座標コピペ対応）
# ============================
st.sidebar.header("発生地点と条件設定")
with st.sidebar.form("point_form"):
    st.caption("Googleマップの座標（例: 34.246099951898415, 133.20578422112848）をそのまま貼り付けできます。")
    latlon_text = st.text_input("座標（緯度, 経度）をコピペ", value="")
    st.markdown("または下の数値入力を使用")
    col_a, col_b = st.columns(2)
    with col_a:
        lat_in = st.number_input("緯度", value=34.257586, format="%.6f")
    with col_b:
        lon_in = st.number_input("経度", value=133.204356, format="%.6f")
    add = st.form_submit_button("発生地点を追加")
    if add:
        parsed = parse_latlon_text(latlon_text)
        if parsed:
            plat, plon = parsed
        else:
            plat, plon = lat_in, lon_in
        if "points" not in st.session_state:
            st.session_state.points = []
        st.session_state.points.append((plat, plon))
        if parsed:
            st.sidebar.success(f"Googleマップ形式から追加: ({plat:.8f}, {plon:.8f})")
        else:
            st.sidebar.success(f"数値入力から追加: ({plat:.6f}, {plon:.6f})")

if st.sidebar.button("登録地点を消去"):
    st.session_state.points = []
    st.sidebar.info("削除しました")

fuel_opts = {"森林（高燃料)": "森林", "草地（中燃料)": "草地", "都市部（低燃料)": "都市部"}
sel_fuel = st.sidebar.selectbox("燃料特性", list(fuel_opts.keys()))
fuel_type = fuel_opts[sel_fuel]

if "points" not in st.session_state:
    st.session_state.points = []

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
st.title("火災拡大シミュレーション（Gemini要約付き）")

# 表示モード選択
mode = st.radio("表示モード", ["2D 地図", "3D 表示"], horizontal=True)

# タブ（時間/日/週/月）
tabs = st.tabs(["時間", "日", "週", "月"])


def run_sim(duration_h: float):
    if not st.session_state.points:
        st.warning("発生地点を追加してください")
        return
    if "weather" not in st.session_state or not st.session_state.weather:
        st.warning("気象データを取得してください")
        return
    lat0, lon0 = st.session_state.points[0]
    wx = st.session_state.weather
    prompt = (
        "あなたは火災拡大シミュレーションの専門家です。\n"
        f"発生地点 緯度{lat0},経度{lon0}\n"
        f"気象:風速{wx.get('windspeed','不明')}m/s,風向{wx.get('winddirection','不明')}度,温度{wx.get('temperature','不明')}℃,湿度{wx.get('humidity','不明')}%,降水{wx.get('precipitation','不明')}mm/h\n"
        f"時間 {duration_h}時間, 燃料 {fuel_type}, 地形10度傾斜, 標高150m, 植生松林草地混在\n"
        "出力はJSONのみ {\"radius_m\":<float>,\"area_sqm\":<float>,\"water_volume_tons\":<float>}"
    )
    pred, raw = gemini_generate(prompt)
    if not pred:
        st.error("Gemini推定に失敗しました")
        if raw:
            with st.expander("生レスポンス"):
                st.json(raw)
        return
    st.subheader("数値結果")
    st.write(pred)
    # 要約
    sum_prompt = (
        "次の火災シミュレーション結果JSONを簡単に説明してください。\n"
        f"{json.dumps(pred,ensure_ascii=False)}"
    )
    summary, _ = gemini_generate(sum_prompt)
    if summary:
        st.subheader("Gemini要約")
        st.write(summary)
    rad = float(pred.get("radius_m",0))
    wd = float(wx.get("winddirection",0) or 0)
    if mode=="2D 地図":
        m = folium.Map(location=[lat0, lon0], zoom_start=13)
        for pt in st.session_state.points:
            folium.Marker(pt, icon=folium.Icon(color="red"), tooltip=f"地点 {pt}").add_to(m)
        sector = create_sector_polygon(lat0, lon0, rad, wd)
        folium.Polygon(sector, color="red", fill=True, fill_opacity=0.4,
                       tooltip=f"半径{rad:.0f}m 面積{pred.get('area_sqm',0):.0f}㎡").add_to(m)
        st_folium(m, width=900, height=600)
    else:
        # pydeck は Mapboxトークン不要のベースでも表示できるスタイルを使用
        ring = [[lon0, lat0]] + [[lon, lat] for lat, lon in create_sector_polygon(lat0, lon0, rad, wd)]
        poly = [{"coordinates": ring}]
        layer = pdk.Layer(
            "PolygonLayer",
            data=poly,
            get_polygon="coordinates",
            get_fill_color="[255, 0, 0, 100]",
            get_line_color="[255,0,0]",
            stroked=True,
            filled=True,
            pickable=True,
        )
        view_state = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=13, pitch=45)
        r = pdk.Deck(layers=[layer], initial_view_state=view_state, map_style="light")
        st.pydeck_chart(r)

# タブごとの処理
with tabs[0]:
    hours = st.slider("時間 (1〜24)", 1, 24, 3)
    if st.button("シミュレーション実行 (時間)"):
        run_sim(hours)

with tabs[1]:
    days = st.slider("日数 (1〜30)", 1, 30, 3)
    if st.button("シミュレーション実行 (日)"):
        run_sim(days*24)

with tabs[2]:
    weeks = st.slider("週数 (1〜52)", 1, 52, 1)
    if st.button("シミュレーション実行 (週)"):
        run_sim(weeks*7*24)

with tabs[3]:
    months = st.slider("月数 (1〜12)", 1, 12, 1)
    if st.button("シミュレーション実行 (月)"):
        run_sim(months*30*24)
