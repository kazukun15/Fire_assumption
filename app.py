# app.py  —  火災延焼範囲予測くん（説明生成つき）
import os
import math
import json
import re
from typing import Optional, Dict, Tuple, List

import requests
import streamlit as st
import folium
from streamlit_folium import st_folium
import pydeck as pdk
import geopandas as gpd
from shapely.geometry import Point
import osmnx as ox

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

# ============================
# JSON抽出ユーティリティ（Geminiが```json ...```で返す場合に対応）
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
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
               f"&current_weather=true&hourly=relativehumidity_2m,precipitation&timezone=auto")
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
# Gemini API呼び出し（数値/JSON用：堅牢パース）
# ============================
def gemini_generate(prompt: str) -> Tuple[Optional[dict], Optional[dict]]:
    if not API_KEY:
        return None, None
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
# Gemini API呼び出し（テキスト用：説明生成）
# ============================
def gemini_generate_text(prompt: str) -> Tuple[Optional[str], Optional[dict]]:
    if not API_KEY:
        return None, None
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
                text = cands[0].get("output", "") or ""
            else:
                parts = cands[0].get("content", {}).get("parts", [])
                for p in parts:
                    text += p.get("text", "") or ""
        return (text or None), raw
    except Exception:
        return None, None

# ============================
# OSM建物データ取得
# ============================
@st.cache_data(show_spinner=False)
def get_osm_buildings(lat: float, lon: float, dist: int = 1000):
    try:
        gdf = ox.geometries_from_point((lat, lon), tags={"building": True}, dist=dist)
        # Polygon系のみに限定（表示・交差簡略化のため）
        if gdf is not None and not gdf.empty:
            gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        return gdf
    except Exception:
        return None

# ============================
# 方角ラベル（日本語8方位）
# ============================
def bearing_to_label(deg: float) -> str:
    dirs = ["北", "北東", "東", "南東", "南", "南西", "西", "北西", "北"]
    idx = int(((deg % 360) + 22.5) // 45)
    return dirs[idx]

# ============================
# UIサイドバー
# ============================
st.sidebar.header("発生地点と条件設定")
with st.sidebar.form("point_form"):
    st.caption("Googleマップの座標（例: 34.246099951898415, 133.20578422112848）をそのまま貼り付け可能。")
    coord_text = st.text_input("緯度,経度（カンマ区切り）", "34.257586,133.204356")
    add = st.form_submit_button("発生地点を追加")
    if add:
        try:
            lat_in, lon_in = [float(x.strip().strip("()")) for x in coord_text.split(",")]
            if "points" not in st.session_state:
                st.session_state.points = []
            st.session_state.points.append((lat_in, lon_in))
            st.sidebar.success(f"地点 ({lat_in},{lon_in}) を追加")
        except Exception:
            st.sidebar.error("緯度経度の形式が不正です。例: 34.246099951898415, 133.20578422112848")

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
        st.session_state.buildings = get_osm_buildings(lat0, lon0)
        if st.session_state.weather:
            st.sidebar.success("気象・建物データを取得しました")
        else:
            st.sidebar.error("気象データの取得に失敗しました")
    else:
        st.sidebar.warning("地点を追加してください")

# ============================
# メインビュー
# ============================
st.title("火災延焼範囲予測くん")

# 表示モード選択
mode = st.radio("表示モード", ["2D 地図", "3D 表示"], horizontal=True)

# タブ（時間/日/週/月）
tabs = st.tabs(["時間", "日", "週", "月"])

# 地図で発火点選択（クリック）
def pick_location():
    m = folium.Map(location=[34.257586, 133.204356], zoom_start=13, tiles="OpenStreetMap")
    if "points" in st.session_state and st.session_state.points:
        for pt in st.session_state.points:
            folium.Marker(pt, icon=folium.Icon(color="red"), tooltip=f"地点 {pt}").add_to(m)
    picked = st_folium(m, width=700, height=500, key="pick_for_point")
    if isinstance(picked, dict) and picked.get("last_clicked"):
        lat = picked["last_clicked"]["lat"]
        lon = picked["last_clicked"]["lng"]
        st.session_state.points = [(lat, lon)]
        st.success(f"クリックした地点を発火点として設定しました ({lat:.5f},{lon:.5f})")

with st.expander("地図で発火点を選択"):
    pick_location()

# ============================
# 説明生成ヘルパ
# ============================
def build_fire_explanation(wx: Dict[str, float], pred: Dict[str, float], duration_h: float,
                           fuel: str, buildings_count: int) -> str:
    """
    Geminiで説明文を生成。API未設定/失敗時はフォールバックの定型文を返す。
    """
    # 入力値の整形
    wind_dir_from = float(wx.get("winddirection") or 0.0)
    windspeed = float(wx.get("windspeed") or 0.0)
    downwind_to = (wind_dir_from + 180.0) % 360.0
    down_label = bearing_to_label(downwind_to)

    radius = float(pred.get("radius_m") or 0.0)
    area = float(pred.get("area_sqm") or 0.0)
    water = float(pred.get("water_volume_tons") or 0.0)
    ros_mps = (radius / (duration_h * 3600.0)) if duration_h > 0 else 0.0
    ros_kmh = ros_mps * 3.6

    # Geminiプロンプト（説明）
    prompt = (
        "あなたは消防・林野火災の専門家かつ記者です。次の事実に基づき、一般向けにわかりやすく、"
        "過度に専門的になりすぎない説明文を日本語で作成してください。"
        "構成は「火災の状況」「延焼の広がり方」「効果的な消火方法」「気象から見た延焼の方角と速度」の4項目です。"
        "箇条書きを適度に使い、重要な数値（半径・面積・速度・方角・水量）はそのまま示してください。\n\n"
        f"- 燃料特性: {fuel}\n"
        f"- 期間: {duration_h} 時間\n"
        f"- 予測半径: {radius:.0f} m, 面積: {area:.0f} m², 必要放水量: {water:.1f} トン\n"
        f"- 風: 風速 {windspeed:.1f} m/s, 風向（FROM）{wind_dir_from:.0f}° → 風下（TO）{down_label} 方向\n"
        f"- 予測延焼速度（概算）: {ros_mps:.2f} m/s（約 {ros_kmh:.2f} km/h）\n"
        f"- 周辺の遮蔽物（建物など）: OSM建物 {buildings_count} 件（存在すれば燃料連続性を分断）\n"
        f"- 気温: {wx.get('temperature','不明')} ℃, 湿度: {wx.get('humidity','不明')} %, 降水: {wx.get('precipitation','不明')} mm/h\n\n"
        "出力は読みやすい短い段落＋箇条書きの組み合わせにしてください。比喩や推測は控えめにし、"
        "与えられた事実から安全側の助言をまとめてください。"
    )

    # Geminiで生成
    if API_KEY:
        text, _ = gemini_generate_text(prompt)
        if text:
            return text

    # フォールバック（Gemini未設定/失敗時）
    msg = []
    msg.append("### 🔥 解説")
    msg.append("**火災の状況**")
    msg.append(f"- 燃料: {fuel}。期間 {duration_h} 時間で、想定半径は約 {radius:.0f} m、面積は約 {area:.0f} m²。")
    msg.append("**延焼の広がり方**")
    msg.append(f"- 風下（{down_label}）側へ優先的に拡大。概算の延焼速度は {ros_mps:.2f} m/s（{ros_kmh:.2f} km/h）。")
    if buildings_count > 0:
        msg.append(f"- 周辺に建物（{buildings_count} 件）があり、燃料の連続性が分断され一部で拡大が抑制される可能性。")
    msg.append("**効果的な消火方法**")
    msg.append("- 風下側の先回り展開、退避路の確保、可燃物の除去ライン。可能なら散水・薬剤で冷却・再燃防止。")
    msg.append("**気象から見た方角と速度**")
    msg.append(f"- 風速 {windspeed:.1f} m/s、風向（FROM）{wind_dir_from:.0f}° → 風下（TO）{down_label}。速度は上記のとおり。")
    msg.append(f"- 必要放水量の目安: 約 {water:.1f} トン。")
    return "\n".join(msg)

# ============================
# シミュレーション本体
# ============================
def run_sim(duration_h: float):
    if not st.session_state.points:
        st.warning("発生地点を追加してください")
        return
    if "weather" not in st.session_state or not st.session_state.weather:
        st.warning("気象データを取得してください")
        return

    lat0, lon0 = st.session_state.points[0]
    wx = st.session_state.weather
    buildings = st.session_state.get("buildings")

    # Geminiプロンプト（数値推定：既存仕様を維持）
    prompt = (
        "あなたは火災拡大シミュレーションの専門家です。OSM建物データを用いて遮蔽物を考慮し、延焼範囲を予測してください。\n"
        f"発生地点: 緯度 {lat0}, 経度 {lon0}\n"
        f"気象: 風速 {wx.get('windspeed','不明')} m/s, 風向 {wx.get('winddirection','不明')} 度, 温度 {wx.get('temperature','不明')} ℃, 湿度 {wx.get('humidity','不明')} %, 降水 {wx.get('precipitation','不明')} mm/h\n"
        f"時間: {duration_h} 時間, 燃料: {fuel_type}, 地形: 10度傾斜, 標高150m, 植生: 松林と草地が混在\n"
        "出力は純粋なJSONのみ。 {\"radius_m\":<float>,\"area_sqm\":<float>,\"water_volume_tons\":<float>}"
    )
    pred, raw = gemini_generate(prompt)
    if not pred:
        st.error("Geminiによる数値推定に失敗しました。")
        if raw:
            with st.expander("Gemini 生レスポンス（参考）"):
                st.json(raw)
        return

    st.subheader("数値結果")
    st.write(pred)

    # 既存の要約（そのまま維持）
    sum_prompt = (
        "次の火災シミュレーション結果JSONを簡単に説明してください。OSM建物データにより遮蔽物が考慮されています。\n"
        f"{json.dumps(pred, ensure_ascii=False)}"
    )
    summary, _ = gemini_generate(sum_prompt)
    if summary:
        st.subheader("Gemini要約")
        st.write(summary)

    # 追加：説明生成（下部に表示するためセッションに保存）
    b_count = 0
    if isinstance(buildings, gpd.GeoDataFrame) and not buildings.empty:
        b_count = len(buildings)
    explanation_text = build_fire_explanation(wx, pred, duration_h, fuel_type, b_count)
    st.session_state["fire_explanation"] = explanation_text

    # 地図描画（既存仕様を維持：建物のオーバーレイ）
    rad = float(pred.get("radius_m", 0) or 0)
    wd = float(wx.get("winddirection", 0) or 0)

    polygons: List[List[Tuple[float, float]]] = []
    if isinstance(buildings, gpd.GeoDataFrame) and not buildings.empty:
        for geom in buildings.geometry:
            try:
                if geom.geom_type == "Polygon":
                    coords = [(y, x) for x, y in geom.exterior.coords]
                    polygons.append(coords)
                elif geom.geom_type == "MultiPolygon":
                    for poly in geom.geoms:
                        coords = [(y, x) for x, y in poly.exterior.coords]
                        polygons.append(coords)
            except Exception:
                continue

    if mode == "2D 地図":
        m = folium.Map(location=[lat0, lon0], zoom_start=15, tiles="OpenStreetMap")
        for pt in st.session_state.points:
            folium.Marker(pt, icon=folium.Icon(color="red"), tooltip=f"地点 {pt}").add_to(m)
        if polygons:
            for poly in polygons:
                folium.Polygon(poly, color="blue", fill=True, fill_opacity=0.4, tooltip="建物").add_to(m)
        st_folium(m, width=900, height=600, key="main_map")
    else:
        layer_buildings = []
        if polygons:
            layer_buildings = [{"polygon": [(x, y) for y, x in poly]} for poly in polygons]
        layer = pdk.Layer(
            "PolygonLayer",
            data=layer_buildings,
            get_polygon="polygon",
            get_fill_color="[100,100,255,150]",
            stroked=True,
            filled=True,
            extruded=True,
            get_elevation=10,
        )
        view_state = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=15, pitch=45)
        deck = pdk.Deck(layers=[layer], initial_view_state=view_state, map_style="light")
        st.pydeck_chart(deck, use_container_width=True, key="main_deck")

# ============================
# タブごとの処理（時間/日/週/月）
# ============================
with tabs[0]:
    hours = st.slider("時間 (1〜24)", 1, 24, 3)
    if st.button("シミュレーション実行 (時間)"):
        run_sim(float(hours))

with tabs[1]:
    days = st.slider("日数 (1〜30)", 1, 30, 3)
    if st.button("シミュレーション実行 (日)"):
        run_sim(float(days) * 24.0)

with tabs[2]:
    weeks = st.slider("週数 (1〜52)", 1, 52, 1)
    if st.button("シミュレーション実行 (週)"):
        run_sim(float(weeks) * 7.0 * 24.0)

with tabs[3]:
    months = st.slider("月数 (1〜12)", 1, 12, 1)
    if st.button("シミュレーション実行 (月)"):
        run_sim(float(months) * 30.0 * 24.0)

# ============================
# 画面下部：説明の表示（セッションにあれば常時表示）
# ============================
st.markdown("---")
st.subheader("📝 解説（Gemini生成）")
if st.session_state.get("fire_explanation"):
    # 生成テキストをそのまま表示（Markdown対応）
    st.markdown(st.session_state["fire_explanation"])
else:
    st.info("シミュレーションを実行すると、ここに「火災の説明／延焼の広がり方／効果的な消火方法／気象から見た方角と速度」の解説が表示されます。")
