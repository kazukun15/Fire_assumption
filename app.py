# app.py — フィーチャー連携で精度向上（2D/3D, OSM + DEM, 結果永続化）
# =============================================================================
# 概要:
#  - 発生地点（複数）+ Open‑Meteo（現在気象）
#  - OSM（道路/水域/建物/土地利用）と DEM（Open‑Elevation）を取得し、
#    風速・燃料・傾斜・障害物密度で物理簡易モデルを補正
#  - さらに同情報を含めて Gemini に数値推定を依頼（JSON）→ 要約生成
#  - 2D（Folium）/ 3D（pydeck）切替、結果・オーバーレイは session_state に永続化
# 依存: streamlit, requests, folium, streamlit-folium, pydeck
# 実行: streamlit run app.py
# =============================================================================

import os
import json
import math
import re
from typing import Dict, Optional, Tuple, List

import requests
import streamlit as st
import folium
from streamlit_folium import st_folium
import pydeck as pdk

# ---------------------------------
# ページ設定
# ---------------------------------
st.set_page_config(page_title="火災拡大シミュレーション（精度向上版）", layout="wide")

# ---------------------------------
# Secrets / API キー
# ---------------------------------
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-001")
API_KEY: Optional[str] = None
try:
    if "general" in st.secrets and "api_key" in st.secrets["general"]:
        API_KEY = st.secrets["general"]["api_key"]
except Exception:
    API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

# ---------------------------------
# ユーティリティ
# ---------------------------------

def extract_json(text: str) -> Optional[dict]:
    """Gemini応答から最初のJSONを抽出。```json フェンスにも対応。"""
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

@st.cache_data(show_spinner=False)
def get_weather(lat: float, lon: float) -> Optional[Dict[str, float]]:
    """Open‑Meteo: 現在気象＋hourlyから湿度/降水（代表時刻）"""
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&current_weather=true&hourly=relativehumidity_2m,precipitation&timezone=auto"
        )
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
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
            rh = hourly.get("relativehumidity_2m", []) or []
            pr = hourly.get("precipitation", []) or []
            if i < len(rh):
                res["humidity"] = rh[i]
            if i < len(pr):
                res["precipitation"] = pr[i]
        return res
    except Exception:
        return None

# ---- DEM（Open‑Elevation）: 近傍5点から傾斜[%]を推定 ----
@st.cache_data(show_spinner=False)
def get_slope_percent(lat: float, lon: float, step_m: float = 200.0) -> Optional[float]:
    try:
        deg = step_m / 111_000.0
        locs = [
            (lat, lon),
            (lat + deg, lon),
            (lat - deg, lon),
            (lat, lon + deg),
            (lat, lon - deg),
        ]
        qs = "|".join([f"{a:.6f},{b:.6f}" for a, b in locs])
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={qs}"
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return None
        js = r.json(); res = js.get("results", [])
        if len(res) < 5:
            return None
        zc = res[0]["elevation"]; zn = res[1]["elevation"]; zs = res[2]["elevation"]; ze = res[3]["elevation"]; zw = res[4]["elevation"]
        dz_ns = (zn - zs) / (2 * step_m)
        dz_ew = (ze - zw) / (2 * step_m)
        slope = math.sqrt(dz_ns ** 2 + dz_ew ** 2) * 100.0  # %
        return float(max(0.0, min(slope, 100.0)))
    except Exception:
        return None

# ---- OSM（Overpass）: 道路/水域/建物/土地利用 ----
@st.cache_data(show_spinner=False)
def get_osm_features(lat: float, lon: float, radius_m: int = 1500) -> Optional[Dict]:
    try:
        q = (
            "[out:json][timeout:25];("
            f"way[\"highway\"](around:{radius_m},{lat},{lon});"
            f"way[\"waterway\"](around:{radius_m},{lat},{lon});"
            f"way[\"landuse\"](around:{radius_m},{lat},{lon});"
            f"way[\"natural\"](around:{radius_m},{lat},{lon});"
            f"way[\"building\"](around:{radius_m},{lat},{lon});"
            ");out geom;"
        )
        r = requests.post("https://overpass-api.de/api/interpreter", data=q, timeout=60)
        if r.status_code != 200:
            return None
        data = r.json()
        # 集計: 延長[m] と 件数
        def length_of_geom(geom: List[Dict]) -> float:
            total = 0.0
            for i in range(len(geom) - 1):
                lat1, lon1 = geom[i]["lat"], geom[i]["lon"]
                lat2, lon2 = geom[i+1]["lat"], geom[i+1]["lon"]
                dy = (lat2 - lat1) * 111_000.0
                dx = (lon2 - lon1) * 111_000.0
                total += math.hypot(dx, dy)
            return total
        stat = {
            "highway_len_m": 0.0,
            "waterway_len_m": 0.0,
            "buildings": 0,
            "landuse_tags": {},  # {'forest': count, 'residential': count, ...}
            "natural_tags": {},  # {'wood':count,'scrub':...}
            "raw": data,
        }
        for el in data.get("elements", []):
            tags = el.get("tags", {}) or {}
            geom = el.get("geometry", []) or []
            if not tags:
                continue
            if "highway" in tags:
                stat["highway_len_m"] += length_of_geom(geom)
            if "waterway" in tags:
                stat["waterway_len_m"] += length_of_geom(geom)
            if "building" in tags:
                stat["buildings"] += 1
            if "landuse" in tags:
                key = tags.get("landuse")
                stat["landuse_tags"][key] = stat["landuse_tags"].get(key, 0) + 1
            if "natural" in tags:
                key = tags.get("natural")
                stat["natural_tags"][key] = stat["natural_tags"].get(key, 0) + 1
        return stat
    except Exception:
        return None

# ---- 半円（風向±90°） ----

def sector_latlon(lat: float, lon: float, radius_m: float, wind_dir_deg: float, steps: int = 60) -> List[Tuple[float, float]]:
    coords = [(lat, lon)]
    start, end = wind_dir_deg - 90.0, wind_dir_deg + 90.0
    for i in range(steps + 1):
        ang = math.radians(start + (end - start) * i / steps)
        north_m = radius_m * math.cos(ang)
        east_m = radius_m * math.sin(ang)
        dlat = north_m / 111_000.0
        dlon = east_m / 111_000.0  # 簡易換算
        coords.append((lat + dlat, lon + dlon))
    return coords


def sector_lonlat(lat: float, lon: float, radius_m: float, wind_dir_deg: float, steps: int = 60) -> List[Tuple[float, float]]:
    ll = sector_latlon(lat, lon, radius_m, wind_dir_deg, steps)
    ring = [(p[1], p[0]) for p in ll]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring

# ---------------------------------
# Gemini 呼び出し（#1 数値 / #2 要約）
# ---------------------------------

def gemini_generate_json(prompt: str) -> Tuple[Optional[dict], Optional[dict], Optional[str]]:
    if not API_KEY:
        return None, None, None
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
        headers = {"Content-Type": "application/json"}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        raw = r.json() if r.content else None
        if r.status_code != 200 or not raw:
            return None, raw, None
        text = ""
        cands = raw.get("candidates") or []
        if cands:
            content = cands[0].get("content") or {}
            parts = content.get("parts") or []
            for p in parts:
                text += p.get("text", "")
            if not text:
                text = cands[0].get("output", "")
        parsed = extract_json(text) or (json.loads(text) if text.strip().startswith("{") else None)
        return parsed, raw, text
    except Exception as e:
        st.error(f"Gemini数値推定中に例外: {e}")
        return None, None, None


def gemini_summarize(json_obj: dict) -> Tuple[Optional[str], Optional[dict], Optional[str]]:
    if not API_KEY:
        return None, None, None
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
        headers = {"Content-Type": "application/json"}
        p = (
            "以下は火災拡大シミュレーションの推定結果JSONです。\n"
            "この数値の意味が直感的に伝わる、日本語の短い説明文を出力してください。\n"
            "専門用語は避け、半径(m)、面積(m²)、必要放水量(トン)の意味が分かるように。\n"
            "JSONは次です:\n" + json.dumps(json_obj, ensure_ascii=False)
        )
        payload = {"contents": [{"parts": [{"text": p}]}]}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        raw = r.json() if r.content else None
        if r.status_code != 200 or not raw:
            return None, raw, None
        text = ""
        cands = raw.get("candidates") or []
        if cands:
            content = cands[0].get("content") or {}
            parts = content.get("parts") or []
            for pt in parts:
                text += pt.get("text", "")
            if not text:
                text = cands[0].get("output", "")
        return (text.strip() or None), raw, text
    except Exception as e:
        st.error(f"Gemini要約中に例外: {e}")
        return None, None, None

# ---------------------------------
# 簡易物理・フィーチャー補正（フォールバック/併用）
# ---------------------------------

def feature_corrected_fallback(wx: Dict[str, float], fuel_label: str, hours: float,
                                slope_pct: Optional[float], osm: Optional[Dict]) -> dict:
    # ベース拡大速度 v0 [m/s]
    v0 = {"森林（高燃料）": 0.30, "草地（中燃料）": 0.60, "都市部（低燃料）": 0.20}.get(fuel_label, 0.40)
    wind = float(wx.get("windspeed") or 0.0)
    rh = float(wx.get("humidity") or 60.0)
    precip = float(wx.get("precipitation") or 0.0)

    # 風の寄与（線形）
    wind_factor = 1.0 + 0.12 * wind

    # 湿度/降水で抑制
    humidity_factor = max(0.6, 1.0 - 0.003 * max(0.0, rh - 30.0))
    precip_factor = max(0.5, 1.0 / (1.0 + precip))

    # 傾斜（上りで促進・下り/平坦で1.0近辺とする簡易係数）
    slope_factor = 1.0
    if slope_pct is not None:
        slope_factor = min(2.0, 1.0 + 0.03 * slope_pct)  # 斜度10%で+30% など

    # OSM: 道路/水域を障害物として抑制、建物密度で風を弱める
    barrier_factor = 1.0
    urban_factor = 1.0
    if osm:
        R = 1.0 / 1000.0  # m→km 係数
        highway_km = (osm.get("highway_len_m") or 0.0) * R
        water_km = (osm.get("waterway_len_m") or 0.0) * R
        buildings = float(osm.get("buildings") or 0)
        # 面積（探索円）
        # radius_m は不明なので highway_km 等は密度とみなし、単純係数化
        barrier_index = 0.15 * highway_km + 0.35 * water_km
        barrier_factor = max(0.5, 1.0 / (1.0 + barrier_index))
        urban_factor = max(0.6, 1.0 - min(0.6, buildings / 400.0))  # 400棟で最大40%低減

    v_eff = v0 * wind_factor * humidity_factor * precip_factor * slope_factor * urban_factor * barrier_factor
    radius_m = max(30.0, v_eff * hours * 3600.0)
    area_sqm = 0.5 * math.pi * radius_m * radius_m
    water_tons = area_sqm * 0.01
    return {"radius_m": radius_m, "area_sqm": area_sqm, "water_volume_tons": water_tons}

# ---------------------------------
# セッション状態
# ---------------------------------
if "points" not in st.session_state:
    st.session_state.points: List[Tuple[float, float]] = []
if "weather" not in st.session_state:
    st.session_state.weather: Optional[Dict[str, float]] = None
if "results" not in st.session_state:
    st.session_state.results = {"hour": None, "day": None, "week": None, "month": None}
if "overlays" not in st.session_state:
    st.session_state.overlays = {"hour": None, "day": None, "week": None, "month": None}

# ---------------------------------
# サイドバー
# ---------------------------------
st.sidebar.header("発生地点と条件設定")
with st.sidebar.form("point_form"):
    lat_in = st.number_input("緯度", value=34.257586, format="%.6f")
    lon_in = st.number_input("経度", value=133.204356, format="%.6f")
    add = st.form_submit_button("発生地点を追加")
    if add:
        st.session_state.points.append((lat_in, lon_in))
        st.sidebar.success(f"地点 ({lat_in:.6f},{lon_in:.6f}) を追加")

if st.sidebar.button("登録地点を消去"):
    st.session_state.points = []
    st.session_state.weather = None
    st.session_state.results = {"hour": None, "day": None, "week": None, "month": None}
    st.session_state.overlays = {"hour": None, "day": None, "week": None, "month": None}
    st.sidebar.info("削除しました")

fuel_options = ["森林（高燃料）", "草地（中燃料）", "都市部（低燃料）"]
fuel_type = st.sidebar.selectbox("燃料特性", fuel_options, index=0)

st.sidebar.subheader("精度向上オプション（Net/地図フィーチャ）")
use_dem = st.sidebar.checkbox("標高・傾斜（Open‑Elevation）を考慮", value=True)
use_osm = st.sidebar.checkbox("OSM（道路/水域/建物/土地利用）を考慮", value=True)
osm_radius_m = st.sidebar.slider("OSM 探索半径 (m)", min_value=500, max_value=3000, value=1500, step=100)
show_osm_overlay = st.sidebar.checkbox("OSMオーバーレイを表示（2D/3D）", value=True)

if st.sidebar.button("⛅ 気象データ取得（代表地点）"):
    if st.session_state.points:
        lat0, lon0 = st.session_state.points[0]
        st.session_state.weather = get_weather(lat0, lon0)
        if st.session_state.weather:
            st.sidebar.success("気象データを取得しました")
        else:
            st.sidebar.error("気象データ取得に失敗しました")
    else:
        st.sidebar.warning("先に発生地点を追加してください")

# ---------------------------------
# メイン: タイトル & 表示モード
# ---------------------------------
st.title("火災拡大シミュレーション（Gemini要約＋フィーチャ補正）")
mode = st.radio("表示モード", ["2D 地図", "3D 表示"], horizontal=True)

# ベース地図（常時表示）
center = st.session_state.points[0] if st.session_state.points else (34.257586, 133.204356)

if mode == "2D 地図":
    base_map = folium.Map(location=[center[0], center[1]], zoom_start=13, control_scale=True)
    for i, (plat, plon) in enumerate(st.session_state.points, start=1):
        folium.Marker([plat, plon], icon=folium.Icon(color="red"), tooltip=f"発火点 {i}: {plat:.5f},{plon:.5f}").add_to(base_map)
    # 既存オーバーレイ（扇形）を再描画
    for key in ("hour", "day", "week", "month"):
        o = st.session_state.overlays.get(key)
        if o:
            poly = sector_latlon(o["lat"], o["lon"], o["radius_m"], o["wind_dir"]) 
            folium.Polygon(poly, color="red", weight=2, fill=True, fill_opacity=0.4,
                           tooltip=f"[{key}] 半径{o['radius_m']:.0f}m 面積{o['area_sqm']:.0f}㎡").add_to(base_map)
    st_folium(base_map, height=520, width=None)
else:
    # 3D: 既存オーバーレイをpydeckで表示
    layers = []
    # 発火点
    if st.session_state.points:
        points_data = [{"lon": p[1], "lat": p[0]} for p in st.session_state.points]
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=points_data,
            get_position='[lon, lat]',
            get_radius=80,
            get_fill_color='[255,0,0]'
        ))
    # 扇形
    polys = []
    for key in ("hour", "day", "week", "month"):
        o = st.session_state.overlays.get(key)
        if o:
            ring = sector_lonlat(o["lat"], o["lon"], o["radius_m"], o["wind_dir"])  # [[lon,lat],...]
            polys.append({"polygon": ring, "key": key})
    if polys:
        layers.append(pdk.Layer(
            "PolygonLayer",
            data=polys,
            get_polygon='polygon',
            get_fill_color='[255, 64, 64, 100]',
            get_line_color='[220, 0, 0]',
            get_line_width_min_pixels=1,
            stroked=True,
            filled=True,
            extruded=False,
            pickable=True
        ))
    # OSMオーバーレイ（3D）: PathLayer（必要時）
    if show_osm_overlay and st.session_state.overlays.get("last_osm_paths"):
        layers.append(pdk.Layer(
            "PathLayer",
            data=st.session_state.overlays["last_osm_paths"],
            get_path='path',
            get_width=2,
            get_color='[80,80,220]'
        ))
    view_state = pdk.ViewState(latitude=center[0], longitude=center[1], zoom=12, pitch=45)
    deck = pdk.Deck(layers=layers, initial_view_state=view_state)
    st.pydeck_chart(deck, use_container_width=True)

# ---------------------------------
# タブ: 時間 / 日 / 週 / 月
# ---------------------------------

def render_result_block(pred: dict, summary: Optional[str], extras: Dict):
    r = float(pred.get("radius_m") or 0)
    a = float(pred.get("area_sqm") or 0)
    w = float(pred.get("water_volume_tons") or 0)
    st.subheader("シミュレーション結果")
    c1, c2, c3 = st.columns(3)
    c1.metric("半径 (m)", f"{r:,.0f}")
    c2.metric("面積 (m²)", f"{a:,.0f}")
    c3.metric("必要放水量 (トン)", f"{w:,.1f}")
    if summary:
        st.subheader("Geminiによる要約")
        st.write(summary)
    with st.expander("フィーチャー入力（参考）"):
        st.json(extras)


def run_sim(duration_hours: float, key: str, container):
    with container:
        if not st.session_state.points:
            st.warning("発生地点を追加してください")
            return
        if not st.session_state.weather:
            st.warning("気象データを取得してください")
            return
        lat0, lon0 = st.session_state.points[0]
        wx = st.session_state.weather

        slope_pct = get_slope_percent(lat0, lon0) if use_dem else None
        osm_stat = get_osm_features(lat0, lon0, radius_m=osm_radius_m) if use_osm else None

        # OSMオーバーレイ生成（2D/3D共通で使う）
        if show_osm_overlay and osm_stat:
            paths = []
            for el in osm_stat.get("raw", {}).get("elements", []):
                tg = el.get("tags", {}) or {}
                geom = el.get("geometry", []) or []
                if not geom:
                    continue
                if ("highway" in tg) or ("waterway" in tg):
                    path = [[pt["lon"], pt["lat"]] for pt in geom]
                    paths.append({"path": path})
            st.session_state.overlays["last_osm_paths"] = paths
        else:
            st.session_state.overlays["last_osm_paths"] = []

        # フィーチャー要約（Gemini #1 に渡す）
        feat_lines = []
        if slope_pct is not None:
            feat_lines.append(f"傾斜 {slope_pct:.1f}%")
        if osm_stat:
            feat_lines.append(f"道路延長 {osm_stat.get('highway_len_m',0):.0f}m, 水域延長 {osm_stat.get('waterway_len_m',0):.0f}m, 建物 {osm_stat.get('buildings',0)} 棟")
            if osm_stat.get("landuse_tags"):
                top_landuse = sorted(osm_stat["landuse_tags"].items(), key=lambda x: -x[1])[0][0]
                feat_lines.append(f"主要土地利用: {top_landuse}")
            if osm_stat.get("natural_tags"):
                top_nat = sorted(osm_stat["natural_tags"].items(), key=lambda x: -x[1])[0][0]
                feat_lines.append(f"自然被覆: {top_nat}")
        feat_text = "; ".join(feat_lines) if feat_lines else "（追加フィーチャなし）"

        # 数値推定プロンプト（JSONのみ; ```禁止）
        p = (
            "あなたは火災拡大シミュレーションの専門家です。\n"
            "次の条件に基づき、火災の拡大を純粋なJSONのみで推定してください。\n"
            f"- 発生地点: 緯度 {lat0}, 経度 {lon0}\n"
            f"- 気象: 風速 {wx.get('windspeed','不明')} m/s, 風向 {wx.get('winddirection','不明')} 度 (北=0, 東=90, 南=180, 西=270)\n"
            f"        温度 {wx.get('temperature','不明')} °C, 湿度 {wx.get('humidity','不明')} %, 降水 {wx.get('precipitation','不明')} mm/h\n"
            f"- シミュレーション時間: {duration_hours} 時間\n"
            f"- 燃料特性: {fuel_type}\n"
            "- 地形/植生/施設（外部データ要約）: \n"
            f"  {feat_text}\n"
            '出力: 他の文字を一切含まず、以下のJSONのみを返すこと。\n{"radius_m": <float>, "area_sqm": <float>, "water_volume_tons": <float>}\n'
        )

        pred, raw_pred, raw_text = gemini_generate_json(p)
        if pred is None:
            # フィーチャー補正付きフォールバック
            pred = feature_corrected_fallback(wx, fuel_type, duration_hours, slope_pct, osm_stat)
            if raw_pred:
                with st.expander("Gemini #1 生JSON（参考）"):
                    st.json(raw_pred)

        # 値の補正
        r = float(pred.get("radius_m") or 0.0)
        a = float(pred.get("area_sqm") or 0.0)
        w = float(pred.get("water_volume_tons") or 0.0)
        if r > 0 and a <= 0:
            a = 0.5 * math.pi * r * r
        if a > 0 and w <= 0:
            w = a * 0.01
        pred = {"radius_m": r, "area_sqm": a, "water_volume_tons": w}

        # 要約
        summary, raw_sum, _ = gemini_summarize(pred)

        # 永続化（結果 + オーバーレイ）
        st.session_state.results[key] = {"pred": pred, "summary": summary, "features": {
            "slope_percent": slope_pct,
            "osm_stats": {k: v for k, v in (osm_stat or {}).items() if k != "raw"},
            "options": {"use_dem": use_dem, "use_osm": use_osm, "osm_radius_m": osm_radius_m}
        }}
        wind_dir = float(wx.get("winddirection") or 0.0)
        st.session_state.overlays[key] = {"lat": lat0, "lon": lon0, "radius_m": r, "area_sqm": a, "wind_dir": wind_dir}

        # 表示
        render_result_block(pred, summary, st.session_state.results[key]["features"])
        with st.expander("Gemini #1 生JSON応答（検証用）"):
            if raw_pred is not None:
                st.json(raw_pred)
            else:
                st.caption("応答なし / キー未設定")
        with st.expander("Gemini #2 生JSON応答（検証用）"):
            if raw_sum is not None:
                st.json(raw_sum)
            else:
                st.caption("応答なし / キー未設定")

# ---- タブUI ----
tab_hour, tab_day, tab_week, tab_month = st.tabs(["時間", "日", "週", "月"])

with tab_hour:
    hours = st.slider("時間（1〜24h）", min_value=1, max_value=24, value=6, step=1, key="slider_hour")
    if st.button("▶ シミュレーション実行（時間）", key="btn_hour"):
        run_sim(duration_hours=float(hours), key="hour", container=tab_hour)
    saved = st.session_state.results.get("hour")
    if saved:
        render_result_block(saved["pred"], saved.get("summary"), saved.get("features", {}))

with tab_day:
    days = st.slider("日数（1〜30）", min_value=1, max_value=30, value=3, step=1, key="slider_day")
    if st.button("▶ シミュレーション実行（日）", key="btn_day"):
        run_sim(duration_hours=float(days) * 24.0, key="day", container=tab_day)
    saved = st.session_state.results.get("day")
    if saved:
        render_result_block(saved["pred"], saved.get("summary"), saved.get("features", {}))

with tab_week:
    weeks = st.slider("週間（1〜52）", min_value=1, max_value=52, value=1, step=1, key="slider_week")
    if st.button("▶ シミュレーション実行（週）", key="btn_week"):
        run_sim(duration_hours=float(weeks) * 7.0 * 24.0, key="week", container=tab_week)
    saved = st.session_state.results.get("week")
    if saved:
        render_result_block(saved["pred"], saved.get("summary"), saved.get("features", {}))

with tab_month:
    months = st.slider("月数（1〜12）", min_value=1, max_value=12, value=1, step=1, key="slider_month")
    if st.button("▶ シミュレーション実行（月）", key="btn_month"):
        run_sim(duration_hours=float(months) * 30.0 * 24.0, key="month", container=tab_month)
    saved = st.session_state.results.get("month")
    if saved:
        render_result_block(saved["pred"], saved.get("summary"), saved.get("features", {}))

