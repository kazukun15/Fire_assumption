# app.py — 火災延焼範囲予測くん（DEM + OSM遮蔽物 + 異方性延焼）
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
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
import osmnx as ox
from pyproj import CRS

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
# ユーティリティ
# ============================
def extract_json(text: str) -> Optional[dict]:
    """Gemini応答が```json ...```でも素JSONでも抽出する"""
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

def utm_crs_for(lat: float, lon: float) -> CRS:
    """地点に適したUTMのCRSを返す（EPSG:326/327）"""
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)

def wind_deg_downwind(wind_from_deg: float) -> float:
    """風向（FROM、0=北）が与えられたときの火の進行主方向（TO）"""
    return (wind_from_deg + 180.0) % 360.0

def angle_diff_deg(a: float, b: float) -> float:
    return ((a - b + 180) % 360) - 180

# ============================
# Open-Meteo（気象）
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
# DEM（標高）取得 + 傾斜/方位の推定
# ============================
@st.cache_data(show_spinner=False)
def fetch_elevation_opentopo(lat: float, lon: float) -> Optional[float]:
    try:
        r = requests.get(
            f"https://api.opentopodata.org/v1/srtm90m?locations={lat},{lon}",
            timeout=12,
        )
        j = r.json()
        if j and j.get("results"):
            val = j["results"][0].get("elevation")
            return float(val) if val is not None else None
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False)
def fetch_elevation_openelev(lat: float, lon: float) -> Optional[float]:
    try:
        r = requests.get(
            f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}",
            timeout=12,
        )
        j = r.json()
        if j and j.get("results"):
            val = j["results"][0].get("elevation")
            return float(val) if val is not None else None
    except Exception:
        pass
    return None

def fetch_elevation(lat: float, lon: float) -> Optional[float]:
    z = fetch_elevation_opentopo(lat, lon)
    if z is not None:
        return z
    return fetch_elevation_openelev(lat, lon)

@st.cache_data(show_spinner=False)
def estimate_slope_aspect(lat: float, lon: float, spacing_m: float = 30.0) -> Optional[Dict[str, float]]:
    """
    中心を含む3x3のサンプルを取り、平面 z = ax + by + c を最小二乗で当て、
    slope[deg] と aspect[deg]（aspectは「下り方向」を0=北,90=東…）を推定。
    """
    # ローカル近似：緯度経度→m
    def ll_to_xy(lat0, lon0, la, lo):
        x = (lo - lon0) * math.cos(math.radians(lat0)) * 111320.0
        y = (la - lat0) * 110540.0
        return x, y

    samples = []
    lat0, lon0 = lat, lon
    # オフセット（-1,0,1）×（-1,0,1）
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            la = lat0 + (dy * spacing_m) / 110540.0
            lo = lon0 + (dx * spacing_m) / (111320.0 * math.cos(math.radians(lat0)))
            z = fetch_elevation(la, lo)
            if z is None:
                return None
            x, y = ll_to_xy(lat0, lon0, la, lo)
            samples.append((x, y, z))

    # 平面当てはめ（最小二乗）
    # 行列 A*[a,b,c]^T = z
    A = []
    Z = []
    for x, y, z in samples:
        A.append([x, y, 1.0])
        Z.append(z)
    # 正規方程式で解く
    try:
        import numpy as np
        A = np.array(A, dtype=float)
        Z = np.array(Z, dtype=float)
        # (A^T A) inv A^T Z
        coeff, *_ = np.linalg.lstsq(A, Z, rcond=None)  # a,b,c
        a, b = coeff[0], coeff[1]  # 勾配ベクトル
        # 勾配の大きさ→傾斜：atan(sqrt(a^2+b^2))  [rad]を度へ
        grad = math.hypot(a, b)
        slope_deg = math.degrees(math.atan(grad))
        # aspect（下り方向）: 勾配ベクトルの向き（x→東, y→北）
        # 下り方向ベクトル = (a, b) の向き。方位角0=北,90=東に合わせる
        # atan2(x成分, y成分)で0=北系にする
        aspect_rad = math.atan2(a, b)  # 通常のatan2(y, x)と入れ替え
        aspect_deg = (math.degrees(aspect_rad) + 360.0) % 360.0
        return {"elevation_m": float(fetch_elevation(lat, lon) or 0.0),
                "slope_deg": float(slope_deg),
                "aspect_deg": float(aspect_deg)}  # 下り方向
    except Exception:
        return None

# ============================
# OSM建物（遮蔽物）
# ============================
@st.cache_data(show_spinner=False)
def get_osm_buildings(lat: float, lon: float, dist: int = 1200):
    try:
        gdf = ox.geometries_from_point((lat, lon), tags={"building": True}, dist=dist)
        # Polygon/Multipolygonのみ
        if gdf is not None and not gdf.empty:
            gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        return gdf
    except Exception:
        return None

def to_local(gdf: gpd.GeoDataFrame, lat: float, lon: float) -> gpd.GeoDataFrame:
    crs_wgs84 = gdf.crs or "EPSG:4326"
    local = utm_crs_for(lat, lon)
    return gdf.to_crs(local)

def point_gdf(lat: float, lon: float, crs="EPSG:4326") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"id":[0]}, geometry=[Point(lon, lat)], crs=crs)

# ============================
# 方向別レイ：DEMと風・遮蔽物で異方性距離
# ============================
def wind_factor(theta_deg: float, spread_dir_deg: float, windspeed: float) -> float:
    """風整合。風下=最大、風上=最小"""
    d = math.radians(angle_diff_deg(theta_deg, spread_dir_deg))
    return max(0.4, 1.0 + 0.12 * windspeed * math.cos(d))

def slope_factor(theta_deg: float, downhill_deg: float, slope_deg: float) -> float:
    """
    傾斜整合。下り方向は抑制、上り方向は促進。
    aspect=下り方向なので、上り方向= (downhill+180)。
    """
    uphill = (downhill_deg + 180.0) % 360.0
    d = math.radians(angle_diff_deg(theta_deg, uphill))
    # 傾斜1度あたり3%程度で強弱（上りで+、下りで-）
    return max(0.5, min(2.0, 1.0 + 0.03 * slope_deg * math.cos(d)))

def build_anisotropic_shape(lat: float, lon: float,
                            base_radius: float,
                            wind_from_deg: float, windspeed: float,
                            dem_info: Optional[Dict[str, float]],
                            buildings: Optional[gpd.GeoDataFrame],
                            angle_steps: int = 240,
                            step_m: float = 30.0,
                            building_buffer_m: float = 2.0) -> List[Tuple[float, float]]:
    """
    異方性（風＋傾斜）× 遮蔽物クリップで境界を求める。
    - 各角度thetaに対し、基準半径×(風係数×傾斜係数)/平均係数 で目標距離を計算。
    - 0..距離までstep_mで進み、建物に当たったら即停止（クリップ）。
    """
    # 角度ごと係数を先に計算し、平均でスケールして面積バランスを取る
    spread_dir = wind_deg_downwind(wind_from_deg)
    downhill = (dem_info or {}).get("aspect_deg", 0.0)  # 下り方向
    slope_deg = (dem_info or {}).get("slope_deg", 0.0)

    thetas = [i * 360.0 / angle_steps for i in range(angle_steps)]
    raw_factors = []
    for th in thetas:
        f = wind_factor(th, spread_dir, windspeed) * slope_factor(th, downhill, slope_deg)
        raw_factors.append(f)
    avg_f = sum(raw_factors) / len(raw_factors) if raw_factors else 1.0
    scale = 1.0 / max(1e-6, avg_f)  # 平均が1になるよう調整

    # 建物ユニオン（ローカル座標）
    building_union_local = None
    local_crs = utm_crs_for(lat, lon)
    origin_pt = point_gdf(lat, lon).to_crs(local_crs).iloc[0].geometry

    if buildings is not None and not buildings.empty:
        try:
            b_local = to_local(buildings, lat, lon).buffer(building_buffer_m)
            building_union_local = unary_union(b_local.geometry)
        except Exception:
            building_union_local = None

    # レイ進行
    boundary_pts: List[Tuple[float, float]] = []
    for th, f in zip(thetas, raw_factors):
        target = base_radius * f * scale
        # 目標距離まで step_m で前進しつつ衝突チェック
        steps = max(1, int(target // step_m))
        hit_point = None
        for s in range(1, steps + 1):
            # UTM平面上で線分を作って交差判定
            dist = min(target, s * step_m)
            rad = math.radians(th)
            dx = dist * math.cos(rad)
            dy = dist * math.sin(rad)
            p = LineString([origin_pt, (origin_pt.x + dx, origin_pt.y + dy)])
            if building_union_local is not None and p.intersects(building_union_local):
                # 交差点までの距離でクリップ
                inter = p.intersection(building_union_local)
                # 交差が複数でも最も近い点を採用
                try:
                    if inter.geom_type == "MultiPoint":
                        pts = list(inter.geoms)
                        inter_pt = min(pts, key=lambda q: origin_pt.distance(q))
                    elif inter.geom_type == "Point":
                        inter_pt = inter
                    else:
                        # ライン等の場合は始点から最近点
                        inter_pt = inter.representative_point()
                    hit_point = inter_pt
                except Exception:
                    hit_point = None
                break

        if hit_point is None:
            # 衝突なし→目標点
            end_x = origin_pt.x + target * math.cos(math.radians(th))
            end_y = origin_pt.y + target * math.sin(math.radians(th))
        else:
            end_x = hit_point.x
            end_y = hit_point.y

        # 地理座標に戻す
        # 逆変換は小GeoDataFrameを作ってto_crsでOK
        gdf_tmp = gpd.GeoDataFrame(geometry=[Point(end_x, end_y)], crs=local_crs).to_crs("EPSG:4326")
        latlon = (gdf_tmp.geometry.iloc[0].y, gdf_tmp.geometry.iloc[0].x)
        boundary_pts.append(latlon)

    # クローズ
    if boundary_pts and boundary_pts[0] != boundary_pts[-1]:
        boundary_pts.append(boundary_pts[0])
    return boundary_pts

# ============================
# Gemini API（堅牢パース）
# ============================
def gemini_generate(prompt: str) -> Tuple[Optional[dict], Optional[dict]]:
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
        return parsed, raw
    except Exception:
        return None, None

# ============================
# フォールバック推定（必ず動く）
# ============================
def fallback_predict(wx: Dict[str, float], fuel_label: str, hours: float, slope_deg: float, wind_dir: float) -> dict:
    # 基本速度（m/s）
    v0 = {"森林": 0.30, "草地": 0.60, "都市部": 0.20}.get(fuel_label, 0.40)
    wind = float(wx.get("windspeed") or 0)
    rh = float(wx.get("humidity") or 60)
    precip = float(wx.get("precipitation") or 0)

    wind_factor = 1.0 + 0.12 * wind
    humidity_factor = max(0.6, 1.0 - 0.003 * max(0.0, rh - 30.0))
    precip_factor = max(0.5, 1.0 / (1.0 + precip))
    # 傾斜の全体係数（簡略）：平均的な上り寄与を加味
    slope_global = 1.0 + 0.01 * slope_deg
    v_eff = v0 * wind_factor * humidity_factor * precip_factor * slope_global

    radius_m = max(40.0, v_eff * hours * 3600.0)
    area_sqm = math.pi * radius_m * radius_m * 0.8  # 異方性でやや広めに（経験的係数）
    water_tons = area_sqm * 0.01
    return {"radius_m": radius_m, "area_sqm": area_sqm, "water_volume_tons": water_tons}

# ============================
# セッション状態
# ============================
if "points" not in st.session_state:
    st.session_state.points: List[Tuple[float, float]] = []
if "weather" not in st.session_state:
    st.session_state.weather: Optional[Dict[str, float]] = None
if "buildings" not in st.session_state:
    st.session_state.buildings = None
if "dem" not in st.session_state:
    st.session_state.dem: Optional[Dict[str, float]] = None
if "last_pred" not in st.session_state:
    st.session_state.last_pred = None
if "last_shape2d" not in st.session_state:
    st.session_state.last_shape2d = None
if "last_shape3d" not in st.session_state:
    st.session_state.last_shape3d = None

# ============================
# UI — サイドバー
# ============================
st.sidebar.header("発生地点と条件設定")
with st.sidebar.form("point_form"):
    st.caption("Googleマップ形式をそのまま貼り付け: 34.246099951898415, 133.20578422112848")
    coord_text = st.text_input("緯度,経度（カンマ区切り）", "34.257586,133.204356")
    add = st.form_submit_button("発生地点を追加（テキスト）")
    if add:
        try:
            lat_in, lon_in = [float(x.strip().strip("()")) for x in coord_text.split(",")]
            st.session_state.points.append((lat_in, lon_in))
            st.sidebar.success(f"追加: ({lat_in:.6f}, {lon_in:.6f})")
        except Exception:
            st.sidebar.error("形式が不正です")

if st.sidebar.button("登録地点を全消去"):
    st.session_state.points = []
    st.session_state.weather = None
    st.session_state.buildings = None
    st.session_state.dem = None
    st.session_state.last_pred = None
    st.session_state.last_shape2d = None
    st.session_state.last_shape3d = None
    st.sidebar.info("削除しました")

fuel_opts = {"森林（高燃料)": "森林", "草地（中燃料)": "草地", "都市部（低燃料)": "都市部"}
fuel_type = fuel_opts[st.sidebar.selectbox("燃料特性", list(fuel_opts.keys()))]

# ============================
# タイトル
# ============================
st.title("火災延焼範囲予測くん")

# ============================
# 地図クリックで発火点設定（任意）
# ============================
with st.expander("🧭 地図で発火点を設定（クリック）", expanded=False):
    init = st.session_state.points[-1] if st.session_state.points else (34.257586, 133.204356)
    m_pick = folium.Map(location=[init[0], init[1]], zoom_start=13, tiles="OpenStreetMap")
    for pt in st.session_state.points:
        folium.Marker(pt, icon=folium.Icon(color="red")).add_to(m_pick)
    ret = st_folium(m_pick, width=720, height=420, key="pickermap")
    if ret and ret.get("last_clicked"):
        lat = ret["last_clicked"]["lat"]
        lon = ret["last_clicked"]["lng"]
        st.session_state.points.append((lat, lon))
        st.success(f"クリック追加: ({lat:.6f},{lon:.6f})")

# ============================
# データ取得（気象+DEM+OSM）
# ============================
colD = st.columns(3)
if colD[0].button("🌤 気象データ取得"):
    if st.session_state.points:
        lat0, lon0 = st.session_state.points[0]
        st.session_state.weather = get_weather(lat0, lon0)
        if st.session_state.weather:
            st.success("気象を取得しました")
        else:
            st.error("気象取得に失敗")

if colD[1].button("⛰ DEM（標高・傾斜・方位）取得"):
    if st.session_state.points:
        lat0, lon0 = st.session_state.points[0]
        dem = estimate_slope_aspect(lat0, lon0, spacing_m=30.0)
        if dem:
            st.session_state.dem = dem
            st.success(f"DEM取得: 標高{dem['elevation_m']:.1f}m / 傾斜{dem['slope_deg']:.1f}° / 方位(下り){dem['aspect_deg']:.0f}°")
        else:
            st.error("DEM取得に失敗")
    else:
        st.warning("発火点を追加してください")

if colD[2].button("🏢 OSM建物（遮蔽物）取得"):
    if st.session_state.points:
        lat0, lon0 = st.session_state.points[0]
        st.session_state.buildings = get_osm_buildings(lat0, lon0, dist=1200)
        if st.session_state.buildings is not None:
            st.success("建物データ取得")
        else:
            st.error("建物取得に失敗")
    else:
        st.warning("発火点を追加してください")

# ============================
# 表示モード・タブ
# ============================
mode = st.radio("表示モード", ["2D 地図", "3D 表示"], horizontal=True)
tabs = st.tabs(["時間", "日", "週", "月"])

# ============================
# シミュレーション
# ============================
def run_sim(total_hours: float):
    if not st.session_state.points:
        st.warning("発火地点を追加してください")
        return
    if not st.session_state.weather:
        st.warning("気象データを取得してください")
        return
    if not st.session_state.dem:
        st.warning("DEMを取得してください（標高・傾斜・方位）")
        return

    lat0, lon0 = st.session_state.points[0]
    wx = st.session_state.weather
    dem = st.session_state.dem
    buildings = st.session_state.buildings

    wind_from = float(wx.get("winddirection") or 0.0)
    windspeed = float(wx.get("windspeed") or 0.0)
    slope_deg = float(dem.get("slope_deg") or 0.0)
    aspect_deg = float(dem.get("aspect_deg") or 0.0)

    # --- Gemini（精度強化プロンプト：DEM/OSM情報込み） ---
    pred = None
    raw = None
    if API_KEY:
        env_text = (
            f"地形: 標高{dem.get('elevation_m', '不明')}m, 傾斜{dem.get('slope_deg', '不明')}°, 下り方向{dem.get('aspect_deg', '不明')}°。\n"
            f"遮蔽物: OSM建物 {'あり' if (buildings is not None and not buildings.empty) else '未取得/少'}。\n"
            "前提: 上り斜面は延焼を促進、下り斜面は抑制。建物は燃料連続性を断ち、延焼を弱める/止める。"
        )
        prompt = (
            "あなたは火災挙動・地理の専門家です。以下の条件から延焼半径・面積・必要放水量を推定し、"
            "純粋なJSONのみを返してください。\n"
            f"- 発火点: 緯度{lat0}, 経度{lon0}\n"
            f"- 気象: 風速{windspeed}m/s, 風向{wind_from}°, 温度{wx.get('temperature','不明')}℃, 湿度{wx.get('humidity','不明')}%, 降水{wx.get('precipitation','不明')}mm/h\n"
            f"- 期間: {total_hours}時間, 燃料: {fuel_type}\n"
            f"- {env_text}\n"
            "出力スキーマ: {\"radius_m\":<float>, \"area_sqm\":<float>, \"water_volume_tons\":<float>}\n"
            "他の文字は出力しないこと。"
        )
        predicted, raw = gemini_generate(prompt)
        if isinstance(predicted, dict) and {"radius_m","area_sqm","water_volume_tons"} <= set(predicted.keys()):
            pred = predicted

    # --- フォールバック（必ず動作） ---
    if pred is None:
        pred = fallback_predict(wx, fuel_type, total_hours, slope_deg, wind_from)
        if raw:
            with st.expander("Gemini生レスポンス（参考）"):
                st.json(raw)

    st.subheader("数値結果")
    c1, c2, c3 = st.columns(3)
    c1.metric("半径 (m)", f"{pred.get('radius_m',0):,.0f}")
    c2.metric("面積 (m²)", f"{pred.get('area_sqm',0):,.0f}")
    c3.metric("必要放水量 (トン)", f"{pred.get('water_volume_tons',0):,.1f}")

    # --- 異方性輪郭の生成（DEM＋風＋建物クリップ） ---
    base_r = float(pred.get("radius_m", 0.0))
    boundary = build_anisotropic_shape(
        lat=lat0, lon=lon0,
        base_radius=base_r,
        wind_from_deg=wind_from, windspeed=windspeed,
        dem_info=dem,
        buildings=buildings,
        angle_steps=240, step_m=30.0, building_buffer_m=2.0
    )

    # 保存（2D/3D）
    st.session_state.last_pred = pred
    st.session_state.last_shape2d = boundary
    st.session_state.last_shape3d = [[lon, lat] for (lat, lon) in boundary]

    # 要約（任意）
    if API_KEY:
        sum_prompt = "次のJSONを専門用語を避けて短く説明: " + json.dumps(pred, ensure_ascii=False)
        summary, _ = gemini_generate(sum_prompt)
        if summary:
            st.subheader("Gemini要約")
            st.write(summary if not isinstance(summary, dict) else json.dumps(summary, ensure_ascii=False))

# ---- タブ操作 ----
with tabs[0]:
    val = st.slider("時間 (1〜24)", 1, 24, 3)
    if st.button("シミュレーション実行（時間）"):
        run_sim(float(val))
with tabs[1]:
    val = st.slider("日数 (1〜30)", 1, 30, 3)
    if st.button("シミュレーション実行（日）"):
        run_sim(float(val) * 24.0)
with tabs[2]:
    val = st.slider("週数 (1〜52)", 1, 52, 1)
    if st.button("シミュレーション実行（週）"):
        run_sim(float(val) * 7.0 * 24.0)
with tabs[3]:
    val = st.slider("月数 (1〜12)", 1, 12, 1)
    if st.button("シミュレーション実行（月）"):
        run_sim(float(val) * 30.0 * 24.0)

# ============================
# 地図表示（常時）
# ============================
center = st.session_state.points[-1] if st.session_state.points else (34.257586, 133.204356)

if mode == "2D 地図":
    m = folium.Map(location=[center[0], center[1]], zoom_start=13, tiles="OpenStreetMap")
    # 発火点
    for pt in st.session_state.points:
        folium.Marker(pt, icon=folium.Icon(color="red"), tooltip=f"発火点 {pt}").add_to(m)
    # 遮蔽物（建物）
    b = st.session_state.buildings
    if b is not None and not b.empty:
        try:
            for geom in b.geometry:
                if geom.geom_type == "Polygon":
                    coords = [(p[1], p[0]) for p in list(geom.exterior.coords)]
                    folium.Polygon(coords, color="blue", fill=True, fill_opacity=0.25, weight=1).add_to(m)
                elif geom.geom_type == "MultiPolygon":
                    for poly in geom.geoms:
                        coords = [(p[1], p[0]) for p in list(poly.exterior.coords)]
                        folium.Polygon(coords, color="blue", fill=True, fill_opacity=0.25, weight=1).add_to(m)
        except Exception:
            pass
    # 延焼輪郭
    if st.session_state.last_shape2d:
        folium.Polygon(st.session_state.last_shape2d, color="red", fill=True, fill_opacity=0.35,
                       tooltip="DEM+OSM考慮の推定延焼輪郭").add_to(m)
    st_folium(m, width=980, height=640, key="main2d")

else:
    # 3D
    layers = []
    # 発火点
    if st.session_state.points:
        pts = [{"lon": p[1], "lat": p[0]} for p in st.session_state.points]
        layers.append(
            pdk.Layer("ScatterplotLayer", data=pts, get_position='[lon, lat]',
                      get_radius=80, get_fill_color='[255,0,0]')
        )
    # 建物押し出し（高さ10m）
    b = st.session_state.buildings
    if b is not None and not b.empty:
        polys = []
        try:
            for geom in b.geometry:
                if geom.geom_type == "Polygon":
                    ring = [[x, y] for (x, y) in list(geom.exterior.coords)]
                    polys.append({"polygon": ring, "elev": 10})
                elif geom.geom_type == "MultiPolygon":
                    for g in geom.geoms:
                        ring = [[x, y] for (x, y) in list(g.exterior.coords)]
                        polys.append({"polygon": ring, "elev": 10})
        except Exception:
            polys = []
        if polys:
            layers.append(
                pdk.Layer(
                    "PolygonLayer",
                    data=polys,
                    get_polygon="polygon",
                    get_fill_color="[100,100,255,140]",
                    get_elevation="elev",
                    extruded=True,
                    stroked=False,
                )
            )
    # 延焼輪郭（押し出し80m）
    if st.session_state.last_shape3d:
        layers.append(
            pdk.Layer(
                "PolygonLayer",
                data=[{"polygon": st.session_state.last_shape3d, "elev": 80}],
                get_polygon="polygon",
                get_fill_color="[240,80,60,150]",
                get_elevation="elev",
                extruded=True,
                stroked=True,
                get_line_color="[255,0,0]"
            )
        )
    view_state = pdk.ViewState(latitude=center[0], longitude=center[1], zoom=12.5, pitch=45)
    deck = pdk.Deck(layers=layers, initial_view_state=view_state, map_style="light")
    st.pydeck_chart(deck, use_container_width=True, key="main3d")
