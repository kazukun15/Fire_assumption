"""
Fire Spread Simulation (v3)
- 常時マップ表示（分析開始前でも）
- 「分析開始」ボタン導入
- 複数の発火点（住所/緯度経度入力 + 地図クリックで追加）
- 気象データ: Open‑Meteo を基本、失敗時は MET Norway (met.no) に自動フォールバック
- 気象取得失敗時は手動設定（一定風向・風速）でシミュレーション可能
- 予測: 風向・風速・降水を反映した簡易楕円モデル（15分ステップ）
- 面積・半径等の時系列出力、CSVエクスポート
- Gemini（任意）による結果要約（st.secrets["gemini"]["api_key"] がある場合のみ）

動作に必要なパッケージ（requirements.txt例）:
streamlit
requests
folium
streamlit-folium
pydeck
shapely>=2.0  # 無ければ自動的に重なり面積は「単純合算」（推定）にフォールバック
google-generativeai  # Gemini要約を使う場合

secrets.toml 例:
[general]
api_key = "<Google API Key (任意: Geocoding/Maps用)>"

[gemini]
api_key = "<Gemini API Key (任意)>"
"""

import os
import math
import json
import urllib.parse
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional, Dict

import requests
import streamlit as st
import folium
from folium import plugins
from streamlit_folium import st_folium
import pydeck as pdk

# shapely は任意
try:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    SHAPELY_OK = True
except Exception:
    SHAPELY_OK = False

# ---------------------------------
# ページ設定
# ---------------------------------
st.set_page_config(page_title="Fire Spread Simulation", layout="wide")

# ---------------------------------
# ユーティリティ（Secrets / API key）
# ---------------------------------

def _get_google_api_key() -> Optional[str]:
    try:
        if "general" in st.secrets and "api_key" in st.secrets["general"]:
            return st.secrets["general"]["api_key"]
    except Exception:
        pass
    try:
        if "api_key" in st.secrets:
            return st.secrets["api_key"]
    except Exception:
        pass
    return os.environ.get("GOOGLE_API_KEY")


def _get_gemini_api_key() -> Optional[str]:
    try:
        if "gemini" in st.secrets and "api_key" in st.secrets["gemini"]:
            return st.secrets["gemini"]["api_key"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")


def _headers_for_osm():
    return {"User-Agent": "fire-spread-sim/1.0 (contact: example@example.com)"}


# ---------------------------------
# 地理/気象 API
# ---------------------------------

@st.cache_data(show_spinner=False)
def geocode_one(text: str) -> Tuple[Optional[float], Optional[float]]:
    """入力文字列を緯度経度に変換。"lat,lon" 形式を優先し、Google→OSMの順に解決。"""
    # 1) 明示的な lat,lon
    try:
        parts = [p.strip() for p in text.split(",")]
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
    except Exception:
        pass

    # 2) Google Geocoding（任意）
    api_key = _get_google_api_key()
    if api_key:
        try:
            addr_enc = urllib.parse.quote(text, safe="")
            url = f"https://maps.googleapis.com/maps/api/geocode/json?address={addr_enc}&key={api_key}"
            r = requests.get(url, timeout=15)
            data = r.json()
            if data.get("status") == "OK" and data.get("results"):
                loc = data["results"][0]["geometry"]["location"]
                return loc["lat"], loc["lng"]
        except Exception:
            pass

    # 3) OSM Nominatim
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": text, "format": "json", "limit": 1}
        r = requests.get(url, params=params, headers=_headers_for_osm(), timeout=15)
        data = r.json()
        if isinstance(data, list) and data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass

    return None, None


@st.cache_data(show_spinner=False)
def fetch_weather_openmeteo(lat: float, lon: float, total_hours: int):
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat:.4f}&longitude={lon:.4f}"
            f"&hourly=winddirection_10m,windspeed_10m,precipitation&windspeed_unit=ms&timezone=UTC&forecast_days=2"
        )
        res = requests.get(url, timeout=30).json()
        if "hourly" not in res or "time" not in res["hourly"]:
            return None
        times = [datetime.fromisoformat(t.replace("Z", "+00:00")) for t in res["hourly"]["time"]]
        wnd  = res["hourly"]["windspeed_10m"]
        wdir = res["hourly"]["winddirection_10m"]
        prcp = res["hourly"]["precipitation"]
        now_utc = datetime.now(timezone.utc)
        now_floor = now_utc.replace(minute=0, second=0, microsecond=0)
        if now_floor < times[0]:
            start_idx = 0
        else:
            try:
                start_idx = times.index(now_floor)
            except ValueError:
                start_idx = next((i for i, t in enumerate(times) if t > now_utc), 0)
        end_idx = min(start_idx + total_hours, len(times) - 1)
        return (
            times[start_idx:end_idx+1],
            wnd[start_idx:end_idx+1],
            wdir[start_idx:end_idx+1],
            prcp[start_idx:end_idx+1],
        )
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def fetch_weather_metno(lat: float, lon: float, total_hours: int):
    """MET Norway (met.no) compact API。User-Agent 必須。"""
    try:
        url = f"https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={lat:.5f}&lon={lon:.5f}"
        headers = {"User-Agent": "fire-spread-sim/1.0 contact: example@example.com"}
        res = requests.get(url, headers=headers, timeout=30).json()
        times_all = []
        windspeed = []
        winddir = []
        precip = []
        for item in res.get("properties", {}).get("timeseries", []):
            t_iso = item.get("time")
            details = item.get("data", {}).get("instant", {}).get("details", {})
            wms = details.get("wind_speed")  # m/s
            wdd = details.get("wind_from_direction")  # degrees
            # 降水は next_1_hours→details→precipitation_amount など（無い場合は0）
            pr = 0.0
            nxt = item.get("data", {}).get("next_1_hours")
            if nxt and "details" in nxt:
                pr = float(nxt["details"].get("precipitation_amount", 0.0))
            if t_iso and (wms is not None) and (wdd is not None):
                times_all.append(datetime.fromisoformat(t_iso.replace("Z", "+00:00")))
                windspeed.append(float(wms))
                winddir.append(float(wdd))
                precip.append(pr)
        if not times_all:
            return None
        now_utc = datetime.now(timezone.utc)
        now_floor = now_utc.replace(minute=0, second=0, microsecond=0)
        if now_floor < times_all[0]:
            start_idx = 0
        else:
            try:
                start_idx = times_all.index(now_floor)
            except ValueError:
                start_idx = next((i for i, t in enumerate(times_all) if t > now_utc), 0)
        end_idx = min(start_idx + total_hours, len(times_all) - 1)
        return (
            times_all[start_idx:end_idx+1],
            windspeed[start_idx:end_idx+1],
            winddir[start_idx:end_idx+1],
            precip[start_idx:end_idx+1],
        )
    except Exception:
        return None


def fetch_weather(lat: float, lon: float, total_hours: int, source: str):
    """選択ソースに従って取得。AUTO は Open‑Meteo→met.no の順に試行。"""
    if source == "Open‑Meteo":
        return fetch_weather_openmeteo(lat, lon, total_hours)
    if source == "MET Norway (met.no)":
        return fetch_weather_metno(lat, lon, total_hours)
    # AUTO
    data = fetch_weather_openmeteo(lat, lon, total_hours)
    if data:
        return data
    return fetch_weather_metno(lat, lon, total_hours)


@st.cache_data(show_spinner=False)
def get_timezone_offset(lat: float, lon: float) -> Tuple[int, str]:
    api_key = _get_google_api_key()
    if api_key:
        try:
            timestamp = int(datetime.now(timezone.utc).timestamp())
            url = (
                f"https://maps.googleapis.com/maps/api/timezone/json?location={lat:.6f},{lon:.6f}&timestamp={timestamp}&key={api_key}"
            )
            r = requests.get(url, timeout=15).json()
            if r.get("status") == "OK":
                return int(r.get("rawOffset", 0)) + int(r.get("dstOffset", 0)), r.get("timeZoneId", "")
        except Exception:
            pass
    # フォールバック（Open‑Meteo）
    try:
        r = requests.get(
            f"https://api.open-meteo.com/v1/timezone?latitude={lat:.6f}&longitude={lon:.6f}",
            timeout=15,
        ).json()
        return int(r.get("utc_offset_seconds", 0)), r.get("timezone", "")
    except Exception:
        return 0, ""


# ---------------------------------
# 延焼シミュレーション（単一点）
# ---------------------------------

def simulate_fire_single(lat: float, lon: float, hours_list, wind_list, dir_list, precip_list,
                         base_speed: float, wind_factor: float):
    """15分刻みでポリゴンを構築。戻り値: (frame_times, polygons, frame_precip)"""
    wind_list = [w * wind_factor for w in wind_list]
    frame_times: List[datetime] = []
    frame_wind: List[float] = []
    frame_dir: List[float] = []
    frame_precip: List[float] = []

    for h in range(len(hours_list) - 1):
        if h == 0:
            frame_times.append(hours_list[h])
            frame_wind.append(wind_list[h])
            frame_dir.append(dir_list[h])
            frame_precip.append(precip_list[h])
        for q in (1, 2, 3):
            frac = q / 4.0
            w_val = wind_list[h] + (wind_list[h + 1] - wind_list[h]) * frac
            p_val = precip_list[h] + (precip_list[h + 1] - precip_list[h]) * frac
            d1, d2 = dir_list[h], dir_list[h + 1]
            delta = ((d2 - d1 + 180) % 360) - 180
            d_val = (d1 + delta * frac) % 360
            frame_times.append(hours_list[h] + timedelta(minutes=15 * q))
            frame_wind.append(w_val)
            frame_dir.append(d_val)
            frame_precip.append(p_val)
        frame_times.append(hours_list[h + 1])
        frame_wind.append(wind_list[h + 1])
        frame_dir.append(dir_list[h + 1])
        frame_precip.append(precip_list[h + 1])

    polygons: List[List[List[float]]] = []  # 各フレームの [ [lon,lat], ... ]
    for idx, t in enumerate(frame_times):
        t_hours = (t - frame_times[0]).total_seconds() / 3600.0
        precip_factor = max(1.0 / (1.0 + frame_precip[idx]), 0.1)
        effective_base = base_speed * precip_factor  # m/s
        k, cross_k = 0.1, 0.05
        S_factor = 1.0 + k * frame_wind[idx]
        U_factor = max(1.0 - k * frame_wind[idx], 0.0)
        cross_factor = max(1.0 - cross_k * frame_wind[idx], 0.3)
        if idx == 0:
            R_down = R_up = R_cross = 20.0
        else:
            time_seconds = t_hours * 3600.0
            R_down = effective_base * S_factor * time_seconds
            R_up   = effective_base * U_factor * time_seconds
            R_cross = effective_base * cross_factor * time_seconds
        front_points, back_points = [], []
        for j in range(31):
            alpha = -math.pi/2 + j * (math.pi / 30)
            front_points.append((R_down * math.cos(alpha), R_cross * math.sin(alpha)))
        for j in range(30, -1, -1):
            alpha = -math.pi/2 + j * (math.pi / 30)
            back_points.append((-R_up * math.cos(alpha), R_cross * math.sin(alpha)))
        if back_points and front_points and back_points[0] == front_points[-1]:
            back_points = back_points[1:]
        poly_local = front_points + back_points
        if poly_local[0] == poly_local[-1]:
            poly_local = poly_local[:-1]
        spread_dir = (frame_dir[idx] + 180.0) % 360.0
        theta = math.radians(spread_dir)
        poly_coords = []  # [lon, lat]
        for (x, y) in poly_local:
            north_offset = x * math.cos(theta) - y * math.sin(theta)
            east_offset  = x * math.sin(theta) + y * math.cos(theta)
            lat_point = lat + north_offset / 110_540.0
            lon_point = lon + east_offset  / (111_320.0 * math.cos(math.radians(lat)))
            poly_coords.append([lon_point, lat_point])
        if poly_coords[0] != poly_coords[-1]:
            poly_coords.append(poly_coords[0])
        polygons.append(poly_coords)
    return frame_times, polygons, frame_precip


def polygon_area_m2(poly_lonlat: List[List[float]], ref_lat: float, ref_lon: float) -> float:
    # 近似: メルカトルではなく単純換算（小領域想定）
    coords_xy = []
    for lonp, latp in poly_lonlat:
        dx = (lonp - ref_lon) * math.cos(math.radians(ref_lat)) * 111_320.0
        dy = (latp - ref_lat) * 110_540.0
        coords_xy.append((dx, dy))
    area = 0.0
    for i in range(len(coords_xy) - 1):
        x1, y1 = coords_xy[i]
        x2, y2 = coords_xy[i + 1]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def union_area_m2(polys_lonlat: List[List[List[float]]]) -> float:
    if SHAPELY_OK:
        try:
            shp_polys = []
            for coords in polys_lonlat:
                shp_polys.append(Polygon([(x, y) for x, y in coords]))  # (lon,lat)
            unioned = unary_union(shp_polys)
            # 緯度依存換算のため代表緯度を使う近似（厳密には投影が必要）
            # ここでは各リングを線形換算する簡易法に留める
            if unioned.is_empty:
                return 0.0
            def ring_area_m2(ring):
                pts = list(ring.coords)
                ref_lat = sum(p[1] for p in pts) / len(pts)
                ref_lon = sum(p[0] for p in pts) / len(pts)
                return polygon_area_m2([[p[0], p[1]] for p in pts], ref_lat, ref_lon)
            if unioned.geom_type == 'Polygon':
                return ring_area_m2(unioned.exterior) - sum(ring_area_m2(i) for i in unioned.interiors)
            elif unioned.geom_type == 'MultiPolygon':
                s = 0.0
                for g in unioned.geoms:
                    s += ring_area_m2(g.exterior) - sum(ring_area_m2(i) for i in g.interiors)
                return s
        except Exception:
            pass
    # フォールバック: 単純合算（重なり無視）
    return sum(polygon_area_m2(poly, ref_lat=poly[0][1], ref_lon=poly[0][0]) for poly in polys_lonlat)


# ---------------------------------
# UI サイドバー
# ---------------------------------

st.sidebar.header("シミュレーション設定")
col_a, col_b = st.sidebar.columns(2)
with col_a:
    hours = st.number_input("予測時間 (h)", min_value=1, max_value=24, value=6, step=1)
with col_b:
    data_source = st.selectbox("気象データソース", ["AUTO", "Open‑Meteo", "MET Norway (met.no)"])

fuel_options = {"草地": 0.6, "森林": 0.3, "低木地帯": 0.4, "都市部": 0.2}
fuel_type = st.sidebar.selectbox("燃料の種類", list(fuel_options.keys()), index=0)
base_speed = fuel_options[fuel_type]
scenario = st.sidebar.selectbox("シナリオ", ["標準", "強風", "初期消火"], index=0)
if scenario == "強風":
    wind_factor = 2.0
elif scenario == "初期消火":
    wind_factor = 1.0
    hours = min(hours, 3)
else:
    wind_factor = 1.0

show_rain = st.sidebar.checkbox("雨雲オーバーレイ", value=True)
use_gemini = st.sidebar.checkbox("Geminiで結果要約を付与", value=False)
run_clicked = st.sidebar.button("🚀 分析開始")
reset_points = st.sidebar.button("🧹 発火点リセット")

# ---------------------------------
# 発火点の入力
# ---------------------------------

if "ignitions" not in st.session_state:
    st.session_state.ignitions: List[Dict] = []

if reset_points:
    st.session_state.ignitions = []

with st.expander("発火点の追加（住所/緯度,経度を1行1点で）"):
    default_text = "Osaka, Japan" if not st.session_state.ignitions else ""
    txt = st.text_area("住所または緯度,経度 (例: 34.68, 135.52)", value=default_text, height=80)
    if st.button("➕ テキストから発火点を追加"):
        new_lines = [s.strip() for s in txt.splitlines() if s.strip()]
        added = 0
        for line in new_lines:
            lat, lon = geocode_one(line)
            if lat is not None and lon is not None:
                st.session_state.ignitions.append({"lat": lat, "lon": lon, "label": line})
                added += 1
        st.success(f"{added} 点を追加しました。")

# ---------------------------------
# 常時マップ表示 + クリック追加
# ---------------------------------

# マップの中心: 既存点の平均、無ければ大阪
if st.session_state.ignitions:
    avg_lat = sum(p["lat"] for p in st.session_state.ignitions) / len(st.session_state.ignitions)
    avg_lon = sum(p["lon"] for p in st.session_state.ignitions) / len(st.session_state.ignitions)
else:
    avg_lat, avg_lon = 34.6937, 135.5023  # Osaka

m = folium.Map(location=[avg_lat, avg_lon], zoom_start=11, tiles="OpenStreetMap", width="100%", height="600")

# 既存発火点を表示
for i, pnt in enumerate(st.session_state.ignitions, 1):
    folium.Marker([pnt["lat"], pnt["lon"]], tooltip=f"発火点 {i}: {pnt.get('label','')}",
                  icon=folium.Icon(color="red", icon="fire", prefix="fa")).add_to(m)

# クリックで追加できるよう、st_folium を用いる
map_ret = st_folium(m, height=600, width=None, returned_objects=["last_clicked"], use_container_width=True)
if map_ret and map_ret.get("last_clicked"):
    lc = map_ret["last_clicked"]
    lat, lon = lc.get("lat"), lc.get("lng")
    if lat and lon:
        st.session_state.ignitions.append({"lat": lat, "lon": lon, "label": f"clicked({lat:.5f},{lon:.5f})"})
        st.toast("発火点を追加しました（マップクリック）", icon="➕")

# ---------------------------------
# 分析（ボタン押下時のみ）
# ---------------------------------

results_table = None
if run_clicked:
    if not st.session_state.ignitions:
        st.warning("発火点を1つ以上追加してください。")
    else:
        # 単一点の代表緯度経度で気象を取得（簡易仕様）。必要であれば各点ごとに取得に拡張可。
        rep = st.session_state.ignitions[0]
        wx = fetch_weather(rep["lat"], rep["lon"], int(hours), data_source)
        manual_used = False
        if not wx:
            st.warning("気象データの取得に失敗しました。手動設定を使用します。")
            # 手動: 一定風速・風向・降水
            with st.form("manual_wind_form"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    man_wspd = st.number_input("風速 (m/s)", min_value=0.0, max_value=60.0, value=3.0, step=0.5)
                with c2:
                    man_wdir = st.number_input("風向 (度: 北=0, 東=90)", min_value=0.0, max_value=359.9, value=270.0, step=1.0)
                with c3:
                    man_prcp = st.number_input("降水 (mm/h)", min_value=0.0, max_value=200.0, value=0.0, step=0.5)
                submitted = st.form_submit_button("この設定で続行")
            if not submitted:
                st.stop()
            now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
            hours_list = [now + timedelta(hours=i) for i in range(int(hours) + 1)]
            wind_list  = [man_wspd for _ in hours_list]
            dir_list   = [man_wdir for _ in hours_list]
            precip_list= [man_prcp for _ in hours_list]
            manual_used = True
        else:
            hours_list, wind_list, dir_list, precip_list = wx

        # シミュレーション: 各発火点のポリゴン列を計算
        all_polys_by_frame: List[List[List[List[float]]]] = []  # frame -> list(polys) -> [ [lon,lat]... ]
        frame_times_ref: List[datetime] = []
        frame_precip_ref: List[float] = []
        for idx_pt, pnt in enumerate(st.session_state.ignitions):
            ftimes, polys, fprec = simulate_fire_single(
                pnt["lat"], pnt["lon"], hours_list, wind_list, dir_list, precip_list,
                base_speed=base_speed, wind_factor=wind_factor
            )
            if idx_pt == 0:
                frame_times_ref = ftimes
                frame_precip_ref = fprec
                all_polys_by_frame = [[poly] for poly in polys]
            else:
                # 既存フレームと同数前提（時刻共通）。
                for i_f in range(len(polys)):
                    all_polys_by_frame[i_f].append(polys[i_f])

        # 統合表示用GeoJSON（各フレームで MultiPolygon）
        frame_iso = [dt.strftime("%Y-%m-%dT%H:%M:%SZ") for dt in frame_times_ref]
        features = []
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [[poly] for poly in all_polys_by_frame[0]] if all_polys_by_frame else []
            },
            "properties": {
                "times": frame_iso,
                "style": {"color": "red", "weight": 1, "fillColor": "red", "fillOpacity": 0.35},
            },
        })

        # 降雨クラウド（時間単位）
        if show_rain:
            hours_iso = [dt.strftime("%Y-%m-%dT%H:%M:%SZ") for dt in hours_list]
            for i, rain_val in enumerate(precip_list):
                if rain_val and rain_val > 0:
                    cloud_radius = 5000.0 + rain_val * 1000.0
                    circle_points = []
                    num_points = 36
                    for deg in range(0, 360, int(360/num_points)):
                        rad = math.radians(deg)
                        east_off = cloud_radius * math.cos(rad)
                        north_off = cloud_radius * math.sin(rad)
                        dlat = north_off / 110_540.0
                        dlon = east_off / (111_320.0 * math.cos(math.radians(avg_lat)))
                        circle_points.append([avg_lon + dlon, avg_lat + dlat])
                    circle_points.append(circle_points[0])
                    features.append({
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": [circle_points]},
                        "properties": {
                            "times": [hours_iso[i]],
                            "style": {"color": "blue", "weight": 0, "fillColor": "blue", "fillOpacity": 0.2},
                        },
                    })

        # 時系列の面積・最大半径などを算出
        rows = []
        for i_f, polys in enumerate(all_polys_by_frame):
            # union 面積
            area_m2 = union_area_m2(polys)
            # 最大半径（各ポリゴンの任意代表点からの最大距離として近似: 中心は代表点=最初の発火点）
            # より厳密には各発火点ごとに中心を変えるべきだが、ここでは代表点基準の簡易実装
            rep_lat, rep_lon = st.session_state.ignitions[0]["lat"], st.session_state.ignitions[0]["lon"]
            max_r = 0.0
            for poly in polys:
                for lonp, latp in poly:
                    dx = (lonp - rep_lon) * math.cos(math.radians(rep_lat)) * 111_320.0
                    dy = (latp - rep_lat) * 110_540.0
                    dist = (dx*dx + dy*dy) ** 0.5
                    if dist > max_r:
                        max_r = dist
            rows.append({
                "utc_time": frame_times_ref[i_f].strftime("%Y-%m-%d %H:%M"),
                "frame_index": i_f,
                "area_m2": area_m2,
                "area_ha": area_m2/10_000.0,
                "max_radius_m": max_r,
                "precip_mm_h": frame_precip_ref[i_f] if i_f < len(frame_precip_ref) else None,
            })

        # 可視化（TimestampedGeoJson を既存マップに重畳）
        m2 = folium.Map(location=[avg_lat, avg_lon], zoom_start=11, tiles="OpenStreetMap", width="100%", height="600")
        for i, pnt in enumerate(st.session_state.ignitions, 1):
            folium.Marker([pnt["lat"], pnt["lon"]], tooltip=f"発火点 {i}: {pnt.get('label','')}",
                          icon=folium.Icon(color="red", icon="fire", prefix="fa")).add_to(m2)

        plugins.TimestampedGeoJson(
            {"type": "FeatureCollection", "features": features},
            period="PT15M", duration="PT1H", add_last_point=False, auto_play=True,
            loop=True, loop_button=True, max_speed=10, progress_bar=True,
        ).add_to(m2)

        st.markdown("### 解析結果（マップ）")
        st_folium(m2, height=600, width=None, use_container_width=True)

        # サマリー
        final = rows[-1]
        st.markdown(
            f"**最終フレーム**  "+
            f"面積: {final['area_ha']:.2f} ha（{final['area_m2']:.0f} ㎡） / 最大半径: {final['max_radius_m']:.0f} m"
        )

        # 拡大速度（m^2/h の近似）
        if len(rows) >= 5:
            dt_hours = 0.25  # 15分
            dA = rows[-1]['area_m2'] - rows[-5]['area_m2']
            growth_rate = dA / (4 * dt_hours)  # m2/h
            st.markdown(f"推定面積拡大速度: **{growth_rate:,.0f} m²/h**")

        # CSV エクスポート
        import pandas as pd
        df = pd.DataFrame(rows)
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("⬇️ 面積・半径の時系列CSVをダウンロード", data=csv, file_name="fire_growth_timeseries.csv", mime="text/csv")
        results_table = df

        # Gemini 要約（任意）
        if use_gemini:
            gk = _get_gemini_api_key()
            if gk:
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=gk)
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    prompt = {
                        "scenario": scenario,
                        "fuel_type": fuel_type,
                        "hours": hours,
                        "points": st.session_state.ignitions,
                        "final_area_m2": final['area_m2'],
                        "final_area_ha": final['area_ha'],
                        "max_radius_m": final['max_radius_m'],
                        "growth_table_head": rows[:8],
                    }
                    resp = model.generate_content([
                        "以下のJSONは火災延焼予測の結果です。短く行政文体でサマリーを書いてください。重要指標(面積, 最大半径, 拡大速度)を数値で示し、根拠の要約も添えてください。",
                        json.dumps(prompt, ensure_ascii=False),
                    ])
                    st.markdown("### Gemini 要約")
                    st.write(resp.text)
                except Exception as e:
                    st.info(f"Gemini要約は利用できませんでした: {e}")
            else:
                st.info("GeminiのAPIキーが設定されていません（[gemini].api_key）。")

# ---------------------------------
# 3Dビュー（任意: 参考表示）
# ---------------------------------
with st.expander("3Dビュー (任意)"):
    if st.session_state.ignitions:
        view_state = pdk.ViewState(latitude=avg_lat, longitude=avg_lon, zoom=11, pitch=45)
        layers = []
        for pnt in st.session_state.ignitions:
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=[{"lon": pnt["lon"], "lat": pnt["lat"]}],
                get_position="[lon, lat]",
                get_radius=100,
                get_color="[255,0,0]",
            ))
        deck = pdk.Deck(map_provider="mapbox", map_style="light-v9", layers=layers, initial_view_state=view_state, height=400)
        st.pydeck_chart(deck, use_container_width=True)
    else:
        st.caption("発火点を追加すると3D表示できます。")

# ---------------------------------
# ヒント
# ---------------------------------
st.info("マップは常時表示されます。発火点は\"住所/緯度経度入力\"または\"地図クリック\"で追加し、\"分析開始\"ボタンで解析します。気象取得に失敗した場合でも手動設定で解析可能です。")
