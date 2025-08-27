"""
Fire Spread Simulation — COMPLETE

要件対応:
- マップは常時表示（分析前でも）
- サイドバーに「🚀 分析開始」ボタン
- 複数発火点（テキストで複数行入力）
- 気象データ: Open‑Meteo → MET Norway (met.no) の順で自動フォールバック
  （両方失敗時は手動パラメータ入力フォームで継続可能）
- 時系列（15分刻み）で延焼ポリゴンを生成し、FoliumのTimestampedGeoJsonで再生
  （各フレーム=1 Feature方式で安全）
- 面積（㎡/ha）、最大半径、拡大速度の算出、CSVダウンロード
- 任意: Gemini による管理文体サマリー（[gemini].api_key がある場合）

requirements.txt 例:
streamlit
requests
folium
pydeck
pandas
shapely>=2.0    # 任意（あると重なりの面積が正確）
google-generativeai  # 任意（Gemini要約を使うとき）

secrets.toml 例:
[general]
api_key = "<Google API Key (任意: Geocoding用)>"

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
import pandas as pd

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


def fetch_weather_auto(lat: float, lon: float, total_hours: int, source: str):
    """選択ソースに従って取得。AUTO は Open‑Meteo→met.no の順に試行。"""
    if source == "Open‑Meteo":
        return fetch_weather_openmeteo(lat, lon, total_hours)
    if source == "MET Norway (met.no)":
        return fetch_weather_metno(lat, lon, total_hours)
    data = fetch_weather_openmeteo(lat, lon, total_hours)
    if data:
        return data
    return fetch_weather_metno(lat, lon, total_hours)


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

    polygons: List[List[List[float]]]= []  # 各フレームの [ [lon,lat], ... ]
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
        spread_dir = (dir_list[min(idx,len(dir_list)-1)] + 180.0) % 360.0
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


def build_timestamped_polygon_features(frame_times: List[datetime], per_frame_polygons: List[List[List[float]]], color="red"):
    features = []
    for i, poly in enumerate(per_frame_polygons):
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [poly]},
            "properties": {
                "times": [frame_times[i].strftime("%Y-%m-%dT%H:%M:%SZ")],
                "style": {"color": color, "weight": 1, "fillColor": color, "fillOpacity": 0.35},
            },
        })
    return features


def polygon_area_m2(poly_lonlat: List[List[float]], ref_lat: float, ref_lon: float) -> float:
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


def union_area_m2(polys_lonlat: List[List[List[float]]], ref_lat: float, ref_lon: float) -> float:
    if SHAPELY_OK:
        try:
            shp_polys = []
            for coords in polys_lonlat:
                pts = [(
                    (lon - ref_lon) * math.cos(math.radians(ref_lat)) * 111_320.0,
                    (lat - ref_lat) * 110_540.0
                ) for lon, lat in coords]
                shp_polys.append(Polygon(pts))
            unioned = unary_union(shp_polys)
            return float(unioned.area)
        except Exception:
            pass
    # フォールバック: 単純合算（重なり無視）
    return sum(polygon_area_m2(poly, ref_lat=ref_lat, ref_lon=ref_lon) for poly in polys_lonlat)


# ---------------------------------
# UI サイドバー
# ---------------------------------

st.sidebar.header("シミュレーション設定")
location_inputs: List[str] = st.sidebar.text_area("火災発生地点（複数行可）", "Osaka, Japan").splitlines()
col1, col2 = st.sidebar.columns(2)
with col1:
    total_hours = st.number_input("予測時間 (h)", min_value=1, max_value=24, value=6)
with col2:
    data_source = st.selectbox("気象データソース", ["AUTO", "Open‑Meteo", "MET Norway (met.no)"])

fuel_options = {"草地": 0.6, "森林": 0.3, "低木地帯": 0.4, "都市部": 0.2}
fuel_type = st.sidebar.selectbox("燃料の種類", list(fuel_options.keys()), index=0)
base_speed = fuel_options[fuel_type]
scenario = st.sidebar.selectbox("シナリオ", ["標準", "強風", "初期消火"], index=0)
if scenario == "強風":
    wind_factor = 2.0
elif scenario == "初期消火":
    wind_factor = 1.0
    total_hours = min(total_hours, 3)
else:
    wind_factor = 1.0

show_rain = st.sidebar.checkbox("雨雲オーバーレイ", value=True)
use_gemini = st.sidebar.checkbox("Geminiで要約", value=False)

analyze = st.sidebar.button("🚀 分析開始")

# ---------------------------------
# 常時マップ表示（入力が住所でも可能な限りジオコーディングしてプロット）
# ---------------------------------

# 代表中心：最初に解決できた地点、なければ大阪
center_lat, center_lon = 34.6937, 135.5023
markers: List[Tuple[float,float,str]] = []
for loc in location_inputs:
    loc = loc.strip()
    if not loc:
        continue
    lat, lon = geocode_one(loc)
    if lat is not None and lon is not None:
        if len(markers) == 0:
            center_lat, center_lon = lat, lon
        markers.append((lat, lon, loc))

m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="OpenStreetMap", width="100%", height="600")
for i, (lat, lon, loc) in enumerate(markers, 1):
    folium.Marker([lat, lon], tooltip=f"発火点 {i}: {loc}", icon=folium.Icon(color="red", icon="fire", prefix="fa")).add_to(m)

st.components.v1.html(m._repr_html_(), height=600, scrolling=False)

# ---------------------------------
# 分析（ボタン押下時）
# ---------------------------------

if analyze:
    if not markers:
        st.warning("発火点を1つ以上入力してください。")
        st.stop()

    all_features = []  # TimestampedGeoJson 用
    per_frame_polys: List[List[List[List[float]]]] = []  # frame -> [polys]
    frame_times_ref: List[datetime] = []
    frame_precip_ref: List[float] = []

    for idx, (lat, lon, loc) in enumerate(markers, 1):
        wx = fetch_weather_auto(lat, lon, int(total_hours), data_source)
        if not wx:
            st.warning(f"気象データを取得できませんでした: {loc}。手動設定で継続します。")
            with st.form(f"manual_wind_form_{idx}"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    man_wspd = st.number_input(f"[{loc}] 風速 (m/s)", min_value=0.0, max_value=60.0, value=3.0, step=0.5)
                with c2:
                    man_wdir = st.number_input(f"[{loc}] 風向 (度: 北=0, 東=90)", min_value=0.0, max_value=359.9, value=270.0, step=1.0)
                with c3:
                    man_prcp = st.number_input(f"[{loc}] 降水 (mm/h)", min_value=0.0, max_value=200.0, value=0.0, step=0.5)
                submitted = st.form_submit_button("この設定で続行")
            if not submitted:
                continue
            now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
            hours_list = [now + timedelta(hours=i) for i in range(int(total_hours) + 1)]
            wind_list  = [man_wspd for _ in hours_list]
            dir_list   = [man_wdir for _ in hours_list]
            precip_list= [man_prcp for _ in hours_list]
        else:
            hours_list, wind_list, dir_list, precip_list = wx

        ftimes, polys, fprec = simulate_fire_single(lat, lon, hours_list, wind_list, dir_list, precip_list,
                                                    base_speed=base_speed, wind_factor=wind_factor)
        # 参照フレーム
        if not frame_times_ref:
            frame_times_ref = ftimes
            frame_precip_ref = fprec
            per_frame_polys = [[polys[i]] for i in range(len(polys))]
        else:
            # 同数前提（異なる場合は切り詰め）
            L = min(len(per_frame_polys), len(polys))
            for i in range(L):
                per_frame_polys[i].append(polys[i])

        # 1点分の時系列 Feature を作成
        all_features.extend(build_timestamped_polygon_features(ftimes, polys, color="red"))

        # 静的な最終ポリゴンも参考表示
        folium.Polygon(polys[-1], color="red", weight=2, fill=True, fill_opacity=0.25,
                       tooltip=f"最終推定範囲: {loc}").add_to(m)

    # 時系列アニメーションマップ
    if all_features:
        m_anim = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="OpenStreetMap", width="100%", height="600")
        plugins.TimestampedGeoJson(
            {"type": "FeatureCollection", "features": all_features},
            period="PT15M", duration="PT15M", add_last_point=False,
            auto_play=True, loop=True, loop_button=True, max_speed=10, progress_bar=True
        ).add_to(m_anim)
        st.markdown("### 時系列アニメーション（解析結果）")
        st.components.v1.html(m_anim._repr_html_(), height=600, scrolling=False)

    # 指標の算出（union 面積、最大半径、拡大速度）
    if per_frame_polys:
        rows = []
        ref_lat, ref_lon = markers[0][0], markers[0][1]
        for i in range(len(per_frame_polys)):
            polys = per_frame_polys[i]
            area_m2 = union_area_m2(polys, ref_lat, ref_lon)
            # 最大半径（代表中心からの最大距離）
            max_r = 0.0
            for poly in polys:
                for lonp, latp in poly:
                    dx = (lonp - ref_lon) * math.cos(math.radians(ref_lat)) * 111_320.0
                    dy = (latp - ref_lat) * 110_540.0
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist > max_r:
                        max_r = dist
            rows.append({
                "utc_time": frame_times_ref[i].strftime("%Y-%m-%d %H:%M"),
                "frame_index": i,
                "area_m2": area_m2,
                "area_ha": area_m2 / 10_000.0,
                "max_radius_m": max_r,
                "precip_mm_h": frame_precip_ref[i] if i < len(frame_precip_ref) else None,
            })

        final = rows[-1]
        st.markdown(
            f"**最終フレーム** — 面積: {final['area_ha']:.2f} ha（{final['area_m2']:.0f} ㎡） / 最大半径: {final['max_radius_m']:.0f} m"
        )

        # 拡大速度（m^2/h の近似、直近1時間＝4コマ）
        if len(rows) >= 5:
            dt_hours = 0.25  # 15分
            dA = rows[-1]['area_m2'] - rows[-5]['area_m2']
            growth_rate = dA / (4 * dt_hours)
            st.markdown(f"推定面積拡大速度: **{growth_rate:,.0f} m²/h**")

        df = pd.DataFrame(rows)
        st.download_button("⬇️ 面積・半径の時系列CSVをダウンロード", data=df.to_csv(index=False).encode('utf-8-sig'),
                           file_name="fire_growth_timeseries.csv", mime="text/csv")

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
                        "hours": int(total_hours),
                        "points": [{"lat": a, "lon": b, "label": c} for a, b, c in markers],
                        "final_area_m2": float(final['area_m2']),
                        "final_area_ha": float(final['area_ha']),
                        "max_radius_m": float(final['max_radius_m']),
                        "growth_rate_m2ph": float(growth_rate) if len(rows) >= 5 else None,
                    }
                    resp = model.generate_content([
                        "以下のJSONは火災延焼予測の結果です。行政文体で簡潔にサマリーを書いてください。重要指標(面積, 最大半径, 拡大速度)を数値で示し、根拠も簡潔に示してください。",
                        json.dumps(prompt, ensure_ascii=False),
                    ])
                    st.markdown("### 要約 (Gemini)")
                    st.write(resp.text)
                except Exception as e:
                    st.info(f"Gemini要約は利用できませんでした: {e}")
            else:
                st.info("GeminiのAPIキーが設定されていません（[gemini].api_key）。")

# ---------------------------------
# 補足: 凡例
# ---------------------------------
legend_html = """
<div style="position: fixed; bottom: 50px; right: 50px; z-index: 9999; background: white; border: 1px solid #888; padding: 10px; opacity: 0.9; font-size: 13px;">
<b>凡例</b><br>
<span style="display:inline-block;width:12px;height:12px;background:red;margin-right:6px;"></span> 推定延焼範囲（時系列）<br>
<i class="fa fa-fire" style="color:red;margin-right:6px;"></i> 発火点
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))
