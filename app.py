import os
import math
import urllib.parse
from datetime import datetime, timedelta, timezone
from typing import Tuple, Optional

import requests
import streamlit as st
import folium
from folium import plugins
import pydeck as pdk

# ============================
# Page config (run early)
# ============================
st.set_page_config(page_title="Fire Spread Simulation", layout="wide")

# ============================
# Secrets / API keys helpers
# ============================

def _get_google_api_key() -> Optional[str]:
    """Fetch Google API key robustly.
    Accepts either:
      [general]\napi_key = "..."
    or top-level: api_key = "..."
    Or env var GOOGLE_API_KEY as fallback.
    """
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


def _headers_for_osm():
    # Nominatim利用時の推奨ヘッダ（任意の連絡先に変更してください）
    return {"User-Agent": "fire-spread-sim/1.0 (contact: example@example.com)"}


# ============================
# External API wrappers (cached)
# ============================

@st.cache_data(show_spinner=False)
def geocode_address(address: str) -> Tuple[Optional[float], Optional[float]]:
    """Geocode an address to (lat, lon).
    - Try parsing "lat, lon" directly
    - Try Google Geocoding API if key exists
    - Fallback to OSM Nominatim
    """
    # If input already looks like "lat, lon"
    try:
        parts = [p.strip() for p in address.split(",")]
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
    except Exception:
        pass

    api_key = _get_google_api_key()
    if api_key:
        try:
            addr_enc = urllib.parse.quote(address, safe="")
            url = f"https://maps.googleapis.com/maps/api/geocode/json?address={addr_enc}&key={api_key}"
            resp = requests.get(url, timeout=15)
            data = resp.json()
            if data.get("status") == "OK" and data.get("results"):
                loc = data["results"][0]["geometry"]["location"]
                return loc["lat"], loc["lng"]
        except Exception:
            pass

    # Fallback to Nominatim
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": address, "format": "json", "limit": 1}
        resp = requests.get(url, params=params, headers=_headers_for_osm(), timeout=15)
        data = resp.json()
        if isinstance(data, list) and data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass

    return None, None


@st.cache_data(show_spinner=False)
def fetch_weather(lat: float, lon: float, total_hours: int):
    """Fetch hourly weather forecast (wind dir/speed, precip) from Open-Meteo (UTC).
    Returns (hours_slice, wind_slice, dir_slice, precip_slice) or None on failure.
    """
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat:.4f}&longitude={lon:.4f}"
            f"&hourly=winddirection_10m,windspeed_10m,precipitation&windspeed_unit=ms&timezone=UTC&forecast_days=2"
        )
        resp = requests.get(url, timeout=30)
        data = resp.json()
        if "hourly" not in data or "time" not in data["hourly"]:
            return None
        times = data["hourly"]["time"]
        wind_speeds = data["hourly"]["windspeed_10m"]
        wind_dirs = data["hourly"]["winddirection_10m"]
        precips = data["hourly"]["precipitation"]
        time_datetimes = [datetime.fromisoformat(t.replace("Z", "+00:00")) for t in times]
        now_utc = datetime.now(timezone.utc)
        now_floor = now_utc.replace(minute=0, second=0, microsecond=0)
        if now_floor < time_datetimes[0]:
            start_idx = 0
        else:
            try:
                start_idx = time_datetimes.index(now_floor)
            except ValueError:
                start_idx = next((i for i, t in enumerate(time_datetimes) if t > now_utc), 0)
        end_idx = min(start_idx + total_hours, len(time_datetimes) - 1)
        hours_slice = time_datetimes[start_idx : end_idx + 1]
        wind_slice = wind_speeds[start_idx : end_idx + 1]
        dir_slice = wind_dirs[start_idx : end_idx + 1]
        precip_slice = precips[start_idx : end_idx + 1]
        return hours_slice, wind_slice, dir_slice, precip_slice
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def get_timezone_offset(lat: float, lon: float) -> Tuple[int, str]:
    """Return (offset_seconds, tz_name). Try Google Time Zone, fallback to Open‑Meteo Timezone API."""
    api_key = _get_google_api_key()
    if api_key:
        try:
            timestamp = int(datetime.now(timezone.utc).timestamp())
            url = (
                f"https://maps.googleapis.com/maps/api/timezone/json?location={lat:.6f},{lon:.6f}&timestamp={timestamp}&key={api_key}"
            )
            resp = requests.get(url, timeout=15)
            data = resp.json()
            if data.get("status") == "OK":
                raw_offset = int(data.get("rawOffset", 0))
                dst_offset = int(data.get("dstOffset", 0))
                tz_id = data.get("timeZoneId", "")
                return raw_offset + dst_offset, tz_id
        except Exception:
            pass
    # Fallback: Open‑Meteo timezone API
    try:
        url = f"https://api.open-meteo.com/v1/timezone?latitude={lat:.6f}&longitude={lon:.6f}"
        r = requests.get(url, timeout=15).json()
        offset = int(r.get("utc_offset_seconds", 0))
        tz = r.get("timezone", "")
        return offset, tz
    except Exception:
        return 0, ""


# ============================
# Core simulation
# ============================

def simulate_fire(lat: float, lon: float, hours_list, wind_list, dir_list, precip_list, base_speed: float, wind_factor: float):
    """Simulate fire spread polygons for 15‑minute frames.
    Returns: frame_times, polygons, frame_precip
    """
    wind_list = [w * wind_factor for w in wind_list]
    frame_times = []
    frame_wind = []
    frame_dir = []
    frame_precip = []

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
            d1 = dir_list[h]
            d2 = dir_list[h + 1]
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

    polygons = []
    for idx, t in enumerate(frame_times):
        t_hours = (t - frame_times[0]).total_seconds() / 3600.0
        precip_factor = 1.0 / (1.0 + frame_precip[idx])
        precip_factor = max(precip_factor, 0.1)
        effective_base = base_speed * precip_factor  # m/s
        k = 0.1
        cross_k = 0.05
        S_factor = 1.0 + k * frame_wind[idx]
        U_factor = max(1.0 - k * frame_wind[idx], 0.0)
        cross_factor = max(1.0 - cross_k * frame_wind[idx], 0.3)

        if idx == 0:
            R_down = R_up = R_cross = 20.0
        else:
            time_seconds = t_hours * 3600.0
            R_down = effective_base * S_factor * time_seconds
            R_up = effective_base * U_factor * time_seconds
            R_cross = effective_base * cross_factor * time_seconds

        front_points = []
        back_points = []
        for j in range(31):
            alpha = -math.pi / 2 + j * (math.pi / 30)
            x = R_down * math.cos(alpha)
            y = R_cross * math.sin(alpha)
            front_points.append((x, y))
        for j in range(30, -1, -1):
            alpha = -math.pi / 2 + j * (math.pi / 30)
            x = -R_up * math.cos(alpha)
            y = R_cross * math.sin(alpha)
            back_points.append((x, y))
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
            east_offset = x * math.sin(theta) + y * math.cos(theta)
            lat_offset = north_offset / 110_540.0
            lon_offset = east_offset / (111_320.0 * math.cos(math.radians(lat)))
            lat_point = lat + lat_offset
            lon_point = lon + lon_offset
            poly_coords.append([lon_point, lat_point])
        if poly_coords[0] != poly_coords[-1]:
            poly_coords.append(poly_coords[0])
        polygons.append(poly_coords)

    return frame_times, polygons, frame_precip


# ============================
# Sidebar UI
# ============================

st.sidebar.header("シミュレーション設定")
location_input = st.sidebar.text_input("火災発生地点", "Osaka, Japan")
fuel_options = {"草地": 0.6, "森林": 0.3, "低木地帯": 0.4, "都市部": 0.2}
fuel_type = st.sidebar.selectbox("燃料の種類", list(fuel_options.keys()), index=0)
base_speed = fuel_options[fuel_type]

scenario_options = ["標準シナリオ", "強風シナリオ", "初期消火シナリオ"]
scenario = st.sidebar.selectbox("シナリオ選択", scenario_options, index=0)
if scenario == "強風シナリオ":
    wind_factor = 2.0
    total_hours = 6
elif scenario == "初期消火シナリオ":
    wind_factor = 1.0
    total_hours = 3
else:
    wind_factor = 1.0
    total_hours = 6

show_rain = st.sidebar.checkbox("雨雲オーバーレイを表示", value=True)
display_mode = st.sidebar.radio("表示モード", ["2D 地図", "3D 表示"], index=0)

# ============================
# Geocode & Weather
# ============================

lat, lon = geocode_address(location_input)
if lat is None or lon is None:
    st.error("入力された地点を特定できませんでした。住所または緯度経度を正しく入力してください。")
    st.stop()

weather_data = fetch_weather(lat, lon, total_hours)
if not weather_data:
    st.error("気象データを取得できませんでした。")
    st.stop()

hours_list, wind_list, dir_list, precip_list = weather_data

# ============================
# Simulation
# ============================

frame_times, fire_polygons, frame_precip = simulate_fire(
    lat, lon, hours_list, wind_list, dir_list, precip_list, base_speed, wind_factor
)
frame_iso = [dt.strftime("%Y-%m-%dT%H:%M:%SZ") for dt in frame_times]

# ============================
# 2D (Folium) or 3D (PyDeck)
# ============================

if display_mode == "2D 地図":
    m = folium.Map(location=[lat, lon], zoom_start=13, tiles="OpenStreetMap", width="100%", height="600")
    folium.Marker(
        location=[lat, lon],
        icon=folium.Icon(color="red", icon="fire", prefix="fa"),
        tooltip="火災発生地点",
    ).add_to(m)

    fire_feature = {
        "type": "Feature",
        "geometry": {"type": "MultiPolygon", "coordinates": [[poly] for poly in fire_polygons]},
        "properties": {
            "times": frame_iso,
            "style": {"color": "red", "weight": 1, "fillColor": "red", "fillOpacity": 0.4},
        },
    }

    features = [fire_feature]

    if show_rain:
        hours_iso = [dt.strftime("%Y-%m-%dT%H:%M:%SZ") for dt in hours_list]
        for i, rain_val in enumerate(precip_list):
            if rain_val and rain_val > 0:
                cloud_radius = 5000.0 + rain_val * 1000.0
                circle_points = []
                num_points = 24
                for deg in range(0, 360, int(360 / num_points)):
                    rad = math.radians(deg)
                    east_off = cloud_radius * math.cos(rad)
                    north_off = cloud_radius * math.sin(rad)
                    dlat = north_off / 110_540.0
                    dlon = east_off / (111_320.0 * math.cos(math.radians(lat)))
                    lat_pt = lat + dlat
                    lon_pt = lon + dlon
                    circle_points.append([lon_pt, lat_pt])
                circle_points.append(circle_points[0])
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": [circle_points]},
                        "properties": {
                            "times": [hours_iso[i]],
                            "style": {
                                "color": "blue",
                                "weight": 0,
                                "fillColor": "blue",
                                "fillOpacity": 0.2,
                            },
                        },
                    }
                )

    plugins.TimestampedGeoJson(
        {"type": "FeatureCollection", "features": features},
        period="PT15M",
        duration="PT1H",
        add_last_point=True,
        auto_play=True,
        loop=True,
        loop_button=True,
        max_speed=10,
        progress_bar=True,
    ).add_to(m)

    legend_html = """
    <div style="
        position: fixed;
        bottom: 50px;
        right: 50px;
        z-index: 9999;
        background-color: white;
        border: 2px solid grey;
        padding: 10px;
        opacity: 0.8;
        font-size: 14px;
    ">
        <b>凡例</b><br>
        <i style="display:inline-block;width:12px;height:12px;background:red;margin-right:5px;"></i> 火災延焼範囲<br>
        <i style="display:inline-block;width:12px;height:12px;background:blue;opacity:0.5;margin-right:5px;"></i> 雨雲領域<br>
        <i class=\"fa fa-fire\" style=\"color:red;margin-right:5px;\"></i> 火災発生地点
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    st.components.v1.html(m._repr_html_(), height=600, scrolling=False)

else:
    offset_sec, tz_name = get_timezone_offset(lat, lon)
    local_times = [(dt + timedelta(seconds=offset_sec)).replace(tzinfo=None) for dt in frame_times]
    selected_time = st.sidebar.select_slider(
        "時刻", options=local_times, value=local_times[-1], format_func=lambda t: t.strftime("%m/%d %H:%M")
    )
    try:
        time_index = local_times.index(selected_time)
    except ValueError:
        time_index = len(local_times) - 1

    layers = []

    polygon_data = [{"coordinates": fire_polygons[time_index]}]
    fire_layer = pdk.Layer(
        "PolygonLayer",
        data=polygon_data,
        get_polygon="coordinates",
        get_fill_color="[255, 50, 50, 100]",
        get_line_color="[255, 0, 0]",
        get_line_width=200,
        stroked=True,
        filled=True,
        extruded=False,
    )
    layers.append(fire_layer)

    ignition_data = [{"lat": lat, "lon": lon, "label": "火災発生地点"}]
    ign_layer = pdk.Layer(
        "ScatterplotLayer",
        data=ignition_data,
        get_position="[lon, lat]",
        get_color="[255, 0, 0]",
        get_radius=100,
        pickable=True,
    )
    layers.append(ign_layer)

    if show_rain:
        rain_val = frame_precip[time_index]
        if rain_val and rain_val > 0:
            cloud_radius = 5000.0 + rain_val * 1000.0
            cloud_coords = []
            for deg in range(0, 360, 10):
                rad = math.radians(deg)
                east_off = cloud_radius * math.cos(rad)
                north_off = cloud_radius * math.sin(rad)
                dlat = north_off / 110_540.0
                dlon = east_off / (111_320.0 * math.cos(math.radians(lat)))
                lat_pt = lat + dlat
                lon_pt = lon + dlon
                cloud_coords.append([lon_pt, lat_pt])
            cloud_coords.append(cloud_coords[0])
            cloud_data = [{"coords": cloud_coords}]
            cloud_layer = pdk.Layer(
                "PolygonLayer",
                data=cloud_data,
                get_polygon="coords",
                get_fill_color="[50, 50, 200, 80]",
                get_line_color="[50, 50, 200]",
                stroked=False,
                filled=True,
            )
            layers.append(cloud_layer)

    view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=12, pitch=45, bearing=0)
    deck = pdk.Deck(
        map_provider="google_maps",
        api_keys={"google_maps": _get_google_api_key() or ""},
        map_style="satellite",
        initial_view_state=view_state,
        layers=layers,
        height=600,
        tooltip={"text": "{label}"},
    )
    st.pydeck_chart(deck, use_container_width=True)

# ============================
# Metrics: area & water need
# ============================

final_polygon = fire_polygons[-1]
ref_lat = lat
coords_xy = []
for lonp, latp in final_polygon:
    dx = (lonp - lon) * math.cos(math.radians(ref_lat)) * 111_320.0
    dy = (latp - lat) * 110_540.0
    coords_xy.append((dx, dy))

area_m2 = 0.0
for i in range(len(coords_xy) - 1):
    x1, y1 = coords_xy[i]
    x2, y2 = coords_xy[i + 1]
    area_m2 += x1 * y2 - x2 * y1
area_m2 = abs(area_m2) / 2.0

area_ha = area_m2 / 10_000.0
water_liters = area_m2 * 10.0
water_cubic = water_liters / 1_000.0

st.markdown(
    f"**延焼面積:** 約{area_ha:.2f}ヘクタール（{area_m2:.0f}㎡）  \n" +
    f"**必要水量:** 約{water_cubic:.1f}立方メートル（{water_liters:,.0f}リットル）"
)
