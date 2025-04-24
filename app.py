import streamlit as st
import requests
import math
import pydeck as pdk
import numpy as np
from io import BytesIO
from PIL import Image
import csv
import re
import google.generativeai as genai

# --- ページ設定 ---
st.set_page_config(page_title="火災拡大シミュレーション＋Gemini報告", layout="wide")

# --- Secrets の読み込み ---
MAPBOX_TOKEN        = st.secrets["mapbox"]["access_token"]
OPENWEATHER_API_KEY = st.secrets["openweather"]["api_key"]
FIRMS_MAP_KEY       = st.secrets["firms"]["map_key"]
GEMINI_API_KEY      = st.secrets["general"]["api_key"]

# --- Gemini 初期化 ---
genai.configure(api_key=GEMINI_API_KEY)
MODEL = genai.GenerativeModel('gemini-1.5-flash')

# --- セッションステート初期化 ---
if "fire_location" not in st.session_state:
    # 初期地点：愛媛県松山市付近
    st.session_state.fire_location = (34.25743760177552, 133.2043209338966)

# --- サイドバー UI ---
st.sidebar.header("🔥 シミュレーション設定")
latlon_str = st.sidebar.text_input(
    "発生地点 (Googleマップ形式: lat, lon)",
    value=f"{st.session_state.fire_location[0]}, {st.session_state.fire_location[1]}"
)
if st.sidebar.button("地点を更新"):
    m = re.match(r"\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*", latlon_str)
    if m:
        st.session_state.fire_location = (float(m[1]), float(m[2]))
        st.sidebar.success("発生地点を更新しました")
    else:
        st.sidebar.error("入力形式エラー: 例 34.2574376, 133.2043209")

fuel_map   = {"森林（高燃料）":1.2, "草地（中燃料）":1.0, "都市部（低燃料）":0.8}
fuel_label = st.sidebar.selectbox("燃料特性", list(fuel_map.keys()))
fuel_coeff = fuel_map[fuel_label]

days = st.sidebar.slider("経過日数 (日)", 1, 7, 4)

show_firms = st.sidebar.checkbox("FIRMSデータを重ねる", value=False)

# --- ヘルパー関数 ---
@st.cache_data(ttl=600)
def get_weather(lat, lon):
    url = (
        f"https://api.openweathermap.org/data/2.5/weather?"
        f"lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
        "&units=metric&lang=ja"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()

@st.cache_data(ttl=600)
def get_firms_area(lat, lon, days, map_key):
    delta = 0.5
    south, north = lat - delta, lat + delta
    west, east   = lon - delta, lon + delta
    bbox = f"{west},{south},{east},{north}"
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/"
        f"{map_key}/VIIRS_SNPP_NRT/{bbox}/{days}"
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
    except:
        return []
    f = StringIO(r.text)
    reader = csv.DictReader(f)
    out = []
    for row in reader:
        try:
            out.append({
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
                "bright": float(row.get("bright_ti4", 0))
            })
        except:
            continue
    return out

def get_elevation(lat, lon):
    """Mapbox Terrain‐RGBで標高取得"""
    zoom = 14
    tx = int((lon + 180) / 360 * 2**zoom)
    ty = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * 2**zoom)
    url = f"https://api.mapbox.com/v4/mapbox.terrain-rgb/{zoom}/{tx}/{ty}.pngraw?access_token={MAPBOX_TOKEN}"
    r = requests.get(url, timeout=10)
    if r.status_code == 200:
        img = Image.open(BytesIO(r.content))
        arr = np.array(img)[128,128,:3].astype(np.int32)
        return -10000 + ((arr[0]*256*256 + arr[1]*256 + arr[2]) * 0.1)
    return 0

def generate_terrain_polygon(lat, lon, radius, wind_dir):
    deg_m = 1/111000
    coords = []
    for deg in np.linspace(wind_dir-90, wind_dir+90, 36):
        rad = math.radians(deg)
        dx = radius * math.sin(rad)
        dy = radius * math.cos(rad)
        plat = lat + dy * deg_m
        plon = lon + dx * deg_m / math.cos(math.radians(lat))
        elev = get_elevation(plat, plon) * 0.3
        coords.append([plon, plat, elev])
    return coords

def summarize_fire(lat, lon, wind, fuel, days, radius, area, water):
    prompt = (
        f"対象地点: 緯度{lat}, 経度{lon}\n"
        f"風速: {wind['speed']}m/s, 風向: {wind['deg']}°\n"
        f"燃料特性: {fuel}, 経過日数: {days}日\n"
        f"延焼半径: {radius:.1f}m, 延焼面積: {area:.1f}m², 消水量: {water:.1f}t\n"
        "これを一般の方向けにわかりやすく日本語で要約してください。"
    )
    resp = MODEL.generate_content(prompt)
    return resp.text.strip()

# --- メイン ---
st.title("🔥 火災拡大シミュレーション with Gemini Report")

lat_c, lon_c = st.session_state.fire_location
weather = get_weather(lat_c, lon_c)
wind    = weather.get("wind", {"speed":0, "deg":0})

st.markdown(
    f"**風速:** {wind['speed']} m/s  **風向:** {wind['deg']}°  "
    f"**燃料:** {fuel_label}  **経過日数:** {days}日"
)

# レイヤー作成
layers = []

# 発生地点マーカー（小さいピン）
layers.append(pdk.Layer(
    "ScatterplotLayer",
    data=[{"position":[lon_c,lat_c]}],
    get_position="position",
    get_color=[0,0,255],
    get_radius=4
))

# 延焼半径・面積・水量算出
base_radius = (250 * fuel_coeff) + 10 * days * fuel_coeff
area_sqm     = math.pi * base_radius**2
water_tons   = (area_sqm / 10000) * 5

# 地形に沿ったポリゴン
polygon = generate_terrain_polygon(lat_c, lon_c, base_radius, wind["deg"])
layers.append(pdk.Layer(
    "PolygonLayer",
    data=[{"polygon": polygon}],
    get_polygon="polygon",
    get_fill_color=[255,0,0,80],
    extruded=False
))

# FIRMS ホットスポット
if show_firms:
    spots = get_firms_area(lat_c, lon_c, days, FIRMS_MAP_KEY)
    ds = []
    for s in spots:
        c = min(max(int((s["bright"]-300)*2),0),255)
        ds.append({
            "position":[s["lon"], s["lat"]],
            "color": [255,255-c,0]
        })
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=ds,
        get_position="position",
        get_fill_color="color",
        get_radius=6000,
        pickable=False
    ))
    st.success(f"FIRMSスポット {len(spots)} 件表示")

# マップ描画
view = pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=12, pitch=45)
st.pydeck_chart(pdk.Deck(
    layers=layers,
    initial_view_state=view,
    map_style="mapbox://styles/mapbox/satellite-streets-v11"
), use_container_width=True)

# Gemini レポート
report = summarize_fire(
    lat_c, lon_c, wind, fuel_label, days,
    base_radius, area_sqm, water_tons
)
st.markdown("## 🔥 Geminiによる火災予測レポート")
st.write(report)

# 気象データ JSON
with st.expander("▼ 気象データ (JSON)"):
    st.json(weather)
