import streamlit as st
import requests
import math
import pydeck as pdk
import numpy as np
from io import BytesIO
from PIL import Image
import google.generativeai as genai

# --- ページ設定 ---
st.set_page_config(page_title="地形に沿った火災拡大シミュレーション (3D DEM版)", layout="wide")

# --- API設定 ---
MAPBOX_TOKEN = st.secrets["mapbox"]["access_token"]
OPENWEATHER_API_KEY = st.secrets["openweather"]["api_key"]
GEMINI_API_KEY = st.secrets["general"]["api_key"]

# Gemini API設定
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- サイドバー設定 ---
st.sidebar.header("設定")
city = st.sidebar.text_input("都市名を入力", "松山市")
radius_default = st.sidebar.slider("初期延焼半径 (m)", 100, 1000, 500, step=50)
wind_direction_default = st.sidebar.slider("風向 (度)", 0, 360, 45, step=5)

# 初期座標（松山市）
lat_center, lon_center = 33.8392, 132.7657

# --- 標高データ取得関数 ---
def get_elevation(lat, lon):
    zoom = 14
    tile_x = int((lon + 180) / 360 * 2**zoom)
    tile_y = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * 2**zoom)
    url = f"https://api.mapbox.com/v4/mapbox.terrain-rgb/{zoom}/{tile_x}/{tile_y}.pngraw?access_token={MAPBOX_TOKEN}"
    response = requests.get(url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        img_array = np.array(img)[128, 128, :3].astype(np.int32)
        elevation = -10000 + ((img_array[0] * 256 * 256 + img_array[1] * 256 + img_array[2]) * 0.1)
        return elevation
    return 0

# --- OpenWeatherMapから天気データ取得 ---
def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&lang=ja&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return {}

# --- Geminiによる延焼範囲予測 ---
def predict_fire_spread(city, weather):
    prompt = f"""
    地形と以下の気象条件に基づき、{city}周辺の火災延焼半径を予測してください。

    気象条件:
    - 気温: {weather['main']['temp']}℃
    - 風速: {weather['wind']['speed']} m/s
    - 風向: {weather['wind']['deg']}度
    - 天気: {weather['weather'][0]['description']}

    延焼半径のみ数字で回答してください（単位はm）。
    """
    response = model.generate_content(prompt)
    try:
        radius = int(response.text.strip())
    except ValueError:
        radius = radius_default
    return radius, weather['wind']['deg']

# --- 延焼範囲のポリゴン生成関数（地形対応） ---
def generate_terrain_polygon(lat, lon, radius, wind_dir_deg):
    coords = []
    num_steps = 36
    deg_per_meter = 1.0 / 111000.0
    for i in range(num_steps + 1):
        angle_deg = wind_dir_deg - 90 + 180 * (i / num_steps)
        angle_rad = math.radians(angle_deg)
        dlat = radius * math.cos(angle_rad) * deg_per_meter
        dlon = radius * math.sin(angle_rad) * deg_per_meter
        plat, plon = lat + dlat, lon + dlon
        elev = get_elevation(plat, plon)
        coords.append([plon, plat, elev * 0.7])
    return coords

# --- 天気情報の取得とGemini予測 ---
weather_data = get_weather(city)
if weather_data:
    predicted_radius, predicted_wind_direction = predict_fire_spread(city, weather_data)
else:
    predicted_radius, predicted_wind_direction = radius_default, wind_direction_default

# --- Pydeckでの3D表示 ---
terrain_polygon = generate_terrain_polygon(lat_center, lon_center, predicted_radius, predicted_wind_direction)
polygon_layer = pdk.Layer(
    "PolygonLayer",
    [{"polygon": terrain_polygon}],
    get_polygon="polygon",
    extruded=True,
    get_fill_color=[255, 100, 0, 160],
    elevation_scale=1,
    pickable=True,
    auto_highlight=True,
)
terrain_layer = pdk.Layer(
    "TerrainLayer",
    data=f"https://api.mapbox.com/v4/mapbox.terrain-rgb/{{z}}/{{x}}/{{y}}.pngraw?access_token={MAPBOX_TOKEN}",
    elevation_decoder={"rScaler":256,"gScaler":256,"bScaler":256,"offset":-10000},
    elevation_scale=1,
)

# --- 天気情報の表示 ---
if weather_data:
    st.sidebar.subheader("現在の気象情報")
    st.sidebar.write(f"天気: {weather_data['weather'][0]['description']}")
    st.sidebar.write(f"気温: {weather_data['main']['temp']} ℃")
    st.sidebar.write(f"風速: {weather_data['wind']['speed']} m/s")

# --- 表示 ---
st.pydeck_chart(pdk.Deck(
    layers=[terrain_layer, polygon_layer],
    initial_view_state=pdk.ViewState(latitude=lat_center, longitude=lon_center, zoom=13, pitch=45),
    map_style="mapbox://styles/mapbox/satellite-streets-v11",
))
