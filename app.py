import streamlit as st
import requests
import math
import pydeck as pdk
import numpy as np
from io import BytesIO
from PIL import Image
import google.generativeai as genai
import time

# --- ページ設定 ---
st.set_page_config(page_title="火災拡大シミュレーション (Gemini & DEM)", layout="wide")

# --- API設定 ---
MAPBOX_TOKEN = st.secrets["mapbox"]["access_token"]
OPENWEATHER_API_KEY = st.secrets["openweather"]["api_key"]
GEMINI_API_KEY = st.secrets["general"]["api_key"]

# Gemini API設定
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- サイドバー設定 ---
st.sidebar.header("火災発生地点の設定")
city = st.sidebar.text_input("都市名を入力", "松山市")
lat_center = st.sidebar.number_input("緯度", value=33.8392, format="%.6f")
lon_center = st.sidebar.number_input("経度", value=132.7657, format="%.6f")

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

# --- 天気データ取得関数 ---
def get_weather(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&lang=ja&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return {}

# --- Geminiによる延焼範囲予測 ---
def predict_fire_spread(lat, lon, weather):
    prompt = f"""
    緯度:{lat}, 経度:{lon}の地点の地形と以下の気象条件に基づき、火災延焼半径を予測してください。

    気象条件:
    - 気温: {weather['main']['temp']}℃
    - 風速: {weather['wind']['speed']} m/s
    - 風向: {weather['wind']['deg']}度
    - 天気: {weather['weather'][0]['description']}

    延焼半径を数字のみで回答してください（単位はm）。
    """
    response = model.generate_content(prompt)
    try:
        radius = int(response.text.strip())
    except ValueError:
        radius = 500
    return radius, weather['wind']['deg']

# --- ポリゴン生成 ---
def generate_polygon(lat, lon, radius, wind_dir_deg):
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
        coords.append([plon, plat, elev * 0.5])  # 表示高さを調整
    return coords

# --- アニメーション表示関数 ---
def animate_fire(lat, lon, radius, wind_dir):
    steps = 20
    for r in np.linspace(0, radius, steps):
        terrain_polygon = generate_polygon(lat, lon, r, wind_dir)
        polygon_layer = pdk.Layer(
            "PolygonLayer",
            [{"polygon": terrain_polygon}],
            extruded=True,
            get_fill_color=[255, 100, 0, 160],
            elevation_scale=1,
        )
        view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=13, pitch=45)
        deck = pdk.Deck(layers=[polygon_layer], initial_view_state=view_state, map_style="mapbox://styles/mapbox/satellite-streets-v11")
        map_area.pydeck_chart(deck)
        time.sleep(0.1)

# --- メイン処理 ---
if st.button("火災シミュレーション開始"):
    weather_data = get_weather(lat_center, lon_center)
    if weather_data:
        predicted_radius, wind_direction = predict_fire_spread(lat_center, lon_center, weather_data)
        st.sidebar.subheader("現在の気象情報")
        st.sidebar.write(f"天気: {weather_data['weather'][0]['description']}")
        st.sidebar.write(f"気温: {weather_data['main']['temp']} ℃")
        st.sidebar.write(f"風速: {weather_data['wind']['speed']} m/s")
        st.sidebar.write(f"予測延焼半径: {predicted_radius} m")
        map_area = st.empty()
        animate_fire(lat_center, lon_center, predicted_radius, wind_direction)
    else:
        st.error("気象データの取得に失敗しました。")
