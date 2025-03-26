import streamlit as st
import requests
import math
import pydeck as pdk
import numpy as np
from io import BytesIO
from PIL import Image
import google.generativeai as genai
import re

# --- ページ設定 ---
st.set_page_config(page_title="火災拡大シミュレーション (Gemini & DEM)", layout="wide")

# --- API設定 ---
MAPBOX_TOKEN = st.secrets["mapbox"]["access_token"]
OPENWEATHER_API_KEY = st.secrets["openweather"]["api_key"]
GEMINI_API_KEY = st.secrets["general"]["api_key"]

# Gemini API設定
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# セッションステートの初期化
if 'points' not in st.session_state:
    st.session_state.points = [(34.25743760177552, 133.2043209338966)]  # 初期位置を設定

# --- サイドバー設定 ---
st.sidebar.header("火災発生地点の設定")
with st.sidebar.form(key='location_form'):
    lat_input = st.number_input("緯度", format="%.6f", value=34.257438)
    lon_input = st.number_input("経度", format="%.6f", value=133.204321)
    add_point = st.form_submit_button("発生地点を設定")
    if add_point:
        st.session_state.points = [(lat_input, lon_input)]
        st.sidebar.success(f"地点 ({lat_input}, {lon_input}) を設定しました。")

fuel_options = {"森林（高燃料）": "森林", "草地（中燃料）": "草地", "都市部（低燃料）": "都市部"}
selected_fuel = st.sidebar.selectbox("燃料特性を選択してください", list(fuel_options.keys()))
fuel_type = fuel_options[selected_fuel]

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
    else:
        st.error(f"気象データの取得に失敗しました。ステータスコード: {response.status_code}, 詳細: {response.text}")
        return {}

# --- Geminiによる延焼範囲予測とレポート作成 ---
def predict_fire_spread(lat, lon, weather, fuel_type):
    prompt = f"""
    地点の緯度:{lat}, 経度:{lon}, 気象条件: 気温:{weather['main']['temp']}℃, 風速:{weather['wind']['speed']}m/s, 風向:{weather['wind']['deg']}度, 天気:{weather['weather'][0]['description']}, 燃料特性:{fuel_type}。
    以下を含む火災延焼予測レポートを作成してください。
    - 延焼半径（m）
    - 延焼範囲（㎡）
    - 必要な放水水量（トン）
    - 推奨される消火設備
    - 予測される火災状況
    """
    response = model.generate_content(prompt)
    return response.text.strip()

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
        coords.append([plon, plat, elev * 0.5])
    return coords

# --- メイン処理 ---
st.title("火災拡大シミュレーション")

lat, lon = st.session_state.points[0]
initial_view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=13, pitch=45)
st.pydeck_chart(pdk.Deck(layers=[], initial_view_state=initial_view_state, map_style="mapbox://styles/mapbox/satellite-streets-v11"))

if st.button("シミュレーション開始"):
    weather_data = get_weather(lat, lon)
    if weather_data:
        report = predict_fire_spread(lat, lon, weather_data, fuel_type)
        st.markdown("### 火災延焼予測レポート")
        st.markdown(report)

        radius_match = re.search(r"延焼半径.*?(\d+)m", report)
        radius = int(radius_match.group(1)) if radius_match else 500

        polygon = generate_polygon(lat, lon, radius, weather_data['wind']['deg'])
        layer = pdk.Layer("PolygonLayer", [{"polygon": polygon}], extruded=True, get_fill_color=[255, 0, 0, 100])
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=initial_view_state, map_style="mapbox://styles/mapbox/satellite-streets-v11"))

        st.sidebar.subheader("現在の気象情報")
        st.sidebar.json(weather_data)
