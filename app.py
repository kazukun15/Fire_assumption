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
    latlon_input = st.text_input("緯度, 経度 (Google Map形式で貼り付けてください)", value="34.257438, 133.204321")
    add_point = st.form_submit_button("発生地点を設定")
    if add_point:
        try:
            lat_str, lon_str = latlon_input.split(',')
            lat_input = float(lat_str.strip())
            lon_input = float(lon_str.strip())
            st.session_state.points = [(lat_input, lon_input)]
            st.sidebar.success(f"地点 ({lat_input}, {lon_input}) を設定しました。")
        except ValueError:
            st.sidebar.error("有効な緯度経度を入力してください（例：34.257438, 133.204321）。")

fuel_options = {"森林（高燃料）": "森林", "草地（中燃料）": "草地", "都市部（低燃料）": "都市部"}
selected_fuel = st.sidebar.selectbox("燃料特性を選択してください", list(fuel_options.keys()))
fuel_type = fuel_options[selected_fuel]

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
def predict_fire_spread(lat, lon, weather, fuel_type, days):
    prompt = f"""
    地点の緯度:{lat}, 経度:{lon}, 気象条件: 気温:{weather['main']['temp']}℃, 風速:{weather['wind']['speed']}m/s, 風向:{weather['wind']['deg']}度, 天気:{weather['weather'][0]['description']}, 燃料特性:{fuel_type}, 発生からの日数:{days}日。
    以下を含む火災延焼予測レポートを作成してください。
    - 延焼半径（m）
    - 延焼範囲（㎡）
    - 必要な放水水量（トン）
    - 推奨される消火設備
    - 予測される火災状況
    """
    response = model.generate_content(prompt)
    return response.text.strip()

# --- 円形ポリゴン生成 ---
def generate_circle_polygon(lat, lon, radius):
    coords = []
    num_steps = 72
    deg_per_meter = 1.0 / 111000.0
    for i in range(num_steps + 1):
        angle_rad = 2 * math.pi * (i / num_steps)
        dlat = radius * math.cos(angle_rad) * deg_per_meter
        dlon = radius * math.sin(angle_rad) * deg_per_meter
        coords.append([lon + dlon, lat + dlat])
    return coords

# --- メイン処理 ---
st.title("火災拡大シミュレーション")

lat, lon = st.session_state.points[0]
initial_view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=13, pitch=0)
marker_layer = pdk.Layer("ScatterplotLayer", data=[{"position": [lon, lat]}], get_position="position", get_color=[255, 0, 0], get_radius=5)

st.sidebar.subheader("シミュレーション日数設定")
days = st.sidebar.slider("日数を選択", 1, 7, 1)

if st.button("シミュレーション開始"):
    weather_data = get_weather(lat, lon)
    if weather_data:
        report = predict_fire_spread(lat, lon, weather_data, fuel_type, days)
        st.markdown("### 火災延焼予測レポート")
        st.markdown(report)

        radius_match = re.search(r"延焼半径.*?(\d+)m", report)
        radius = int(radius_match.group(1)) if radius_match else 500

        polygon = generate_circle_polygon(lat, lon, radius)
        polygon_layer = pdk.Layer("PolygonLayer", [{"polygon": polygon}], extruded=False, get_fill_color=[255, 0, 0, 100])
        st.pydeck_chart(pdk.Deck(layers=[polygon_layer, marker_layer], initial_view_state=initial_view_state, map_style="mapbox://styles/mapbox/satellite-streets-v11"))

        st.sidebar.subheader("現在の気象情報")
        st.sidebar.json(weather_data)

else:
    st.pydeck_chart(pdk.Deck(layers=[marker_layer], initial_view_state=initial_view_state, map_style="mapbox://styles/mapbox/satellite-streets-v11"))
