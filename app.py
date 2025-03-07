import streamlit as st
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point
import geopandas as gpd
import openai
import requests

# ページ設定
st.set_page_config(page_title="火災拡大シミュレーション", layout="wide")

# サイドバー：APIキーとモデル名の設定
st.sidebar.title("設定")
API_KEY = st.sidebar.text_input("APIキーを入力してください", type="password")
MODEL_NAME = "gemini-2.0-flash-001"

# サイドバー：火災発生地点の入力
st.sidebar.title("火災発生地点の入力")
with st.sidebar.form(key='location_form'):
    lat = st.number_input("緯度", format="%f")
    lon = st.number_input("経度", format="%f")
    add_point = st.form_submit_button("発生地点を追加")
    if add_point:
        if 'points' not in st.session_state:
            st.session_state.points = []
        st.session_state.points.append((lat, lon))
        st.sidebar.success(f"地点 ({lat}, {lon}) を追加しました。")

# メインエリア：タイトル
st.title("火災拡大シミュレーション")

# 地図の表示
m = folium.Map(location=[35.681236, 139.767125], zoom_start=12)

# 発生地点の表示
if 'points' in st.session_state:
    for point in st.session_state.points:
        folium.Marker(location=point, icon=folium.Icon(color='red')).add_to(m)

# 地図を表示
st_folium(m, width=700, height=500)

# 気象データの取得関数
def get_weather(lat=35.681236, lng=139.767125):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&current_weather=true"
    response = requests.get(url)
    data = response.json()
    return {
        'wind_speed': data['current_weather']['windspeed'],
        'wind_direction': data['current_weather']['winddirection']
    }

# 消火水量の計算関数
def calculate_water_volume(area_sqm):
    # 1平方メートルあたり0.5立方メートルの水を必要とする（仮定）
    water_volume_cubic_m = area_sqm * 0.5
    water_volume_tons = water_volume_cubic_m  # 1立方メートル = 1トン
    return water_volume_tons

# 火災拡大予測関数
def predict_fire_spread(points, weather, duration_hours, api_key, model_name):
    # OpenAI APIの設定
    openai.api_key = api_key

    # 発生地点の座標を文字列に変換
    points_str = ', '.join([f"({lat}, {lon})" for lat, lon in points])

    # プロンプトの作成
    prompt = f"""
    以下の条件で火災の炎症範囲を予測してください。

    発生地点: {points_str}
    風速: {weather['wind_speed']} m/s
    風向: {weather['wind_direction']}度
    時間経過: {duration_hours} 時間

    以下を算出してください:
    1. 予測される炎症範囲の半径（メートル）
    2. 炎症範囲のおおよその面積（平方メートル）
    3. 必要な消火水量（トン）

    出力はJSON形式で以下のように返してください：
    {{
        "radius_m": 値,
        "area_sqm": 値,
        "water_volume_tons": 値
    }}
    """

    # OpenAI APIを使用して予測を取得
    response = openai.Completion.create(
        engine=model_name,
        prompt=prompt,
        max_tokens=150
    )

    # 応答の解析
    prediction = response.choices[0].text.strip()
    prediction_json = eval(prediction)  # 注意：evalの使用はセキュリティ上のリスクがあります。安全な方法で解析してください。

    # 地理的範囲の作成（簡易円形）
    gdf_points = gpd.GeoSeries([Point(lon, lat) for lat, lon in points], crs="EPSG:4326")
    centroid = gdf_points.unary_union.centroid
    buffer = centroid.buffer(prediction_json['radius_m'] / 111000)  # 簡易緯度経度変換

    area_coordinates = [(coord[1], coord[0]) for coord in buffer.exterior.coords]

    return {
        'radius_m': prediction_json['radius_m'],
        'area_sqm': prediction_json['area_sqm'],
        'water_volume_tons': prediction_json['water_volume_tons'],
        'area_coordinates': area_coordinates
    }

# 気象データの取得
if st.button("気象データ取得"):
    if 'points' in st.session_state and len(st.session_state.points) > 0:
        lat, lon = st.session_state.points[0]  # 最初の地点の気象データを取得
        weather_data = get_weather(lat, lon)
        st.session_state.weather_data = weather_data
        st.write(f"取得した気象データ: {weather_data}")
    else:
        st.warning("発生地点を追加してください。")

# 消火なしシミュレーションの設定
st.write("## 消火活動が行われない場合のシミュレーション")

tab1, tab2, tab3 = st.tabs(["日単位", "週単位", "月単位"])

with tab1:
    days = st.slider("日数を選択", 1, 30, 1)
    duration_hours = days * 24
    unit = "日"

with tab2:
    weeks = st.slider("週数を選択", 1, 52, 1)
    duration_hours = weeks * 7 * 24
    unit = "週"

with tab3:
    months = st.slider("月数を選択", 1, 12, 1)
    duration_hours = months * 30 * 24
    unit = "月"

# シミュレーションの
::contentReference[oaicite:4]{index=4}
 
