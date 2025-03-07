import streamlit as st
import folium
from streamlit_folium import st_folium
from utils import get_weather, predict_fire_spread, calculate_water_volume
from shapely.geometry import Point, Polygon
import geopandas as gpd

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

# シミュレーションの実行
if st.button("シミュレーション実行"):
    if 'weather_data' not in st.session_state or len(st.session_state.points) == 0:
        st.warning("発生地点と気象データが必要です。")
    else:
        prediction = predict_fire_spread(
            points=st.session_state.points,
            weather=st.session_state.weather_data,
            duration_hours=duration_hours,
            api_key=API_KEY,
            model_name=MODEL_NAME
        )
        st.write(f"予測結果（{days if unit == '日' else weeks if unit == '週' else months} {unit}後）:")
        st.write(f"拡大範囲の半径: {prediction['radius_m']:.2f} m")
        st.write(f"拡大面積: {prediction['area_sqm']:.2f} 平方メートル")
        st.write(f"必要な消火水量: {prediction['water_volume_tons']:.2f} トン")

        # 範囲を地図に表示
        folium.Polygon(prediction['area_coordinates'], color="red", fill=True, fill_opacity=0.5).add_to(m)

        # 地図の再表示
        st_folium(m, width=700, height=500)

