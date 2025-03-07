import streamlit as st
import folium
from streamlit_folium import st_folium
from utils import get_weather, predict_fire_spread, calculate_water_volume
from shapely.geometry import Point, Polygon
import geopandas as gpd

st.set_page_config(page_title="火災拡大シミュレーション", layout="wide")

st.title("火災拡大シミュレーション")

# 地図の表示
m = folium.Map(location=[35.681236, 139.767125], zoom_start=12)
map_data = st_folium(m, width=700, height=500)

# 発火地点の取得
if 'points' not in st.session_state:
    st.session_state.points = []

if map_data['last_clicked']:
    latlng = map_data['last_clicked']
    st.session_state.points.append((latlng['lat'], latlng['lng']))

st.write(f"発生地点: {st.session_state.points}")

# 気象データの取得
if st.button("気象データ取得"):
    weather_data = get_weather()
    st.session_state.weather_data = weather_data
    st.write(f"取得した気象データ: {weather_data}")

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
            duration_hours=duration_hours
        )
        st.write(f"予測結果（{duration_hours/24} {unit}後）:")
        st.write(f"拡大範囲の半径: {prediction['radius_m']} m")
        st.write(f"拡大面積: {prediction['area_sqm']} 平方メートル")
        st.write(f"必要な消火水量: {prediction['water_volume_tons']} トン")

        # 範囲を地図に表示
        folium.Polygon(prediction['area_coordinates'], color="red", fill=True, fill_opacity=0.5).add_to(m)

# 地図の再表示
st_folium(m, width=700, height=500)
