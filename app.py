import streamlit as st
import folium
from streamlit_folium import st_folium
from utils import get_weather, predict_fire_spread
from shapely.geometry import Point, Polygon
import geopandas as gpd

st.title("火災拡大シミュレーション")

# タブの作成
tabs = st.tabs(["日単位", "週単位", "月単位"])

# スライダーの設定
with tabs[0]:
    days = st.slider("日数を選択", 1, 30, 1)
with tabs[1]:
    weeks = st.slider("週数を選択", 1, 52, 1)
with tabs[2]:
    months = st.slider("月数を選択", 1, 12, 1)

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

# シミュレーションの実行
if st.button("シミュレーション実行"):
    if 'weather_data' not in st.session_state or len(st.session_state.points) == 0:
        st.warning("発生地点と気象データが必要です。")
    else:
        # 選択された時間単位に応じて期間を設定
        if 'days' in locals():
            duration_hours = days * 24
        elif 'weeks' in locals():
            duration_hours = weeks * 7 * 24
        elif 'months' in locals():
            duration_hours = months * 30 * 24

        prediction = predict_fire_spread(
            points=st.session_state.points,
            weather=st.session_state.weather_data,
            duration_hours=duration_hours
        )
        st.write("予測結果:")
        st.write(f"拡大範囲の半径: {prediction['radius_m']} m")
        st.write(f"拡大面積: {prediction['area_sqm']} 平方メートル")

        # 範囲を地図に表示
        folium.Polygon(prediction['area_coordinates'], color="red", fill=True, fill_opacity=0.5).add_to(m)

# 地図の再表示
st_folium(m, width=700, height=500)
