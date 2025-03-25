import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import pydeck as pdk
import time

st.set_page_config(page_title="🔥 火災拡大シミュレーション（全機能版）", layout="wide")

# デフォルト座標
default_lat = 34.257493583590986
default_lon = 133.20437169456872

# セッションステート初期化
state_defaults = {
    'simulation_run': False,
    'weather_data': None,
    'points': [(default_lat, default_lon)],
    'map_center': [default_lat, default_lon],
    'map_zoom': 13
}
for key, val in state_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# サイドバー設定
st.sidebar.header("🔥 シミュレーション設定")
fuel_type = st.sidebar.selectbox("燃料タイプ", ["森林", "草地", "都市部"])
scenario = st.sidebar.selectbox("消火シナリオ", ["通常の消火活動あり", "消火活動なし"])
map_style = st.sidebar.selectbox("地図スタイル", ["カラー", "ダーク"])
show_raincloud = st.sidebar.checkbox("雨雲オーバーレイを表示", value=False)

st.sidebar.header("📍 発生地点設定")
lat = st.sidebar.number_input("緯度", value=default_lat, format="%.6f")
lon = st.sidebar.number_input("経度", value=default_lon, format="%.6f")
if st.sidebar.button("発生地点を設定"):
    st.session_state.points = [(lat, lon)]
    st.session_state.map_center = [lat, lon]
    st.success("発生地点を更新しました。")

# メインタイトル
st.title("🔥 火災拡大シミュレーション（全機能版）")

# 基本地図生成（初期状態と再読み込み）
m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom, control_scale=True)
folium.CircleMarker(location=st.session_state.points[0], radius=5, color="red", popup="発生地点").add_to(m)

# 初期地図表示（状態保持対応）
map_data = st_folium(m, width=700, height=500)
if map_data and map_data.get("last_center") and map_data.get("last_zoom"):
    st.session_state.map_center = [
        map_data["last_center"]["lat"], 
        map_data["last_center"]["lng"]
    ]
    st.session_state.map_zoom = map_data["last_zoom"]

# 気象データ取得
if st.button("気象データ取得"):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        response = requests.get(url).json()
        st.session_state.weather_data = response.get("current_weather", {})
        st.success("気象データ取得成功")
        st.write(st.session_state.weather_data)
    except Exception as e:
        st.error(f"気象データ取得エラー: {e}")

# シミュレーション実行
if st.button("シミュレーション実行"):
    st.session_state.simulation_run = True

    # 仮のシミュレーション結果
    radius_m = 500
    area_ha = 3.14 * radius_m**2 / 10000
    water_volume_tons = area_ha * 5

    # シミュレーション結果マップ
    sim_map = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom, control_scale=True)
    folium.CircleMarker(
        location=st.session_state.points[0], radius=5, color="red", popup="発生地点"
    ).add_to(sim_map)
    folium.Circle(
        location=st.session_state.points[0],
        radius=radius_m,
        color="blue",
        fill=True,
        fill_opacity=0.4,
        popup=f"延焼範囲: {radius_m}m"
    ).add_to(sim_map)

    # 雨雲オーバーレイ
    if show_raincloud:
        folium.raster_layers.ImageOverlay(
            image="https://tile.openweathermap.org/map/clouds_new/10/900/380.png?appid=YOUR_API_KEY",
            bounds=[[lat-0.05, lon-0.05], [lat+0.05, lon+0.05]],
            opacity=0.4
        ).add_to(sim_map)

    # 結果マップ表示（状態保持対応）
    sim_map_data = st_folium(sim_map, width=700, height=500)
    if sim_map_data and sim_map_data.get("last_center") and sim_map_data.get("last_zoom"):
        st.session_state.map_center = [
            sim_map_data["last_center"]["lat"], 
            sim_map_data["last_center"]["lng"]
        ]
        st.session_state.map_zoom = sim_map_data["last_zoom"]

    # DEM 3D表示（pydeck）
    MAPBOX_TOKEN = st.secrets["mapbox"]["access_token"]
    terrain_layer = pdk.Layer(
        "TerrainLayer",
        data=f"https://api.mapbox.com/v4/mapbox.terrain-rgb/{{z}}/{{x}}/{{y}}.pngraw?access_token={MAPBOX_TOKEN}",
        elevationDecoder={"rScaler":6553.6,"gScaler":25.6,"bScaler":0.1,"offset":-10000},
        elevationScale=1,
        meshMaxError=4,
        minZoom=0, maxZoom=15
    )
    view_state = pdk.ViewState(
        latitude=lat, longitude=lon, zoom=st.session_state.map_zoom, pitch=45
    )
    deck = pdk.Deck(
        layers=[terrain_layer], initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/satellite-streets-v11" if map_style=="カラー" else "mapbox://styles/mapbox/dark-v10",
        mapbox_key=MAPBOX_TOKEN
    )
    st.subheader("🗺️ 地形3D表示")
    st.pydeck_chart(deck)

    # 詳細レポート表示
    st.subheader("📃 詳細レポート")
    st.markdown(f"""
    - 燃料タイプ：{fuel_type}
    - 消火シナリオ：{scenario}
    - 延焼半径：{radius_m}m
    - 延焼面積：{area_ha:.2f} ha
    - 必要消火水量：{water_volume_tons:.2f} トン
    """)

    # アニメーション表示
    st.subheader("▶️ 延焼範囲アニメーション")
    placeholder = st.empty()
    for r in range(0, radius_m+1, 50):
        anim_map = folium.Map(location=st.session_state.points[0], zoom_start=st.session_state.map_zoom, control_scale=True)
        folium.CircleMarker(location=st.session_state.points[0], radius=5, color="red").add_to(anim_map)
        folium.Circle(location=st.session_state.points[0], radius=r, color='orange', fill=True, fill_opacity=0.5).add_to(anim_map)
        with placeholder.container():
            st_folium(anim_map, width=700, height=500)
        time.sleep(0.1)

# シミュレーション後の再表示 (再読み込み時)
elif st.session_state.simulation_run:
    st.info("前回のシミュレーション結果を表示中")
    st_folium(sim_map, width=700, height=500)
    st.pydeck_chart(deck)
