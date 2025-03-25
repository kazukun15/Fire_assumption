import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import pydeck as pdk
import time

st.set_page_config(page_title="火災拡大シミュレーション（全修正版）", layout="wide")

# デフォルトの座標
default_lat = 34.257493583590986
default_lon = 133.20437169456872

# セッションステートの初期化
if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = None
if 'points' not in st.session_state:
    st.session_state.points = [(default_lat, default_lon)]

# サイドバー
st.sidebar.header("🔥 シミュレーション設定")
fuel_type = st.sidebar.selectbox("燃料タイプ", ["森林", "草地", "都市部"])
scenario = st.sidebar.selectbox("消火シナリオ", ["通常の消火活動あり", "消火活動なし"])
map_style = st.sidebar.selectbox("地図スタイル", ["カラー", "ダーク"])
show_raincloud = st.sidebar.checkbox("雨雲オーバーレイを表示", value=False)

# 発生地点設定（追加）
st.sidebar.header("📍 発生地点の設定")
lat = st.sidebar.number_input("緯度", value=default_lat, format="%.6f")
lon = st.sidebar.number_input("経度", value=default_lon, format="%.6f")
if st.sidebar.button("発生地点を設定"):
    st.session_state.points = [(lat, lon)]
    st.success("発生地点を更新しました。")

# メインタイトル
st.title("🔥 火災拡大シミュレーション")

# 初期地図（必ず表示）
initial_map = folium.Map(location=st.session_state.points[0], zoom_start=13, control_scale=True)
folium.CircleMarker(location=st.session_state.points[0], radius=5, color="red", popup="発生地点").add_to(initial_map)
st.subheader("📍 初期地図表示（シミュレーション前）")
st_folium(initial_map, width=700, height=500)

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

# シミュレーション実行（1つだけ表示）
if st.button("シミュレーション実行"):
    st.session_state.simulation_run = True

    # シミュレーション結果 (API結果を想定)
    radius_m = 500
    area_ha = 3.14 * radius_m**2 / 10000
    water_volume_tons = area_ha * 5

    # シミュレーションマップ
    sim_map = folium.Map(location=st.session_state.points[0], zoom_start=13, control_scale=True)
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

    # 雨雲オーバーレイ（オプション）
    if show_raincloud:
        folium.raster_layers.ImageOverlay(
            image="https://tile.openweathermap.org/map/clouds_new/10/900/380.png?appid=YOUR_KEY",
            bounds=[[lat-0.05, lon-0.05], [lat+0.05, lon+0.05]],
            opacity=0.4
        ).add_to(sim_map)

    # シミュレーション地図表示
    st.subheader("🔥 シミュレーション結果表示")
    st_folium(sim_map, width=700, height=500)

    # DEM 3D表示 (pydeck)
    MAPBOX_TOKEN = st.secrets["mapbox"]["access_token"]
    terrain_layer = pdk.Layer(
        "TerrainLayer",
        data=f"https://api.mapbox.com/v4/mapbox.terrain-rgb/{{z}}/{{x}}/{{y}}.pngraw?access_token={MAPBOX_TOKEN}",
        minZoom=0, maxZoom=15,
        meshMaxError=4,
        elevationDecoder={"rScaler":6553.6,"gScaler":25.6,"bScaler":0.1,"offset":-10000},
        elevationScale=1
    )
    view_state = pdk.ViewState(
        latitude=lat, longitude=lon, zoom=13, pitch=45, bearing=0
    )
    deck = pdk.Deck(
        layers=[terrain_layer], initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/satellite-streets-v11" if map_style=="カラー" else "mapbox://styles/mapbox/dark-v10",
        mapbox_key=MAPBOX_TOKEN
    )
    st.subheader("🗺️ 地形（DEM）3D表示")
    st.pydeck_chart(deck)

    # 詳細レポート
    st.subheader("📃 詳細レポート")
    st.markdown(f"""
    - 燃料タイプ：{fuel_type}
    - 消火シナリオ：{scenario}
    - 延焼半径：{radius_m}m
    - 推定延焼面積：{area_ha:.2f} ha
    - 必要消火水量：{water_volume_tons:.2f}トン
    """)

    # アニメーション表示
    st.subheader("▶️ 延焼範囲アニメーション")
    animation_placeholder = st.empty()
    for r in range(0, radius_m+1, 50):
        anim_map = folium.Map(location=st.session_state.points[0], zoom_start=13, control_scale=True)
        folium.CircleMarker(location=st.session_state.points[0], radius=5, color="red").add_to(anim_map)
        folium.Circle(location=st.session_state.points[0], radius=r, color='orange', fill=True, fill_opacity=0.5).add_to(anim_map)

        with animation_placeholder.container():
            st_folium(anim_map, width=700, height=500)
        time.sleep(0.1)

# シミュレーション済みの場合の再表示（sim_mapを存在確認してから）
if st.session_state.simulation_run and 'sim_map' in locals():
    st.info("前回のシミュレーション結果再表示中")
    st_folium(sim_map, width=700, height=500)
    st.pydeck_chart(deck)
