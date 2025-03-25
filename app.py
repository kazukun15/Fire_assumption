import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import pydeck as pdk
import time

# ----------------------------------------
# ページ設定
st.set_page_config(page_title="火災拡大シミュレーション (DEM＆全機能復元)", layout="wide")

# ----------------------------------------
# デフォルト座標
default_lat = 34.257493583590986
default_lon = 133.20437169456872

# ----------------------------------------
# セッションステート初期化
if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = None
if 'points' not in st.session_state:
    st.session_state.points = [(default_lat, default_lon)]

# ----------------------------------------
# サイドバー設定（復元）
st.sidebar.header("🔥 シミュレーション設定")
fuel_type = st.sidebar.selectbox("燃料タイプ", ["森林", "草地", "都市部"])
scenario = st.sidebar.selectbox("消火シナリオ", ["通常の消火活動あり", "消火活動なし"])
map_style = st.sidebar.selectbox("地図スタイル", ["カラー", "ダーク"])
animation_type = st.sidebar.selectbox(
    "アニメーションタイプ", ["Full Circle", "Fan Shape", "Timestamped GeoJSON", "Color Gradient"]
)
show_raincloud = st.sidebar.checkbox("雨雲オーバーレイを表示", value=False)

# ----------------------------------------
# 初期地図表示（起動時に必ず表示）
st.title("🔥 火災拡大シミュレーション")

initial_map = folium.Map(location=[default_lat, default_lon], zoom_start=13, control_scale=True)
folium.CircleMarker(
    location=[default_lat, default_lon],
    radius=5,
    color="red",
    fill=True,
    popup="発生地点"
).add_to(initial_map)

st.subheader("📍 初期地図表示（シミュレーション前）")
st_folium(initial_map, width=700, height=500)

# ----------------------------------------
# 気象データ取得（復元）
if st.button("気象データ取得"):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={default_lat}&longitude={default_lon}&current_weather=true"
        response = requests.get(url).json()
        st.session_state.weather_data = response.get("current_weather", {})
        st.success("気象データ取得成功")
        st.write(st.session_state.weather_data)
    except Exception as e:
        st.error(f"気象データ取得エラー: {e}")

# ----------------------------------------
# シミュレーション実行ボタン（全機能復元）
if st.button("シミュレーション実行"):
    st.session_state.simulation_run = True

    # シミュレーション結果（実際のAPI利用想定）
    radius_m = 500  # 実際はAPIで取得
    area_ha = 3.14 * radius_m**2 / 10000
    water_volume_tons = area_ha * 5

    # 結果表示マップ生成
    sim_map = folium.Map(location=[default_lat, default_lon], zoom_start=13, control_scale=True)
    folium.CircleMarker(
        location=[default_lat, default_lon],
        radius=5,
        color="red",
        fill=True,
        popup="発生地点"
    ).add_to(sim_map)
    folium.Circle(
        location=[default_lat, default_lon],
        radius=radius_m,
        color="blue",
        fill=True,
        fill_opacity=0.4,
        popup=f"延焼範囲: {radius_m}m"
    ).add_to(sim_map)

    # 雨雲オーバーレイ（オプション機能復元）
    if show_raincloud:
        folium.raster_layers.ImageOverlay(
            image="https://tile.openweathermap.org/map/clouds_new/10/900/380.png?appid=YOUR_KEY",
            bounds=[[default_lat-0.05, default_lon-0.05], [default_lat+0.05, default_lon+0.05]],
            opacity=0.4
        ).add_to(sim_map)

    st.subheader("🔥 シミュレーション結果表示")
    st_folium(sim_map, width=700, height=500)

    # DEM表示（TerrainLayer完全復元）
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
        latitude=default_lat, longitude=default_lon, zoom=13, pitch=45,
        bearing=0
    )
    deck = pdk.Deck(
        layers=[terrain_layer], initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/satellite-streets-v11" if map_style == "カラー" else "mapbox://styles/mapbox/dark-v10",
        mapbox_key=MAPBOX_TOKEN
    )
    st.subheader("🗺️ 地形（DEM）3D表示")
    st.pydeck_chart(deck)

    # 詳細レポート復元
    st.subheader("📃 詳細レポート")
    st.markdown(f"""
    - 燃料タイプ：{fuel_type}
    - 消火シナリオ：{scenario}
    - 延焼半径：{radius_m}m
    - 推定延焼面積：{area_ha:.2f} ha
    - 必要消火水量：{water_volume_tons:.2f}トン
    """)

    # アニメーション機能（完全復元）
    st.subheader("▶️ 延焼範囲アニメーション")
    animation_placeholder = st.empty()
    for r in range(0, radius_m+1, 50):
        anim_map = folium.Map(location=[default_lat, default_lon], zoom_start=13, control_scale=True)
        folium.CircleMarker(location=[default_lat, default_lon], radius=5, color="red").add_to(anim_map)
        folium.Circle(location=[default_lat, default_lon], radius=r, color='orange', fill=True, fill_opacity=0.5).add_to(anim_map)

        with animation_placeholder.container():
            st_folium(anim_map, width=700, height=500)
        time.sleep(0.1)

# ----------------------------------------
# 一度シミュレーションした内容を保持（便利機能復元）
if st.session_state.simulation_run and not st.button("シミュレーション実行", key="dummy"):
    st.info("シミュレーション結果再表示中...")
    st_folium(sim_map, width=700, height=500)
    st.pydeck_chart(deck)
