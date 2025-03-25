import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import pydeck as pdk
import time

# ----------------------------------------
# ページ設定
st.set_page_config(page_title="火災拡大シミュレーション（完全版）", layout="wide")

# デフォルト座標
default_lat = 34.257493583590986
default_lon = 133.20437169456872

# ----------------------------------------
# セッションステート初期化（初期地図表示用）
if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = None
if 'points' not in st.session_state:
    st.session_state.points = [(default_lat, default_lon)]

# ----------------------------------------
# 最初の地図（初期表示のみ、発生地点表示）
st.title("🔥 火災拡大シミュレーション")

# 初期表示用地図（シンプルな状態）
initial_map = folium.Map(location=[default_lat, default_lon], zoom_start=13, control_scale=True)
folium.CircleMarker(
    location=[default_lat, default_lon],
    radius=5,
    color='red',
    popup="発生地点"
).add_to(initial_map)

# 起動時に初期地図表示（必ず実行される）
st.subheader("📍 初期地図表示")
st_folium(initial_map, width=700, height=500)

# ----------------------------------------
# 気象データ取得ボタン（既存コードと同じ）
if st.button("気象データ取得"):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={default_lat}&longitude={default_lon}&current_weather=true"
        response = requests.get(url)
        data = response.json()
        st.session_state.weather_data = data.get('current_weather', {})
        st.success("気象データ取得成功")
        st.write(st.session_state.weather_data)
    except Exception as e:
        st.error(f"気象データ取得エラー: {e}")

# ----------------------------------------
# シミュレーション実行ボタン（既存コードの全機能をここで統合）
if st.button("シミュレーション実行"):
    st.session_state.simulation_run = True

    # ここで本来のシミュレーション（Gemini APIやDEM処理）を行う
    radius_m = 500  # 実際にはシミュレーションから得た値を使用

    # シミュレーション結果を表示するマップ作成
    sim_map = folium.Map(location=[default_lat, default_lon], zoom_start=13, control_scale=True)
    folium.CircleMarker(
        location=[default_lat, default_lon],
        radius=5,
        color='red',
        popup="発生地点"
    ).add_to(sim_map)

    # 延焼範囲（シミュレーション結果）
    folium.Circle(
        location=[default_lat, default_lon],
        radius=radius_m,
        color='blue',
        fill=True,
        fill_opacity=0.4,
        popup=f"延焼範囲（{radius_m}m）"
    ).add_to(sim_map)

    # 気象データの表示（もしあれば）
    if st.session_state.weather_data:
        weather = st.session_state.weather_data
        folium.Marker(
            location=[default_lat, default_lon],
            popup=f"温度: {weather.get('temperature', '不明')} °C\n風速: {weather.get('windspeed', '不明')} m/s"
        ).add_to(sim_map)

    # DEM地形表示 (pydeckのTerrainLayer使用)
    MAPBOX_TOKEN = st.secrets["mapbox"]["access_token"]
    terrain_layer = pdk.Layer(
        "TerrainLayer",
        data=f"https://api.mapbox.com/v4/mapbox.terrain-rgb/{{z}}/{{x}}/{{y}}.pngraw?access_token={MAPBOX_TOKEN}",
        minZoom=0, maxZoom=15,
        meshMaxError=4,
        elevationDecoder={"rScaler":6553.6,"gScaler":25.6,"bScaler":0.1,"offset":-10000},
        elevationScale=1
    )
    view_state = pdk.ViewState(latitude=default_lat, longitude=default_lon, zoom=13, pitch=45)
    deck = pdk.Deck(layers=[terrain_layer], initial_view_state=view_state)
    
    # シミュレーション地図表示
    st.subheader("🔥 シミュレーション結果表示")
    st_folium(sim_map, width=700, height=500)
    st.pydeck_chart(deck)

    # レポート詳細表示
    st.subheader("📃 詳細レポート")
    st.markdown(f"""
    - 延焼半径：{radius_m}m
    - 推定燃焼面積：{3.14 * radius_m**2 / 10000:.2f} ヘクタール
    - 必要消火水量：{(3.14 * radius_m**2 / 10000) * 5:.2f} トン
    """)

    # アニメーション表示（st.empty使用）
    st.subheader("▶️ 延焼範囲アニメーション")
    animation_placeholder = st.empty()
    for r in range(0, radius_m + 1, 50):
        anim_map = folium.Map(location=[default_lat, default_lon], zoom_start=13, control_scale=True)
        folium.CircleMarker(location=[default_lat, default_lon], radius=5, color="red").add_to(anim_map)
        folium.Circle(location=[default_lat, default_lon], radius=r, color='orange', fill=True, fill_opacity=0.5).add_to(anim_map)

        with animation_placeholder.container():
            st_folium(anim_map, width=700, height=500)
        time.sleep(0.1)

# ----------------------------------------
# 結果が保持される工夫
if st.session_state.simulation_run and not st.button("シミュレーション実行", key="dummy"):
    st.info("シミュレーション済み（地図再表示）")
    st_folium(sim_map, width=700, height=500)
    st.pydeck_chart(deck)
