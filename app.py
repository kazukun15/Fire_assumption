import streamlit as st
import folium
from streamlit_folium import st_folium
import google.generativeai as genai
import requests
import json
import math
import re
import pydeck as pdk
from shapely.geometry import Point
import geopandas as gpd
import time

# ここを修正：demjson -> demjson3
import demjson3 as demjson

# ページ設定
st.set_page_config(page_title="火災拡大シミュレーション (3Dカラム版)", layout="wide")

# secretsからAPIキーを取得（[general]セクションのapi_key）
API_KEY = st.secrets["general"]["api_key"]
MODEL_NAME = "gemini-2.0-flash-001"

# google.generativeai (Gemini) の初期設定
genai.configure(api_key=API_KEY)

# セッションステートの初期化
if 'points' not in st.session_state:
    st.session_state.points = []
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = {}

# サイドバー：火災発生地点の入力
st.sidebar.title("火災発生地点の入力")
with st.sidebar.form(key='location_form'):
    lat_input = st.number_input("緯度", format="%.6f", value=34.257586)
    lon_input = st.number_input("経度", format="%.6f", value=133.204356)
    add_point = st.form_submit_button("発生地点を追加")
    if add_point:
        st.session_state.points.append((lat_input, lon_input))
        st.sidebar.success(f"地点 ({lat_input}, {lon_input}) を追加しました。")

# サイドバー：登録地点消去ボタン
if st.sidebar.button("登録地点を消去"):
    st.session_state.points = []
    st.sidebar.info("全ての発生地点を削除しました。")

# サイドバー：燃料特性の選択
st.sidebar.title("燃料特性の選択")
fuel_options = {
    "森林（高燃料）": "森林",
    "草地（中燃料）": "草地",
    "都市部（低燃料）": "都市部"
}
selected_fuel = st.sidebar.selectbox("燃料特性を選択してください", list(fuel_options.keys()))
fuel_type = fuel_options[selected_fuel]

# メインエリア：タイトル
st.title("火災拡大シミュレーション（Gemini要約＋3Dカラム表示）")

# ベースマップの作成（初期位置）
initial_location = [34.257586, 133.204356]
base_map = folium.Map(location=initial_location, zoom_start=12)
for point in st.session_state.points:
    folium.Marker(location=point, icon=folium.Icon(color='red')).add_to(base_map)
st_folium(base_map, width=700, height=500)

# --- 関数定義 ---

def extract_json(text: str) -> dict:
    """
    テキストからJSONオブジェクトを抽出する（多様なパターンに対応）。
    jsonライブラリと demjson の両方でパースを試みる。
    """
    try:
        text = text.strip()
        # まずは不要な文字や改行を削除
        # 正規表現でJSON部分を抽出
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                # 標準のjsonライブラリで解析を試みる
                return json.loads(json_str)
            except json.JSONDecodeError:
                try:
                    # demjsonライブラリで解析を試みる
                    return demjson.decode(json_str)
                except demjson.JSONDecodeError:
                    st.error("demjsonによるJSON解析にも失敗しました。")
                    return {}
        else:
            st.error("JSON文字列が見つかりませんでした。")
            return {}
    except Exception as e:
        st.error(f"JSON抽出中にエラーが発生しました: {e}")
        return {}

def get_weather(lat, lon):
    """
    Open-Meteo APIから指定緯度・経度の気象情報を取得する関数。
    温度、風速、風向、湿度、降水量などの情報を返す。
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&current_weather=true&"
        f"hourly=relativehumidity_2m,precipitation&timezone=auto"
    )
    response = requests.get(url)
    st.write("Open-Meteo API ステータスコード:", response.status_code)
    data = response.json()
    current = data.get("current_weather", {})
    result = {
        'temperature': current.get("temperature"),
        'windspeed': current.get("windspeed"),
        'winddirection': current.get("winddirection"),
        'weathercode': current.get("weathercode")
    }
    current_time = current.get("time")
    if current_time and "hourly" in data:
        times = data["hourly"].get("time", [])
        if current_time in times:
            idx = times.index(current_time)
            result["humidity"] = data["hourly"].get("relativehumidity_2m", [])[idx]
            result["precipitation"] = data["hourly"].get("precipitation", [])[idx]
    return result

def create_half_circle_coords(center_lat, center_lon, radius_m, wind_direction_deg):
    """
    風向きを考慮した半円形（扇形）の座標列を生成（[lon, lat]形式）。
    """
    deg_per_meter = 1.0 / 111000.0
    start_angle = wind_direction_deg - 90
    end_angle = wind_direction_deg + 90
    num_steps = 36
    coords = []
    coords.append([center_lon, center_lat])
    for i in range(num_steps + 1):
        angle_deg = start_angle + (end_angle - start_angle) * i / num_steps
        angle_rad = math.radians(angle_deg)
        offset_y = radius_m * math.cos(angle_rad)
        offset_x = radius_m * math.sin(angle_rad)
        offset_lat = offset_y * deg_per_meter
        offset_lon = offset_x * deg_per_meter
        new_lat = center_lat + offset_lat
        new_lon = center_lon + offset_lon
        coords.append([new_lon, new_lat])
    return coords

def gemini_generate_text(prompt, api_key, model_name):
    """
    Gemini API のエンドポイントにリクエストを送り、テキスト生成を行う。
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    response = requests.post(url, headers=headers, json=data)
    st.write("Gemini API ステータスコード:", response.status_code)
    raw_json = None
    try:
        raw_json = response.json()
    except Exception:
        st.error("Gemini APIレスポンスのJSONパース失敗")
    if response.status_code == 200 and raw_json:
        candidates = raw_json.get("candidates", [])
        if candidates:
            generated_text = candidates[0].get("output", "").strip()
            return generated_text, raw_json
        else:
            return None, raw_json
    else:
        return None, raw_json

def predict_fire_spread(points, weather, duration_hours, api_key, model_name, fuel_type):
    """
    Gemini API を利用して火災拡大予測を行う関数。
    """
    rep_lat, rep_lon = points[0]
    wind_speed = weather['windspeed']
    wind_dir = weather['winddirection']
    temperature = weather.get("temperature", "不明")
    humidity_info = f"{weather.get('humidity', '不明')}%"
    precipitation_info = f"{weather.get('precipitation', '不明')} mm/h"
    slope_info = "10度程度の傾斜"
    elevation_info = "標高150m程度"
    vegetation_info = "松林と草地が混在"

    # プロンプト
    detailed_prompt = (
        "あなたは火災拡大シミュレーションの専門家です。以下の条件に基づき、火災の拡大予測を数値で出力してください。\n"
        f"- 発生地点: 緯度 {rep_lat}, 経度 {rep_lon}\n"
        f"- 気象条件: 風速 {wind_speed} m/s, 風向 {wind_dir} 度, 時間経過 {duration_hours} 時間, 温度 {temperature}°C, "
        f"湿度 {humidity_info}, 降水量 {precipitation_info}\n"
        f"- 地形情報: 傾斜 {slope_info}, 標高 {elevation_info}\n"
        f"- 植生: {vegetation_info}\n"
        f"- 燃料特性: {fuel_type}\n"
        "求める出力（絶対に純粋なJSON形式のみ、他のテキストを含むな）:\n"
        '{"radius_m": <火災拡大半径（m）>, "area_sqm": <拡大面積（m²）>, "water_volume_tons": <消火水量（トン）>}\n'
        "例:\n"
        '{"radius_m": 650.00, "area_sqm": 1327322.89, "water_volume_tons": 475.50}\n'
        "もしこの形式と異なる場合は、必ずエラーを出力してください。"
    )

    generated_text, raw_json = gemini_generate_text(detailed_prompt, api_key, model_name)
    st.write("### Gemini API 生JSON応答")
    if raw_json:
        with st.expander("生JSON応答 (折りたたみ)"):
            st.json(raw_json)
    else:
        st.warning("Gemini APIからJSON形式の応答が得られませんでした。")

    if not generated_text:
        st.error("Gemini APIから有効な応答が得られませんでした。")
        return None

    # 予測結果JSONを抽出
    prediction_json = extract_json(generated_text)
    if not prediction_json:
        st.error("予測結果の解析に失敗しました。返却されたテキストを確認してください。")
        st.markdown(f"`json\n{generated_text}\n`")
        return None

    # JSONオブジェクトの必須キーを確認
    required_keys = ["radius_m", "area_sqm", "water_volume_tons"]
    if not all(key in prediction_json for key in required_keys):
        st.error(f"JSONに必須キー {required_keys} が含まれていません。")
        return None

    return prediction_json

def run_simulation(duration_hours, time_label):
    if 'weather_data' not in st.session_state or not st.session_state.weather_data:
        st.error("気象データが取得されていません。")
        return
    if len(st.session_state.points) == 0:
        st.error("発生地点が設定されていません。")
        return

    with st.spinner("シミュレーション実行中..."):
        prediction_json = predict_fire_spread(
            points=st.session_state.points,
            weather=st.session_state.weather_data,
            duration_hours=duration_hours,
            api_key=API_KEY,
            model_name=MODEL_NAME,
            fuel_type=fuel_type
        )

    if prediction_json is None:
        return

    radius_m = prediction_json.get("radius_m", 0)
    area_sqm = prediction_json.get("area_sqm", 0)
    water_volume_tons = prediction_json.get("water_volume_tons", 0)

    # 結果表示
    st.write(f"### シミュレーション結果 ({time_label})")
    st.write(f"**半径**: {radius_m:.2f} m")
    st.write(f"**面積**: {area_sqm:.2f} m²")
    st.write("#### 必要放水量")
    st.info(f"{water_volume_tons:.2f} トン")

    # スライダーで延焼進捗を指定
    progress = st.slider("延焼進捗 (%)", 0, 100, 100, key="progress_slider")
    fraction = progress / 100.0
    current_radius = radius_m * fraction

    # 中心と風向
    lat_center, lon_center = st.session_state.points[0]
    wind_dir = st.session_state.weather_data["winddirection"]

    coords = create_half_circle_coords(lat_center, lon_center, current_radius, wind_dir)

    # 3Dカラムレイヤー用データ
    col_data = []
    for c in coords:
        col_data.append({
            "lon": c[0],
            "lat": c[1],
            "height": 100  # 高さ
        })

    # ColumnLayer で3Dカラム表示
    column_layer = pdk.Layer(
        "ColumnLayer",
        data=col_data,
        get_position='[lon, lat]',
        get_elevation='height',  # カラムの高さ
        get_radius=30,           # カラムの太さ
        elevation_scale=1,
        get_fill_color='[200, 30, 30, 200]',
        pickable=True,
        auto_highlight=True,
    )
    view_state = pdk.ViewState(
        latitude=lat_center,
        longitude=lon_center,
        zoom=13,
        pitch=45  # 3D感を出すために少し角度を付ける
    )
    deck = pdk.Deck(layers=[column_layer], initial_view_state=view_state)
    st.pydeck_chart(deck)

# 気象データ取得ボタン
if st.button("気象データ取得"):
    if len(st.session_state.points) > 0:
        lat_weather, lon_weather = st.session_state.points[0]
        weather_data = get_weather(lat_weather, lon_weather)
        st.session_state.weather_data = weather_data
        st.write(f"取得した気象データ: {weather_data}")
    else:
        st.warning("発生地点を追加してください。")

st.write("## 消火活動が行われない場合のシミュレーション")

# 日/週/月単位でのシミュレーションタブ
tab_day, tab_week, tab_month = st.tabs(["日単位", "週単位", "月単位"])

with tab_day:
    days = st.slider("日数を選択", 1, 30, 1, key="days_slider")
    if st.button("シミュレーション実行 (日単位)", key="sim_day"):
        duration = days * 24
        run_simulation(duration, f"{days} 日後")

with tab_week:
    weeks = st.slider("週数を選択", 1, 52, 1, key="weeks_slider")
    if st.button("シミュレーション実行 (週単位)", key="sim_week"):
        duration = weeks * 7 * 24
        run_simulation(duration, f"{weeks} 週後")

with tab_month:
    months = st.slider("月数を選択", 1, 12, 1, key="months_slider")
    if st.button("シミュレーション実行 (月単位)", key="sim_month"):
        duration = months * 30 * 24
        run_simulation(duration, f"{months} ヶ月後")
