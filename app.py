import streamlit as st
import folium
from streamlit_folium import st_folium
import google.generativeai as genai
import requests
import json
import math
import re
import pydeck as pdk
import time
import demjson3 as demjson  # Python3 用の demjson ライブラリ
from shapely.geometry import Point
import geopandas as gpd

# ページ設定
st.set_page_config(page_title="火災拡大シミュレーション (2D/3Dレポート版)", layout="wide")

# secretsからAPIキーを取得（[general]セクション）
API_KEY = st.secrets["general"]["api_key"]
MODEL_NAME = "gemini-2.0-flash-001"

# Gemini API の初期設定
genai.configure(api_key=API_KEY)

# セッションステートの初期化
if 'points' not in st.session_state:
    st.session_state.points = []
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = {}

# -----------------------------
# サイドバー：発生地点の入力
# -----------------------------
st.sidebar.title("火災発生地点の入力")
with st.sidebar.form(key='location_form'):
    lat_input = st.number_input("緯度", format="%.6f", value=34.257586)
    lon_input = st.number_input("経度", format="%.6f", value=133.204356)
    add_point = st.form_submit_button("発生地点を追加")
    if add_point:
        st.session_state.points.append((lat_input, lon_input))
        st.sidebar.success(f"地点 ({lat_input}, {lon_input}) を追加しました。")

if st.sidebar.button("登録地点を消去"):
    st.session_state.points = []
    st.sidebar.info("全ての発生地点を削除しました。")

# -----------------------------
# サイドバー：燃料特性の選択
# -----------------------------
st.sidebar.title("燃料特性の選択")
fuel_options = {
    "森林（高燃料）": "森林",
    "草地（中燃料）": "草地",
    "都市部（低燃料）": "都市部"
}
selected_fuel = st.sidebar.selectbox("燃料特性を選択してください", list(fuel_options.keys()))
fuel_type = fuel_options[selected_fuel]

# -----------------------------
# メインエリア：タイトルと初期マップ（2D）
# -----------------------------
st.title("火災拡大シミュレーション（Gemini要約＋2D/3Dレポート表示）")
initial_location = [34.257586, 133.204356]
base_map = folium.Map(location=initial_location, zoom_start=12)
for point in st.session_state.points:
    folium.Marker(location=point, icon=folium.Icon(color='red')).add_to(base_map)
st_folium(base_map, width=700, height=500)

# -----------------------------
# 関数定義
# -----------------------------
def extract_json(text: str) -> dict:
    """
    テキストからJSONオブジェクトを抽出する（多様なパターンに対応）。
    まず直接 json.loads() を試み、失敗した場合は正規表現で抽出し、
    demjson3 を利用して解析を試みる。
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pattern = r"\{.*\}"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                try:
                    return demjson.decode(json_str)
                except demjson.JSONDecodeError:
                    st.error("demjsonによるJSON解析に失敗しました。")
                    return {}
        else:
            st.error("JSON文字列が見つかりませんでした。")
            return {}

@st.cache_data(show_spinner=False)
def get_weather(lat, lon):
    """
    Open-Meteo APIから指定緯度・経度の気象情報を取得する。
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

@st.cache_data(show_spinner=False)
def gemini_generate_text(prompt, api_key, model_name):
    """
    Gemini API にリクエストを送り、テキスト生成を行う関数。
    生のJSON応答も返す。
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
        st.error("Gemini APIレスポンスのJSONパースに失敗しました。")
    if response.status_code == 200 and raw_json:
        candidates = raw_json.get("candidates", [])
        if candidates:
            generated_text = candidates[0].get("output", "").strip()
            return generated_text, raw_json
        else:
            return None, raw_json
    else:
        return None, raw_json

def create_half_circle_polygon(center_lat, center_lon, radius_m, wind_direction_deg):
    """
    風向きを考慮した半円形（扇形）の座標列を生成する。
    pydeck用に [lon, lat] の形式で返す。
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

def predict_fire_spread(points, weather, duration_hours, api_key, model_name, fuel_type):
    """
    Gemini API を利用して火災拡大予測を行う関数。
    最新の気象データおよび下記の条件に基づいてシミュレーションを実施します。
    
    条件:
      - 発生地点: 緯度 {rep_lat}, 経度 {rep_lon}
      - 気象条件: 風速 {wind_speed} m/s, 風向 {wind_dir} 度, 時間経過 {duration_hours} 時間,
        温度 {temperature}°C, 湿度 {humidity_info}, 降水量 {precipitation_info}
      - 地形情報: 傾斜 {slope_info}, 標高 {elevation_info}
      - 植生: {vegetation_info}
      - 燃料特性: {fuel_type}
    
    出力形式（厳密にこれのみ）:
    {
      "radius_m": <火災拡大半径（m）>,
      "area_sqm": <拡大面積（m²）>,
      "water_volume_tons": <消火水量（トン）>
    }
    
    出力例:
    {
      "radius_m": 650.00,
      "area_sqm": 1327322.89,
      "water_volume_tons": 475.50
    }
    
    もしこの形式と異なる場合は、必ずエラーを出力してください。
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

    detailed_prompt = (
        "以下の最新気象データに基づいて、火災拡大シミュレーションを実施してください。\n"
        "【条件】\n"
        f"・発生地点: 緯度 {rep_lat}, 経度 {rep_lon}\n"
        f"・気象条件: 温度 {temperature}°C, 風速 {wind_speed} m/s, 風向 {wind_dir} 度, "
        f"湿度 {humidity_info}, 降水量 {precipitation_info}\n"
        f"・地形情報: 傾斜 {slope_info}, 標高 {elevation_info}\n"
        f"・植生: {vegetation_info}\n"
        f"・燃料特性: {fuel_type}\n"
        "【求める出力】\n"
        "絶対に純粋なJSON形式のみを出力してください（他のテキストを含むな）。\n"
        "出力形式:\n"
        '{"radius_m": <火災拡大半径（m）>, "area_sqm": <拡大面積（m²）>, "water_volume_tons": <消火水量（トン）>}\n'
        "例:\n"
        '{"radius_m": 650.00, "area_sqm": 1327322.89, "water_volume_tons": 475.50}\n'
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

    prediction_json = extract_json(generated_text)
    if not prediction_json:
        st.error("予測結果の解析に失敗しました。返されたテキストを確認してください。")
        st.markdown(f"`json\n{generated_text}\n`")
        return None

    required_keys = ["radius_m", "area_sqm", "water_volume_tons"]
    if not all(key in prediction_json for key in required_keys):
        st.error(f"JSONオブジェクトに必須キー {required_keys} が含まれていません。")
        return None

    return prediction_json

def gemini_summarize_data(json_data, api_key, model_name):
    json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
    summary_prompt = (
        "あなたはデータをわかりやすく説明するアシスタントです。\n"
        "次の火災拡大シミュレーション結果のJSONを確認し、その内容を一般の方が理解しやすい日本語で要約してください。\n"
        "```json\n" + json_str + "\n```\n"
        "短く簡潔な説明文でお願いします。"
    )
    summary_text, _ = gemini_generate_text(summary_prompt, API_KEY, model_name)
    return summary_text or "要約が取得できませんでした。"

def run_simulation(duration_hours, time_label):
    if 'weather_data' not in st.session_state or not st.session_state.weather_data:
        st.error("気象データが取得されていません。")
        return
    if len(st.session_state.points) == 0:
        st.error("発生地点が設定されていません。")
        return

    with st.spinner(f"{time_label}のシミュレーションを実行中..."):
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

    st.write(f"### シミュレーション結果 ({time_label})")
    st.write
