import streamlit as st
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point
import geopandas as gpd
import requests
import json
import math

# ページ設定
st.set_page_config(page_title="火災拡大シミュレーション", layout="wide")

# グローバル設定（secretsからAPIキーを取得）
API_KEY = st.secrets["general"]["api_key"]
MODEL_NAME = "gemini-2.0-flash-001"  # 使用するモデル名

# サイドバー：発生地点の入力
st.sidebar.title("火災発生地点の入力")
with st.sidebar.form(key='location_form'):
    lat_input = st.number_input("緯度", format="%.6f", value=34.257586)
    lon_input = st.number_input("経度", format="%.6f", value=133.204356)
    add_point = st.form_submit_button("発生地点を追加")
    if add_point:
        if 'points' not in st.session_state:
            st.session_state.points = []
        st.session_state.points.append((lat_input, lon_input))
        st.sidebar.success(f"地点 ({lat_input}, {lon_input}) を追加しました。")

# サイドバー：登録地点消去ボタン
if st.sidebar.button("登録地点を消去"):
    st.session_state.points = []
    st.sidebar.info("全ての発生地点を削除しました。")

# サイドバー： 燃料特性の選択
st.sidebar.title("燃料特性の選択")
fuel_options = {
    "森林（高燃料）": "森林",
    "草地（中燃料）": "草地",
    "都市部（低燃料）": "都市部"
}
selected_fuel = st.sidebar.selectbox("燃料特性を選択してください", list(fuel_options.keys()))
fuel_type = fuel_options[selected_fuel]

# メインエリア：タイトル
st.title("火災拡大シミュレーション（半円形 + 詳細プロンプト + JSON表示）")

# セッションに発生地点リストが無い場合は初期化
if 'points' not in st.session_state:
    st.session_state.points = []

# ベースマップの作成（初期位置は指定座標）
initial_location = [34.257586, 133.204356]
m = folium.Map(location=initial_location, zoom_start=12)
for point in st.session_state.points:
    folium.Marker(location=point, icon=folium.Icon(color='red')).add_to(m)
st_folium(m, width=700, height=500)

# --- 関数定義 ---

def get_weather(lat, lon):
    """
    指定した緯度・経度の現在の気象情報を、Open-Meteo APIから取得する関数。
    温度、風速、風向、天気コード、湿度、降水量を取得。
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&current_weather=true&"
        f"hourly=relativehumidity_2m,precipitation&timezone=auto"
    )
    response = requests.get(url)
    # Open-Meteo API のステータスコードを表示
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

def create_half_circle_polygon(center_lat, center_lon, radius_m, wind_direction_deg):
    """
    風向きの方向を中心とした ±90° の半円形（扇形）Polygon を作成する関数。
    wind_direction_deg: 0=北, 90=東, 180=南, 270=西（度数）
    radius_m: 半径（メートル）
    """
    deg_per_meter = 1.0 / 111000.0
    start_angle = wind_direction_deg - 90
    end_angle = wind_direction_deg + 90
    num_steps = 36
    coords = []
    coords.append((center_lat, center_lon))
    for i in range(num_steps + 1):
        angle_deg = start_angle + (end_angle - start_angle) * i / num_steps
        angle_rad = math.radians(angle_deg)
        offset_y_m = radius_m * math.cos(angle_rad)
        offset_x_m = radius_m * math.sin(angle_rad)
        offset_lat = offset_y_m * deg_per_meter
        offset_lon = offset_x_m * deg_per_meter
        new_lat = center_lat + offset_lat
        new_lon = center_lon + offset_lon
        coords.append((new_lat, new_lon))
    return coords

def gemini_generate_text(prompt, api_key, model_name):
    """
    Gemini API のエンドポイントにリクエストを送り、テキスト生成を行う関数。
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
    except Exception as e:
        st.error("レスポンスのJSONパースに失敗しました。")
    if response.status_code == 200 and raw_json:
        candidates = raw_json.get("candidates", [])
        if candidates:
            generated_text = candidates[0].get("output", "").strip()
            return generated_text, raw_json
        else:
            return None, raw_json
    else:
        return None, raw_json

def predict_fire_spread(points, weather, duration_hours, api_key, model_name):
    """
    Gemini API を利用して火災拡大の予測を行う関数。
    出力は以下の JSON 形式:
      {"radius_m": <float>, "area_sqm": <float>, "water_volume_tons": <float>}
    燃料特性 (fuel_type) もプロンプトに含む。
    """
    rep_lat, rep_lon = points[0]
    wind_speed = weather['windspeed']
    wind_dir = weather['winddirection']

    slope_info = "10度程度の傾斜"
    elevation_info = "標高150m程度"
    vegetation_info = "松林と草地が混在"
    humidity_info = f"相対湿度 {weather.get('humidity', '不明')}%"
    precipitation_info = f"{weather.get('precipitation', '不明')} mm/h"

    detailed_prompt = f"""
あなたは火災拡大シミュレーションの専門家です。以下の条件に基づき、火災の拡大予測を数値で出力してください。
条件:
- 発生地点: 緯度 {rep_lat}, 経度 {rep_lon}
- 気象条件: 風速 {wind_speed} m/s, 風向 {wind_dir} 度 (0=北,90=東,180=南,270=西), 時間経過 {duration_hours} 時間, 温度 {weather.get("temperature", "不明")}°C, 湿度 {humidity_info}, 降水量 {precipitation_info}
- 地形情報: 傾斜 {slope_info}, 標高 {elevation_info}
- 植生: {vegetation_info}
- 燃料特性: {fuel_type}
求める出力（純粋なJSON形式のみ、他のテキストを含むな）:
{{"radius_m": <火災拡大半径（m）>, "area_sqm": <拡大面積（m²）>, "water_volume_tons": <消火水量（トン）>}}
例:
{{"radius_m": 331.45, "area_sqm": 345069.36, "water_volume_tons": 123.45}}
"""
    generated_text, raw_json = gemini_generate_text(detailed_prompt, api_key, model_name)
    st.write("### Gemini API 生JSON応答")
    if raw_json:
        st.json(raw_json)
    else:
        st.warning("Gemini APIからJSON形式の応答が得られませんでした。")
    if not generated_text:
        st.error("Gemini APIから有効な応答が得られませんでした。")
        return None
    try:
        prediction_json = json.loads(generated_text)
    except Exception as e:
        st.error("予測結果の解析に失敗しました。返されたテキストを確認してください。")
        st.write("返却されたテキスト:")
        st.write(generated_text)
        return None
    return prediction_json

def run_simulation(duration_hours, time_label):
    if 'weather_data' not in st.session_state:
        st.error("気象データが取得されていません。")
        return
    if len(st.session_state.points) == 0:
        st.error("発生地点が設定されていません。")
        return

    prediction_json = predict_fire_spread(
        points=st.session_state.points,
        weather=st.session_state.weather_data,
        duration_hours=duration_hours,
        api_key=API_KEY,
        model_name=MODEL_NAME
    )
    if prediction_json is None:
        return

    radius_m = prediction_json.get("radius_m", 0)
    area_sqm = prediction_json.get("area_sqm", 0)
    water_volume_tons = prediction_json.get("water_volume_tons", 0)

    st.write(f"### シミュレーション結果 ({time_label})")
    st.write(f"半径: {radius_m:.2f} m")
    st.write(f"面積: {area_sqm:.2f} m²")
    st.write("#### 消火水量")
    st.info(f"{water_volume_tons:.2f} トン")

    lat_center, lon_center = st.session_state.points[0]
    wind_dir = st.session_state.weather_data["winddirection"]

    coords = create_half_circle_polygon(lat_center, lon_center, radius_m, wind_dir)
    m_sim = folium.Map(location=[lat_center, lon_center], zoom_start=13)
    folium.Polygon(
        locations=coords,
        color="red",
        fill=True,
        fill_opacity=0.4,
        tooltip=f"半径: {radius_m:.2f} m / 面積: {area_sqm:.2f} m²"
    ).add_to(m_sim)
    for pt in st.session_state.points:
        folium.Marker(location=pt, icon=folium.Icon(color='red')).add_to(m_sim)
    st_folium(m_sim, width=700, height=500)

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
