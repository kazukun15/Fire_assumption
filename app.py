import streamlit as st
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point
import geopandas as gpd
import requests
import json
import math
import numpy as np

# ページ設定
st.set_page_config(page_title="火災拡大シミュレーション", layout="wide")

# グローバル設定（secretsからAPIキーを取得）
API_KEY = st.secrets["general"]["api_key"]
MODEL_NAME = "gemini-2.0-flash-001"  # 使用するモデル名

# サイドバー：火災発生地点の入力
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

# メインエリア：タイトル
st.title("火災拡大シミュレーション（半円形表示 + 詳細プロンプト）")

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
    指定した緯度・経度の現在の気象情報を取得する関数（Open-Meteo APIを利用）
    """
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    response = requests.get(url)
    data = response.json()
    return {
        'wind_speed': data['current_weather']['windspeed'],
        'wind_direction': data['current_weather']['winddirection']  # 0=北,90=東,180=南,270=西(概念)
    }

def gemini_generate_text(prompt, api_key, model_name):
    """
    Gemini API のエンドポイントに対してリクエストを送り、テキスト生成を行う関数
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        candidates = result.get("candidates", [])
        if candidates:
            generated_text = candidates[0].get("output", "").strip()
            return generated_text
        else:
            st.error("API 応答に候補が含まれていません。")
            return None
    else:
        st.error(f"APIリクエストに失敗しました。ステータスコード: {response.status_code}\n{response.text}")
        return None

def create_half_circle_polygon(center_lat, center_lon, radius_m, wind_direction_deg):
    """
    風向きの方向を中心とした ±90° の半円形（扇形）Polygon を作成する。
    wind_direction_deg: 0=北, 90=東, 180=南, 270=西（度数）
    radius_m: 半径（メートル）
    """
    # 1度 ≈ 111,000m (緯度方向)
    # 経度方向は緯度によって縮尺が変わるが、簡易的に同一係数で計算
    deg_per_meter = 1.0 / 111000.0

    # 半円（扇形）を描画する角度範囲
    # wind_direction_deg - 90° から wind_direction_deg + 90° まで
    start_angle = wind_direction_deg - 90
    end_angle = wind_direction_deg + 90
    num_steps = 36  # ステップ数（細かくするほど滑らかな扇形）

    # 扇形の座標を格納するリスト
    coords = []
    # 中心点を (lat, lon) -> (y, x) の順で入れておく
    center_point = (center_lat, center_lon)

    # 最初に中心点を追加（Polygonを閉じるために使う場合もある）
    coords.append(center_point)

    # 指定した角度範囲をステップごとにループ
    for i in range(num_steps + 1):
        angle_deg = start_angle + (end_angle - start_angle) * i / num_steps
        # ラジアンに変換
        angle_rad = math.radians(angle_deg)
        # メートル単位で x, y のオフセットを計算（y軸=北向き, x軸=東向き）
        # ただし angle_deg=0 は北向きではなく、気象庁などの方角定義に合わせるなら要調整
        # ここでは 0=北、90=東 という前提で計算
        offset_y_m = radius_m * math.cos(angle_rad)
        offset_x_m = radius_m * math.sin(angle_rad)

        # 緯度・経度への変換（簡易計算）
        offset_lat = offset_y_m * deg_per_meter
        offset_lon = offset_x_m * deg_per_meter

        # 実際の座標
        new_lat = center_lat + offset_lat
        new_lon = center_lon + offset_lon
        coords.append((new_lat, new_lon))

    # 戻り値は Polygon 化できるよう (lat, lon) のリスト
    return coords

def predict_fire_spread(points, weather, duration_hours, api_key, model_name):
    """
    Gemini API を利用して、火災拡大の予測を行う関数
    出力はJSON形式で {"radius_m": 値, "area_sqm": 値, "water_volume_tons": 値} を想定
    """
    # 複数地点がある場合、最初の地点を代表地点とする（例）
    rep_lat, rep_lon = points[0]
    wind_speed = weather['wind_speed']
    wind_dir = weather['wind_direction']

    # 例: 地形や植生、湿度などの情報を仮に固定して入れる
    # 実際には別途APIやデータベースから取得可能
    slope_info = "10度程度の傾斜"
    vegetation_info = "松林と草地が混在"
    humidity_info = "相対湿度 60%"
    elevation_info = "標高 150m 程度"

    # 詳細なプロンプトを作成
    detailed_prompt = f"""
あなたは高度な火災シミュレーションモデルです。
以下の情報を踏まえて、可能な限り詳細に火災の広がりを分析してください。

【火災発生地点】
- 緯度: {rep_lat}
- 経度: {rep_lon}

【気象条件】
- 風速: {wind_speed} m/s
- 風向: {wind_dir} 度 (0=北,90=東,180=南,270=西)
- 時間経過: {duration_hours} 時間

【地形・植生・湿度など】
- 地形傾斜: {slope_info}
- 植生: {vegetation_info}
- 湿度: {humidity_info}
- 標高: {elevation_info}

【求めたい情報】
- 1. 予測される炎症範囲の半径 (メートル)
- 2. 炎症範囲のおおよその面積 (平方メートル)
- 3. 必要な消火水量 (トン)

必ず以下のJSON形式のみで出力してください:
{{
  "radius_m": <float>,
  "area_sqm": <float>,
  "water_volume_tons": <float>
}}
"""

    # Gemini API を呼び出し
    generated_text = gemini_generate_text(detailed_prompt, api_key, model_name)
    if not generated_text:
        st.error("Gemini APIからの応答がありません。")
        return None

    # JSON解析
    try:
        prediction_json = json.loads(generated_text)
    except Exception as e:
        st.error("予測結果の解析に失敗しました。JSON形式を確認してください。")
        st.write("Gemini API応答:")
        st.write(generated_text)
        return None

    return prediction_json

# 気象データ取得ボタン
if st.button("気象データ取得"):
    if len(st.session_state.points) > 0:
        # 1つ目の発生地点を基準に気象情報を取得
        lat_weather, lon_weather = st.session_state.points[0]
        weather_data = get_weather(lat_weather, lon_weather)
        st.session_state.weather_data = weather_data
        st.write(f"取得した気象データ: {weather_data}")
    else:
        st.warning("発生地点を追加してください。")

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

    # 結果を取得
    radius_m = prediction_json["radius_m"]
    area_sqm = prediction_json["area_sqm"]
    water_volume_tons = prediction_json["water_volume_tons"]

    # 表示
    st.write(f"### シミュレーション結果 ({time_label})")
    st.write(f"半径: {radius_m:.2f} m")
    st.write(f"面積: {area_sqm:.2f} m²")
    # 消火水量を別枠で表示
    st.write("#### 消火水量")
    st.info(f"{water_volume_tons:.2f} トン")

    # 地図表示（半円形で可視化）
    lat_center, lon_center = st.session_state.points[0]
    wind_dir = st.session_state.weather_data["wind_direction"]

    coords = create_half_circle_polygon(lat_center, lon_center, radius_m, wind_dir)
    # folium で描画
    m_sim = folium.Map(location=[lat_center, lon_center], zoom_start=13)
    folium.Polygon(
        locations=coords,
        color="red",
        fill=True,
        fill_opacity=0.4,
        tooltip=f"半径: {radius_m:.2f}m / 面積: {area_sqm:.2f}m²"
    ).add_to(m_sim)
    # 発生地点も表示
    for pt in st.session_state.points:
        folium.Marker(location=pt, icon=folium.Icon(color='red')).add_to(m_sim)

    st_folium(m_sim, width=700, height=500)

st.write("## 消火活動が行われない場合のシミュレーション")

# タブによる時間単位の切替
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
