import streamlit as st
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point
import geopandas as gpd
import requests
import json

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
st.title("火災拡大シミュレーション")

# セッションに発生地点リストが無い場合は初期化
if 'points' not in st.session_state:
    st.session_state.points = []

# ベースマップの作成（初期位置は指定座標）
initial_location = [34.25758634545399, 133.20435568517337]
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
        'wind_direction': data['current_weather']['winddirection']
    }

def calculate_water_volume(area_sqm):
    """
    火災拡大面積（平方メートル）に対して、必要な消火水量を計算する（1平方メートルあたり0.5立方メートル）
    1立方メートル = 1トンとして換算
    """
    water_volume_cubic_m = area_sqm * 0.5
    water_volume_tons = water_volume_cubic_m
    return water_volume_tons

def gemini_generate_text(prompt, api_key, model_name):
    """
    Gemini API のエンドポイントに対してリクエストを送り、テキスト生成を行う関数
    生成されたテキストと生のAPI応答（JSON）をタプルで返す
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        result = response.json()
        candidates = result.get("candidates", [])
        if candidates:
            generated_text = candidates[0].get("output", "").strip()
            return generated_text, result
        else:
            st.error("API 応答に候補が含まれていません。")
            return None, result
    else:
        st.error(f"APIリクエストに失敗しました。ステータスコード: {response.status_code}\n{response.text}")
        return None, None

def predict_fire_spread(points, weather, duration_hours, api_key, model_name):
    """
    Gemini API を利用して、火災拡大の予測を行う関数
    ※出力はJSON形式で {"radius_m": 値, "area_sqm": 値, "water_volume_tons": 値} を想定
    """
    points_str = ', '.join([f"({lat}, {lon})" for lat, lon in points])
    prompt = f"""
    以下の条件で火災の炎症範囲を予測してください。

    発生地点: {points_str}
    風速: {weather['wind_speed']} m/s
    風向: {weather['wind_direction']}度
    時間経過: {duration_hours} 時間

    以下を算出してください:
    1. 予測される炎症範囲の半径（メートル）
    2. 炎症範囲のおおよその面積（平方メートル）
    3. 必要な消火水量（トン）

    出力はJSON形式で以下のように返してください：
    {{
        "radius_m": 値,
        "area_sqm": 値,
        "water_volume_tons": 値
    }}
    """
    generated_text, raw_response = gemini_generate_text(prompt, api_key, model_name)
    if generated_text is None:
        st.error("テキスト生成に失敗しました。")
        if raw_response is not None:
            st.json(raw_response)
        return None
    try:
        prediction_json = json.loads(generated_text)
    except Exception as e:
        st.error("予測結果の解析に失敗しました。APIの応答内容を確認してください。")
        st.json(raw_response)  # 生のJSON応答を表示
        return None

    # 発生地点群の重心を求め、そこを中心に半径分のバッファ（円）を作成
    gdf_points = gpd.GeoSeries([Point(lon, lat) for lat, lon in points], crs="EPSG:4326")
    centroid = gdf_points.unary_union.centroid
    # 1度 ≒111,000mとして、バッファの半径（度）を算出
    buffer = centroid.buffer(prediction_json['radius_m'] / 111000)
    area_coordinates = [(coord[1], coord[0]) for coord in buffer.exterior.coords]

    return {
        'radius_m': prediction_json['radius_m'],
        'area_sqm': prediction_json['area_sqm'],
        'water_volume_tons': prediction_json['water_volume_tons'],
        'area_coordinates': area_coordinates
    }

# --- UI 操作 ---

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

# シミュレーション結果表示用の関数
def run_simulation(duration_hours, time_label):
    if 'weather_data' not in st.session_state:
        st.error("気象データが取得されていません。")
        return
    if len(st.session_state.points) == 0:
        st.error("発生地点が設定されていません。")
        return
    simulation = predict_fire_spread(
        points=st.session_state.points,
        weather=st.session_state.weather_data,
        duration_hours=duration_hours,
        api_key=API_KEY,
        model_name=MODEL_NAME
    )
    if simulation is None:
        return
    st.write(f"### シミュレーション結果 ({time_label})")
    st.write(f"拡大範囲の半径: {simulation['radius_m']:.2f} m")
    st.write(f"拡大面積: {simulation['area_sqm']:.2f} 平方メートル")
    st.write(f"必要な消火水量: {simulation['water_volume_tons']:.2f} トン")
    
    # シミュレーション結果の領域を表示する新たな地図を作成（初期位置は指定座標）
    m_sim = folium.Map(location=initial_location, zoom_start=12)
    for point in st.session_state.points:
        folium.Marker(location=point, icon=folium.Icon(color='red')).add_to(m_sim)
    folium.Polygon(simulation['area_coordinates'], color="red", fill=True, fill_opacity=0.5).add_to(m_sim)
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
