import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import json
import math
import re
import demjson3 as demjson  # Python3 用の demjson のフォーク
import pydeck as pdk
import time

# --- ページ設定 ---
st.set_page_config(page_title="火災拡大シミュレーション", layout="wide")

# --- API設定 ---
API_KEY = st.secrets["general"]["api_key"]
MODEL_NAME = "gemini-2.0-flash-001"

# --- gemini_generate_text 関数 ---
@st.cache_data(show_spinner=False)
def gemini_generate_text(prompt, api_key, model_name):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(url, headers=headers, json=data)
    st.write("Gemini API ステータスコード:", response.status_code)
    try:
        raw_json = response.json()
    except Exception:
        st.error("Gemini APIレスポンスのJSONパースに失敗しました。")
        return None
    if response.status_code == 200 and raw_json:
        candidates = raw_json.get("candidates", [])
        if candidates:
            content = candidates[0].get("content")
            if content and "parts" in content:
                return content["parts"][0].get("text", "").strip()
    st.error(f"Gemini APIエラー: {raw_json}")
    return None

# --- extract_json 関数 ---
def extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pattern = r"```json\s*(\{[\s\S]*?\})\s*```"
        match = re.search(pattern, text)
        if match:
            json_str = match.group(1)
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

# -----------------------------
# サイドバー入力（グローバル変数として定義）
# -----------------------------
lat = st.sidebar.number_input("緯度", value=34.257586)
lon = st.sidebar.number_input("経度", value=133.204356)
fuel_type = st.sidebar.selectbox("燃料特性", ["森林", "草地", "都市部"])

# -----------------------------
# メインUI
# -----------------------------
st.title("火災拡大シミュレーション（Gemini要約＋2D/3Dレポート表示）")

# 2Dマップの初期表示（入力値を中心とする）
base_map = folium.Map(location=[lat, lon], zoom_start=12)
st_folium(base_map, width=700, height=500)

# -----------------------------
# シミュレーション実行関数
# -----------------------------
def create_half_circle_polygon(center_lat, center_lon, radius_m, wind_direction_deg):
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

def predict_fire_spread(lat, lon, duration_hours, api_key, model_name, fuel_type):
    prompt = (
        f"以下の最新気象データに基づいて、火災拡大シミュレーションを実施してください。\n"
        f"【条件】\n"
        f"・発生地点: 緯度 {lat}, 経度 {lon}\n"
        f"・燃料特性: {fuel_type}\n"
        f"・シミュレーション時間: {duration_hours} 時間\n"
        "【求める出力】\n"
        "絶対に純粋なJSON形式のみを出力してください（他のテキストを含むな）。\n"
        "出力形式:\n"
        "{'radius_m': 数値, 'area_sqm': 数値, 'water_volume_tons': 数値}\n"
        "例:\n"
        "{'radius_m': 650.00, 'area_sqm': 1327322.89, 'water_volume_tons': 475.50}\n"
    )
    response_text = gemini_generate_text(prompt, api_key, model_name)
    if not response_text:
        st.error("Gemini APIから有効な応答が得られませんでした。")
        return None
    result = extract_json(response_text)
    required_keys = ["radius_m", "area_sqm", "water_volume_tons"]
    if not all(key in result for key in required_keys):
        st.error(f"JSONオブジェクトに必須キー {required_keys} が含まれていません。")
        return None
    return result

def gemini_summarize_data(json_data, api_key, model_name):
    json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
    summary_prompt = (
        "あなたはデータをわかりやすく説明するアシスタントです。\n"
        "次の火災拡大シミュレーション結果のJSONを確認し、その内容を一般の方が理解しやすい日本語で要約してください。\n"
        "```json\n" + json_str + "\n```\n"
        "短く簡潔な説明文でお願いします。"
    )
    summary_text = gemini_generate_text(summary_prompt, API_KEY, model_name)
    return summary_text or "要約が取得できませんでした。"

def run_simulation(duration_hours, time_label):
    # シミュレーション結果を保持
    result = predict_fire_spread(lat, lon, duration_hours, API_KEY, MODEL_NAME, fuel_type)
    if result is None:
        return
    try:
        radius_m = float(result.get("radius_m", 0))
    except (KeyError, ValueError):
        st.error("JSONに 'radius_m' の数値が見つかりません。")
        return
    area_sqm = result.get("area_sqm", "不明")
    water_volume_tons = result.get("water_volume_tons", "不明")

    st.write(f"### シミュレーション結果 ({time_label})")
    st.write(f"**半径**: {radius_m:.2f} m")
    st.write(f"**面積**: {area_sqm} ㎡")
    st.write("#### 必要放水量")
    st.info(f"{water_volume_tons} トン")

    summary_text = gemini_summarize_data(result, API_KEY, MODEL_NAME)
    st.write("#### Geminiによる要約")
    st.info(summary_text)

    # 延焼進捗スライダー
    progress = st.slider("延焼進捗 (%)", 0, 100, 100, key="progress_slider")
    fraction = progress / 100.0
    current_radius = radius_m * fraction

    # 2D Folium 地図描写
    coords = create_half_circle_polygon(lat, lon, current_radius, st.session_state.weather_data.get("winddirection", 0))
    folium_map = folium.Map(location=[lat, lon], zoom_start=13)
    folium.Marker([lat, lon], popup="火災発生地点", icon=folium.Icon(color='red')).add_to(folium_map)
    folium.Polygon(locations=coords, color="red", fill=True, fill_opacity=0.5).add_to(folium_map)
    st.write("#### Folium 地図（延焼範囲）")
    st_folium(folium_map, width=700, height=500)

    # 3D pydeck 表示（3Dカラム）
    col_data = []
    scale_factor = 50  # 放水量に基づくスケール例
    for c in coords:
        col_data.append({
            "lon": c[0],
            "lat": c[1],
            "height": float(water_volume_tons) / scale_factor if water_volume_tons != "不明" else 100
        })
    column_layer = pdk.Layer(
        "ColumnLayer",
        data=col_data,
        get_position='[lon, lat]',
        get_elevation='height',
        get_radius=30,
        elevation_scale=1,
        get_fill_color='[200, 30, 30, 200]',
        pickable=True,
        auto_highlight=True,
    )
    view_state = pdk.ViewState(
        latitude=lat,
        longitude=lon,
        zoom=13,
        pitch=45
    )
    deck = pdk.Deck(layers=[column_layer], initial_view_state=view_state)
    st.write("#### pydeck 3Dカラム表示")
    st.pydeck_chart(deck)

# -----------------------------
# 気象データ取得ボタン
# -----------------------------
if st.button("気象データ取得"):
    weather_data = get_weather(lat, lon)
    if weather_data:
        st.session_state.weather_data = weather_data
        st.write(f"取得した気象データ: {weather_data}")
    else:
        st.error("気象データの取得に失敗しました。")

# -----------------------------
# シミュレーション実行（タブ切替）
# -----------------------------
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
