import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import json
import math
import re
import demjson3 as demjson
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
    """
    テキストからJSONオブジェクトを抽出する（多様なパターンに対応）。
    マークダウン形式のコードブロックに含まれるJSON部分も抽出できるようにします。
    """
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

# --- シミュレーション結果をセッションに保持する ---
if "simulation_result" not in st.session_state:
    st.session_state.simulation_result = None

# --- メインUI ---
st.title("火災拡大シミュレーション（Gemini要約＋2D/3Dレポート表示）")

with st.sidebar:
    lat = st.number_input("緯度", value=34.257586)
    lon = st.number_input("経度", value=133.204356)
    fuel_type = st.selectbox("燃料特性", ["森林", "草地", "都市部"])

# 2D マップ表示（初期状態）
initial_location = [34.257586, 133.204356]
base_map = folium.Map(location=initial_location, zoom_start=12)
st_folium(base_map, width=700, height=500)

# --- シミュレーション実行 ---
def run_simulation(duration_hours, time_label):
    if not st.session_state.get("weather_data"):
        st.error("気象データが取得されていません。")
        return
    if not st.session_state.get("points"):
        st.error("発生地点が設定されていません。")
        return

    with st.spinner(f"{time_label}のシミュレーションを実行中..."):
        result_json = predict_fire_spread(
            lat, lon, duration_hours, API_KEY, MODEL_NAME, fuel_type
        )
    if result_json is None:
        return

    st.session_state.simulation_result = result_json  # 結果をセッションに保持

    # 結果の各値を取得
    try:
        radius_m = float(result_json.get("radius_m", 0))
    except (KeyError, ValueError):
        st.error("JSONに 'radius_m' の数値が見つかりません。")
        return
    area_sqm = result_json.get("area_sqm", "不明")
    water_volume_tons = result_json.get("water_volume_tons", "不明")

    st.write(f"### シミュレーション結果 ({time_label})")
    st.write(f"**半径**: {radius_m:.2f} m")
    st.write(f"**面積**: {area_sqm} ㎡")
    st.write("#### 必要放水量")
    st.info(f"{water_volume_tons} トン")

    summary_text = gemini_summarize_data(result_json, API_KEY, MODEL_NAME)
    st.write("#### Geminiによる要約")
    st.info(summary_text)

    # 2D Folium 地図で延焼範囲を描写
    folium_map = folium.Map(location=[lat, lon], zoom_start=13)
    folium.Marker([lat, lon], popup="火災発生地点", icon=folium.Icon(color='red')).add_to(folium_map)
    coords = create_half_circle_polygon(lat, lon, radius_m, st.session_state.weather_data["winddirection"])
    folium.Polygon(locations=coords, color="red", fill=True, fill_opacity=0.5).add_to(folium_map)
    st.write("#### Folium 地図（延焼範囲）")
    st_folium(folium_map, width=700, height=500)

    # 3D pydeck 表示（3Dカラム）
    col_data = []
    scale_factor = 50  # 水量に基づくスケール例
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

# Gemini API を利用したシミュレーション実行関数（改良版）
def predict_fire_spread(lat, lon, duration_hours, api_key, model_name, fuel_type):
    prompt = (
        f"以下の最新気象データに基づいて、火災拡大シミュレーションを実施してください。\n"
        "【条件】\n"
        f"・発生地点: 緯度 {lat}, 経度 {lon}\n"
        f"・燃料特性: {fuel_type}\n"
        f"・シミュレーション時間: {duration_hours} 時間\n"
        "【求める出力】\n"
        "絶対に純粋なJSON形式のみを出力してください（他のテキストを含むな）。\n"
        "出力形式:\n"
        '{"radius_m": 数値, "area_sqm": 数値, "water_volume_tons": 数値}\n'
        "例:\n"
        '{"radius_m": 650.00, "area_sqm": 1327322.89, "water_volume_tons": 475.50}\n'
    )
    response_text = gemini_generate_text(prompt, api_key, model_name)
    if not response_text:
        st.error("Gemini APIから有効な応答が得られませんでした。")
        return None
    result_json = extract_json(response_text)
    required_keys = ["radius_m", "area_sqm", "water_volume_tons"]
    if not all(key in result_json for key in required_keys):
        st.error(f"JSONオブジェクトに必須キー {required_keys} が含まれていません。")
        return None
    return result_json

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
