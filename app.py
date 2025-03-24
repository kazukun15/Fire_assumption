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
import demjson3 as demjson  # Python3 用の demjson のフォーク
from shapely.geometry import Point
import geopandas as gpd

# --- ページ設定 ---
st.set_page_config(page_title="火災拡大シミュレーション (2D/3D レポート＆マッピング版)", layout="wide")

# --- API設定 ---
API_KEY = st.secrets["general"]["api_key"]
MODEL_NAME = "gemini-2.0-flash-001"

# --- Gemini API の初期設定 ---
genai.configure(api_key=API_KEY)

# --- セッションステートの初期化 ---
if 'points' not in st.session_state:
    st.session_state.points = []
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = {}

# -----------------------------
# サイドバー入力（グローバル変数として）
# -----------------------------
lat = st.sidebar.number_input("緯度", value=34.257586)
lon = st.sidebar.number_input("経度", value=133.204356)
fuel_type = st.sidebar.selectbox("燃料特性", ["森林", "草地", "都市部"])
if st.sidebar.button("発生地点を追加"):
    st.session_state.points.append((lat, lon))
    st.sidebar.success(f"地点 ({lat}, {lon}) を追加しました。")
if st.sidebar.button("登録地点を消去"):
    st.session_state.points = []
    st.sidebar.info("全ての発生地点を削除しました。")

# -----------------------------
# メインUI：タイトルと初期マップ（2D表示）
# -----------------------------
st.title("火災拡大シミュレーション（Gemini要約＋2D/3D レポート＆マッピング版）")
base_map = folium.Map(location=[lat, lon], zoom_start=12)
for point in st.session_state.points:
    folium.Marker(location=point, icon=folium.Icon(color='red')).add_to(base_map)
st_folium(base_map, width=700, height=500)

# -----------------------------
# 関数定義
# -----------------------------
def extract_json(text: str) -> dict:
    """
    テキストからJSONオブジェクトを抽出する関数。
    まず直接 json.loads() を試み、失敗した場合は
    マークダウン形式のコードブロック（```json ... ```）または、
    最初に現れる { ... } 部分を抽出して解析を試みます。
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    pattern_md = r"```json\s*(\{[\s\S]*?\})\s*```"
    match = re.search(pattern_md, text)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                return demjson.decode(json_str)
            except Exception as e:
                st.error(f"demjsonによるJSON解析に失敗しました: {e}")
                return {}
    pattern = r"\{[\s\S]*\}"
    match = re.search(pattern, text)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                return demjson.decode(json_str)
            except Exception as e:
                st.error(f"demjsonによるJSON解析に失敗しました: {e}")
                return {}
    st.error("有効なJSON文字列が見つかりませんでした。")
    return {}

# キャッシュ設定は必要に応じて調整。気象データはキャッシュOK。
@st.cache_data(show_spinner=False)
def get_weather(lat, lon):
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

# キャッシュを一時解除（デバッグのため毎回新たに取得）
def gemini_generate_text(prompt, api_key, model_name):
    st.write("【Gemini送信プロンプト】")
    st.code(prompt, language="text")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
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
            # 応答例に合わせ、"content" の "parts" 配列から取得
            generated_text = candidates[0].get("content", {}).get("parts", [])[0].get("text", "").strip()
            return generated_text, raw_json
        else:
            return None, raw_json
    else:
        return None, raw_json

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

def predict_fire_spread(points, weather, duration_hours, api_key, model_name, fuel_type):
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
        '{"radius_m": 数値, "area_sqm": 数値, "water_volume_tons": 数値}\n'
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
    summary_text = gemini_generate_text(summary_prompt, API_KEY, model_name)
    return summary_text or "要約が取得できませんでした。"

def convert_json_for_map(original_json, center_lat, center_lon):
    """
    取得したJSON（例: "radius_m", "area_sqm", "water_volume_tons"）を元に、
    中心点 (center_lat, center_lon) をもとに円形の境界の座標リストを生成するJSON形式に変換するため、Gemini API に送信します。
    
    出力形式の例:
    {"coordinates": [[緯度, 経度], [緯度, 経度], ...]}
    他のテキストは一切含まないこと。
    入力JSON:
    {original_json}
    """
    prompt = (
        "以下のJSONは火災拡大の予測結果です。これを元に、中心点 ("
        f"{center_lat}, {center_lon}) を中心とした円形の境界を表す座標リストを生成してください。\n"
        "出力は必ず以下の形式にしてください。\n"
        '{"coordinates": [[緯度, 経度], [緯度, 経度], ...]}\n'
        "他のテキストは一切含まないこと。\n"
        "入力JSON:\n" + json.dumps(original_json)
    )
    with st.spinner("座標変換中..."):
        converted_text, raw = gemini_generate_text(prompt, API_KEY, MODEL_NAME)
    if not converted_text:
        st.error("座標変換用のGemini API応答が得られませんでした。")
        return None
    converted_json = extract_json(converted_text)
    return converted_json

def run_simulation(duration_hours, time_label):
    if not st.session_state.get("weather_data"):
        st.error("気象データが取得されていません。")
        return
    if not st.session_state.get("points"):
        st.error("発生地点が設定されていません。")
        return

    with st.spinner(f"{time_label}のシミュレーションを実行中..."):
        result = predict_fire_spread(st.session_state.points, st.session_state.weather_data, duration_hours, API_KEY, MODEL_NAME, fuel_type)
    
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
    st.write(f"**面積**: {area_sqm:.2f} m²")
    st.write("#### 必要放水量")
    st.info(f"{water_volume_tons:.2f} トン")
    
    summary_text = gemini_summarize_data(result, API_KEY, MODEL_NAME)
    st.write("#### Geminiによる要約")
    st.info(summary_text)
    
    progress = st.slider("延焼進捗 (%)", 0, 100, 100, key="progress_slider")
    fraction = progress / 100.0
    current_radius = radius_m * fraction
    
    lat_center, lon_center = st.session_state.points[0]
    wind_dir = st.session_state.weather_data.get("winddirection", 0)
    
    # JSON を再度 Gemini に送り、地図描写用の形式に変換する
    with st.spinner("座標変換中..."):
        converted = convert_json_for_map(result, lat_center, lon_center)
    if converted and "coordinates" in converted:
        coords = converted["coordinates"]
    else:
        coords = create_half_circle_polygon(lat_center, lon_center, current_radius, wind_dir)
    
    # 2D 表示/3D 表示の切り替え
    map_mode = st.radio("表示モード", ("2D", "3D"), key="map_mode")
    
    if map_mode == "2D":
        folium_map = folium.Map(location=[lat_center, lon_center], zoom_start=13)
        folium.Marker(location=[lat_center, lon_center], popup="火災発生地点", icon=folium.Icon(color="red")).add_to(folium_map)
        folium.Polygon(locations=coords, color="red", fill=True, fill_opacity=0.5).add_to(folium_map)
        st.write("#### Folium 地図（延焼範囲）")
        st_folium(folium_map, width=700, height=500)
    else:
        col_data = []
        scale_factor = 50  # 放水量に基づくスケール例
        try:
            water_val = float(water_volume_tons)
        except:
            water_val = 100
        for c in coords:
            col_data.append({
                "lon": c[0],
                "lat": c[1],
                "height": water_val / scale_factor
            })
        column_layer = pdk.Layer(
            "ColumnLayer",
            data=col_data,
            get_position='[lon, lat]',
            get_elevation='height',
            get_radius=30,
            elevation_scale=1,
            get_fill_color='[200, 30, 30, 100]',  # アルファ値100で透明度を上げる
            pickable=True,
            auto_highlight=True,
        )
        view_state = pdk.ViewState(
            latitude=lat_center,
            longitude=lon_center,
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

st.write("## 消火活動が行われない場合のシミュレーション")

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
