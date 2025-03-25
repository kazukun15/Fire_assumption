import streamlit as st
import folium
from streamlit_folium import st_folium
import google.generativeai as genai
import requests
import json
import math
import re
import pydeck as pdk
import demjson3 as demjson  # Python3 用 demjson のフォーク
import time
import xml.etree.ElementTree as ET
from shapely.geometry import Point
import geopandas as gpd

# --- ページ設定 ---
st.set_page_config(page_title="火災拡大シミュレーション (2D/3D DEM＆雨雲オーバーレイ版)", layout="wide")

# --- API設定 ---
API_KEY = st.secrets["general"]["api_key"]
MODEL_NAME = "gemini-2.0-flash-001"
try:
    YAHOO_APPID = st.secrets["yahoo"]["appid"]
except Exception:
    YAHOO_APPID = None
    st.warning("Yahoo! API の appid が設定されていません。雨雲オーバーレイ機能は無効です。")
try:
    TAVILY_TOKEN = st.secrets["tavily"]["api_key"]
except Exception:
    TAVILY_TOKEN = None
    st.warning("Tavily のトークンが設定されていません。Tavily検証機能は無効です。")
try:
    MAPBOX_TOKEN = st.secrets["mapbox"]["access_token"]
except Exception:
    MAPBOX_TOKEN = None
    st.warning("Mapbox のアクセストークンが設定されていません。DEM 表示機能は無効です。")

# --- Gemini API の初期設定 ---
genai.configure(api_key=API_KEY)

# --- セッションステートの初期化 ---
if 'points' not in st.session_state:
    st.session_state.points = []
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = {}

# -----------------------------
# verify_with_tavily 関数（グローバル定義）
# -----------------------------
def verify_with_tavily(radius, wind_direction, water_volume):
    if not TAVILY_TOKEN:
        return ["Tavilyのトークンが設定されていないため、検証できません。"]
    try:
        url = "https://api.tavily.com/search"
        query = "火災 拡大半径 一般的"
        payload = {
            "query": query,
            "topic": "fire",
            "search_depth": "basic",
            "chunks_per_source": 3,
            "max_results": 1,
            "time_range": None,
            "days": 3,
            "include_answer": True,
            "include_raw_content": False,
            "include_images": False,
            "include_image_descriptions": False,
            "include_domains": [],
            "exclude_domains": []
        }
        headers = {
            "Authorization": f"Bearer {TAVILY_TOKEN}",
            "Content-Type": "application/json"
        }
        response = requests.post(url, json=payload, headers=headers)
        result = response.json()
        messages = []
        if "answer" in result and result["answer"]:
            messages.append(f"Tavily検索結果: {result['answer']}")
        else:
            messages.append("Tavily検索結果が見つかりませんでした。")
        return messages
    except Exception as e:
        st.error(f"Tavily検証中エラー: {e}")
        return ["Tavily検証中にエラーが発生しました。"]

# -----------------------------
# ヘルパー関数：燃焼日数に応じた色生成（1日→緑、10日以上→赤）
def get_color_by_days(days):
    ratio = min(days / 10, 1)
    r = int(255 * ratio)
    g = int(255 * (1 - ratio))
    b = 0
    return (r, g, b, 150)

def rgb_to_hex(color):
    r, g, b, a = color
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

# -----------------------------
# サイドバー入力（シミュレーション設定）
# -----------------------------
with st.sidebar.expander("シミュレーション設定", expanded=True):
    default_lat = st.number_input("初期中心 緯度", value=34.257586)
    default_lon = st.number_input("初期中心 経度", value=133.204356)
    fuel_type = st.selectbox("燃料特性", ["森林", "草地", "都市部"])
    scenario = st.radio("シナリオ選択", ("消火活動なし", "通常の消火活動あり"))
    display_mode = st.radio("表示モード", ("2D", "3D"))
    show_raincloud = st.checkbox("雨雲オーバーレイ表示", value=True)

if st.sidebar.button("発生地点を追加"):
    if "center" in st.session_state:
        new_point = st.session_state["center"]
        if isinstance(new_point, dict):
            new_point = [new_point.get("lat"), new_point.get("lng")]
    else:
        new_point = [default_lat, default_lon]
    st.session_state.points.append(new_point)
    st.sidebar.success(f"発生地点 {new_point} を追加しました。")
if st.sidebar.button("登録地点を消去"):
    st.session_state.points = []
    st.sidebar.info("全ての発生地点を削除しました。")

# -----------------------------
# 初期マップ表示（地図は一枚、上部に配置、十字は削除）
# -----------------------------
st.title("火災拡大シミュレーション＆雨雲オーバーレイ")
if st.session_state.points:
    center = st.session_state.points[-1]
else:
    center = [default_lat, default_lon]
if isinstance(center, dict):
    center = [center.get("lat"), center.get("lng")]
base_map = folium.Map(location=center, zoom_start=12)
for point in st.session_state.points:
    if isinstance(point, dict):
        point = [point.get("lat"), point.get("lng")]
    folium.Marker(location=point, icon=folium.Icon(color='red')).add_to(base_map)
map_data = st_folium(base_map, width=700, height=500)
if "center" in map_data:
    st.session_state["center"] = map_data["center"]

# -----------------------------
# 気象情報日本語表示関数
# -----------------------------
def display_weather_info(weather):
    try:
        temp = weather.get("temperature", "不明")
        wind_speed = weather.get("windspeed", "不明")
        wind_dir = weather.get("winddirection", "不明")
        humidity = weather.get("humidity", "不明")
        precipitation = weather.get("precipitation", "不明")
        info = f"""
**現在の気象情報**  
- 温度: {temp} °C  
- 風速: {wind_speed} m/s  
- 風向: {wind_dir} 度  
- 湿度: {humidity}%  
- 降水量: {precipitation} mm/h  
        """
        st.markdown(info)
    except Exception as e:
        st.error(f"気象情報表示中エラー: {e}")

# -----------------------------
# 雨雲データ取得関数（Yahoo! Weather API 使用例）
# -----------------------------
def get_raincloud_data(lat, lon):
    if not YAHOO_APPID:
        st.warning("Yahoo! API の appid が設定されていません。雨雲オーバーレイは無効です。")
        return None
    try:
        url = "https://map.yahooapis.jp/weather/V1/place"
        params = {
            "appid": YAHOO_APPID,
            "lat": lat,
            "lon": lon,
            "output": "xml"
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            clouds = root.find(".//clouds")
            if clouds is not None and clouds.get("img"):
                image_url = clouds.get("img")
                bounds = [[lat-1, lon-1], [lat+1, lon+1]]
                return {"image_url": image_url, "bounds": bounds}
        st.error("Yahoo! Weather API で雨雲情報が取得できませんでした。")
        return None
    except Exception as e:
        st.error(f"雨雲データ取得中エラー: {e}")
        return None

# -----------------------------
# 既存関数群
# -----------------------------
def extract_json(text: str) -> dict:
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
                st.error(f"demjson解析失敗: {e}")
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
                st.error(f"demjson解析失敗: {e}")
                return {}
    st.error("有効なJSONが見つかりませんでした。")
    return {}

@st.cache_data(show_spinner=False)
def get_weather(lat, lon):
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&current_weather=true&"
            f"hourly=relativehumidity_2m,precipitation&timezone=auto"
        )
        response = requests.get(url, timeout=10)
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
    except Exception as e:
        st.error(f"気象データ取得中エラー: {e}")
        return {}

def gemini_generate_text(prompt, api_key, model_name):
    try:
        st.write("【Gemini送信プロンプト】")
        st.code(prompt, language="text")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        response = requests.post(url, headers=headers, json=data, timeout=15)
        st.write("Gemini API ステータスコード:", response.status_code)
        raw_json = response.json()
        if response.status_code == 200 and raw_json:
            candidates = raw_json.get("candidates", [])
            if candidates:
                generated_text = candidates[0].get("content", {}).get("parts", [])[0].get("text", "").strip()
                return generated_text, raw_json
            else:
                return None, raw_json
        else:
            return None, raw_json
    except Exception as e:
        st.error(f"Gemini API呼び出し中エラー: {e}")
        return None, None

def create_half_circle_polygon(center_lat, center_lon, radius_m, wind_direction_deg):
    try:
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
    except Exception as e:
        st.error(f"座標生成中エラー: {e}")
        return []

def predict_fire_spread(points, weather, duration_hours, api_key, model_name, fuel_type):
    try:
        rep_lat, rep_lon = points[0]
        wind_speed = weather['windspeed']
        wind_dir = weather['winddirection']
        temperature = weather.get("temperature", "不明")
        humidity_info = f"{weather.get('humidity', '不明')}%"
        precipitation_info = f"{weather.get('precipitation', '不明')} mm/h"
        slope_info = "10度程度の傾斜"
        elevation_info = "標高150m程度"
        vegetation_info = "松林と草地混在"
        
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
    except Exception as e:
        st.error(f"火災拡大予測中エラー: {e}")
        return None

def gemini_summarize_data(json_data, api_key, model_name):
    try:
        json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
        summary_prompt = (
            "以下の火災拡大シミュレーション結果をもとに、火災の拡大の様子と、"
            "必要な消火水量および消火設備の提案を、一般の方が理解しやすい文章で説明してください。"
        )
        summary_text = gemini_generate_text(summary_prompt, api_key, model_name)
        return summary_text or "要約が取得できませんでした。"
    except Exception as e:
        st.error(f"要約生成中エラー: {e}")
        return "要約が取得できませんでした。"

def convert_json_for_map(original_json, center_lat, center_lon):
    try:
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
        if not converted_json or "coordinates" not in converted_json:
            st.error("座標変換結果が期待通りではありません。")
            return None
        return converted_json
    except Exception as e:
        st.error(f"座標変換中エラー: {e}")
        return None

def get_mountain_shape(center_lat, center_lon, radius_m):
    try:
        circle_coords = create_half_circle_polygon(center_lat, center_lon, radius_m, 0)
        mountain_coords = []
        for lon_val, lat_val in circle_coords:
            mountain_coords.append([lon_val + 0.0005 * math.sin(lat_val), lat_val + 0.0005 * math.cos(lon_val)])
        return mountain_coords
    except Exception as e:
        st.error(f"山岳形状生成中エラー: {e}")
        return create_half_circle_polygon(center_lat, center_lon, radius_m, 0)

def suggest_firefighting_equipment(terrain_info, effective_area_ha, extinguish_days):
    suggestions = []
    if effective_area_ha > 50:
        suggestions.append("大型消火車")
        suggestions.append("航空機")
        suggestions.append("消火ヘリ")
    else:
        suggestions.append("消火車")
        suggestions.append("消防ポンプ")
    if "傾斜" in terrain_info:
        suggestions.append("山岳消火装備")
    suggestions.append(f"消火日数の目安: 約 {extinguish_days:.1f} 日")
    return ", ".join(suggestions)

# 3D 表示でも延焼範囲は平面のポリゴンとして表示するための PolygonLayer
def get_flat_polygon_layer(coords, water_volume, color):
    polygon_data = [{"polygon": coords}]
    layer = pdk.Layer(
        "PolygonLayer",
        data=polygon_data,
        get_polygon="polygon",
        get_fill_color=str(color),
        pickable=True,
        auto_highlight=True,
    )
    return layer

# DEM 表示用 TerrainLayer（Mapbox）
def get_terrain_layer():
    if not MAPBOX_TOKEN:
        return None
    terrain_layer = pdk.Layer(
        "TerrainLayer",
        data=f"https://api.mapbox.com/v4/mapbox.terrain-rgb/{{z}}/{{x}}/{{y}}.pngraw?access_token={MAPBOX_TOKEN}",
        minZoom=0,
        maxZoom=15,
        meshMaxError=4,
        elevationDecoder={
            "rScaler": 256,
            "gScaler": 256,
            "bScaler": 256,
            "offset": -10000
        },
        elevationScale=1,
        getTerrainRGB=True,
    )
    return terrain_layer

# -----------------------------
# シミュレーション実行（期間は固定：10日＝240時間）
# -----------------------------
def run_simulation(time_label):
    duration_hours = 240  # 固定：10日間
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
    try:
        area_sqm = float(result.get("area_sqm", 0))
        area_ha = area_sqm / 10000.0
    except (KeyError, ValueError):
        area_sqm = "不明"
        area_ha = "不明"
    water_volume_tons = result.get("water_volume_tons", "不明")
    
    # 地図の中心は発生地点の最初の位置
    lat_center, lon_center = st.session_state.points[0]
    # 燃焼継続日数に応じた色（duration_hours/24）
    burn_days = duration_hours / 24
    color_rgba = get_color_by_days(burn_days)
    color_hex = rgb_to_hex(color_rgba)
    
    # 延焼形状：燃料特性に応じた形状
    if fuel_type == "森林":
        shape_coords = get_mountain_shape(lat_center, lon_center, radius_m)
    else:
        shape_coords = create_half_circle_polygon(lat_center, lon_center, radius_m, st.session_state.weather_data.get("winddirection", 0))
    
    # 地図（1枚だけ表示、延焼範囲は平面ポリゴンとして表示）
    if display_mode == "2D":
        final_map = folium.Map(location=[lat_center, lon_center], zoom_start=13)
        # 発火地点のマーカー（十字は削除）
        folium.Marker(location=[lat_center, lon_center], icon=folium.Icon(color="red")).add_to(final_map)
        if fuel_type == "森林":
            folium.Polygon(locations=shape_coords, color=color_hex, fill=True, fill_opacity=0.5).add_to(final_map)
        else:
            folium.Circle(location=[lat_center, lon_center], radius=radius_m, color=color_hex, fill=True, fill_opacity=0.5).add_to(final_map)
    else:
        # 3D 表示：延焼範囲は平面のポリゴンとして表示（PolygonLayer）＋背景に DEM
        polygon_layer = get_flat_polygon_layer(shape_coords, water_volume_tons, color_rgba)
        layers = [polygon_layer]
        if MAPBOX_TOKEN:
            terrain_layer = get_terrain_layer()
            if terrain_layer:
                layers.append(terrain_layer)
        view_state = pdk.ViewState(
            latitude=lat_center,
            longitude=lon_center,
            zoom=13,
            pitch=45,
            bearing=0,
            mapStyle="mapbox://styles/mapbox/light-v10" if MAPBOX_TOKEN else None,
        )
        final_map = None
        deck = pdk.Deck(layers=layers, initial_view_state=view_state)
    
    # ユーザー向けレポート（JSON 部分はすべて除去し、読みやすい文章）
    summary_text = gemini_summarize_data(result, API_KEY, MODEL_NAME)
    report_text = f"""
**シミュレーション結果：**

- 火災拡大半径: {radius_m:.2f} m  
- 拡大面積: {area_ha if isinstance(area_ha, str) else f'{area_ha:.2f}'} ヘクタール  
- 必要な消火水量: {water_volume_tons} トン  

【シナリオ別結果】  
"""
    if scenario == "通常の消火活動あり":
        suppression_factor = 0.5
        effective_radius = radius_m * suppression_factor
        effective_area = math.pi * (effective_radius ** 2)
        effective_area_ha = effective_area / 10000.0
        extinguish_days = effective_area_ha / 20.0
        terrain_info = "傾斜10度, 標高150m, 松林と草地混在"
        equipment_suggestions = suggest_firefighting_equipment(terrain_info, effective_area_ha, extinguish_days)
        report_text += f"""
・効果適用後の延焼半径: {effective_radius:.2f} m  
・効果適用後の延焼面積: {effective_area_ha:.2f} ヘクタール  
・推定消火完了日数: {extinguish_days:.1f} 日  
・推奨消火設備: {equipment_suggestions}
"""
    # レイアウト：上部に地図、下部にレポート
    st.markdown("---")
    st.subheader("シミュレーション結果マップ")
    if display_mode == "2D":
        st_folium(final_map, width=700, height=500)
    else:
        st.pydeck_chart(deck, key="pydeck_chart_" + str(time.time()))
    st.subheader("シミュレーションレポート")
    st.markdown(report_text)

# -----------------------------
# 気象データ取得ボタン
# -----------------------------
if st.button("気象データ取得"):
    weather_data = get_weather(default_lat, default_lon)
    if weather_data:
        st.session_state.weather_data = weather_data
        st.write("取得した気象データ（日本語表示）:")
        display_weather_info(weather_data)
    else:
        st.error("気象データの取得に失敗しました。")

st.write("## 消火活動が行われない場合のシミュレーション")
if st.button("シミュレーション実行"):
    run_simulation("10日後")
