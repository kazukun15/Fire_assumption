import streamlit as st
import folium
from streamlit_folium import st_folium
from folium import plugins
import google.generativeai as genai
import requests
import json
import math
import re
import pydeck as pdk
import demjson3 as demjson
import time
import xml.etree.ElementTree as ET
from shapely.geometry import Point
import geopandas as gpd

# --- ページ設定 ---
st.set_page_config(page_title="火災拡大シミュレーション (DEM＆自動表示)", layout="wide")

# --- API設定 ---
API_KEY = st.secrets["general"]["api_key"]
MODEL_NAME = "gemini-2.0-flash-001"
try:
    YAHOO_APPID = st.secrets["yahoo"]["appid"]
except Exception:
    YAHOO_APPID = None
try:
    TAVILY_TOKEN = st.secrets["tavily"]["api_key"]
except Exception:
    TAVILY_TOKEN = None
try:
    MAPBOX_TOKEN = st.secrets["mapbox"]["access_token"]
except Exception:
    MAPBOX_TOKEN = None

# --- Gemini API 初期設定 ---
genai.configure(api_key=API_KEY)

# --- セッションステート初期化 ---
if 'points' not in st.session_state:
    # 発生地点をデフォルトで設定
    st.session_state.points = [(34.257493583590986, 133.20437169456872)]
if 'weather_data' not in st.session_state:
    # 初期値として気象データを自動取得（取得に失敗した場合はダミーデータを設定）
    default_weather = {
        "temperature": 25,
        "windspeed": 3,
        "winddirection": 90,
        "humidity": 50,
        "precipitation": 0
    }
    try:
        st.session_state.weather_data = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude=34.257493583590986&longitude=133.20437169456872&current_weather=true&hourly=relativehumidity_2m,precipitation&timezone=auto", timeout=10
        ).json().get("current_weather", default_weather)
    except Exception:
        st.session_state.weather_data = default_weather

# --- グローバル変数 ---
default_lat = 34.257493583590986
default_lon = 133.20437169456872

# サイドバー（任意で変更可能）
fuel_type = st.sidebar.selectbox("燃料タイプ", ["森林", "草地", "都市部"])
scenario = st.sidebar.selectbox("消火シナリオ", ["通常の消火活動あり", "消火活動なし"])
map_style_choice = st.sidebar.selectbox("地図スタイル", ["カラー", "ダーク"])
if map_style_choice == "カラー":
    map_style_url = "mapbox://styles/mapbox/satellite-streets-v11"
else:
    map_style_url = "mapbox://styles/mapbox/dark-v10"
animation_type = st.sidebar.selectbox("アニメーションタイプ", 
                                      ["Full Circle", "Fan Shape", "Timestamped GeoJSON", "Color Gradient"])
show_raincloud = st.sidebar.checkbox("雨雲オーバーレイ", value=False)

def display_weather_info(weather_data):
    st.markdown("**現在の気象情報:**")
    st.write(f"温度: {weather_data.get('temperature', '不明')} °C")
    st.write(f"風速: {weather_data.get('windspeed', '不明')} m/s")
    st.write(f"風向: {weather_data.get('winddirection', '不明')} 度")
    st.write(f"湿度: {weather_data.get('humidity', '不明')} %")
    st.write(f"降水量: {weather_data.get('precipitation', '不明')} mm/h")

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

# ----- 以下、各種関数 -----
def verify_with_tavily(radius, wind_direction, water_volume):
    if not TAVILY_TOKEN:
        return ["Tavilyのトークンが設定されていません。"]
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
        headers = {"Authorization": f"Bearer {TAVILY_TOKEN}", "Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers)
        result = response.json()
        messages = []
        if "answer" in result and result["answer"]:
            messages.append(f"Tavily検索結果: {result['answer']}")
        else:
            messages.append("Tavily検索結果が見つかりません。")
        return messages
    except Exception as e:
        st.error(f"Tavily検証中エラー: {e}")
        return ["Tavily検証中にエラー発生。"]

def get_color_by_days(days):
    ratio = min(days / 10, 1)
    r = int(255 * ratio)
    g = int(255 * (1 - ratio))
    b = 0
    return (r, g, b, 150)

def rgb_to_hex(color):
    r, g, b, a = color
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def get_weather_open_meteo(lat, lon):
    try:
        url = (f"https://api.open-meteo.com/v1/forecast?"
               f"latitude={lat}&longitude={lon}&current_weather=true&"
               f"hourly=relativehumidity_2m,precipitation&timezone=auto")
        response = requests.get(url, timeout=10)
        st.write("Open-Meteo API ステータスコード:", response.status_code)
        data = response.json()
        current = data.get("current_weather", {})
        result = {
            "temperature": current.get("temperature", "不明"),
            "windspeed": current.get("windspeed", "不明"),
            "winddirection": current.get("winddirection", "不明")
        }
        current_time = current.get("time")
        if current_time and "hourly" in data:
            times = data["hourly"].get("time", [])
            if current_time in times:
                idx = times.index(current_time)
                result["humidity"] = data["hourly"].get("relativehumidity_2m", ["不明"])[idx]
                result["precipitation"] = data["hourly"].get("precipitation", ["不明"])[idx]
        return result
    except Exception as e:
        st.error(f"気象データ取得中エラー: {e}")
        return {}

@st.cache_data(show_spinner=False)
def get_weather(lat, lon):
    return get_weather_open_meteo(lat, lon)

def gemini_generate_text(prompt, api_key, model_name):
    with st.expander("Gemini送信プロンプト（折りたたみ）"):
        st.code(prompt, language="text")
    try:
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

# ----- ポリゴン生成 -----
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

def create_fan_polygon(center_lat, center_lon, radius_m, wind_direction_deg, angle=60):
    try:
        deg_per_meter = 1.0 / 111000.0
        start_angle = wind_direction_deg - angle/2
        end_angle = wind_direction_deg + angle/2
        num_steps = 20
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
        coords.append([center_lon, center_lat])
        return coords
    except Exception as e:
        st.error(f"扇状ポリゴン生成中エラー: {e}")
        return []

def generate_timestamped_features(center_lat, center_lon, max_radius, steps, wind_direction, polygon_func):
    features = []
    base_time = time.time()
    for i in range(steps):
        t_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(base_time + i * 60))
        current_radius = max_radius * (i+1)/steps
        polygon_coords = polygon_func(center_lat, center_lon, current_radius, wind_direction) if polygon_func else []
        feature = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [polygon_coords]},
            "properties": {"times": [t_iso]}
        }
        features.append(feature)
    return features

def fallback_fire_spread(points, weather, duration_hours, fuel_type):
    rep_lat, rep_lon = points[0]
    wind_speed = weather.get("windspeed", 1)
    radius_m = 10 + wind_speed * duration_hours * 50
    area_sqm = math.pi * radius_m**2
    water_volume_tons = area_sqm / 10000.0 * 5
    return {"radius_m": radius_m, "area_sqm": area_sqm, "water_volume_tons": water_volume_tons}

def predict_fire_spread(points, weather, duration_hours, api_key, model_name, fuel_type):
    try:
        rep_lat, rep_lon = points[0]
        wind_speed = weather.get("windspeed", "不明")
        wind_dir = weather.get("winddirection", "不明")
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
        with st.expander("生JSON応答（折りたたみ）"):
            st.json(raw_json)
        if not generated_text:
            raise Exception("Gemini APIから有効な応答が得られませんでした。")
        prediction_json = extract_json(generated_text)
        required_keys = ["radius_m", "area_sqm", "water_volume_tons"]
        if not prediction_json or not all(key in prediction_json for key in required_keys):
            raise Exception("Gemini APIの結果が不完全です。")
        return prediction_json
    except Exception as e:
        st.warning("Gemini APIによる分析が完了しなかったため、フォールバックシミュレーションを使用します。")
        return fallback_fire_spread(points, weather, duration_hours, fuel_type)

def gemini_summarize_data(json_data, api_key, model_name):
    return "シミュレーション結果に基づくレポートを表示します。"

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
        for coord in circle_coords:
            lon_val, lat_val = coord
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

def get_flat_polygon_layer(coords, water_volume, color):
    polygon_data = [{"polygon": coords}]
    layer = pdk.Layer(
        "PolygonLayer",
        data=polygon_data,
        get_polygon="polygon",
        get_fill_color=list(color),
        pickable=True,
        auto_highlight=True,
    )
    return layer

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
            "rScaler": 256 * 256 * 0.1,
            "gScaler": 256 * 0.1,
            "bScaler": 0.1,
            "offset": -10000
        },
        elevationScale=1,
        getTerrainRGB=True,
    )
    return terrain_layer

def get_raincloud_data(lat, lon):
    return {
        "image_url": "https://www.example.com/raincloud.png",
        "bounds": [[lat - 0.05, lon - 0.05], [lat + 0.05, lon + 0.05]]
    }

# ----- シミュレーション実行（72時間＝3日間） -----
def run_simulation(time_label):
    duration_hours = 72
    if not st.session_state.get("weather_data"):
        st.error("気象データが取得されていません。")
        return
    if not st.session_state.get("points"):
        st.error("発生地点が設定されていません。")
        return

    with st.spinner(f"{time_label}のシミュレーションを実行中..."):
        result = predict_fire_spread(
            st.session_state.points,
            st.session_state.weather_data,
            duration_hours,
            API_KEY,
            MODEL_NAME,
            fuel_type
        )
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
    
    lat_center, lon_center = st.session_state.points[0]
    burn_days = duration_hours / 24
    color_rgba = get_color_by_days(burn_days)
    color_hex = rgb_to_hex(color_rgba)
    
    # 静的延焼範囲（最大可能性：消火活動なし）
    if fuel_type == "森林":
        static_coords = get_mountain_shape(lat_center, lon_center, radius_m)
    else:
        static_coords = create_half_circle_polygon(lat_center, lon_center, radius_m,
                                                    st.session_state.weather_data.get("winddirection", 0))
    
    # 1. Foliumマップ生成（静的レイヤーと動的レイヤーを統合）
    m = folium.Map(location=[lat_center, lon_center], zoom_start=13, tiles="OpenStreetMap", control_scale=True)
    # 静的レイヤー：延焼範囲
    folium.Polygon(locations=static_coords, color=color_hex, fill=True, fill_opacity=0.5,
                   tooltip="最大延焼範囲 (静的)").add_to(m)
    # 発生地点：小さな赤丸
    folium.CircleMarker(location=[lat_center, lon_center], radius=5, color="red", fill=True, fill_color="red",
                        tooltip="発生地点").add_to(m)
    # 動的レイヤー用 FeatureGroupの追加
    dynamic_fg = folium.FeatureGroup(name="Dynamic Animation")
    m.add_child(dynamic_fg)
    
    # 2. 静的結果とレポートの表示（画面上部に表示）
    st.subheader("シミュレーション結果 (静的表示)")
    st.markdown(f"""
**シミュレーション結果：**

- **最大延焼半径**: {radius_m:.2f} m  
- **拡大面積**: {area_ha if isinstance(area_ha, str) else f'{area_ha:.2f}'} ヘクタール  
- **必要な消火水量**: {water_volume_tons} トン  
- **燃焼継続日数**: {burn_days:.1f} 日  
""")
    report_text = f"""
### 詳細レポート
1. **地形について**  
   - 傾斜は約10度、標高は150m程度と仮定（DEMの自動取得はMapbox TerrainLayerで実現可能）。  
   - 対象地域は松林と草地が混在し、燃料タイプは「{fuel_type}」です。  
   - 地形の複雑さにより、火災は斜面に沿って急速に延焼する可能性があります。
2. **延焼の仕方**  
   - 風向 {st.session_state.weather_data.get("winddirection", "不明")} 度、風速 {st.session_state.weather_data.get("windspeed", "不明")} m/s により、火災は風下側へ不均一に延焼。  
   - 最大可能性の延焼範囲は全方向に広がると仮定。
3. **可能性について**  
   - 早期消火の重要性と、継続的なモニタリングが求められます。
"""
    st.markdown(report_text)
    
    # 3. アニメーション表示（同一マップ上で動的レイヤーを更新）
    st.subheader("延焼範囲アニメーション")
    animation_placeholder = st.empty()  # アニメーション用プレースホルダー
    if st.button("延焼範囲アニメーション開始", key="anim_start"):
        if animation_type == "Full Circle":
            for r in range(0, int(radius_m) + 1, max(1, int(radius_m)//20)):
                dynamic_fg._children.clear()
                folium.Circle(location=[lat_center, lon_center], radius=r,
                              color=color_hex, fill=True, fill_opacity=0.5).add_to(dynamic_fg)
                with animation_placeholder.container():
                    st_folium(m, width=700, height=500)
                time.sleep(0.1)
        elif animation_type == "Fan Shape":
            wind_direction = st.session_state.weather_data.get("winddirection", 0)
            for r in range(0, int(radius_m) + 1, max(1, int(radius_m)//20)):
                dynamic_fg._children.clear()
                fan_coords = create_fan_polygon(lat_center, lon_center, r, wind_direction, angle=60)
                folium.Polygon(locations=fan_coords, color=color_hex, fill=True, fill_opacity=0.5).add_to(dynamic_fg)
                with animation_placeholder.container():
                    st_folium(m, width=700, height=500)
                time.sleep(0.1)
        elif animation_type == "Timestamped GeoJSON":
            steps = 20
            wind_direction = st.session_state.weather_data.get("winddirection", 0)
            features = generate_timestamped_features(lat_center, lon_center, radius_m, steps, wind_direction,
                                                     lambda lat, lon, r, wd: create_half_circle_polygon(lat, lon, r, 0))
            geojson = {"type": "FeatureCollection", "features": features}
            dynamic_fg._children.clear()
            plugins.TimestampedGeoJson(geojson, period="PT1M", add_last_point=True, loop=True, auto_play=True).add_to(m)
            with animation_placeholder.container():
                st_folium(m, width=700, height=500)
        elif animation_type == "Color Gradient":
            steps = max(1, int(radius_m)//20)
            for i, r in enumerate(range(0, int(radius_m) + 1, steps)):
                ratio = i / (radius_m/steps) if (radius_m/steps) != 0 else 0
                r_val = 255
                g_val = int(255 * ratio)
                b_val = 0
                dynamic_color = (r_val, g_val, b_val, 150)
                dynamic_hex = rgb_to_hex(dynamic_color)
                dynamic_fg._children.clear()
                folium.Circle(location=[lat_center, lon_center], radius=r,
                              color=dynamic_hex, fill=True, fill_opacity=0.5).add_to(dynamic_fg)
                with animation_placeholder.container():
                    st_folium(m, width=700, height=500)
                time.sleep(0.1)
    
    # 4. 雨雲オーバーレイ（オプション）
    if show_raincloud:
        rain_data = get_raincloud_data(lat_center, lon_center)
        if rain_data:
            folium.raster_layers.ImageOverlay(
                image=rain_data["image_url"],
                bounds=rain_data["bounds"],
                opacity=0.4,
                interactive=True,
                cross_origin=False,
                zindex=1,
            ).add_to(m)
    
    st.subheader("最終地図（静的＋動的）")
    st_folium(m, width=700, height=500)

# ----- サイドバー：発生地点の設定 -----
st.sidebar.subheader("発生地点の設定")
lat_input = st.sidebar.text_input("緯度", value=str(default_lat))
lon_input = st.sidebar.text_input("経度", value=str(default_lon))
if st.sidebar.button("発生地点を設定"):
    try:
        lat_val = float(lat_input)
        lon_val = float(lon_input)
        st.session_state.points = [(lat_val, lon_val)]
        st.success("発生地点が設定されました。")
    except ValueError:
        st.error("有効な数値を入力してください。")

# ----- 気象データの取得 -----
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
    run_simulation("3日後")
