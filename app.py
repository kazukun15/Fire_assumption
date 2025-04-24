import streamlit as st
import requests
import csv
import math
import re
import pydeck as pdk
import folium
from streamlit_folium import st_folium
import numpy as np
from io import StringIO, BytesIO
from PIL import Image
import google.generativeai as genai

# --- ページ設定 ---
st.set_page_config(page_title="火災シミュレーション＋楕円モデル＋双方向地点設定＋Geminiレポート", layout="wide")

# --- Secrets の読み込み ---
MAPBOX_TOKEN        = st.secrets["mapbox"]["access_token"]
OPENWEATHER_API_KEY = st.secrets["openweather"]["api_key"]
FIRMS_MAP_KEY       = st.secrets["firms"]["map_key"]
GEMINI_API_KEY      = st.secrets["general"]["api_key"]

# --- Gemini 初期化 ---
genai.configure(api_key=GEMINI_API_KEY)
MODEL = genai.GenerativeModel('gemini-1.5-flash')

# --- セッションステート初期化 ---
if "fire_location" not in st.session_state:
    st.session_state.fire_location = (34.25743760177552, 133.2043209338966)

# --- サイドバー UI ---
st.sidebar.header("🔥 発生地点・シミュレーション設定")

# (1) テキスト入力による地点設定
latlon_text = st.sidebar.text_input(
    "発生地点 (lat, lon)",
    value=f"{st.session_state.fire_location[0]}, {st.session_state.fire_location[1]}"
)
if st.sidebar.button("テキストで更新"):
    m = re.match(r"\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*", latlon_text)
    if m:
        st.session_state.fire_location = (float(m[1]), float(m[2]))
        st.sidebar.success("発生地点をテキストで更新しました")
    else:
        st.sidebar.error("形式エラー：例 34.2574376, 133.2043209338966")

st.sidebar.markdown("---")
fuel_map   = {"森林（高燃料）":1.2, "草地（中燃料）":1.0, "都市部（低燃料）":0.8}
fuel_label = st.sidebar.selectbox("燃料特性", list(fuel_map.keys()))
fuel_coeff = fuel_map[fuel_label]

days       = st.sidebar.slider("経過日数 (日)", 1, 7, 4)
show_firms = st.sidebar.checkbox("FIRMSデータを重ねる", value=False)

# --- キャッシュ化したデータ取得関数 ---
@st.cache_data(ttl=600)
def get_weather(lat, lon):
    url = (
        f"https://api.openweathermap.org/data/2.5/weather?"
        f"lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
        "&units=metric&lang=ja"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=600)
def get_firms_area(lat, lon, days, map_key):
    delta = 0.5
    south, north = lat - delta, lat + delta
    west,  east   = lon - delta, lon + delta
    bbox = f"{west},{south},{east},{north}"
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/"
        f"{map_key}/VIIRS_SNPP_NRT/{bbox}/{days}"
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
    except:
        return []
    f = StringIO(r.text)
    reader = csv.DictReader(f)
    out = []
    for row in reader:
        try:
            out.append({
                "lat":   float(row["latitude"]),
                "lon":   float(row["longitude"]),
                "bright": float(row.get("bright_ti4", 0))
            })
        except:
            continue
    return out

def get_elevation(lat, lon):
    zoom = 14
    tx = int((lon + 180) / 360 * 2**zoom)
    ty = int((1 - math.log(math.tan(math.radians(lat)) + 1/math.cos(math.radians(lat))) / math.pi) / 2 * 2**zoom)
    url = f"https://api.mapbox.com/v4/mapbox.terrain-rgb/{zoom}/{tx}/{ty}.pngraw?access_token={MAPBOX_TOKEN}"
    r = requests.get(url, timeout=10)
    if r.status_code == 200:
        img = Image.open(BytesIO(r.content))
        arr = np.array(img)[128,128,:3].astype(np.int32)
        return -10000 + ((arr[0]*256*256 + arr[1]*256 + arr[2]) * 0.1)
    return 0

# --- 楕円モデルによる延焼範囲生成 ---
def generate_realistic_spread(lat, lon, base_radius, wind_speed, wind_dir, fuel_coeff):
    a = base_radius * fuel_coeff * (1 + wind_speed / 3.0)
    b = base_radius * fuel_coeff
    deg_per_m = 1 / 111000
    theta0 = math.radians(wind_dir)
    coords = []
    steps = 60
    for i in range(steps + 1):
        t = 2 * math.pi * i / steps
        x0 = a * math.cos(t)
        y0 = b * math.sin(t)
        x = x0 * math.cos(theta0) - y0 * math.sin(theta0)
        y = x0 * math.sin(theta0) + y0 * math.cos(theta0)
        lat_i = lat + (y * deg_per_m)
        lon_i = lon + (x * deg_per_m) / math.cos(math.radians(lat))
        elev = get_elevation(lat_i, lon_i) * 0.3
        coords.append([lon_i, lat_i, elev])
    return coords

# --- Gemini要約レポート生成 ---
def summarize_fire(lat, lon, place, wind, fuel, days, radius, area, water):
    prompt = (
        f"以下は火災シミュレーションの結果です。\n"
        f"- 発生地点: 緯度{lat}, 経度{lon}\n"
        f"- 風速: {wind['speed']} m/s, 風向: {wind['deg']}°\n"
        f"- 燃料特性: {fuel}\n"
        f"- 経過日数: {days}日\n"
        f"- 延焼半径: {radius:.1f} m\n"
        f"- 延焼面積: {area:.1f} ㎡\n"
        f"- 必要放水量: {water:.1f} トン\n\n"
        "これを踏まえ、以下を含む一般の方が理解しやすい日本語レポートを作成してください：\n"
        "1. 効果的な消火方法（推奨装備・戦術）\n"
        "2. 消火を重点的に行うべき場所（風下側要所、住宅密集地境界など）\n"
        "3. 今後予想される火勢の動きやリスクの変化\n"
        "4. 簡潔なまとめと提案\n"
    )
    resp = MODEL.generate_content(prompt)
    return resp.text.strip()

# --- メイン画面：Foliumクリックで地点設定 ---
st.subheader("▶ 発生地点を地図でクリック設定")
m = folium.Map(location=st.session_state.fire_location, zoom_start=12)
map_data = st_folium(m, width=700, height=400, returned_objects=["last_clicked"])
if map_data and map_data.get("last_clicked"):
    lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
    st.session_state.fire_location = (lat, lon)
    st.success(f"発生地点をマップで設定: {lat:.6f}, {lon:.6f}")

# --- シミュレーション結果表示 ---
st.subheader("🔥 火災シミュレーション結果")
lat_c, lon_c = st.session_state.fire_location
weather = get_weather(lat_c, lon_c)
wind    = weather.get("wind", {"speed":0, "deg":0})
place = f"{lat_c:.6f}, {lon_c:.6f}"  # 逆ジオコーディング省略

st.markdown(
    f"**発生地点:** {place}  \n"
    f"**風速:** {wind['speed']} m/s   **風向:** {wind['deg']}°   "
    f"**燃料:** {fuel_label}   **経過日数:** {days}日"
)

# レイヤー準備
layers = [ pdk.Layer("ScatterplotLayer",
    data=[{"position":[lon_c,lat_c]}],
    get_position="position", get_color=[0,0,255], get_radius=4) ]

# 延焼範囲
base_radius = (250 * fuel_coeff) + 10 * days * fuel_coeff
area_sqm     = math.pi * base_radius**2
water_tons   = (area_sqm / 10000) * 5
polygon = generate_realistic_spread(lat_c, lon_c, base_radius,
                                    wind["speed"], wind["deg"], fuel_coeff)
layers.append(pdk.Layer("PolygonLayer",
    data=[{"polygon": polygon}],
    get_polygon="polygon",
    get_fill_color=[255,0,0,80],
    extruded=False))

# FIRMS
if show_firms:
    spots = get_firms_area(lat_c, lon_c, days, FIRMS_MAP_KEY)
    pts = []
    for s in spots:
        c = min(max(int((s["bright"]-300)*2),0),255)
        pts.append({"position":[s["lon"],s["lat"]],"color":[255,255-c,0]})
    layers.append(pdk.Layer("ScatterplotLayer",
        data=pts,get_position="position",get_fill_color="color",
        get_radius=6000,pickable=False))
    st.success(f"FIRMSスポット: {len(spots)} 件表示")

# 地図描画
view = pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=12, pitch=45)
st.pydeck_chart(pdk.Deck(layers=layers,
    initial_view_state=view,
    map_style="mapbox://styles/mapbox/satellite-streets-v11"),
    use_container_width=True)

# Gemini レポート
report = summarize_fire(
    lat_c, lon_c, place, wind, fuel_label, days,
    base_radius, area_sqm, water_tons)
st.markdown("## 🔥 Gemini 要約レポート")
st.write(report)

# 生気象データ JSON
with st.expander("▼ 気象データ (JSON)"):
    st.json(weather)
