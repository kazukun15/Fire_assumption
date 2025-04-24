import streamlit as st
import requests, csv, math, re
import pydeck as pdk
from io import StringIO

# --- ページ設定 ---
st.set_page_config(page_title="火災拡大シミュレーション＋FIRMS", layout="wide")

# --- Secrets ---
MAPBOX_TOKEN = st.secrets["mapbox"]["access_token"]
OPENWEATHER_API_KEY = st.secrets["openweather"]["api_key"]

# --- セッションステート初期化 ---
if "fire_location" not in st.session_state:
    st.session_state.fire_location = (34.25743760177552, 133.2043209338966)

# --- 入力パネル (サイドバー) ---
st.sidebar.header("🔥 シミュレーション設定")

# 発生地点
latlon_str = st.sidebar.text_input(
    "発生地点 (lat, lon)",
    value=f"{st.session_state.fire_location[0]}, {st.session_state.fire_location[1]}"
)
if st.sidebar.button("地点を更新"):
    m = re.match(r"\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*", latlon_str)
    if m:
        st.session_state.fire_location = (float(m[1]), float(m[2]))
        st.sidebar.success("地点を更新しました。")
    else:
        st.sidebar.error("形式エラー：例 33.9884, 133.0449")

# 燃料特性
fuel_map = {"森林（高燃料）":1.2, "草地（中燃料）":1.0, "都市部（低燃料）":0.8}
fuel_label = st.sidebar.selectbox("燃料特性", list(fuel_map.keys()))
fuel_coeff = fuel_map[fuel_label]

# 経過日数
days = st.sidebar.slider("経過日数", 1, 7, 1)

st.sidebar.markdown("---")
st.sidebar.header("🛰 FIRMS設定")
show_firms = st.sidebar.checkbox("FIRMSを表示", value=False)

# --- キャッシュ化したデータ取得関数 ---
@st.cache_data(ttl=600)
def get_weather(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=600)
def get_firms():
    url = "https://firms.modaps.eosdis.nasa.gov/active_fire/c6.1/viirs24.txt"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    f = StringIO(r.text)
    reader = csv.DictReader(f)
    pts = []
    for row in reader:
        try:
            lat = float(row["latitude"]); lon = float(row["longitude"])
            bri = float(row.get("bright_ti4",0))
            pts.append((lat, lon, bri))
        except:
            continue
    return pts

# --- 延焼範囲算出関数 ---
def generate_ellipse(lat0, lon0, base_r, wind_speed, wind_dir):
    a = base_r * (1 + wind_speed/2)  # 長径
    b = base_r                      # 短径
    deg_per_m = 1/111000
    theta = math.radians(wind_dir)
    coords = []
    steps = 60
    for i in range(steps+1):
        t = 2*math.pi*i/steps
        x0 = a*math.cos(t); y0 = b*math.sin(t)
        x =  x0*math.cos(theta) - y0*math.sin(theta)
        y =  x0*math.sin(theta) + y0*math.cos(theta)
        lat = lat0 + (y*deg_per_m)
        lon = lon0 + (x*deg_per_m)/math.cos(math.radians(lat0))
        coords.append([lon, lat])
    return coords

# --- マップ初期化 ---
lat_c, lon_c = st.session_state.fire_location
view = pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=11, pitch=0)

# --- メイン表示 ---
st.title("火災拡大シミュレーション with FIRMS")

# 1. 天気データ取得＆表示
with st.spinner("気象データ取得中…"):
    weather = get_weather(lat_c, lon_c)
wind = weather.get("wind", {})
st.markdown(f"**⚙️ 風速:** {wind.get('speed', '–')} m/s  **風向:** {wind.get('deg','–')}°  **燃料:** {fuel_label}  **日数:** {days}日")

# 2. 基本マップレイヤー
layers = []
# 発生地点マーカー
layers.append(pdk.Layer(
    "ScatterplotLayer",
    data=[{"position":[lon_c,lat_c]}],
    get_position="position", get_color=[0,0,255], get_radius=5
))

# 3. 延焼範囲表示
base_radius = 200 * fuel_coeff + 30 * days * fuel_coeff
ellipse = generate_ellipse(lat_c, lon_c, base_radius, wind.get("speed",0), wind.get("deg",0))
layers.append(pdk.Layer(
    "PolygonLayer",
    data=[{"polygon": ellipse}],
    get_polygon="polygon", get_fill_color=[255,0,0,80], extruded=False
))

# 4. FIRMS表示（オプション）
if show_firms:
    pts = get_firms()
    data = []
    for la, lo, bri in pts:
        # 明るさ→色変換 (高いほど赤寄り)
        c = min(max(int((bri-300)*2),0),255)
        color = [255, 255-c, 0]
        data.append({"position":[lo,la], "color": color})
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position="position", get_fill_color="color",
        get_radius=8000, pickable=False
    ))

# 5. pydeck描画
deck = pdk.Deck(
    layers=layers,
    initial_view_state=view,
    map_style="mapbox://styles/mapbox/light-v10"
)
st.pydeck_chart(deck, use_container_width=True)

# 6. FIRMS件数表示
if show_firms:
    st.success(f"FIRMSホットスポット: {len(pts)} 件を表示")

# 7. 気象データのJSONを折りたたみ
with st.expander("▼ 生気象データ (JSON)"):
    st.json(weather)
