import streamlit as st
import requests, csv, math, re
import pydeck as pdk
from io import StringIO

# --- ページ設定 ---
st.set_page_config(page_title="火災拡大シミュレーション＋FIRMS API", layout="wide")

# --- Secrets の読み込み ---
MAPBOX_TOKEN       = st.secrets["mapbox"]["access_token"]
OPENWEATHER_API_KEY= st.secrets["openweather"]["api_key"]
FIRMS_MAP_KEY      = st.secrets["firms"]["map_key"]

# セッションステート初期化：発生地点 (愛媛県付近)
if "fire_location" not in st.session_state:
    st.session_state.fire_location = (34.25743760177552, 133.2043209338966)

# --- サイドバー UI ---
st.sidebar.header("🔥 シミュレーション設定")

# 発生地点入力（Google マップ形式）
latlon_str = st.sidebar.text_input(
    "発生地点 (lat, lon)", 
    value=f"{st.session_state.fire_location[0]}, {st.session_state.fire_location[1]}"
)
if st.sidebar.button("地点を更新"):
    m = re.match(r"\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*", latlon_str)
    if m:
        st.session_state.fire_location = (float(m[1]), float(m[2]))
        st.sidebar.success("発生地点を更新しました。")
    else:
        st.sidebar.error("形式エラー：例 33.9884, 133.0449")

# 燃料特性
fuel_map = {"森林（高燃料）":1.2, "草地（中燃料）":1.0, "都市部（低燃料）":0.8}
fuel_label = st.sidebar.selectbox("燃料特性", list(fuel_map.keys()))
fuel_coeff = fuel_map[fuel_label]

# 日数（FIRMS API 取得期間にも使う）
days = st.sidebar.slider("経過日数", 1, 7, 1)

# FIRMS 表示トグル
show_firms = st.sidebar.checkbox("FIRMSホットスポットを重ねる", value=False)

# --- キャッシュ化したデータ取得関数 ---
@st.cache_data(ttl=600)
def get_weather(lat, lon):
    url = (
        f"https://api.openweathermap.org/data/2.5/weather?"
        f"lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric&lang=ja"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=600)
def get_firms_area(lat, lon, days, map_key):
    """
    FIRMS API の /api/area/ エンドポイントを使って
    [west,south,east,north] の範囲内で past <days> 日分の
    VIIRS_SNPP_NRT ホットスポットを取得する。
    """
    # 中心から ±0.5度 の範囲を例示
    delta = 0.5
    south, north = lat - delta, lat + delta
    west, east  = lon - delta, lon + delta
    bbox = f"{west},{south},{east},{north}"
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/"
        f"{map_key}/VIIRS_SNPP_NRT/{bbox}/{days}"
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        st.error(f"FIRMS API HTTPエラー ({r.status_code})")
        return []
    except requests.exceptions.RequestException as e:
        st.error(f"FIRMS API ネットワークエラー: {e}")
        return []

    # CSV パース
    f = StringIO(r.text)
    reader = csv.DictReader(f)
    out = []
    for row in reader:
        try:
            la = float(row["latitude"])
            lo = float(row["longitude"])
            bri= float(row.get("bright_ti4",0))
            out.append({"lat": la, "lon": lo, "bright": bri})
        except:
            continue
    return out

# --- 延焼範囲算出関数 ---
def generate_ellipse(lat0, lon0, base_r, wind_speed, wind_dir):
    a = base_r * (1 + wind_speed/2)  # 風下方向の長径
    b = base_r                      # 短径
    deg_per_m = 1/111000
    theta = math.radians(wind_dir)
    coords = []
    steps = 60
    for i in range(steps+1):
        t = 2*math.pi * i/steps
        x0, y0 = a*math.cos(t), b*math.sin(t)
        x =  x0*math.cos(theta) - y0*math.sin(theta)
        y =  x0*math.sin(theta) + y0*math.cos(theta)
        lat = lat0 + y*deg_per_m
        lon = lon0 + x*deg_per_m / math.cos(math.radians(lat0))
        coords.append([lon, lat])
    return coords

# --- メイン処理 ---
st.title("火災拡大シミュレーション + FIRMS")

# 天気取得
lat_c, lon_c = st.session_state.fire_location
weather = get_weather(lat_c, lon_c)
wind = weather.get("wind", {})
st.markdown(
    f"**風速:** {wind.get('speed','–')} m/s  **風向:** {wind.get('deg','–')}°  "
    f"**燃料:** {fuel_label}  **日数:** {days}日"
)

# ベースマップ用レイヤー
layers = [
    pdk.Layer(
        "ScatterplotLayer",
        data=[{"position":[lon_c,lat_c]}],
        get_position="position",
        get_color=[0,0,255],
        get_radius=5
    )
]

# 延焼範囲表示
base_radius = (200 * fuel_coeff) + (30 * days * fuel_coeff)
ellipse = generate_ellipse(lat_c, lon_c, base_radius, wind.get("speed",0), wind.get("deg",0))
layers.append(
    pdk.Layer(
        "PolygonLayer",
        data=[{"polygon": ellipse}],
        get_polygon="polygon",
        get_fill_color=[255,0,0,80],
        extruded=False
    )
)

# FIRMS ホットスポット重ねる
if show_firms:
    firms = get_firms_area(lat_c, lon_c, days, FIRMS_MAP_KEY)
    pts = []
    for item in firms:
        c = min(max(int((item["bright"]-300)*2),0),255)
        color = [255, 255-c, 0]
        pts.append({"position":[item["lon"], item["lat"]], "color": color})
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=pts,
            get_position="position",
            get_fill_color="color",
            get_radius=5000,
            pickable=False
        )
    )
    st.success(f"FIRMSホットスポットを {len(firms)} 件表示しました。")

# pydeck 表示
view = pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=11, pitch=0)
deck = pdk.Deck(
    layers=layers,
    initial_view_state=view,
    map_style="mapbox://styles/mapbox/light-v10",
    tooltip={"text": "{position}"}
)
st.pydeck_chart(deck, use_container_width=True)

# JSON 表示
with st.expander("▼ 天気データ (JSON)"):
    st.json(weather)
