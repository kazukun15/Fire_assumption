import streamlit as st
import requests, csv, math, re
import pydeck as pdk
from io import StringIO
import google.generativeai as genai

# --- ページ設定 ---
st.set_page_config(page_title="火災拡大シミュレーション＋Gemini報告", layout="wide")

# --- Secrets の読み込み ---
MAPBOX_TOKEN        = st.secrets["mapbox"]["access_token"]
OPENWEATHER_API_KEY = st.secrets["openweather"]["api_key"]
FIRMS_MAP_KEY       = st.secrets["firms"]["map_key"]
GEMINI_API_KEY      = st.secrets["general"]["api_key"]

# Gemini 初期化
genai.configure(api_key=GEMINI_API_KEY)
MODEL = genai.GenerativeModel('gemini-1.5-flash')

# --- セッションステート初期化 ---
if "fire_location" not in st.session_state:
    st.session_state.fire_location = (34.25743760177552, 133.2043209338966)

# --- サイドバー UI ---
st.sidebar.header("🔥 シミュレーション設定")
latlon_in = st.sidebar.text_input(
    "発生地点 (lat, lon)",
    value=f"{st.session_state.fire_location[0]}, {st.session_state.fire_location[1]}"
)
if st.sidebar.button("地点更新"):
    m = re.match(r"\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*", latlon_in)
    if m:
        st.session_state.fire_location = (float(m[1]), float(m[2]))
        st.sidebar.success("地点を更新しました")
    else:
        st.sidebar.error("形式エラー")

fuel_map    = {"森林（高燃料）":1.2, "草地（中燃料）":1.0, "都市部（低燃料）":0.8}
fuel_label  = st.sidebar.selectbox("燃料特性", list(fuel_map.keys()))
fuel_coeff  = fuel_map[fuel_label]
days        = st.sidebar.slider("経過日数", 1, 7, 1)
show_firms  = st.sidebar.checkbox("FIRMSホットスポットを重ねる", value=False)

# --- キャッシュ化したデータ取得関数 ---
@st.cache_data(ttl=600)
def get_weather(lat, lon):
    url = (f"https://api.openweathermap.org/data/2.5/weather?"
           f"lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric&lang=ja")
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=600)
def get_firms_area(lat, lon, days, map_key):
    delta = 0.5
    bbox = f"{lon-delta},{lat-delta},{lon+delta},{lat+delta}"
    url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/"
           f"{map_key}/VIIRS_SNPP_NRT/{bbox}/{days}")
    try:
        r = requests.get(url, timeout=15); r.raise_for_status()
    except:
        return []
    f = StringIO(r.text)
    reader = csv.DictReader(f)
    out=[]
    for row in reader:
        try:
            out.append({
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
                "bright": float(row.get("bright_ti4", 0))
            })
        except:
            continue
    return out

# --- 延焼範囲算出関数 ---
def generate_ellipse(lat0, lon0, base_r, wind_speed, wind_dir):
    a=base_r*(1+wind_speed/2); b=base_r
    deg_m=1/111000; θ=math.radians(wind_dir)
    pts=[]; steps=60
    for i in range(steps+1):
        t=2*math.pi*i/steps
        x0,y0=a*math.cos(t), b*math.sin(t)
        x= x0*math.cos(θ)-y0*math.sin(θ)
        y= x0*math.sin(θ)+y0*math.cos(θ)
        lat=lat0+y*deg_m
        lon=lon0+x*deg_m/math.cos(math.radians(lat0))
        pts.append([lon,lat])
    return pts

# --- Gemini 要約関数 ---
def summarize_fire(lat, lon, wind, fuel, days, radius_m, area_sqm, water_tons):
    prompt = (
        f"以下の火災シミュレーション結果を一般の人がわかりやすく"
        f"日本語で要約してください。\n"
        f"- 発生地点: 緯度{lat}, 経度{lon}\n"
        f"- 風速: {wind.get('speed',0)} m/s, 風向: {wind.get('deg',0)}°\n"
        f"- 燃料特性: {fuel}\n"
        f"- 経過日数: {days}日\n"
        f"- 延焼半径: {radius_m:.1f} m\n"
        f"- 延焼面積: {area_sqm:.1f} m²\n"
        f"- 必要放水量: {water_tons:.1f} トン\n"
    )
    resp = MODEL.generate_content(prompt)
    return resp.text.strip()

# --- メイン表示 ---
st.title("火災拡大シミュレーション + Gemini報告")

lat_c, lon_c = st.session_state.fire_location
weather = get_weather(lat_c, lon_c)
wind    = weather.get("wind", {})

st.markdown(
    f"**風速:** {wind.get('speed','–')} m/s  "
    f"**風向:** {wind.get('deg','–')}°  "
    f"**燃料:** {fuel_label}  **日数:** {days}日"
)

# レイヤー準備
layers = [
    pdk.Layer("ScatterplotLayer",
              data=[{"position":[lon_c,lat_c]}],
              get_position="position", get_color=[0,0,255], get_radius=5)
]
# 延焼範囲
base_r = 200*fuel_coeff + 30*days*fuel_coeff
ellipse = generate_ellipse(lat_c, lon_c, base_r,
                           wind.get("speed",0), wind.get("deg",0))
# 面積計算
# m → deg conversion approximate, skip precise GIS area calc for brevity
area = math.pi * base_r**2

layers.append(
    pdk.Layer("PolygonLayer",
              data=[{"polygon": ellipse}],
              get_polygon="polygon", get_fill_color=[255,0,0,80],
              extruded=False)
)

# FIRMS
if show_firms:
    firms = get_firms_area(lat_c, lon_c, days, FIRMS_MAP_KEY)
    pts=[]
    for fpt in firms:
        c=min(max(int((fpt["bright"]-300)*2),0),255)
        pts.append({"position":[fpt["lon"],fpt["lat"]],
                    "color":[255,255-c,0]})
    layers.append(
        pdk.Layer("ScatterplotLayer",
                  data=pts,
                  get_position="position",
                  get_fill_color="color",
                  get_radius=5000,
                  pickable=False)
    )
    st.success(f"FIRMSスポット {len(firms)} 件表示")

# pydeck 描画
view = pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=11, pitch=0)
st.pydeck_chart(pdk.Deck(layers=layers,
                        initial_view_state=view,
                        map_style="mapbox://styles/mapbox/light-v10"))

# Geminiレポート
radius_m = base_r
area_sqm = area
water_tons = area / 10000 * 5  # 仮の係数: 1haあたり5t
report = summarize_fire(lat_c, lon_c, wind,
                        fuel_label, days,
                        radius_m, area_sqm, water_tons)
st.markdown("## 🔥 Geminiによる火災予測レポート")
st.write(report)

# 元データ確認
with st.expander("▼ 気象データ (JSON)"):
    st.json(weather)
