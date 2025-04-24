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
st.set_page_config(page_title="火災シミュレーション＋グラデーション範囲＋双方向地点＋Geminiレポート", layout="wide")

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

# --- データ取得関数 ---
@st.cache_data(ttl=600)
def get_weather(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric&lang=ja"
    r = requests.get(url, timeout=10); r.raise_for_status(); return r.json()

@st.cache_data(ttl=600)
def get_firms_area(lat, lon, days, map_key):
    delta = 0.5
    bbox = f"{lon-delta},{lat-delta},{lon+delta},{lat+delta}"
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/{map_key}/VIIRS_SNPP_NRT/{bbox}/{days}"
    try:
        r = requests.get(url, timeout=15); r.raise_for_status()
    except:
        return []
    f = StringIO(r.text)
    reader = csv.DictReader(f)
    return [{"lat":float(r["latitude"]),"lon":float(r["longitude"]),"bright":float(r.get("bright_ti4",0))} for r in reader if r.get("latitude")]

def get_elevation(lat, lon):
    zoom,deg_m = 14,1/111000
    tx = int((lon+180)/360*2**zoom)
    ty = int((1-math.log(math.tan(math.radians(lat))+1/math.cos(math.radians(lat)))/math.pi)/2*2**zoom)
    url = f"https://api.mapbox.com/v4/mapbox.terrain-rgb/{zoom}/{tx}/{ty}.pngraw?access_token={MAPBOX_TOKEN}"
    r = requests.get(url, timeout=10)
    if r.status_code==200:
        img=np.array(Image.open(BytesIO(r.content)))[128,128,:3].astype(np.int32)
        return -10000+((img[0]*256*256+img[1]*256+img[2])*0.1)
    return 0

# --- 楕円モデル延焼範囲＋グラデーション ---
def make_gradient_layers(lat, lon, base_radius, wind_speed, wind_dir, fuel_coeff, rings=10):
    layers=[]
    for i in range(1, rings+1):
        frac = i / rings
        r = base_radius * frac
        a = r * fuel_coeff * (1 + wind_speed/3.0)
        b = r * fuel_coeff
        theta = math.radians(wind_dir)
        coords=[]
        for t in np.linspace(0,2*math.pi,60):
            x0,y0=a*math.cos(t), b*math.sin(t)
            x = x0*math.cos(theta)-y0*math.sin(theta)
            y = x0*math.sin(theta)+y0*math.cos(theta)
            lat_i=lat+y*(1/111000)
            lon_i=lon+x*(1/111000)/math.cos(math.radians(lat))
            coords.append([lon_i,lat_i])
        alpha=int(80*(1-frac))
        layers.append(pdk.Layer("PolygonLayer",
            data=[{"polygon":coords}],
            get_polygon="polygon",
            get_fill_color=[255,0,0,alpha],
            extruded=False))
    return layers

# --- Gemini要約 ---
def summarize_fire(lat, lon, wind, fuel, days, radius, area, water):
    prompt=(f"火災地点緯度{lat},経度{lon}\n風速{wind['speed']}m/s風向{wind['deg']}°\n"
            f"燃料{fuel}経過{days}日\n延焼半径{radius:.1f}m面積{area:.1f}㎡水量{water:.1f}t\n"
            "1.効果的消火方法 2.重点消火場所 3.将来の火勢 4.まとめ提案")
    return MODEL.generate_content(prompt).text.strip()

# --- メイン画面: Foliumクリック ---
st.subheader("▶ 地図クリックで地点設定")
m=folium.Map(location=st.session_state.fire_location,zoom_start=12)
md=st_folium(m,width=700,height=400,returned_objects=["last_clicked"])
if md and md.get("last_clicked"):
    lat,lon=md["last_clicked"]["lat"],md["last_clicked"]["lng"]
    st.session_state.fire_location=(lat,lon)
    st.success(f"地点設定: {lat:.6f},{lon:.6f}")

# --- シミュ結果 ---
st.subheader("🔥 シミュレーション結果")
lat_c,lon_c=st.session_state.fire_location
weather=get_weather(lat_c,lon_c)
wind=weather.get("wind",{"speed":0,"deg":0})
st.markdown(f"**風速**{wind['speed']}m/s **風向**{wind['deg']}° **燃料**{fuel_label} **日数**{days}日")

# レイヤー
layers=[pdk.Layer("ScatterplotLayer",
    data=[{"position":[lon_c,lat_c]}],
    get_position="position",get_color=[0,0,255],get_radius=4)]

base_radius=(250*fuel_coeff)+10*days*fuel_coeff
area_sqm=math.pi*base_radius**2
water_tons=(area_sqm/10000)*5

# グラデーション楕円
layers+=make_gradient_layers(lat_c,lon_c,base_radius,
    wind["speed"],wind["deg"],fuel_coeff,rings=12)

# FIRMS
if show_firms:
    spots=get_firms_area(lat_c,lon_c,days,FIRMS_MAP_KEY)
    pts=[{"position":[s["lon"],s["lat"]],
          "color":[255,255-min(max(int((s["bright"]-300)*2),0),255),0]}
         for s in spots]
    layers.append(pdk.Layer("ScatterplotLayer",
        data=pts,get_position="position",get_fill_color="color",
        get_radius=6000,pickable=False))
    st.success(f"FIRMS: {len(spots)}件")

view=pdk.ViewState(latitude=lat_c,longitude=lon_c,zoom=12,pitch=45)
st.pydeck_chart(pdk.Deck(layers=layers,
    initial_view_state=view,
    map_style="mapbox://styles/mapbox/satellite-streets-v11"),
    use_container_width=True)

# Geminiレポート
report=summarize_fire(lat_c,lon_c,wind,fuel_label,days,base_radius,area_sqm,water_tons)
st.markdown("## 🔥 Gemini 要約レポート")
st.write(report)

# 生気象データ
with st.expander("▼ 気象データ (JSON)"):
    st.json(weather)
