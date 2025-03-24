import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import json
import re
import demjson3 as demjson

# --- ページ設定 ---
st.set_page_config(page_title="火災拡大シミュレーション", layout="wide")

# API設定
API_KEY = st.secrets["general"]["api_key"]
MODEL_NAME = "gemini-2.0-flash-001"

# --- gemini_generate_text 関数 ---
@st.cache_data(show_spinner=False)
def gemini_generate_text(prompt, api_key, model_name):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    response = requests.post(url, headers=headers, json=data)

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
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return demjson.decode(json_str)
    st.error("有効なJSON文字列が見つかりませんでした。"); return {}

# --- メインUI ---
st.title("火災拡大シミュレーション")

with st.sidebar:
    lat = st.number_input("緯度", value=34.257586)
    lon = st.number_input("経度", value=133.204356)
    fuel_type = st.selectbox("燃料特性", ["森林", "草地", "都市部"])

map_placeholder = st.empty()

if st.button("シミュレーション実行"):
    prompt = (
        f"火災シミュレーションを実施します。\n"
        f"発生地点: 緯度{lat}, 経度{lon}\n燃料特性: {fuel_type}\n"
        "結果をJSONで: {'radius_m':数値,'area_sqm':数値,'water_volume_tons':数値}"
    )

    response_text = gemini_generate_text(prompt, API_KEY, MODEL_NAME)
    if not response_text:
        st.stop()

    result = extract_json(response_text)
    if not result:
        st.stop()

    radius_m = float(result["radius_m"])

    m = folium.Map(location=[lat, lon], zoom_start=14)
    folium.Marker([lat, lon], popup="火災発生地点", icon=folium.Icon(color='red')).add_to(m)
    folium.Circle(
        [lat, lon], radius=radius_m, color='orange', fill=True, fill_opacity=0.4
    ).add_to(m)

    with map_placeholder:
        st_folium(m, width=700, height=500)

    st.write(f"半径: {result['radius_m']}m")
    st.write(f"面積: {result['area_sqm']}㎡")
    st.write(f"必要な消火水量: {result['water_volume_tons']}トン")
