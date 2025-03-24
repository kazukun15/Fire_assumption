import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import json
import math
import re
import demjson3 as demjson  # Python3 用の demjson のフォーク

# --- ページ設定 ---
st.set_page_config(page_title="火災拡大シミュレーション", layout="wide")

# --- API設定 ---
API_KEY = st.secrets["general"]["api_key"]
MODEL_NAME = "gemini-2.0-flash-001"

# --- gemini_generate_text 関数 ---
@st.cache_data(show_spinner=False)
def gemini_generate_text(prompt, api_key, model_name):
    """
    Gemini API にリクエストを送り、テキスト生成を行います。
    正常なら、応答の content 部分（マークダウンのコードブロック内のテキスト）を返します。
    """
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
    テキストからJSONオブジェクトを抽出します。
    マークダウンのコードブロック形式（```json ... ```）に対応し、
    直接 json.loads() でパースできない場合は demjson3 で解析を試みます。
    """
    text = text.strip()
    # まず、コードブロック形式を対象とした正規表現を試す
    pattern_md = r"```json\s*(\{[\s\S]*?\})\s*```"
    match = re.search(pattern_md, text)
    if match:
        json_str = match.group(1)
    else:
        # コードブロック形式でない場合、最初に現れる { ... } を抽出
        pattern = r"\{[\s\S]*\}"
        match = re.search(pattern, text)
        if match:
            json_str = match.group(0)
        else:
            st.error("JSON文字列が見つかりませんでした。")
            return {}
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            return demjson.decode(json_str)
        except demjson.JSONDecodeError:
            st.error("JSON解析に失敗しました。")
            return {}

# --- メインUI ---
st.title("火災拡大シミュレーション")

with st.sidebar:
    lat = st.number_input("緯度", value=34.257586)
    lon = st.number_input("経度", value=133.204356)
    fuel_type = st.selectbox("燃料特性", ["森林", "草地", "都市部"])

# --- シミュレーション実行 ---
if st.button("シミュレーション実行"):
    # プロンプト例（Gemini に対しては、指定の形式で JSON を返すように強く指示）
    prompt = (
        f"火災シミュレーションを実施します。\n"
        f"発生地点: 緯度 {lat}, 経度 {lon}\n"
        f"燃料特性: {fuel_type}\n"
        "結果をJSONで出力してください。形式は以下の通りです。\n"
        "{'radius_m': 数値, 'area_sqm': 数値, 'water_volume_tons': 数値}\n"
        "絶対に純粋なJSON形式のみを出力してください（他のテキストは含むな）。"
    )
    response_text = gemini_generate_text(prompt, API_KEY, MODEL_NAME)
    if not response_text:
        st.stop()

    # JSON抽出
    result = extract_json(response_text)
    if not result:
        st.stop()

    # テキストレポートとして表示
    try:
        radius_m = float(result["radius_m"])
    except (KeyError, ValueError):
        st.error("JSONに 'radius_m' の数値が見つかりません。")
        st.stop()

    st.write("### 予測結果レポート")
    st.write(f"半径: {result.get('radius_m', '不明')} m")
    st.write(f"面積: {result.get('area_sqm', '不明')} ㎡")
    st.write(f"必要な消火水量: {result.get('water_volume_tons', '不明')} トン")

    # 2D Folium 地図に延焼範囲を描写
    m = folium.Map(location=[lat, lon], zoom_start=14)
    folium.Marker([lat, lon], popup="火災発生地点", icon=folium.Icon(color='red')).add_to(m)
    folium.Circle(
        [lat, lon], radius=radius_m, color='orange', fill=True, fill_opacity=0.4
    ).add_to(m)
    st.write("#### Folium 地図（延焼範囲）")
    st_folium(m, width=700, height=500)

    # 3D pydeck 表示用：各座標点から3Dカラムを生成（ここでは単純な例として円を 3D カラムに変換）
    def create_circle_columns(center_lat, center_lon, radius, num_points=36, height=100):
        columns = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            col_lat = center_lat + (radius / 111000.0) * math.cos(angle)
            col_lon = center_lon + (radius / 111000.0) * math.sin(angle)
            columns.append({
                "lat": col_lat,
                "lon": col_lon,
                "height": height
            })
        return columns

    col_data = create_circle_columns(lat, lon, radius_m, height=result.get("water_volume_tons", 0)/50)
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
        zoom=14,
        pitch=45
    )
    deck = pdk.Deck(layers=[column_layer], initial_view_state=view_state)
    st.write("#### pydeck 3Dカラム表示")
    st.pydeck_chart(deck)
