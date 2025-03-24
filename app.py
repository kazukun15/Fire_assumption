import streamlit as st
import folium
from streamlit_folium import st_folium
import json
import re
import demjson3 as demjson  # Python3 用の demjson のフォーク
import math
import pydeck as pdk

st.title("火災拡大シミュレーション レポート＆描写")

# サンプルのGemini API応答（"text"フィールドにマークダウン形式のJSONが含まれる）
sample_response = '''"text":"```json
{
  "radius_m": 750.00,
  "area_sqm": 1935064.81,
  "water_volume_tons": 634.17
}
```"'''

# "text": の部分を取り除く（必要なら）
if sample_response.startswith('"text":'):
    sample_response = sample_response.split(":", 1)[1].strip().strip('"')

def extract_json(text: str) -> dict:
    """
    テキストからJSONオブジェクトを抽出する関数。
    ・まず直接 json.loads() を試み、失敗した場合は
      マークダウン形式のコードブロック（```json ... ```）または
      最初に現れる { ... } 部分を抽出して解析します。
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # マークダウン形式のコードブロックを対象にする
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
    # それ以外の場合、最初に現れる { ... } 部分を抽出
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

# JSON抽出を実施
result = extract_json(sample_response)

# 抽出結果をテキストレポートとして表示
st.write("### 抽出されたJSON内容")
st.json(result)

radius_m = result.get("radius_m", 0)
area_sqm = result.get("area_sqm", 0)
water_volume_tons = result.get("water_volume_tons", 0)

st.write("### 予測結果レポート")
st.write(f"半径: {radius_m} m")
st.write(f"面積: {area_sqm} ㎡")
st.write(f"必要な消火水量: {water_volume_tons} トン")

# --- 2D 描写: Folium 地図 ---
# 例として、固定の中心点（ここではサンプル用に設定）を利用
center = [34.257586, 133.204356]
m = folium.Map(location=center, zoom_start=14)
folium.Marker(center, popup="火災発生地点", icon=folium.Icon(color='red')).add_to(m)
# JSON から取得した半径を利用して円を描写
folium.Circle(location=center, radius=radius_m, color='orange', fill=True, fill_opacity=0.4).add_to(m)
st.write("### Folium 地図（延焼範囲）")
st_folium(m, width=700, height=500)

# --- 3D 描写: pydeck ColumnLayer ---
def create_circle_columns(center, radius, num_points=36, height=100):
    """
    中心点を基に、円周上に均等に配置された点から3Dカラムを生成するサンプル関数。
    """
    columns = []
    lat, lon = center
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        col_lat = lat + (radius / 111000.0) * math.cos(angle)
        col_lon = lon + (radius / 111000.0) * math.sin(angle)
        columns.append({"lat": col_lat, "lon": col_lon, "height": height})
    return columns

# 例として、water_volume_tons に応じた高さスケールを利用
height_value = water_volume_tons / 50 if water_volume_tons else 100
col_data = create_circle_columns(center, radius_m, num_points=36, height=height_value)

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
    latitude=center[0],
    longitude=center[1],
    zoom=14,
    pitch=45
)

deck = pdk.Deck(layers=[column_layer], initial_view_state=view_state)
st.write("### pydeck 3D 表示")
st.pydeck_chart(deck)
