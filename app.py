import streamlit as st
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point
import geopandas as gpd
import requests
import json
import re

st.set_page_config(page_title="火災拡大シミュレーション", layout="wide")

# ダミーの応答例（本来はAPIから取得する想定）
dummy_response_text = """
json
{
  "radius_m": 331.45,
  "area_sqm": 345069.36,
  "water_volume_tons": 999.99
}

**計算:**

1.  **半径:**
    `radius_m = 5 * 13.5 * sqrt(24) = 5 * 13.5 * 4.899 = 331.45 m`

2.  **面積:**
    `area_sqm = pi * 331.45^2 = 3.14159 * 109869.0025 = 345069.36 m^2`

ここはJSON以外の解説や計算手順が書かれているが、表示しない。
"""

# --- JSON抽出用の関数 ---
def extract_json_block(text: str) -> str:
    """
    文字列中の「json\n{ ... }」部分のみを抜き出して返す。
    見つからなければ None を返す。
    """
    pattern = r"(?s)json\s*\{(.*?)\}"
    match = re.search(pattern, text)
    if match:
        # JSONの中身だけ取り出して再度 "{" + 中身 + "}" の形にする
        return "{" + match.group(1) + "}"
    return None

# --- メイン処理 ---
st.title("火災拡大シミュレーション（JSON抽出デモ）")

# 1) 受け取ったテキストから JSON 部分を抽出
json_str = extract_json_block(dummy_response_text)
if json_str is None:
    st.error("JSON部分が見つかりませんでした。")
else:
    try:
        # 2) JSONを解析し、数値を取得
        data = json.loads(json_str)
        radius_m = data["radius_m"]
        area_sqm = data["area_sqm"]
        water_volume_tons = data["water_volume_tons"]

        # 3) 地図上に範囲を表示
        #    - 発火地点を仮に固定 [34.257586, 133.204356] とする
        #    - Shapely で円形バッファを作り folium で可視化
        st.write("### 地図上で火災範囲を可視化")
        center = (34.257586, 133.204356)
        m = folium.Map(location=center, zoom_start=14)

        # EPSG:4326 (経緯度)で radius_m メートルのバッファを作る場合、
        # 1度 ≈ 111,000m なので、(radius_m / 111000) 度のバッファを作る
        gdf_center = gpd.GeoSeries([Point(center[1], center[0])], crs="EPSG:4326")
        buffer_deg = radius_m / 111000
        polygon = gdf_center.buffer(buffer_deg).iloc[0]

        # folium.Polygon で描画するため、(lat, lon) の順に座標を取り出す
        coords = [(lat, lon) for lon, lat in polygon.exterior.coords]
        folium.Polygon(
            locations=coords,
            color="red",
            fill=True,
            fill_opacity=0.4,
            tooltip=f"半径: {radius_m}m / 面積: {area_sqm}m^2"
        ).add_to(m)

        # 地図を表示
        st_folium(m, width=700, height=500)

        # 4) 消火水量は別枠で表示
        st.write("### 消火水量")
        st.info(f"{water_volume_tons} トン")

    except (KeyError, json.JSONDecodeError) as e:
        st.error("JSONの解析に失敗しました。フォーマットを確認してください。")
        st.write(e)
