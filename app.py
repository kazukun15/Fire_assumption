import streamlit as st
import json
import re
import math
import pydeck as pdk
import demjson3 as demjson  # Python3 用の demjson のフォーク

# -----------------------------
# JSON抽出関数（多様なパターンに対応）
# -----------------------------
def extract_json(text: str) -> dict:
    """
    テキストからJSONオブジェクトを抽出する。
    まず直接 json.loads() を試み、失敗した場合は正規表現で抽出し、demjson3で解析を試みる。
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pattern = r"\{.*\}"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                try:
                    return demjson.decode(json_str)
                except demjson.JSONDecodeError:
                    st.error("JSON解析に失敗しました。")
                    return {}
        else:
            st.error("JSON文字列が見つかりませんでした。")
            return {}

# -----------------------------
# 半円形ポリゴン生成関数
# -----------------------------
def create_half_circle_polygon(center_lat, center_lon, radius_m, wind_direction_deg):
    """
    指定された中心地点、半径、風向きに基づいて半円形の座標列を生成する。
    戻り値は pydeck 用に [lon, lat] 形式の座標リスト。
    """
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

# -----------------------------
# テスト用：Gemini API の応答をシミュレーションするサンプルテキスト
# -----------------------------
sample_text = '''```json
{"radius_m": 650.00, "area_sqm": 1327322.89, "water_volume_tons": 475.50}
```'''

# 抽出して辞書化
prediction = extract_json(sample_text)

# テキストレポートとして表示
st.write("### 抽出された予測結果のJSON")
st.write(prediction)
st.write(f"**半径**: {prediction.get('radius_m', 0)} m")
st.write(f"**面積**: {prediction.get('area_sqm', 0)} m²")
st.write(f"**必要放水量**: {prediction.get('water_volume_tons', 0)} トン")

# -----------------------------
# pydeck による延焼範囲（半円形）の3D表示
# -----------------------------
# 固定の中心地点（例として指定）
center_lat = 34.257586
center_lon = 133.204356
# 風向きは例として 90°（東）とする
wind_direction = 90

# JSONから取得した半径を使用
radius = prediction.get("radius_m", 0)

# 半円形の座標列を生成
coords = create_half_circle_polygon(center_lat, center_lon, radius, wind_direction)

# pydeckのPolygonLayerで表示するためのデータ作成
polygon_data = [{"coordinates": [coords]}]

# pydeckのViewState設定
view_state = pdk.ViewState(
    latitude=center_lat,
    longitude=center_lon,
    zoom=12,
    pitch=45
)

# PolygonLayer を作成して延焼範囲を表示
polygon_layer = pdk.Layer(
    "PolygonLayer",
    data=polygon_data,
    get_polygon="coordinates",
    get_fill_color="[255, 0, 0, 100]",
    pickable=True,
    auto_highlight=True,
)

deck = pdk.Deck(layers=[polygon_layer], initial_view_state=view_state)
st.write("### pydeckによる3D 延焼範囲表示")
st.pydeck_chart(deck)
