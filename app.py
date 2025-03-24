import streamlit as st
import folium
from streamlit_folium import st_folium
import json
import re

# サンプルのテキスト（Gemini APIの応答例の一部を模擬）
sample_text = '''```json
{
  "radius_m": 785.00,
  "area_sqm": 1935064.81,
  "water_volume_tons": 695.00
}
```'''

def extract_json(text: str) -> dict:
    """
    テキストからJSONオブジェクトを抽出する関数（多様なパターンに対応）。
    まず直接 json.loads() を試み、失敗した場合はマークダウン形式のコードブロックから抽出する。
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pattern = r"```json\s*(\{.*?\})\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                st.error("JSONの解析に失敗しました。")
                return {}
        else:
            st.error("JSON文字列が見つかりませんでした。")
            return {}

# JSON抽出実行
data = extract_json(sample_text)

if data:
    # 抽出結果の数値をテキストで表示
    radius_m = data.get("radius_m", 0)
    area_sqm = data.get("area_sqm", 0)
    water_volume_tons = data.get("water_volume_tons", 0)
    
    st.write("### 抽出された予測結果")
    st.write(f"半径: {radius_m} m")
    st.write(f"面積: {area_sqm} m²")
    st.write(f"必要放水量: {water_volume_tons} トン")
    
    # 地図表示：例として固定の中心点を利用（ここでは東京駅付近）
    center = [35.681236, 139.767125]
    m = folium.Map(location=center, zoom_start=12)
    
    # 半径をもとに円を描画
    folium.Circle(
        location=center,
        radius=radius_m,  # JSONで抽出した半径
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.5
    ).add_to(m)
    
    st.write("### 延焼範囲を示す Folium 地図")
    st_folium(m, width=700, height=500)
else:
    st.error("JSONの抽出に失敗しました。")
