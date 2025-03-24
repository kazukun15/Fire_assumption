import streamlit as st
import json
import re
import demjson3 as demjson  # Python3 用の demjson のフォーク

def extract_json(text: str) -> dict:
    """
    入力テキストから、マークダウン形式のコードブロック内にある
    JSON部分を抽出して辞書型に変換します。
    まず直接 json.loads() を試み、失敗した場合は正規表現と demjson3 を利用します。
    """
    text = text.strip()
    try:
        # 直接パースを試みる
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # マークダウン形式のコードブロックから抽出（```json ... ```）
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
                st.error(f"demjsonによる解析に失敗しました: {e}")
                return {}
    else:
        st.error("有効なJSON文字列が見つかりませんでした。")
        return {}

# サンプル応答（"text" フィールド内にマークダウン形式でJSONが含まれている）
sample_response = '''"text":"```json
{
  "radius_m": 750.00,
  "area_sqm": 1767145.87,
  "water_volume_tons": 634.17
}
```"'''

# "text": の部分を取り除く（もし必要であれば）
if sample_response.startswith('"text":'):
    sample_response = sample_response.split(":", 1)[1].strip().strip('"')

# JSON抽出を実施
result = extract_json(sample_response)

# 結果をテキストとして表示
st.write("### 抽出されたJSON内容")
st.text(json.dumps(result, indent=2))
