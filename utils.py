import requests
from shapely.geometry import Point
import geopandas as gpd
import openai

def get_weather(lat=35.681236, lng=139.767125):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&current_weather=true"
    response = requests.get(url)
    data = response.json()
    return {
        'wind_speed': data['current_weather']['windspeed'],
        'wind_direction': data['current_weather']['winddirection']
    }

def predict_fire_spread(points, weather, duration_hours, api_key, model_name):
    # OpenAI APIの設定
    openai.api_key = api_key

    # 発生地点の座標を文字列に変換
    points_str = ', '.join([f"({lat}, {lon})" for lat, lon in points])

    # プロンプトの作成
    prompt = f"""
    以下の条件で火災の炎症範囲を予測してください。

    発生地点: {points_str}
    風速: {weather['wind_speed']} m/s
    風向: {weather['wind_direction']}度
    時間経過: {duration_hours} 時間

    以下を算出してください:
    1. 予測される炎症範囲の半径（メートル）
    2. 炎症範囲のおおよその面積（平方メートル）
    3. 必要な消火水量（トン）

    出力はJSON形式で以下のように返してください：
    {{
        "radius_m": 値,
        "area_sqm": 値,
        "water_volume_tons": 値
    }}
    """

    # OpenAI APIを使用して予測を取得
    response = openai.Completion.create(
        engine=model_name,
        prompt=
::contentReference[oaicite:4]{index=4}
 
