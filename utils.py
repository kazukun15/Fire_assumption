import google.generativeai as genai
from shapely.geometry import Point
import geopandas as gpd

def predict_fire_spread(points, weather):
    model = genai.GenerativeModel('gemini-pro')
    
    prompt = f"""
    以下の条件で火災の炎症範囲を予測してください。
    
    発生地点: {points}
    風速: {weather['wind_speed']} m/s
    風向: {weather['wind_direction']}度
    
    以下を算出:
    1. 予測される炎症範囲の半径（m）
    2. 拡大までにかかる時間（分単位）
    3. 炎症範囲のおおよその面積（平方メートル）
    
    出力はJSON形式で以下のように返してください：
    {{
        "radius_m": 値,
        "spread_time_min": 値,
        "area_sqm": 値
    }}
    """
    
    response = model.generate_content(prompt)
    result = response.text
    prediction = eval(result)  # JSON形式に変換

    # 地理的範囲の作成（簡易円形）
    gdf_points = gpd.GeoSeries([Point(lng, lat) for lat, lng in points], crs="EPSG:4326")
    centroid = gdf_points.unary_union.centroid
    buffer = centroid.buffer(prediction["radius_m"] / 111000)  # 簡易緯度経度変換
    
    return {
        'radius_m': prediction["radius_m"],
        'spread_time_min': prediction["spread_time_min"],
        'area_sqm': prediction["spread_area_m2"],
        'area_coordinates': [(coord[1], coord[0]) for coord in buffer.exterior.coords]
    }

