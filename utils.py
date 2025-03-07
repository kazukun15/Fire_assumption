import requests
from shapely.geometry import Point
import geopandas as gpd
import google.generativeai as genai

def get_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    response = requests.get(url)
    data = response.json()
    return {
        'wind_speed': data['current_weather']['windspeed'],
        'wind_direction': data['current_weather']['winddirection']
    }

def predict_fire_spread(points, weather, duration_hours, api_key, model_name):
    # Gemini APIの設定
    genai.configure(api_key=api_key)

    # 簡易的な火災拡大モデル
    base_spread_rate = 0.1  # 時速10%の拡大率（仮定）
    wind_factor = 1 + (weather['wind_speed'] / 10)  # 風速による拡大係数（仮定）
    spread_rate = base_spread_rate * wind_factor
    radius_m = spread_rate * duration_hours * 1000  # 拡大半径（メートル）

    # 地理的範囲の作成（簡易円形）
    gdf_points = gpd.GeoSeries([Point(lon, lat) for lat, lon in points], crs="EPSG:4326")
    centroid = gdf_points.unary_union.centroid
    buffer = centroid.buffer(radius_m / 111000)  # 簡易緯度経度変換

    area_sqm = buffer.area * (111000 ** 2)  # 面積（平方メートル）

    # 必要な消火水量の計算
    water_volume_tons = calculate_water_volume(area_sqm)

    return {
        'radius_m': radius_m,
        'area_sqm': area_sqm,
        'water_volume_tons': water_volume_tons,
        'area_coordinates': [(coord[1], coord[0]) for coord in buffer.exterior.coords]

::contentReference[oaicite:4]{index=4}
 
