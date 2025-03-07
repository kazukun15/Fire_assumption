import requests
from shapely.geometry import Point
import geopandas as gpd

def get_weather(lat=35.681236, lng=139.767125):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&current=wind_speed_10m,wind_direction_10m"
    response = requests.get(url)
    data = response.json()
    return {
        'wind_speed': data['current']['wind_speed_10m'],
        'wind_direction': data['current']['wind_direction_10m']
    }

def predict_fire_spread(points, weather, duration_hours):
    # 簡易的な火災拡大モデル
    base_spread_rate = 0.1  # 時速10%の拡大率（仮定）
    wind_factor = 1 + (weather['wind_speed'] / 10)  # 風速による拡大係数（仮定）
    spread_rate = base_spread_rate * wind_factor
    radius_m = spread_rate * duration_hours * 1000  # 拡大半径（メートル）

    # 地理的範囲の作成（簡易円形）
    gdf_points = gpd.GeoSeries([Point(lng, lat) for lat, lng in points], crs="EPSG:4326")
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
    }

def calculate_water_volume(area_sqm):
    # 1平方メートルあたり0.5立方メートルの水を必要とする（仮定）
    water_volume_cubic_m = area_sqm * 0.5
    water_volume_tons = water_volume_cubic_m  # 1立方メートル = 1トン
    return water_volume_tons
