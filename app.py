import streamlit as st
import requests
import math
import urllib.parse
from datetime import datetime, timedelta, timezone
import folium
from folium import plugins
import pydeck as pdk

# Set up the Streamlit page configuration
if __name__ == '__main__':
    st.set_page_config(page_title="Fire Spread Simulation", layout="wide")

    # Caching for external API calls
    @st.cache_data
    def geocode_address(address: str):
        """Geocode an address to latitude and longitude using Google Geocoding API."""
        api_key = st.secrets["api_key"]  # Google API key stored in Streamlit secrets
        # URL encode the address
        addr_enc = urllib.parse.quote(address, safe='')
        url = f"https://maps.googleapis.com/maps/api/geocode/json?address={addr_enc}&key={api_key}"
        resp = requests.get(url)
        data = resp.json()
        if data.get("status") == "OK" and data.get("results"):
            loc = data["results"][0]["geometry"]["location"]
            return loc["lat"], loc["lng"]
        else:
            return None, None

    @st.cache_data
    def fetch_weather(lat: float, lon: float, total_hours: int):
        """Fetch hourly weather forecast data (wind direction, wind speed, precipitation) from Open-Meteo API."""
        # Request weather forecast (up to 2 days to cover simulation duration)
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat:.4f}&longitude={lon:.4f}"
               f"&hourly=winddirection_10m,windspeed_10m,precipitation&windspeed_unit=ms&timezone=UTC&forecast_days=2")
        resp = requests.get(url)
        data = resp.json()
        # Ensure we have required data
        if "hourly" not in data or "time" not in data["hourly"]:
            return None
        times = data["hourly"]["time"]
        wind_speeds = data["hourly"]["windspeed_10m"]
        wind_dirs = data["hourly"]["winddirection_10m"]
        precips = data["hourly"]["precipitation"]
        # Parse times to datetime objects (UTC)
        time_datetimes = [datetime.fromisoformat(t.replace("Z", "+00:00")) for t in times]
        # Find the starting index (use the current or nearest future hour as start)
        now_utc = datetime.now(timezone.utc)
        # Use the current hour (floor) as start point
        now_floor = now_utc.replace(minute=0, second=0, microsecond=0)
        if now_floor < time_datetimes[0]:
            start_idx = 0
        else:
            try:
                start_idx = time_datetimes.index(now_floor)
            except ValueError:
                # If exact hour not found, find the first future hour
                start_idx = next((i for i, t in enumerate(time_datetimes) if t > now_utc), 0)
        # Determine end index based on simulation hours
        end_idx = start_idx + total_hours
        if end_idx >= len(time_datetimes):
            end_idx = len(time_datetimes) - 1
        # Slice the hourly data to the needed range
        hours_slice = time_datetimes[start_idx:end_idx+1]
        wind_slice = wind_speeds[start_idx:end_idx+1]
        dir_slice = wind_dirs[start_idx:end_idx+1]
        precip_slice = precips[start_idx:end_idx+1]
        return hours_slice, wind_slice, dir_slice, precip_slice

    @st.cache_data
    def get_timezone_offset(lat: float, lon: float):
        """Get timezone offset (in seconds) for the given location using Google Time Zone API."""
        api_key = st.secrets["api_key"]
        # Use current time as reference timestamp
        timestamp = int(datetime.now(timezone.utc).timestamp())
        url = f"https://maps.googleapis.com/maps/api/timezone/json?location={lat:.6f},{lon:.6f}&timestamp={timestamp}&key={api_key}"
        resp = requests.get(url)
        data = resp.json()
        if data.get("status") == "OK":
            raw_offset = data.get("rawOffset", 0)
            dst_offset = data.get("dstOffset", 0)
            tz_id = data.get("timeZoneId", "")
            return raw_offset + dst_offset, tz_id
        else:
            return 0, ""

    def simulate_fire(lat: float, hours_list, wind_list, dir_list, precip_list, base_speed: float, wind_factor: float):
        """Simulate fire spread polygons for each time frame based on weather and fuel parameters."""
        # Apply scenario wind factor to wind speeds
        wind_list = [w * wind_factor for w in wind_list]
        # Prepare frame lists for 15-minute intervals
        frame_times = []
        frame_wind = []
        frame_dir = []
        frame_precip = []
        for h in range(len(hours_list) - 1):
            if h == 0:
                frame_times.append(hours_list[h])
                frame_wind.append(wind_list[h])
                frame_dir.append(dir_list[h])
                frame_precip.append(precip_list[h])
            # Interpolate 15-min steps within hour h to h+1
            for q in [1, 2, 3]:
                frac = q / 4.0
                # Linear interpolate wind speed and precipitation
                w_val = wind_list[h] + (wind_list[h+1] - wind_list[h]) * frac
                p_val = precip_list[h] + (precip_list[h+1] - precip_list[h]) * frac
                # Interpolate wind direction the shortest way (circular interpolation)
                d1 = dir_list[h]
                d2 = dir_list[h+1]
                delta = ((d2 - d1 + 180) % 360) - 180
                d_val = (d1 + delta * frac) % 360
                frame_times.append(hours_list[h] + timedelta(minutes=15 * q))
                frame_wind.append(w_val)
                frame_dir.append(d_val)
                frame_precip.append(p_val)
            # Append the next hour point
            frame_times.append(hours_list[h+1])
            frame_wind.append(wind_list[h+1])
            frame_dir.append(dir_list[h+1])
            frame_precip.append(precip_list[h+1])
        # Generate fire polygon for each frame (list of [lon, lat] points)
        polygons = []
        for idx, t in enumerate(frame_times):
            # Compute time elapsed in hours from the start of simulation
            t_hours = (t - frame_times[0]).total_seconds() / 3600.0
            # Compute effective base spread rate (m/s) adjusted for precipitation
            precip_factor = 1.0 / (1.0 + frame_precip[idx])
            if precip_factor < 0.1:
                precip_factor = 0.1  # ensure some spread even in heavy rain
            effective_base = base_speed * precip_factor  # m/s
            # Wind influence factors
            k = 0.1
            cross_k = 0.05
            # Spread factors
            S_factor = 1.0 + k * frame_wind[idx]
            U_factor = 1.0 - k * frame_wind[idx]
            if U_factor < 0.0:
                U_factor = 0.0
            cross_factor = 1.0 - cross_k * frame_wind[idx]
            if cross_factor < 0.3:
                cross_factor = 0.3
            # Travel distances in each direction (meters)
            if idx == 0:
                # small initial radius for ignition point
                R_down = R_up = R_cross = 20.0
            else:
                time_seconds = t_hours * 3600.0
                R_down = effective_base * S_factor * time_seconds
                R_up   = effective_base * U_factor * time_seconds
                R_cross = effective_base * cross_factor * time_seconds
            # Construct polygon in local coordinates (x: downwind axis, y: crosswind axis)
            front_points = []
            back_points = []
            # Front half (from left cross, through front, to right cross)
            for j in range(31):
                alpha = -math.pi/2 + j * (math.pi / 30)  # -90° to +90°
                x = R_down * math.cos(alpha)
                y = R_cross * math.sin(alpha)
                front_points.append((x, y))
            # Back half (from right cross, through back, to left cross)
            for j in range(30, -1, -1):
                alpha = -math.pi/2 + j * (math.pi / 30)
                x = -R_up * math.cos(alpha)
                y = R_cross * math.sin(alpha)
                back_points.append((x, y))
            # Remove duplicate right cross point at the junction of front and back halves
            if back_points and front_points and back_points[0] == front_points[-1]:
                back_points = back_points[1:]
            # Combine points
            poly_local = front_points + back_points
            # Remove duplicate left cross point if present
            if poly_local[0] == poly_local[-1]:
                poly_local = poly_local[:-1]
            # Rotate local coordinates to global (bearing = spread direction)
            spread_dir = (frame_dir[idx] + 180.0) % 360.0  # wind from X -> fire spreads towards X+180
            theta = math.radians(spread_dir)
            poly_coords = []
            for (x, y) in poly_local:
                # Rotate (x, y) where x is along spread_dir and y is spread_dir+90 (to the right)
                north_offset = x * math.cos(theta) - y * math.sin(theta)
                east_offset  = x * math.sin(theta) + y * math.cos(theta)
                # Convert offsets (m) to lat/lon degrees
                lat_offset = north_offset / 110540.0  # ~110.54 km per degree latitude
                lon_offset = east_offset / (111320.0 * math.cos(math.radians(lat)))
                lat_point = lat + lat_offset
                lon_point = lon + lon_offset
                poly_coords.append([lon_point, lat_point])
            # Ensure polygon is closed (first = last)
            if poly_coords[0] != poly_coords[-1]:
                poly_coords.append(poly_coords[0])
            polygons.append(poly_coords)
        return frame_times, polygons

    # Sidebar UI elements
    st.sidebar.header("シミュレーション設定")
    # Fire outbreak location input
    location_input = st.sidebar.text_input("火災発生地点", "Osaka, Japan")
    # Fuel type selection
    fuel_options = {"草地": 0.6, "森林": 0.3, "低木地帯": 0.4, "都市部": 0.2}  # base spread rates (m/s) for each fuel
    fuel_type = st.sidebar.selectbox("燃料の種類", list(fuel_options.keys()), index=0)
    base_speed = fuel_options[fuel_type]
    # Scenario selection
    scenario_options = ["標準シナリオ", "強風シナリオ", "初期消火シナリオ"]
    scenario = st.sidebar.selectbox("シナリオ選択", scenario_options, index=0)
    # Determine scenario parameters
    if scenario == "強風シナリオ":
        wind_factor = 2.0  # double wind effect
        total_hours = 6
    elif scenario == "初期消火シナリオ":
        wind_factor = 1.0
        total_hours = 3    # fire is contained after 3 hours
    else:
        wind_factor = 1.0
        total_hours = 6
    # Rain cloud overlay toggle
    show_rain = st.sidebar.checkbox("雨雲オーバーレイを表示", value=True)
    # Display mode (2D or 3D)
    display_mode = st.sidebar.radio("表示モード", ["2D 地図", "3D 表示"], index=0)

    # Geocode the location input to lat, lon
    lat, lon = None, None
    if location_input:
        # If input looks like "lat, lon"
        try:
            parts = [p.strip() for p in location_input.split(",")]
            if len(parts) == 2:
                lat_val = float(parts[0])
                lon_val = float(parts[1])
                lat, lon = lat_val, lon_val
        except:
            lat, lon = None, None
        if lat is None or lon is None:
            lat, lon = geocode_address(location_input)
    if lat is None or lon is None:
        st.error("入力された地点を特定できませんでした。住所または緯度経度を正しく入力してください。")
        st.stop()

    # Fetch weather data for the location
    weather_data = fetch_weather(lat, lon, total_hours)
    if not weather_data:
        st.error("気象データを取得できませんでした。")
        st.stop()
    hours_list, wind_list, dir_list, precip_list = weather_data

    # Simulate fire spread
    frame_times, fire_polygons = simulate_fire(lat, hours_list, wind_list, dir_list, precip_list, base_speed, wind_factor)

    # Prepare data for 2D map (Folium) and 3D view (PyDeck)
    # Create time strings for each frame in ISO format (UTC)
    frame_iso = [dt.strftime("%Y-%m-%dT%H:%M:%SZ") for dt in frame_times]

    # Prepare Folium 2D map with fire spread animation and optional rain overlay
    if display_mode == "2D 地図":
        # Center map at fire location
        m = folium.Map(location=[lat, lon], zoom_start=13, tiles="OpenStreetMap", width="100%", height="600")
        # Add fire starting point marker (fire icon)
        folium.Marker(location=[lat, lon], 
                      icon=folium.Icon(color="red", icon="fire", prefix="fa"), 
                      tooltip="火災発生地点").add_to(m)
        # Build geojson features for fire spread (MultiPolygon with times)
        fire_feature = {
            "type": "Feature",
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [[poly] for poly in fire_polygons]  # each poly is one frame
            },
            "properties": {
                "times": frame_iso,
                "style": {
                    "color": "red",
                    "weight": 1,
                    "fillColor": "red",
                    "fillOpacity": 0.4
                }
            }
        }
        features = [fire_feature]
        # Add rain cloud overlay features if enabled
        if show_rain:
            # Use hourly data points for rain overlay
            hours_iso = [dt.strftime("%Y-%m-%dT%H:%M:%SZ") for dt in hours_list]
            for i, rain_val in enumerate(precip_list):
                # Only add if there is precipitation at this hour
                if rain_val and rain_val > 0:
                    # Define radius in meters for rain cloud (base 5km + 1km per mm/h)
                    cloud_radius = 5000.0 + rain_val * 1000.0
                    # Generate a circle polygon around the fire location
                    circle_points = []
                    num_points = 24
                    for deg in range(0, 360, int(360 / num_points)):
                        rad = math.radians(deg)
                        # x = east offset, y = north offset
                        east_off = cloud_radius * math.cos(rad)
                        north_off = cloud_radius * math.sin(rad)
                        # Convert to lat/lon
                        dlat = north_off / 110540.0
                        dlon = east_off / (111320.0 * math.cos(math.radians(lat)))
                        lat_pt = lat + dlat
                        lon_pt = lon + dlon
                        circle_points.append([lon_pt, lat_pt])
                    # Close the circle
                    circle_points.append(circle_points[0])
                    rain_feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [circle_points]
                        },
                        "properties": {
                            "times": [hours_iso[i]],
                            "style": {
                                "color": "blue",
                                "weight": 0,
                                "fillColor": "blue",
                                "fillOpacity": 0.2
                            }
                        }
                    }
                    features.append(rain_feature)
        # Add time-enabled geojson to map
        plugins.TimestampedGeoJson(
            {
                "type": "FeatureCollection",
                "features": features
            },
            period="PT15M",
            duration="PT1H",
            add_last_point=True,
            auto_play=True,
            loop=True,
            loop_button=True,
            max_speed=10,
            progress_bar=True
        ).add_to(m)
        # Add legend (color legend for fire area and rain)
        legend_html = """
        <div style="
            position: fixed;
            bottom: 50px;
            right: 50px;
            z-index: 9999;
            background-color: white;
            border: 2px solid grey;
            padding: 10px;
            opacity: 0.8;
            font-size: 14px;
        ">
            <b>凡例</b><br>
            <i style="display:inline-block;width:12px;height:12px;background:red;margin-right:5px;"></i> 火災延焼範囲<br>
            <i style="display:inline-block;width:12px;height:12px;background:blue;opacity:0.5;margin-right:5px;"></i> 雨雲領域<br>
            <i class="fa fa-fire" style="color:red;margin-right:5px;"></i> 火災発生地点
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        # Render the map in Streamlit
        st.components.v1.html(m._repr_html_(), height=600, scrolling=False)

    # Prepare 3D Deck.gl visualization
    else:
        # Determine local timezone offset for labeling times
        offset_sec, tz_name = get_timezone_offset(lat, lon)
        # Create list of local time labels for frames
        local_times = [(dt + timedelta(seconds=offset_sec)).replace(tzinfo=None) for dt in frame_times]
        # Select slider for time (discrete steps)
        selected_time = st.sidebar.select_slider("時刻", options=local_times, value=local_times[-1],
                                                 format_func=lambda t: t.strftime("%m/%d %H:%M"))
        # Find index of selected time in frame list
        try:
            time_index = local_times.index(selected_time)
        except ValueError:
            time_index = len(local_times) - 1
        # Prepare PyDeck layers
        layers = []
        # Fire area polygon at selected time
        polygon_data = [{"coordinates": fire_polygons[time_index]}]  # single polygon
        fire_layer = pdk.Layer(
            "PolygonLayer",
            data=polygon_data,
            get_polygon="coordinates",
            get_fill_color="[255, 50, 50, 100]",
            get_line_color="[255, 0, 0]",
            get_line_width=200,
            stroked=True,
            filled=True,
            extruded=False
        )
        layers.append(fire_layer)
        # Ignition point marker
        ignition_data = [{"lat": lat, "lon": lon}]
        ign_layer = pdk.Layer(
            "ScatterplotLayer",
            data=ignition_data,
            get_position=["lon", "lat"],
            get_color="[255, 0, 0]",
            get_radius=100,
            tooltip="火災発生地点"
        )
        layers.append(ign_layer)
        # Rain cloud overlay as circle (if enabled and precipitation at selected time)
        if show_rain:
            rain_val = frame_precip[time_index] if 'frame_precip' in locals() else 0
            if rain_val and rain_val > 0:
                cloud_radius = 5000.0 + rain_val * 1000.0
                # Generate circle geometry (approximate with 36 points)
                cloud_coords = []
                for deg in range(0, 360, 10):
                    rad = math.radians(deg)
                    east_off = cloud_radius * math.cos(rad)
                    north_off = cloud_radius * math.sin(rad)
                    dlat = north_off / 110540.0
                    dlon = east_off / (111320.0 * math.cos(math.radians(lat)))
                    lat_pt = lat + dlat
                    lon_pt = lon + dlon
                    cloud_coords.append([lon_pt, lat_pt])
                cloud_coords.append(cloud_coords[0])
                cloud_data = [{"coords": cloud_coords}]
                cloud_layer = pdk.Layer(
                    "PolygonLayer",
                    data=cloud_data,
                    get_polygon="coords",
                    get_fill_color="[50, 50, 200, 80]",
                    get_line_color="[50, 50, 200]",
                    stroked=False,
                    filled=True
                )
                layers.append(cloud_layer)
        # Define the initial view state (center on fire location, zoom to fit fire spread)
        # Compute a suitable zoom and view based on final fire polygon
        # Use pydeck utility to compute viewport from fire polygon points
        all_points = [p for poly in fire_polygons for p in poly]
        # Convert list of [lon,lat] to list of [lat, lon] for compute_view (expects lat,lon order in points list)
        point_list = [[pt[1], pt[0]] for pt in all_points]
        try:
            view_state = pdk.data_utils.viewport_helpers.compute_view(point_list)
        except Exception:
            # Fallback to default view
            view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=12)
        # Adjust view: set a tilt for 3D perspective
        view_state.pitch = 45
        view_state.bearing = 0
        # Create deck.gl map
        deck = pdk.Deck(
            map_provider="google_maps",
            api_keys={"google_maps": st.secrets["api_key"]},
            map_style="satellite",
            initial_view_state=view_state,
            layers=layers,
            height=600,
            tooltip={"text": "{tooltip}"}
        )
        st.pydeck_chart(deck, use_container_width=True)

    # Display calculated total burned area and water required
    # Compute area of final fire polygon (last frame) in square meters
    final_polygon = fire_polygons[-1]
    # Use shoelace formula on planar coordinates (approximate using lat as reference)
    ref_lat = lat  # use fire origin latitude as reference for scaling
    coords_xy = []
    for lonp, latp in final_polygon:
        # Convert lat/lon to local coordinates (meters) relative to fire origin
        dx = (lonp - lon) * math.cos(math.radians(ref_lat)) * 111320.0
        dy = (latp - lat) * 110540.0
        coords_xy.append((dx, dy))
    # Shoelace formula for polygon area
    area_m2 = 0.0
    for i in range(len(coords_xy) - 1):
        x1, y1 = coords_xy[i]
        x2, y2 = coords_xy[i + 1]
        area_m2 += x1 * y2 - x2 * y1
    area_m2 = abs(area_m2) / 2.0
    # Convert area to hectares
    area_ha = area_m2 / 10000.0
    # Estimate water required (assuming ~10 L/m²)
    water_liters = area_m2 * 10.0
    water_cubic = water_liters / 1000.0
    # Display results
    st.markdown(f"**延焼面積:** 約{area_ha:.2f}ヘクタール（{area_m2:.0f}㎡）  \n" +
                f"**必要水量:** 約{water_cubic:.1f}立方メートル（{water_liters:,.0f}リットル）")
