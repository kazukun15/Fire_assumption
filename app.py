import pydeck as pdk

view_state = pdk.ViewState(
    latitude=35.0,
    longitude=139.0,
    zoom=10,
    pitch=45,
    bearing=0,
    mapStyle="mapbox://styles/mapbox/light-v10"  # Mapbox スタイル URL を指定
)

layer = pdk.Layer(
    "ColumnLayer",
    data=[{"lat": 35.0, "lon": 139.0, "height": 100}],
    get_position='[lon, lat]',
    get_elevation='height',
    elevation_scale=1,
    get_radius=500,
    get_fill_color='[200, 30, 30, 200]',
    pickable=True,
    auto_highlight=True,
)

r = pdk.Deck(layers=[layer], initial_view_state=view_state)
r.to_html("mapbox_3d_map.html")
