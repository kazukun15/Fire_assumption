def run_simulation(time_label):
    duration_hours = 72  # 3日間
    if not st.session_state.get("weather_data"):
        st.error("気象データが取得されていません。")
        return
    if not st.session_state.get("points"):
        st.error("発生地点が設定されていません。")
        return

    with st.spinner(f"{time_label}のシミュレーションを実行中..."):
        result = predict_fire_spread(
            st.session_state.points,
            st.session_state.weather_data,
            duration_hours,
            API_KEY,
            MODEL_NAME,
            fuel_type
        )
    
    if result is None:
        return
    
    try:
        radius_m = float(result.get("radius_m", 0))
    except (KeyError, ValueError):
        st.error("JSONに 'radius_m' の数値が見つかりません。")
        return
    try:
        area_sqm = float(result.get("area_sqm", 0))
        area_ha = area_sqm / 10000.0
    except (KeyError, ValueError):
        area_sqm = "不明"
        area_ha = "不明"
    water_volume_tons = result.get("water_volume_tons", "不明")
    
    lat_center, lon_center = st.session_state.points[0]
    burn_days = duration_hours / 24
    color_rgba = get_color_by_days(burn_days)
    color_hex = rgb_to_hex(color_rgba)
    
    # 基本の延焼範囲（消火活動なしの場合の結果）
    if fuel_type == "森林":
        shape_coords = get_mountain_shape(lat_center, lon_center, radius_m)
    else:
        shape_coords = create_half_circle_polygon(
            lat_center, lon_center, radius_m,
            st.session_state.weather_data.get("winddirection", 0)
        )
    
    # シナリオに応じた効果を適用
    if scenario == "通常の消火活動あり":
        suppression_factor = 0.5
        effective_radius = radius_m * suppression_factor
        if fuel_type == "森林":
            effective_shape_coords = get_mountain_shape(lat_center, lon_center, effective_radius)
        else:
            effective_shape_coords = create_half_circle_polygon(
                lat_center, lon_center, effective_radius,
                st.session_state.weather_data.get("winddirection", 0)
            )
        # 上記の結果を延焼範囲として採用する
        shape_coords = effective_shape_coords
    else:
        effective_radius = radius_m  # 消火活動なしの場合はそのまま

    # アニメーション表示（各ループ毎に st.empty() のコンテナを更新）
    if st.button("延焼範囲アニメーション開始"):
        anim_placeholder = st.empty()
        final_map = None
        for cycle in range(2):
            for r in range(0, int(radius_m) + 1, max(1, int(radius_m)//20)):
                try:
                    m_anim = folium.Map(
                        location=[lat_center, lon_center],
                        zoom_start=13,
                        tiles="OpenStreetMap",
                        control_scale=True
                    )
                    folium.Marker(
                        location=[lat_center, lon_center],
                        icon=folium.Icon(color="red")
                    ).add_to(m_anim)
                    if fuel_type == "森林":
                        poly = get_mountain_shape(lat_center, lon_center, r)
                        folium.Polygon(
                            locations=poly,
                            color=color_hex,
                            fill=True,
                            fill_opacity=0.5
                        ).add_to(m_anim)
                    else:
                        folium.Circle(
                            location=[lat_center, lon_center],
                            radius=r,
                            color=color_hex,
                            fill=True,
                            fill_opacity=0.5
                        ).add_to(m_anim)
                    anim_placeholder.empty()
                    with anim_placeholder:
                        st_folium(m_anim, width=700, height=500)
                    time.sleep(0.1)
                    final_map = m_anim
                except Exception as e:
                    st.error(f"アニメーション中エラー: {e}")
                    break
        if final_map is not None:
            st_folium(final_map, width=700, height=500)
    
    # 3D DEM 表示のため pydeck でマップを作成（表示する延焼範囲は効果適用後のもの）
    polygon_layer = get_flat_polygon_layer(shape_coords, water_volume_tons, color_rgba)
    layers = [polygon_layer]
    if MAPBOX_TOKEN:
        terrain_layer = get_terrain_layer()
        if terrain_layer:
            layers.append(terrain_layer)
    view_state = pdk.ViewState(
        latitude=lat_center,
        longitude=lon_center,
        zoom=13,
        pitch=45,
        bearing=0,
        mapStyle="mapbox://styles/mapbox/satellite-streets-v11"
    )
    deck = pdk.Deck(layers=layers, initial_view_state=view_state)
    
    # 詳細なレポート生成
    report_text = f"""
**シミュレーション結果：**

- **火災拡大半径**: {radius_m:.2f} m  
- **拡大面積**: {area_ha if isinstance(area_ha, str) else f'{area_ha:.2f}'} ヘクタール  
- **必要な消火水量**: {water_volume_tons} トン  
- **燃焼継続日数**: {burn_days:.1f} 日

---

### 詳細レポート

#### 1. 地形について
- **傾斜・標高**: 本シミュレーションでは、傾斜は約10度、標高は150m程度と仮定しています。  
- **植生**: 対象地域は松林と草地が混在しており、選択された燃料タイプは「{fuel_type}」です。  
- **地形の影響**: 地形の複雑さにより、火災は斜面に沿って急速に延焼する可能性があり、延焼パターンが変動します。

#### 2. 延焼の仕方
- **風向と延焼**: 現在の風向は {st.session_state.weather_data.get("winddirection", "不明")} 度、風速は {st.session_state.weather_data.get("windspeed", "不明")} m/s です。これにより、火災は風下側に向かって不均一に延焼すると予測されます。  
- **燃料の影響**: 「{fuel_type}」の燃料特性により、火災の延焼速度や燃焼の強度が決まり、延焼半径や面積にも大きな影響を与えます。  
- **延焼パターン**: 消火活動なしの場合、火災は広範囲に拡大しますが、通常の消火活動ありの場合、消火対策の効果により延焼半径は約 {suppression_factor * 100:.0f}% に抑えられると想定されます。

#### 3. 可能性について
- **消火活動の効果**:
  - **通常の消火活動あり**: 延焼半径が効果的に半減（例: {effective_radius:.2f} m）し、延焼面積も小さくなるため、迅速な消火活動が実施された場合のリスクが低減します。
  - **消火活動なし**: 消防活動が行われない場合、火災はそのまま拡大し、被害が大きくなる可能性があります。
- **リスク評価**: 延焼パターンや燃焼継続日数から、早期の消火活動が極めて重要であると判断され、地形や気象条件を踏まえた最適な対策が求められます。
- **将来的なシナリオ**: 気象条件の変化や地形の多様性により火災の挙動は大きく変動するため、継続的なモニタリングとシナリオごとの対策計画が不可欠です。
"""
    
    st.markdown("---")
    st.subheader("シミュレーション結果マップ (3D DEM表示)")
    st.pydeck_chart(deck, key="pydeck_chart_" + str(time.time()))
    st.subheader("シミュレーションレポート")
    st.markdown(report_text)
    
    if show_raincloud:
        rain_data = get_raincloud_data(lat_center, lon_center)
        if rain_data:
            from folium.raster_layers import ImageOverlay
            st.markdown("#### 雨雲オーバーレイ")
            m_overlay = folium.Map(
                location=[lat_center, lon_center],
                zoom_start=13,
                tiles="OpenStreetMap",
                control_scale=True
            )
            folium.Marker(
                location=[lat_center, lon_center],
                icon=folium.Icon(color="red")
            ).add_to(m_overlay)
            overlay = ImageOverlay(
                image=rain_data["image_url"],
                bounds=rain_data["bounds"],
                opacity=0.4,
                interactive=True,
                cross_origin=False,
                zindex=1,
            )
            overlay.add_to(m_overlay)
            st_folium(m_overlay, width=700, height=500)
