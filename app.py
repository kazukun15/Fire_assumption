import math
import requests
import streamlit as st
import pydeck as pdk
import folium
from streamlit_folium import st_folium
from branca.element import Template, MacroElement  # 凡例用

# タイトル
st.title("火災拡大シミュレーションアプリ")

# サイドバーの入力フォーム
with st.sidebar:
    st.header("シミュレーション設定")
    st.markdown("""
        火災が発生した地点の座標や風速などを設定してください。\n
        必要に応じて現在の気象データを取得し、火災の拡大範囲を計算します。
    """)
    # 緯度・経度入力
    lat = st.number_input("火災発生地点の緯度 (Latitude)", value=34.6937,
                          format="%.6f")
    lon = st.number_input("火災発生地点の経度 (Longitude)", value=135.5023,
                          format="%.6f")
    # 気象データ自動取得の選択
    use_live_weather = st.checkbox("現在の気象データを使用する（風速を自動取得）", value=True)
    # 風速入力（手動入力用）
    manual_wind = None
    if not use_live_weather:
        manual_wind = st.number_input("風速 (m/s)", min_value=0.0, value=5.0, step=0.1,
                                      help="現地の推定風速を入力してください。")
    # シミュレーション時間入力
    duration = st.slider("シミュレーション時間（時間）", min_value=1, max_value=48, value=24,
                         help="開始から何時間後までシミュレーションするか選択してください。")
    # マップ表示モード選択
    view_mode = st.selectbox("マップ表示モード", options=["2D", "3D"], index=0,
                             help="2D（平面地図）または3D（立体地図）の表示を選択できます。")
    # フォーム送信ボタン
    submitted = st.button("シミュレーション実行")

# シミュレーション実行時の処理
if submitted:
    # 風速データ取得
    if use_live_weather:
        # Open-Meteo APIから現在の風速と風向を取得
        url = (f"https://api.open-meteo.com/v1/forecast?"
               f"latitude={lat}&longitude={lon}"
               f"&current_weather=true&wind_speed_unit=ms")
        try:
            res = requests.get(url, timeout=5)
            res.raise_for_status()
            weather = res.json().get("current_weather", {})
            wind_speed = weather.get("windspeed")         # m/s（APIパラメータで単位をm/sに指定）
            wind_direction = weather.get("winddirection") # 0-360度表現（北=0, 東=90 等）
            if wind_speed is None:
                # 念のためNoneチェック（API応答に値がない場合）
                raise ValueError("風速データが取得できませんでした")
        except Exception as e:
            # エラー時はメッセージを出し、手動入力値を使う（なければ処理中断）
            st.error("気象データの取得に失敗しました。手動入力した風速を使用します。")
            if manual_wind is None:
                st.stop()  # 手動風速もない場合は処理を中断
            wind_speed = manual_wind
            wind_direction = None
    else:
        # 手動入力モードならユーザ入力値を使用
        wind_speed = float(manual_wind) if manual_wind is not None else 0.0
        wind_direction = None

    # 火災拡大シミュレーション計算
    # 風速 (m/s) を基に、指定時間後の延焼半径を計算（単純モデル）
    hours = float(duration)
    base_speed = wind_speed        # 基本風速（m/s）
    dist_downwind = base_speed * 3600 * hours  # 風下方向の延焼距離（m）
    radius = dist_downwind * (4.0/3.0)         # 全方向に広がる半径（風下距離の4/3を半径に仮定）
    area = math.pi * (radius ** 2)             # 延焼範囲の面積（m^2）
    water_vol = area * 0.001                   # 必要な放水量（トン）= 面積 × 1mm降水 (0.001m)

    # 各時間ステップごとの半径リスト（1時間ごと）
    radii_over_time = []
    for t in range(1, duration + 1):
        r_t = (base_speed * 3600 * t) * (4.0/3.0)
        radii_over_time.append(r_t)

    # 結果をセッションステートに保存（再実行時に利用）
    st.session_state["sim_results"] = {
        "lat": float(lat),
        "lon": float(lon),
        "wind_speed": float(wind_speed),
        "wind_direction": float(wind_direction) if wind_direction is not None else None,
        "duration": int(duration),
        "radius_final": float(radius),
        "area": float(area),
        "water_vol": float(water_vol),
        "radii_over_time": radii_over_time
    }
    # アニメーション用の時間スライダー値を初期化（最終時刻に設定）
    st.session_state["hour_slider"] = int(duration)

# シミュレーション結果の表示
if "sim_results" in st.session_state:
    # 結果データを取得
    sim = st.session_state["sim_results"]
    lat0 = sim["lat"]
    lon0 = sim["lon"]
    wind_speed = sim["wind_speed"]
    wind_dir = sim["wind_direction"]  # 方位（度）
    duration = sim["duration"]
    radii_over_time = sim["radii_over_time"]

    # タブで地図表示とレポート表示を分離
    tab_map, tab_report = st.tabs(["地図で確認", "レポート表示"])

    # **地図表示タブ**
    with tab_map:
        st.subheader("延焼範囲の地図表示")
        # 現在の風速やシミュレーション条件を表示
        wind_info = f"風速: {wind_speed:.2f} m/s"
        if wind_dir is not None:
            # 風向を方角に変換
            directions = ["北", "北北東", "北東", "東北東", "東", "東南東", "南東",
                          "南南東", "南", "南南西", "南西", "西南西", "西", "西北西", "北西", "北北西"]
            # 0度を北とし、22.5度刻みで16方位に分類
            idx = int((wind_dir + 11.25) % 360 // 22.5)
            wind_info += f" （風向: {directions[idx]}・{wind_dir:.0f}°）"
        st.write(f"**{wind_info}, シミュレーション時間: {duration} 時間**")

        # 時間スライダー（現在時刻の選択）
        hour = st.slider("経過時間 (h)", 1, duration, 
                         key="hour_slider",
                         help="火災発生からの経過時間を選択できます。")
        current_radius = radii_over_time[hour - 1]

        # 「アニメーション再生」ボタン
        animate = st.button("▶️ アニメーション再生")
        # 地図表示用プレースホルダー
        map_placeholder = st.empty()

        if animate:
            # アニメーション再生: 1時間から最終時間まで順次表示
            for t in range(1, duration + 1):
                frame_radius = radii_over_time[t - 1]
                # 地図を更新描画（2Dまたは3D）
                if view_mode == "2D":
                    # Folium地図を生成
                    m = folium.Map(location=[lat0, lon0], zoom_start=7)
                    # 延焼範囲（同心円3段階で表示）
                    # 大（黄）, 中（橙）, 小（赤）の円を重ねて描画
                    folium.Circle(location=[lat0, lon0], radius=frame_radius,
                                  color="yellow", fill=True, fill_opacity=0.2).add_to(m)
                    folium.Circle(location=[lat0, lon0], radius=frame_radius * 2/3,
                                  color="orange", fill=True, fill_opacity=0.3).add_to(m)
                    folium.Circle(location=[lat0, lon0], radius=frame_radius * 1/3,
                                  color="red", fill=True, fill_opacity=0.4).add_to(m)
                    # 火災発生地点のマーカー
                    folium.Marker(location=[lat0, lon0], 
                                  tooltip="火災発生地点",
                                  icon=folium.Icon(color="red", icon="fire", prefix="fa")
                                 ).add_to(m)
                    # 凡例をHTMLテンプレートで追加
                    legend_html = """
                    {% macro html(this, kwargs) %}
                    <div id='maplegend' class='maplegend'
                         style='position: absolute; z-index:9999; background-color: rgba(255, 255, 255, 0.7);
                                border-radius: 5px; padding: 10px; font-size: 12px; right: 20px; bottom: 20px;'>
                      <div class='legend-title'>凡例（影響の強さ）</div>
                      <div class='legend-scale'>
                        <ul class='legend-labels'>
                          <li><span style='background:red;opacity:0.7;'></span>強い影響 (赤)</li>
                          <li><span style='background:orange;opacity:0.7;'></span>中程度の影響 (橙)</li>
                          <li><span style='background:yellow;opacity:0.7;'></span>軽微な影響 (黄)</li>
                        </ul>
                      </div>
                    </div>
                    <style type='text/css'>
                      .maplegend .legend-title {font-weight: bold; margin-bottom: 5px;}
                      .maplegend .legend-scale ul {margin: 0; padding: 0;}
                      .maplegend .legend-scale ul li {list-style: none; line-height: 18px; margin-bottom: 2px;}
                      .maplegend .legend-scale ul li span {display: inline-block; width: 12px; height: 12px; margin-right: 6px;}
                    </style>
                    {% endmacro %}
                    """
                    macro = MacroElement()
                    macro._template = Template(legend_html)
                    m.get_root().add_child(macro)
                    # プレースホルダーに地図描画
                    with map_placeholder.container():
                        st_folium(m, width=700, height=500)
                        st.caption(f"経過時間: {t} 時間")
                else:
                    # PyDeck地図を生成（3D散布図レイヤー）
                    # 3段階の円をポイントデータとして設定（大,中,小半径）
                    data = [
                        {"pos": [lon0, lat0], "radius": frame_radius,        "color": [255, 255, 0, 80]},  # 黄
                        {"pos": [lon0, lat0], "radius": frame_radius * 2/3, "color": [255, 165, 0, 150]}, # 橙
                        {"pos": [lon0, lat0], "radius": frame_radius * 1/3, "color": [255, 0, 0, 200]}    # 赤
                    ]
                    layers = [
                        pdk.Layer(
                            "ScatterplotLayer",
                            data,
                            get_position="pos",
                            get_radius="radius",
                            get_fill_color="color",
                            radius_min_pixels=1,
                            radius_max_pixels=100,
                            opacity=0.3
                        ),
                        pdk.Layer(
                            "ScatterplotLayer",
                            [{"pos": [lon0, lat0]}],
                            get_position="pos",
                            get_radius=50,  # 中心マーカー用にごく小さな円
                            get_fill_color=[0, 0, 0],
                            radius_min_pixels=5,  # 常に少なくともピクセルサイズ5で表示
                        )
                    ]
                    # ビューポート設定（範囲に合わせてズーム調整）
                    # 経度方向の範囲からおおまかにズームレベル算出
                    lon_span = 2 * (frame_radius / (111000 * math.cos(math.radians(lat0))))
                    zoom_level = max(1, min(15, math.log2(360 / (lon_span if lon_span != 0 else 360))))
                    view_state = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=zoom_level, pitch=45)
                    deck = pdk.Deck(layers=layers, initial_view_state=view_state)
                    # プレースホルダーに地図描画
                    with map_placeholder.container():
                        st.pydeck_chart(deck)
                        st.caption(f"経過時間: {t} 時間")
                # 若干のディレイを入れて次フレームへ
                st.sleep(0.5)
        else:
            # 静止表示: スライダーで選択した時刻hourの状態を表示
            if view_mode == "2D":
                # Folium地図生成（選択時刻の半径で円表示）
                m = folium.Map(location=[lat0, lon0], zoom_start=7)
                folium.Circle(location=[lat0, lon0], radius=current_radius,
                              color="yellow", fill=True, fill_opacity=0.2).add_to(m)
                folium.Circle(location=[lat0, lon0], radius=current_radius * 2/3,
                              color="orange", fill=True, fill_opacity=0.3).add_to(m)
                folium.Circle(location=[lat0, lon0], radius=current_radius * 1/3,
                              color="red", fill=True, fill_opacity=0.4).add_to(m)
                folium.Marker(location=[lat0, lon0],
                              tooltip="火災発生地点",
                              icon=folium.Icon(color="red", icon="fire", prefix="fa")
                             ).add_to(m)
                # 凡例（HTMLマクロ前述と同じ）
                macro = MacroElement()
                macro._template = Template(legend_html)
                m.get_root().add_child(macro)
                map_placeholder = st_folium(m, width=700, height=500)
            else:
                # PyDeck地図生成（選択時刻の半径）
                data = [
                    {"pos": [lon0, lat0], "radius": current_radius,        "color": [255, 255, 0, 80]},
                    {"pos": [lon0, lat0], "radius": current_radius * 2/3, "color": [255, 165, 0, 150]},
                    {"pos": [lon0, lat0], "radius": current_radius * 1/3, "color": [255, 0, 0, 200]}
                ]
                layers = [
                    pdk.Layer("ScatterplotLayer", data,
                              get_position="pos",
                              get_radius="radius",
                              get_fill_color="color",
                              radius_min_pixels=1, radius_max_pixels=100, opacity=0.3),
                    pdk.Layer("ScatterplotLayer", [{"pos": [lon0, lat0]}],
                              get_position="pos",
                              get_radius=50, get_fill_color=[0, 0, 0],
                              radius_min_pixels=5)
                ]
                lon_span = 2 * (current_radius / (111000 * math.cos(math.radians(lat0))))
                zoom_level = max(1, min(15, math.log2(360 / (lon_span if lon_span != 0 else 360))))
                view_state = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=zoom_level, pitch=45)
                deck = pdk.Deck(layers=layers, initial_view_state=view_state)
                map_placeholder = st.pydeck_chart(deck)
            # 凡例（PyDeck用）をテキスト表示
            st.markdown("**凡例:**  🟥 強い影響（赤）&nbsp;&nbsp;🟧 中程度（橙）&nbsp;&nbsp;🟨 軽微（黄）")

    # **レポート表示タブ**
    with tab_report:
        st.subheader("シミュレーション結果レポート")
        # 数値結果をJSON形式で表示
        result_dict = {
            "radius_m": sim["radius_final"],
            "area_sqm": sim["area"],
            "water_volume_tons": sim["water_vol"]
        }
        st.json(result_dict, expanded=True)
        # 解説テキストの表示
        st.markdown("**解説:** 火災発生地点から風下方向へ広がる火災の距離を基に、全方向への延焼半径を計算しています。上記の結果では、風速{:.1f} m/s（約{:.1f} km/h）の条件で{}時間後に火災が半径約{:.0f} mまで拡大すると仮定しました。これは非常に単純化したモデルであり、実際の火災の広がり方は地形・植生・湿度・風向きの変化など多くの要因で大きく異なります。".format(
            wind_speed, wind_speed * 3.6, duration, sim["radius_final"]
        ))
        st.markdown(
            "計算された延焼範囲の面積は約{:.0f} 平方メートルに及びます。この面積に対し一様に1mmの雨が降ったと仮定すると、水{:.0f}トンが必要になる計算になります。\
            \n\n**計算根拠:** 1) 風速を秒速から時速に換算し ({} m/s = {:.1f} km/h)、その速度で{}時間進む距離を求めました（約{:.1f} km）。\
            2) 得られた距離を延焼半径と仮定し、円形範囲の面積をπr²で算出しました。\
            3) 面積に対し1mmの降水量を全域に与えると仮定し、水量を面積×0.001mとして算出しました。\n\n※本シミュレーションは非常に概略的なモデルに基づいており、実際の延焼速度・範囲を正確に予測するものではありません。大規模火災時には専門機関の分析結果や最新の現場情報に従って判断してください。".format(
            sim["area"], sim["water_vol"], wind_speed, wind_speed * 3.6, duration, (wind_speed * 3.6 * duration)
        ))
