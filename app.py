# app.py
# =============================================================================
# 火災拡大シミュレーション（Gemini要約付き）- 単一ファイル完成版
# 目的:
#  - Streamlit で発火点と気象を入力
#  - Gemini API で {radius_m, area_sqm, water_volume_tons} をJSON推定
#  - そのJSONを再度Geminiに渡して一般向け要約
#  - 風向±90°の扇形ポリゴンをFoliumで可視化
# 重要:
#  - Secrets: st.secrets["general"]["api_key"] を Gemini の API Key として利用（なければ環境変数 GOOGLE_API_KEY / GEMINI_API_KEY）
#  - Open-Meteo から現在気象＋hourly（湿度/降水）取得（timezone=auto）
#  - 例外時も落ちない堅牢設計（生JSONを折りたたみ表示）
# 依存:
#  - streamlit, folium, streamlit-folium, requests
# 実行:
#  - streamlit run app.py
# =============================================================================

import os
import json
import math
import re
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict

import requests
import streamlit as st
import folium
from streamlit_folium import st_folium


# ==========================
# ページ設定
# ==========================
st.set_page_config(page_title="火災拡大シミュレーション（Gemini要約付き）", layout="wide")


# ==========================
# Secrets / API キー（堅牢化）
# ==========================
def _get_api_key() -> Optional[str]:
    """st.secrets['general']['api_key'] → env(GOOGLE_API_KEY/GEMINI_API_KEY) の順で取得。"""
    try:
        if "general" in st.secrets and "api_key" in st.secrets["general"]:
            return st.secrets["general"]["api_key"]
    except Exception:
        pass
    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")


API_KEY = _get_api_key()
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-001")


# ==========================
# ユーティリティ
# ==========================
def extract_json(text: str) -> Optional[dict]:
    """
    Gemini応答が Markdown の ```json ... ``` に入る可能性があるため、
    最初に見つかった JSON オブジェクトを抽出して dict にして返す。
    """
    if not text:
        return None
    # ```json ... ``` フェンス優先
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text)
    if not m:
        # フェンスなしの素の { ... }
        m = re.search(r"(\{[\s\S]*\})", text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def get_weather(lat: float, lon: float) -> Optional[Dict[str, float]]:
    """
    Open-Meteo から現在の気象を取得。timezone=auto。
    取得項目: temperature, windspeed, winddirection, humidity, precipitation
    """
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            "&current_weather=true&hourly=relativehumidity_2m,precipitation&timezone=auto"
        )
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            st.warning(f"Open-Meteo取得に失敗しました（HTTP {r.status_code}）。")
            return None

        data = r.json()
        cur = data.get("current_weather", {}) or {}
        res = {
            "temperature": cur.get("temperature"),
            "windspeed": cur.get("windspeed"),
            "winddirection": cur.get("winddirection"),
            "weathercode": cur.get("weathercode"),
        }

        # 現在時刻に対応する hourly の湿度・降水
        t = cur.get("time")
        hourly = data.get("hourly", {}) or {}
        times = hourly.get("time", []) or []
        if t in times:
            i = times.index(t)
            rh = hourly.get("relativehumidity_2m", []) or []
            pr = hourly.get("precipitation", []) or []
            if i < len(rh):
                res["humidity"] = rh[i]
            if i < len(pr):
                res["precipitation"] = pr[i]
        return res
    except Exception as e:
        st.error(f"気象取得中にエラー: {e}")
        return None


def create_sector_polygon(lat: float, lon: float, radius_m: float, wind_dir_deg: float, steps: int = 60) -> List[Tuple[float, float]]:
    """
    風向を中心に ±90° の扇形ポリゴン（Folium 用に [lat, lon] 順）。
    簡易換算: 1度 ≒ 111,000 m（高緯度補正なし）。
    """
    coords = [(lat, lon)]
    start_deg = wind_dir_deg - 90.0
    end_deg = wind_dir_deg + 90.0
    deg_per_meter = 1.0 / 111_000.0

    for i in range(steps + 1):
        ang = math.radians(start_deg + (end_deg - start_deg) * i / steps)
        north_m = radius_m * math.cos(ang)
        east_m = radius_m * math.sin(ang)
        dlat = north_m * deg_per_meter
        dlon = east_m * deg_per_meter  # 指示どおり cos(lat) は掛けない
        coords.append((lat + dlat, lon + dlon))
    return coords


def gemini_generate_json(prompt: str, api_key: str, model_name: str) -> Tuple[Optional[dict], Optional[dict], Optional[str]]:
    """
    Gemini REST で JSON を生成。
    - candidates[0].content.parts[].text または candidates[0].output の両対応。
    - 戻り値: (parsed_json, raw_json, raw_text)
    """
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        raw = r.json() if r.content else None

        if r.status_code != 200 or not raw:
            return None, raw, None

        # candidates[0].content.parts[].text を連結
        text = ""
        cands = raw.get("candidates") or []
        if cands:
            # まず content.parts[].text を試す
            content = cands[0].get("content") or {}
            parts = content.get("parts") or []
            for p in parts:
                text += p.get("text", "")
            # 次に output（古い/別バリアント）をフォールバック
            if not text:
                text = cands[0].get("output", "")

        parsed = extract_json(text) or (json.loads(text) if text.strip().startswith("{") else None)
        return parsed, raw, text
    except Exception as e:
        st.error(f"Gemini呼び出し中に例外: {e}")
        return None, None, None


def gemini_summarize(json_obj: dict, api_key: str, model_name: str) -> Tuple[Optional[str], Optional[dict], Optional[str]]:
    """
    1回目で得た推定JSONをそのまま渡して一般向け短文を生成。
    - 戻り値: (summary_text, raw_json, raw_text)
    """
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}

        # ここではトリプルクォートや ``` を使わず、連結で構築
        p = ""
        p += "以下は火災拡大シミュレーションの推定結果JSONです。\n"
        p += "この数値の意味が直感的に伝わる、日本語の短い説明文を出力してください。\n"
        p += "専門用語は避け、半径(m)、面積(m²)、必要放水量(トン)の意味が分かるように。\n"
        p += "JSONは次です:\n"
        p += json.dumps(json_obj, ensure_ascii=False)

        payload = {"contents": [{"parts": [{"text": p}]}]}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        raw = r.json() if r.content else None
        if r.status_code != 200 or not raw:
            return None, raw, None

        # テキスト抽出（parts[].text 優先 → output フォールバック）
        text = ""
        cands = raw.get("candidates") or []
        if cands:
            content = cands[0].get("content") or {}
            parts = content.get("parts") or []
            for pt in parts:
                text += pt.get("text", "")
            if not text:
                text = cands[0].get("output", "")
        return (text.strip() or None), raw, text
    except Exception as e:
        st.error(f"Gemini要約中に例外: {e}")
        return None, None, None


def fallback_estimation(weather: Dict[str, float], fuel_type: str, hours: float) -> dict:
    """
    Gemini失敗時の簡易推定（半円モデル）。
    - 燃料別ベース拡大速度 v0 [m/s]
    - 風の寄与を線形係数で加味
    - area ≈ 0.5 * π * r^2
    - water ≈ 0.01 ton/m²（= 10 L/m²）
    """
    wind = float(weather.get("windspeed") or 0.0)
    base = {"森林（高燃料）": 0.30, "草地（中燃料）": 0.60, "都市部（低燃料）": 0.20}.get(fuel_type, 0.40)
    v_eff = base * (1.0 + 0.12 * wind)
    radius_m = max(30.0, v_eff * hours * 3600.0)
    area_sqm = 0.5 * math.pi * radius_m * radius_m
    water_tons = area_sqm * 0.01
    return {"radius_m": radius_m, "area_sqm": area_sqm, "water_volume_tons": water_tons}


# ==========================
# セッション状態
# ==========================
if "points" not in st.session_state:
    st.session_state.points: List[Tuple[float, float]] = []
if "weather_data" not in st.session_state:
    st.session_state.weather_data: Optional[Dict[str, float]] = None


# ==========================
# サイドバー UI
# ==========================
st.sidebar.title("火災発生地点の入力")
with st.sidebar.form(key="location_form"):
    lat_input = st.number_input("緯度", format="%.6f", value=34.257586)
    lon_input = st.number_input("経度", format="%.6f", value=133.204356)
    add_point = st.form_submit_button("発生地点を追加")
    if add_point:
        st.session_state.points.append((lat_input, lon_input))
        st.sidebar.success(f"地点 ({lat_input:.6f}, {lon_input:.6f}) を追加しました。")

if st.sidebar.button("登録地点を消去"):
    st.session_state.points = []
    st.session_state.weather_data = None
    st.sidebar.info("全ての発生地点を削除しました。")

st.sidebar.title("燃料特性の選択")
fuel_options = ["森林（高燃料）", "草地（中燃料）", "都市部（低燃料）"]
fuel_type = st.sidebar.selectbox("燃料特性を選択してください", fuel_options, index=0)

st.sidebar.divider()
get_wx_clicked = st.sidebar.button("⛅ 気象データ取得（Open-Meteo）")


# ==========================
# メインビュー：ベースマップ（常時表示）
# ==========================
st.title("火災拡大シミュレーション（Gemini要約付き）")

center = st.session_state.points[0] if st.session_state.points else (34.257586, 133.204356)
base_map = folium.Map(location=[center[0], center[1]], zoom_start=13, control_scale=True)
for i, (plat, plon) in enumerate(st.session_state.points, start=1):
    folium.Marker(location=[plat, plon], icon=folium.Icon(color="red"),
                  tooltip=f"発火点 {i}: {plat:.5f}, {plon:.5f}").add_to(base_map)
st_folium(base_map, height=520, width=None)


# ==========================
# 気象データ取得
# ==========================
if get_wx_clicked:
    if not st.session_state.points:
        st.warning("先に発生地点を1つ以上登録してください。")
    else:
        rep_lat, rep_lon = st.session_state.points[0]
        wx = get_weather(rep_lat, rep_lon)
        if wx:
            st.session_state.weather_data = wx
            st.success("気象データを取得・保持しました（代表地点は最初の発火点）。")
        else:
            st.session_state.weather_data = None
            st.error("気象データの取得に失敗しました。後で再実行してください。")

if st.session_state.weather_data:
    with st.expander("取得済み気象（代表地点の現在値）", expanded=True):
        st.json(st.session_state.weather_data)


# ==========================
# タブ：日 / 週 / 月
# ==========================
tab_day, tab_week, tab_month = st.tabs(["日", "週", "月"])


def run_simulation(duration_hours: float, tab_area):
    """指定時間（時間単位）でGemini推定 → 要約 → 地図描画＆数値表示。"""
    with tab_area:
        # 前提チェック
        if not st.session_state.points:
            st.warning("発生地点が未設定です。サイドバーから追加してください。")
            return
        if not st.session_state.weather_data:
            st.warning("気象が未取得です。「⛅ 気象データ取得」を先に実行してください。")
            return

        rep_lat, rep_lon = st.session_state.points[0]
        wx = st.session_state.weather_data

        # 数値推定プロンプト（JSONのみ要求）— 連結で構築（```禁止）
        p = ""
        p += "あなたは火災拡大シミュレーションの専門家です。\n"
        p += "次の条件に基づき、火災の拡大を純粋なJSONのみで推定してください。\n"
        p += "条件:\n"
        p += f"- 発生地点: 緯度 {rep_lat}, 経度 {rep_lon}\n"
        p += f"- 気象: 風速 {wx.get('windspeed','不明')} m/s, 風向 {wx.get('winddirection','不明')} 度 (北=0, 東=90, 南=180, 西=270)\n"
        p += f"        温度 {wx.get('temperature','不明')} °C, 湿度 {wx.get('humidity','不明')} %, 降水 {wx.get('precipitation','不明')} mm/h\n"
        p += f"- シミュレーション時間: {duration_hours} 時間\n"
        p += f"- 燃料特性: {fuel_type}\n"
        p += "- 地形: 10度程度の傾斜 / 標高150m\n"
        p += "- 植生: 松林と草地が混在\n"
        p += "出力: 他の文字を一切含まず、以下のJSONのみを返すこと。\n"
        p += '{"radius_m": <float>, "area_sqm": <float>, "water_volume_tons": <float>}\n'

        pred = None
        raw_pred = None
        raw_pred_text = None

        if API_KEY:
            pred, raw_pred, raw_pred_text = gemini_generate_json(p, API_KEY, MODEL_NAME)
        else:
            st.info("Gemini APIキーが未設定のため、簡易推定で補完します（Secrets または環境変数で設定可）。")

        if pred is None:
            st.info("Gemini推定に失敗したため、簡易推定で補完しました。")
            pred = fallback_estimation(wx, fuel_type, duration_hours)

        # 値の正規化・相互補完
        radius_m = float(pred.get("radius_m") or 0.0)
        area_sqm = float(pred.get("area_sqm") or 0.0)
        water_t  = float(pred.get("water_volume_tons") or 0.0)
        if radius_m > 0 and area_sqm <= 0:
            area_sqm = 0.5 * math.pi * radius_m * radius_m
        if area_sqm > 0 and water_t <= 0:
            water_t = area_sqm * 0.01

        # 要約（一般向け短文）
        summary = None
        raw_sum = None
        raw_sum_text = None
        if API_KEY:
            summary, raw_sum, raw_sum_text = gemini_summarize(
                {"radius_m": radius_m, "area_sqm": area_sqm, "water_volume_tons": water_t},
                API_KEY, MODEL_NAME
            )

        # 可視化（風向±90° 扇形）
        wind_dir = float(wx.get("winddirection") or 0.0)
        sector = create_sector_polygon(rep_lat, rep_lon, radius_m, wind_dir)
        result_map = folium.Map(location=[rep_lat, rep_lon], zoom_start=13, control_scale=True)
        folium.Marker(location=[rep_lat, rep_lon], icon=folium.Icon(color="red"), tooltip="代表発火点").add_to(result_map)
        folium.Polygon(
            locations=sector,
            color="red",
            weight=2,
            fill=True,
            fill_opacity=0.4,
            tooltip=f"半径: {radius_m:.0f} m / 面積: {area_sqm:.0f} m²"
        ).add_to(result_map)

        # 数値と要約
        st.subheader("シミュレーション結果")
        c1, c2, c3 = st.columns(3)
        c1.metric("半径 (m)", f"{radius_m:,.0f}")
        c2.metric("面積 (m²)", f"{area_sqm:,.0f}")
        c3.metric("必要放水量 (トン)", f"{water_t:,.1f}")

        st.subheader("地図（想定拡大範囲：風向±90°の扇形）")
        st_folium(result_map, height=520, width=None)

        if summary:
            st.subheader("Geminiによる要約")
            st.write(summary)

        # デバッグ/検証用 生JSON 折りたたみ
        with st.expander("Gemini #1 生JSON応答（検証用）"):
            if raw_pred is not None:
                st.json(raw_pred)
            else:
                st.caption("応答なし / キー未設定")

        with st.expander("Gemini #2 生JSON応答（検証用）"):
            if raw_sum is not None:
                st.json(raw_sum)
            else:
                st.caption("応答なし / キー未設定")


with tab_day:
    days = st.slider("日数（1〜30）", min_value=1, max_value=30, value=3, step=1)
    if st.button("▶ シミュレーション実行（日タブ）"):
        run_simulation(duration_hours=float(days) * 24.0, tab_area=tab_day)

with tab_week:
    weeks = st.slider("週間（1〜52）", min_value=1, max_value=52, value=1, step=1)
    if st.button("▶ シミュレーション実行（週タブ）"):
        run_simulation(duration_hours=float(weeks) * 7.0 * 24.0, tab_area=tab_week)

with tab_month:
    months = st.slider("月数（1〜12）", min_value=1, max_value=12, value=1, step=1)
    if st.button("▶ シミュレーション実行（月タブ）"):
        # 単純化：1か月=30日換算
        run_simulation(duration_hours=float(months) * 30.0 * 24.0, tab_area=tab_month)


# ==========================
# フッター
# ==========================
st.caption("ヒント: 先に「⛅ 気象データ取得」を行ってから、各タブの『シミュレーション実行』を押してください。")
