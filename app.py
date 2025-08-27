# app.py — 結果が消えない永続化対応版（見た目はそのまま）
# =============================================================================
# 目的:
#  - Streamlit で火災拡大シミュレーション（Gemini数値→Gemini要約）
#  - 扇形ポリゴンをFoliumに描画
#  - ボタン後の再実行でも**結果が消えない**ように session_state に保存し再描画
# 仕様:
#  - サイドバー: 発生地点追加/全消去/燃料選択/気象取得
#  - メイン: ベースマップ（常時表示）/ 日・週・月タブ（スライダー＋実行）
#  - 数値（半径/面積/水量）、要約、地図ツールチップ
# 重要:
#  - Secrets: st.secrets["general"]["api_key"]（未設定でも落ちない）
#  - Gemini応答は parts[].text / output 両対応 + ```json 抽出
# 依存: streamlit, folium, streamlit-folium, requests
# 実行: streamlit run app.py
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

# ============================
# ページ設定
# ============================
st.set_page_config(page_title="火災拡大シミュレーション（Gemini要約付き）", layout="wide")

# ============================
# Secrets / APIキー取得（堅牢化）
# ============================
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-001")
API_KEY: Optional[str] = None
try:
    if "general" in st.secrets and "api_key" in st.secrets["general"]:
        API_KEY = st.secrets["general"]["api_key"]
except Exception:
    API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

# ============================
# JSON抽出ユーティリティ
# ============================

def extract_json(text: str) -> Optional[dict]:
    """Gemini応答から最初のJSONを抽出して dict に。```json フェンスにも対応。"""
    if not text:
        return None
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text)
    if not m:
        m = re.search(r"(\{[\s\S]*\})", text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None

# ============================
# Open‑Meteo（現在気象）
# ============================

@st.cache_data(show_spinner=False)
def get_weather(lat: float, lon: float) -> Optional[Dict[str, float]]:
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            "&current_weather=true&hourly=relativehumidity_2m,precipitation&timezone=auto"
        )
        resp = requests.get(url, timeout=20)
        if resp.status_code != 200:
            return None
        data = resp.json()
        cur = data.get("current_weather", {}) or {}
        res = {
            "temperature": cur.get("temperature"),
            "windspeed": cur.get("windspeed"),
            "winddirection": cur.get("winddirection"),
            "weathercode": cur.get("weathercode"),
        }
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
    except Exception:
        return None

# ============================
# Gemini REST 呼び出し
# ============================

def gemini_generate_json(prompt: str) -> Tuple[Optional[dict], Optional[dict], Optional[str]]:
    """数値推定用: JSONを期待。parts[].text / output 両対応。戻り値=(parsed, raw, text)。"""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
        headers = {"Content-Type": "application/json"}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        raw = r.json() if r.content else None
        if r.status_code != 200 or not raw:
            return None, raw, None
        text = ""
        cands = raw.get("candidates") or []
        if cands:
            content = cands[0].get("content") or {}
            parts = content.get("parts") or []
            for p in parts:
                text += p.get("text", "")
            if not text:
                text = cands[0].get("output", "")
        parsed = extract_json(text) or (json.loads(text) if text.strip().startswith("{") else None)
        return parsed, raw, text
    except Exception as e:
        st.error(f"Gemini呼び出し中に例外: {e}")
        return None, None, None


def gemini_summarize(json_obj: dict) -> Tuple[Optional[str], Optional[dict], Optional[str]]:
    """要約用: 一般向けの短文を生成。戻り値=(summary, raw, text)。"""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
        headers = {"Content-Type": "application/json"}
        p = (
            "以下は火災拡大シミュレーションの推定結果JSONです。\n"
            "この数値の意味が直感的に伝わる、日本語の短い説明文を出力してください。\n"
            "専門用語は避け、半径(m)、面積(m²)、必要放水量(トン)の意味が分かるように。\n"
            "JSONは次です:\n" + json.dumps(json_obj, ensure_ascii=False)
        )
        payload = {"contents": [{"parts": [{"text": p}]}]}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        raw = r.json() if r.content else None
        if r.status_code != 200 or not raw:
            return None, raw, None
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

# ============================
# 扇形ポリゴン
# ============================

def create_sector_polygon(lat: float, lon: float, radius_m: float, wind_dir_deg: float, steps: int = 60) -> List[Tuple[float, float]]:
    coords = [(lat, lon)]
    start, end = wind_dir_deg - 90.0, wind_dir_deg + 90.0
    for i in range(steps + 1):
        ang = math.radians(start + (end - start) * i / steps)
        north_m = radius_m * math.cos(ang)
        east_m = radius_m * math.sin(ang)
        dlat = north_m / 111_000.0
        dlon = east_m / 111_000.0  # 指示に従い高緯度補正なし
        coords.append((lat + dlat, lon + dlon))
    return coords

# ============================
# セッション状態（永続化の要）
# ============================
if "points" not in st.session_state:
    st.session_state.points: List[Tuple[float, float]] = []
if "weather" not in st.session_state:
    st.session_state.weather: Optional[Dict[str, float]] = None
# タブごとの最新結果を保持（再実行しても再表示できる）
if "results" not in st.session_state:
    st.session_state.results = {"day": None, "week": None, "month": None}
# タブごとの扇形オーバーレイのパラメータ（再描画用）
if "overlays_by_tab" not in st.session_state:
    st.session_state.overlays_by_tab = {"day": None, "week": None, "month": None}

# ============================
# サイドバー UI
# ============================
st.sidebar.header("発生地点と条件設定")
with st.sidebar.form("point_form"):
    lat_in = st.number_input("緯度", value=34.257586, format="%.6f")
    lon_in = st.number_input("経度", value=133.204356, format="%.6f")
    add = st.form_submit_button("発生地点を追加")
    if add:
        st.session_state.points.append((lat_in, lon_in))
        st.sidebar.success(f"地点 ({lat_in:.6f},{lon_in:.6f}) を追加")

if st.sidebar.button("登録地点を消去"):
    st.session_state.points = []
    st.session_state.weather = None
    # 結果とオーバーレイもクリア
    st.session_state.results = {"day": None, "week": None, "month": None}
    st.session_state.overlays_by_tab = {"day": None, "week": None, "month": None}
    st.sidebar.info("削除しました")

fuel_opts = {"森林（高燃料)": "森林（高燃料）", "草地（中燃料)": "草地（中燃料）", "都市部（低燃料)": "都市部（低燃料）"}
sel_fuel = st.sidebar.selectbox("燃料特性", list(fuel_opts.keys()))
fuel_type = fuel_opts[sel_fuel]

if st.sidebar.button("⛅ 気象データ取得"):
    if st.session_state.points:
        lat0, lon0 = st.session_state.points[0]
        st.session_state.weather = get_weather(lat0, lon0)
        if st.session_state.weather:
            st.sidebar.success("気象データを取得しました（代表: 最初の発火点）")
        else:
            st.sidebar.error("気象データの取得に失敗しました")
    else:
        st.sidebar.warning("先に地点を追加してください")

# ============================
# メインビュー（ベースマップは常時表示）
# ============================
st.title("火災拡大シミュレーション（Gemini要約付き）")
center = st.session_state.points[0] if st.session_state.points else (34.257586, 133.204356)
base_map = folium.Map(location=[center[0], center[1]], zoom_start=13, control_scale=True)
# マーカー
for i, (plat, plon) in enumerate(st.session_state.points, start=1):
    folium.Marker([plat, plon], icon=folium.Icon(color="red"), tooltip=f"発火点 {i}: {plat:.5f},{plon:.5f}").add_to(base_map)
# 既存オーバーレイを**毎回**再描画（これで再実行でも消えない）
for tab_key in ("day", "week", "month"):
    o = st.session_state.overlays_by_tab.get(tab_key)
    if o:
        sector = create_sector_polygon(o["lat"], o["lon"], o["radius_m"], o["wind_dir"])  # 再生成
        folium.Polygon(sector, color="red", fill=True, fill_opacity=0.4,
                       tooltip=f"半径{o['radius_m']:.0f}m 面積{o['area_sqm']:.0f}㎡").add_to(base_map)
# 地図表示
st_folium(base_map, width=900, height=560)

# ============================
# タブ（日/週/月）
# ============================
tab_day, tab_week, tab_month = st.tabs(["日", "週", "月"])


def render_result_block(pred: dict, summary: Optional[str]):
    """数値3指標と要約を同じ見た目で表示（永続化でも使う）。"""
    radius_m = float(pred.get("radius_m", 0) or 0)
    area_sqm = float(pred.get("area_sqm", 0) or 0)
    water_t  = float(pred.get("water_volume_tons", 0) or 0)
    st.subheader("シミュレーション結果")
    c1, c2, c3 = st.columns(3)
    c1.metric("半径 (m)", f"{radius_m:,.0f}")
    c2.metric("面積 (m²)", f"{area_sqm:,.0f}")
    c3.metric("必要放水量 (トン)", f"{water_t:,.1f}")
    if summary:
        st.subheader("Geminiによる要約")
        st.write(summary)


def run_sim(duration_hours: float, tab_key: str, container):
    with container:
        if not st.session_state.points:
            st.warning("発生地点を追加してください")
            return
        if not st.session_state.weather:
            st.warning("気象データを取得してください")
            return
        lat0, lon0 = st.session_state.points[0]
        wx = st.session_state.weather
        # 数値推定プロンプト（JSONのみ要求）
        p = (
            "あなたは火災拡大シミュレーションの専門家です。\n"
            "次の条件に基づき、火災の拡大を純粋なJSONのみで推定してください。\n"
            f"- 発生地点: 緯度 {lat0}, 経度 {lon0}\n"
            f"- 気象: 風速 {wx.get('windspeed','不明')} m/s, 風向 {wx.get('winddirection','不明')} 度 (北=0, 東=90, 南=180, 西=270)\n"
            f"        温度 {wx.get('temperature','不明')} °C, 湿度 {wx.get('humidity','不明')} %, 降水 {wx.get('precipitation','不明')} mm/h\n"
            f"- シミュレーション時間: {duration_hours} 時間\n"
            f"- 燃料特性: {fuel_type}\n"
            "- 地形: 10度程度の傾斜 / 標高150m\n"
            "- 植生: 松林と草地が混在\n"
            '出力: 他の文字を一切含まず、以下のJSONのみを返すこと。\n{"radius_m": <float>, "area_sqm": <float>, "water_volume_tons": <float>}\n'
        )
        pred = None
        raw_pred = None
        if API_KEY:
            pred, raw_pred, _ = gemini_generate_json(p)
        if pred is None:
            # フォールバック簡易推定
            wind = float(wx.get("windspeed") or 0.0)
            base = {"森林（高燃料）": 0.30, "草地（中燃料）": 0.60, "都市部（低燃料）": 0.20}.get(fuel_type, 0.40)
            v_eff = base * (1.0 + 0.12 * wind)
            r = max(30.0, v_eff * duration_hours * 3600.0)
            a = 0.5 * math.pi * r * r
            w = a * 0.01
            pred = {"radius_m": r, "area_sqm": a, "water_volume_tons": w}
            if raw_pred:
                with st.expander("Gemini #1 生JSON（失敗時の参照用）"):
                    st.json(raw_pred)
        # 値の相互補完
        r = float(pred.get("radius_m") or 0.0)
        a = float(pred.get("area_sqm") or 0.0)
        w = float(pred.get("water_volume_tons") or 0.0)
        if r > 0 and a <= 0:
            a = 0.5 * math.pi * r * r
        if a > 0 and w <= 0:
            w = a * 0.01
        pred = {"radius_m": r, "area_sqm": a, "water_volume_tons": w}
        # 要約
        summary = None
        raw_sum = None
        if API_KEY:
            summary, raw_sum, _ = gemini_summarize(pred)
        # --- ここが肝: 結果とオーバーレイを session_state に保存 ---
        st.session_state.results[tab_key] = {"pred": pred, "summary": summary}
        wind_dir = float(wx.get("winddirection") or 0.0)
        st.session_state.overlays_by_tab[tab_key] = {
            "lat": lat0, "lon": lon0, "radius_m": r, "area_sqm": a, "wind_dir": wind_dir,
        }
        # 表示（見た目は既存と同じ）
        render_result_block(pred, summary)
        # デバッグ用 生JSON 折りたたみ
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

# --- タブUI ---
with tab_day:
    days = st.slider("日数（1〜30）", min_value=1, max_value=30, value=3, step=1, key="slider_day")
    if st.button("▶ シミュレーション実行（日タブ）", key="btn_day"):
        run_sim(duration_hours=float(days) * 24.0, tab_key="day", container=tab_day)
    # 直近結果の自動再表示（再実行でも消えない）
    saved = st.session_state.results.get("day")
    if saved:
        render_result_block(saved["pred"], saved.get("summary"))

with tab_week:
    weeks = st.slider("週間（1〜52）", min_value=1, max_value=52, value=1, step=1, key="slider_week")
    if st.button("▶ シミュレーション実行（週タブ）", key="btn_week"):
        run_sim(duration_hours=float(weeks) * 7.0 * 24.0, tab_key="week", container=tab_week)
    saved = st.session_state.results.get("week")
    if saved:
        render_result_block(saved["pred"], saved.get("summary"))

with tab_month:
    months = st.slider("月数（1〜12）", min_value=1, max_value=12, value=1, step=1, key="slider_month")
    if st.button("▶ シミュレーション実行（月タブ）", key="btn_month"):
        # 単純化: 1か月=30日
        run_sim(duration_hours=float(months) * 30.0 * 24.0, tab_key="month", container=tab_month)
    saved = st.session_state.results.get("month")
    if saved:
        render_result_block(saved["pred"], saved.get("summary"))

# ============================
# ASCIIイメージ図（参考）
# ============================
st.markdown(
    """
```
[User] ──入力(緯度/経度/燃料)──▶ [Streamlit UI] ──保存▶ session_state
    │                                        │
    │  「気象取得」                           ▼
    └────────────────────────────▶ [Open-Meteo API] ─▶ weather
                                             │
「シミュレーション実行」                      ▼
      ┌───────────────────────────────────────────────────────┐
      │   [Gemini#1 数値推定]  (条件→JSONのみ)                │
      │    └▶ {"radius_m","area_sqm","water_volume_tons"}     │
      └───────────────────────────────────────────────────────┘
                               │
                               ▼
                    [Gemini#2 要約生成]（一般向け短文）
                               │
                               ▼
                [Folium] 半円ポリゴン表示（風向±90°, 半径R）
                               │
                               ▼
          [UI] 数値 & 要約表示 / 生JSON折畳み（※結果は永続化され保持）
```
"""
)
