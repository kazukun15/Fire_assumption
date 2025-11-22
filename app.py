# -*- coding: utf-8 -*-
"""
Fire Spread Simulator Pro (Streamlit + Gemini 2.5 Flash Ensemble)
----------------------------------------------------------------
- 物理モデル + Gemini 2.5 Flash を組み合わせたハイブリッド火災拡大シミュレーション
- Gemini を複数視点で並列実行し、重み付きアンサンブルで総合判断
- 発生源の指定: 地図クリック / 住所検索 / 緯度・経度入力
- OpenWeather の気象情報を取得して解析に反映
- 発生源からの延焼を、地図上で時間スライダーで表示（自動再生機能は削除）
- 予測時間は「分・時間・日」の単位で指定可能（内部では分に換算）
- Gemini の数値結果を「世界一の消防士・災害スペシャリスト」として
  現状評価・延焼可能性・消火アドバイスの形で解説

■ 必要ライブラリ
- streamlit
- numpy
- matplotlib
- google-generativeai
- requests
- folium
- streamlit-folium

■ 起動
streamlit run app.py

■ .streamlit/secrets.toml 例
[general]
api_key = "YOUR_GOOGLE_API_KEY"               # Gemini 用（Google API Key）

[mapbox]
access_token = "YOUR_MAPBOX_ACCESS_TOKEN"     # ジオコーディング用

[openweather]
api_key = "YOUR_OPENWEATHER_API_KEY"         # 気象情報取得用
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager as fm
import requests
import urllib.parse
import folium
import google.generativeai as genai

# ---- streamlit_folium の安全なインポート ----
try:
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

# ------------------------- ページ設定 / グローバル -------------------------
st.set_page_config(
    page_title="Fire Spread Simulator Pro",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- 日本語フォント設定 & グラフテーマ（□対策＋見た目調整） ----
def configure_matplotlib_for_japanese() -> None:
    """
    利用可能な日本語フォントを自動検出して設定。
    見つからない場合は sans-serif のまま。
    """
    try:
        available = {f.name for f in fm.fontManager.ttflist}
        candidates = [
            "IPAexGothic",
            "IPAPGothic",
            "Noto Sans CJK JP",
            "Noto Sans JP",
            "Yu Gothic",
            "YuGothic",
            "MS Gothic",
            "MS UI Gothic",
        ]
        for name in candidates:
            if name in available:
                matplotlib.rcParams["font.family"] = name
                break
        else:
            matplotlib.rcParams["font.family"] = "sans-serif"
    except Exception:
        matplotlib.rcParams["font.family"] = "sans-serif"

    matplotlib.rcParams["axes.unicode_minus"] = False

    # グラフの見た目（Streamlit ライトテーマ寄り）
    base_bg = "#f0f2f6"
    matplotlib.rcParams["figure.facecolor"] = base_bg
    matplotlib.rcParams["axes.facecolor"] = "#ffffff"
    matplotlib.rcParams["axes.edgecolor"] = "#cccccc"
    matplotlib.rcParams["grid.color"] = "#dddddd"
    matplotlib.rcParams["grid.alpha"] = 0.6
    matplotlib.rcParams["axes.grid"] = True
    matplotlib.rcParams["axes.titlesize"] = 12
    matplotlib.rcParams["axes.labelsize"] = 11

configure_matplotlib_for_japanese()

# ---- 軽いCSSで全体を少し整える ----
CUSTOM_CSS = """
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
div[data-testid="stMetric"] > div {white-space: nowrap;}
h3, h4 { margin-top: 0.6rem; }
.small { font-size: 0.92rem; opacity: 0.8; }
"""
st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

# ------------------------------ ドメインモデル ------------------------------
@dataclass
class Inputs:
    duration_min: float
    wind_speed_ms: float
    wind_dir_deg: float
    rel_humidity: float
    air_temp_c: float
    slope_percent: float
    fuel_class: str
    init_radius_m: float
    attack_duration_min: float
    app_rate_lpm_per_m: float
    efficiency: float

@dataclass
class Outputs:
    radius_m: float
    area_sqm: float
    water_volume_tons: float
    ellipse_a_m: float
    ellipse_b_m: float
    perimeter_m: float

# ------------------------------ 物理モデル用パラメータ ------------------------------
BASE_RATE_BY_FUEL = {
    "grass": 8.0,
    "shrub": 3.0,
    "timber": 0.6,
}
HUMIDITY_K = 1.1
WIND_A = 0.10
WIND_B = 0.010
SLOPE_K = 4.0
LB_C = 0.30
LB_MAX = 5.0
EPS = 1e-9

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# ------------------------------ 物理モデル ------------------------------
def humidity_factor(rh: float) -> float:
    f = math.exp(-HUMIDITY_K * max(0.0, rh - 30.0) / 70.0)
    if rh < 30.0:
        f = 1.0 + 0.02 * (30.0 - rh)
    return clamp(f, 0.25, 1.6)

def wind_factor(u_ms: float) -> float:
    f = 1.0 + WIND_A * u_ms + WIND_B * (u_ms ** 2)
    return clamp(f, 1.0, 6.0)

def slope_factor(slope_percent: float) -> float:
    tan_th = slope_percent / 100.0
    f = 1.0 + SLOPE_K * tan_th
    return clamp(f, 1.0, 5.0)

def base_rate(fuel: str) -> float:
    return BASE_RATE_BY_FUEL.get(fuel, BASE_RATE_BY_FUEL["grass"])

def ros_m_per_min(inp: Inputs) -> float:
    r0 = base_rate(inp.fuel_class)
    f_h = humidity_factor(inp.rel_humidity)
    f_w = wind_factor(inp.wind_speed_ms)
    f_s = slope_factor(inp.slope_percent)
    return max(EPS, r0 * f_h * f_w * f_s)

def length_breadth_ratio(u_ms: float) -> float:
    """
    風速に応じた長軸/短軸比。風が強いほど前後に細長くなる。
    """
    return clamp(1.0 + LB_C * u_ms, 1.0, LB_MAX)

def ellipse_axes(ros: float, t_min: float, init_r: float, u_ms: float) -> Tuple[float, float]:
    A = ros * t_min + init_r   # 半長軸（m）
    lb = length_breadth_ratio(u_ms)
    B = max(EPS, A / lb)       # 半短軸（m）
    return A, B

def ellipse_area_perimeter(a: float, b: float) -> Tuple[float, float]:
    area = math.pi * a * b
    h = ((a - b) ** 2) / ((a + b) ** 2 + EPS)
    perimeter = math.pi * (a + b) * (1 + (3*h)/(10 + math.sqrt(4 - 3*h + EPS)))
    return area, perimeter

def water_requirement_ton(perimeter_m: float, app_rate_lpm_per_m: float, duration_min: float, efficiency: float) -> float:
    liters = app_rate_lpm_per_m * perimeter_m * duration_min
    liters_eff = liters / max(efficiency, 0.05)
    return liters_eff / 1000.0

def run_physical_model(inp: Inputs) -> Outputs:
    ros = ros_m_per_min(inp)
    A, B = ellipse_axes(ros, inp.duration_min, inp.init_radius_m, inp.wind_speed_ms)
    area, perimeter = ellipse_area_perimeter(A, B)
    r_equiv = math.sqrt(area / math.pi)
    water_ton = water_requirement_ton(
        perimeter, inp.app_rate_lpm_per_m, inp.attack_duration_min, inp.efficiency
    )
    return Outputs(
        radius_m=r_equiv,
        area_sqm=area,
        water_volume_tons=water_ton,
        ellipse_a_m=A,
        ellipse_b_m=B,
        perimeter_m=perimeter,
    )

# ------------------------------ 外部API: ジオコーディング & 気象 ------------------------------
def geocode_address_mapbox(address: str) -> Optional[Tuple[float, float]]:
    try:
        token = st.secrets["mapbox"]["access_token"]
    except Exception:
        st.error("Mapbox の access_token が secrets.toml に設定されていません。")
        return None

    try:
        q = urllib.parse.quote(address)
        url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{q}.json"
        params = {"access_token": token, "limit": 1, "language": "ja"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        features = data.get("features", [])
        if not features:
            st.warning("住所から位置を特定できませんでした。")
            return None
        coords = features[0]["center"]
        lon, lat = coords[0], coords[1]
        return lat, lon
    except Exception as e:
        st.error(f"ジオコーディング中にエラーが発生しました: {e}")
        return None

def fetch_openweather(lat: float, lon: float) -> Optional[Dict[str, float]]:
    try:
        api_key = st.secrets["openweather"]["api_key"]
    except Exception:
        st.error("OpenWeather の api_key が secrets.toml に設定されていません。")
        return None

    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": api_key,
            "units": "metric",
            "lang": "ja",
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        main = data.get("main", {})
        wind = data.get("wind", {})
        weather = {
            "temp_c": float(main.get("temp", 0.0)),
            "humidity": float(main.get("humidity", 0.0)),
            "wind_speed": float(wind.get("speed", 0.0)),
            "wind_deg": float(wind.get("deg", 0.0)) if "deg" in wind else None,
            "description": data.get("weather", [{}])[0].get("description", ""),
        }
        return weather
    except Exception as e:
        st.error(f"気象情報取得中にエラーが発生しました: {e}")
        return None

# ------------------------------ Gemini 2.5 Flash 設定 ------------------------------
def get_gemini_model() -> Optional[genai.GenerativeModel]:
    try:
        api_key = st.secrets["general"]["api_key"]
        if not api_key:
            st.warning("general.api_key が設定されていないため、Gemini 解析は無効です。", icon="⚠️")
            return None
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        return model
    except Exception as e:
        st.error(f"Gemini モデル初期化でエラーが発生しました: {e}")
        return None

def build_gemini_prompt(
    inputs: Inputs,
    physical: Outputs,
    role_desc: str,
    origin: Optional[Tuple[float, float]],
    weather: Optional[Dict[str, float]],
) -> str:
    if origin is not None:
        lat, lon = origin
        origin_str = f"緯度 {lat:.5f}, 経度 {lon:.5f}"
    else:
        origin_str = "発生源位置: 未指定"

    if weather is not None:
        wstr = (
            f"気温 {weather['temp_c']:.1f} ℃, "
            f"相対湿度 {weather['humidity']:.0f} %, "
            f"風速 {weather['wind_speed']:.1f} m/s, "
            f"風向(deg) {weather.get('wind_deg', 'N/A')}, "
            f"天気: {weather.get('description', '')}"
        )
    else:
        wstr = "外部気象データ: 未取得（入力された値のみで評価）"

    return f"""
あなたは世界一の消防士であり、災害対応のスペシャリストです。
あなたの視点: {role_desc}

以下の条件で、火災の拡大と必要水量を評価し、
「現在の状況」「延焼の可能性」「消火・対応のポイント」を数値ベースで判断してください。

[発生源位置]
- {origin_str}

[外部気象情報(OpenWeather)]
- {wstr}

[入力条件（ユーザー入力）]
- 燃料種: {inputs.fuel_class}
- 予測時間: {inputs.duration_min:.1f} 分
- 風速(入力値): {inputs.wind_speed_ms:.1f} m/s
- 風向(入力値): {inputs.wind_dir_deg:.0f} 度 (0=北, 90=東)
- 相対湿度(入力値): {inputs.rel_humidity:.0f} %
- 気温(入力値): {inputs.air_temp_c:.1f} ℃
- 斜面勾配: {inputs.slope_percent:.1f} %
- 初期半径: {inputs.init_radius_m:.1f} m
- 散水比率: {inputs.app_rate_lpm_per_m:.2f} L/min/m
- 初期攻勢時間: {inputs.attack_duration_min:.1f} 分
- 散水効率: {inputs.efficiency:.2f}

[物理モデルからの参考値]
- 等価半径 radius_m: {physical.radius_m:.2f} m
- 延焼面積 area_sqm: {physical.area_sqm:.2f} m2
- 必要水量 water_volume_tons: {physical.water_volume_tons:.2f} ton
- 楕円長軸 ellipse_a_m: {physical.ellipse_a_m:.2f} m
- 楕円短軸 ellipse_b_m: {physical.ellipse_b_m:.2f} m
- 周長 perimeter_m: {physical.perimeter_m:.2f} m

[タスク]
- 上記の物理モデル結果をベースラインとし、あなたの専門的判断により、
  安全率や不確実性、燃料・気象条件を考慮して、**最大 ±30% の範囲**で補正した推定値を出してください。
- ロールごとの考え方:
  - 安全マージン重視: radius, area, water_volume をやや大きめに（+10〜+30%）補正しやすくする。
  - 資機材効率重視: water_volume をやや小さめに（-10〜-25%）補正しつつ、安全上必要な最低限を維持。
  - バランス型: 物理モデル付近（±15% 程度）に収まるように調整。

[重要な制約]
1. 出力は **1行の JSON オブジェクトのみ** で返してください。説明文やコメント、コードブロックは一切付けないこと。
2. JSON のキーは **必ず** 次の6つだけにしてください:
   "radius_m", "area_sqm", "water_volume_tons", "ellipse_a_m", "ellipse_b_m", "perimeter_m"
3. 単位:
   - radius_m, ellipse_a_m, ellipse_b_m, perimeter_m は [m]
   - area_sqm は [m2]
   - water_volume_tons は [ton]
4. 各値は物理モデル結果の 0.7〜1.3 倍の範囲に収めてください。

JSON:
""".strip()

def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines()
            if not line.strip().startswith("```")
        ).strip()
    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        return text[start:end]
    return text

def call_gemini_variant(
    model: genai.GenerativeModel,
    inputs: Inputs,
    physical: Outputs,
    role_id: str,
    role_desc: str,
    temperature: float,
    origin: Optional[Tuple[float, float]],
    weather: Optional[Dict[str, float]],
) -> Dict:
    prompt = build_gemini_prompt(inputs, physical, role_desc, origin, weather)
    try:
        response = model.generate_content(
            prompt,
            generation_config={"temperature": temperature, "max_output_tokens": 256},
        )
        text = _extract_json(response.text or "")
        data = json.loads(text)
        for key in [
            "radius_m",
            "area_sqm",
            "water_volume_tons",
            "ellipse_a_m",
            "ellipse_b_m",
            "perimeter_m",
        ]:
            if key not in data:
                raise ValueError(f"missing key {key}")
        return {
            "role_id": role_id,
            "ok": True,
            "raw_text": text,
            "data": data,
        }
    except Exception as e:
        return {
            "role_id": role_id,
            "ok": False,
            "error": str(e),
            "raw_text": "",
            "data": {
                "radius_m": physical.radius_m,
                "area_sqm": physical.area_sqm,
                "water_volume_tons": physical.water_volume_tons,
                "ellipse_a_m": physical.ellipse_a_m,
                "ellipse_b_m": physical.ellipse_b_m,
                "perimeter_m": physical.perimeter_m,
            },
        }

def run_gemini_ensemble(
    inputs: Inputs,
    origin: Optional[Tuple[float, float]],
    weather: Optional[Dict[str, float]],
) -> Tuple[Outputs, Dict]:
    physical = run_physical_model(inputs)
    model = get_gemini_model()
    if model is None:
        meta = {
            "mode": "physical_only",
            "physical": physical.__dict__,
            "ensemble_details": [],
            "origin": origin,
            "weather": weather,
        }
        return physical, meta

    roles = [
        ("balanced", "総合バランス型", 0.4, 0.5),
        ("safety", "安全マージン重視", 0.3, 0.3),
        ("resource", "資機材効率重視", 0.2, 0.2),
    ]

    results: List[Dict] = []
    with ThreadPoolExecutor(max_workers=len(roles)) as ex:
        futures = []
        for role_id, desc, temp, weight in roles:
            futures.append(
                ex.submit(
                    call_gemini_variant,
                    model,
                    inputs,
                    physical,
                    role_id,
                    desc,
                    temp,
                    origin,
                    weather,
                )
            )
        for fut in as_completed(futures):
            results.append(fut.result())

    def aggregate_field(field: str) -> float:
        num = 0.0
        den = 0.0
        for role, (_, _, _, weight) in zip(results, roles):
            value = float(role["data"][field])
            num += weight * value
            den += weight
        if den <= 0:
            return getattr(physical, field)
        return num / den

    agg = Outputs(
        radius_m=aggregate_field("radius_m"),
        area_sqm=aggregate_field("area_sqm"),
        water_volume_tons=aggregate_field("water_volume_tons"),
        ellipse_a_m=aggregate_field("ellipse_a_m"),
        ellipse_b_m=aggregate_field("ellipse_b_m"),
        perimeter_m=aggregate_field("perimeter_m"),
    )

    meta = {
        "mode": "gemini_ensemble",
        "physical": physical.__dict__,
        "ensemble_details": results,
        "origin": origin,
        "weather": weather,
    }
    return agg, meta

# ------------------------------ Geoユーティリティ（延焼楕円→緯度経度） ------------------------------
def meters_to_latlon(lat0: float, lon0: float, dx_m: float, dy_m: float) -> Tuple[float, float]:
    """
    原点(lat0, lon0) からのオフセット dx,dy[m] を緯度経度に変換
    dx: 東向き[m], dy: 北向き[m]
    """
    R = 6378137.0
    dlat = (dy_m / R) * (180.0 / math.pi)
    dlon = (dx_m / (R * math.cos(math.radians(lat0)))) * (180.0 / math.pi)
    return lat0 + dlat, lon0 + dlon

def ellipse_polygon_latlon(
    lat0: float,
    lon0: float,
    a_m: float,
    b_m: float,
    wind_dir_deg: float,
    center_shift_factor: float,
    n_points: int = 120,
) -> List[Tuple[float, float]]:
    """
    物理モデルの楕円 (a,b, 風向) を地理座標のポリゴン(緯度経度列)に変換。
    - X軸: 東, Y軸: 北
    - 風向: 0°=北, 90°=東
    - center_shift_factor に応じて、楕円の“中心”を風下側へシフトさせる。
      → 発生源（lat0,lon0）は楕円のやや後端に来るので、現実の火頭が風下側に伸びた形に近づく。
    """
    # 基本楕円（原点中心）
    t = np.linspace(0, 2 * np.pi, n_points)
    x = a_m * np.cos(t)
    y = b_m * np.sin(t)

    # 風向に応じて回転（0°=北, 90°=東）
    theta = math.radians(90.0 - wind_dir_deg)
    rot = np.array([[math.cos(theta), -math.sin(theta)],
                    [math.sin(theta),  math.cos(theta)]])
    xy = rot @ np.vstack([x, y])

    # 風下方向（長軸正方向）に楕円中心をシフト
    shift_dist = center_shift_factor * a_m
    shift_x = shift_dist * math.cos(theta)
    shift_y = shift_dist * math.sin(theta)

    poly = []
    for dx, dy in zip(xy[0], xy[1]):
        lat, lon = meters_to_latlon(lat0, lon0, dx + shift_x, dy + shift_y)
        poly.append((lat, lon))
    return poly

# ------------------------------ UI ユーティリティ ------------------------------
def metric_block(col, label: str, value: float, unit: str, precision: int = 2):
    col.metric(label, f"{value:,.{precision}f} {unit}")

def to_json(outputs: Outputs) -> str:
    payload = {
        "radius_m": round(outputs.radius_m, 2),
        "area_sqm": round(outputs.area_sqm, 2),
        "water_volume_tons": round(outputs.water_volume_tons, 2),
        "ellipse_a_m": round(outputs.ellipse_a_m, 2),
        "ellipse_b_m": round(outputs.ellipse_b_m, 2),
        "perimeter_m": round(outputs.perimeter_m, 2),
    }
    return json.dumps(payload, ensure_ascii=False)

def pct_diff(new: float, base: float) -> float:
    if abs(base) < EPS:
        return 0.0
    return (new / base - 1.0) * 100.0

# ------------------------------ セッション初期化 ------------------------------
if "origin_lat" not in st.session_state:
    st.session_state["origin_lat"] = 35.681236  # 東京駅付近
if "origin_lon" not in st.session_state:
    st.session_state["origin_lon"] = 139.767125
if "weather_info" not in st.session_state:
    st.session_state["weather_info"] = None
if "anim_t_sel" not in st.session_state:
    st.session_state["anim_t_sel"] = 0.0  # スライダーの現在値を保持

# ------------------------------ メインUI ------------------------------
st.title("Fire Spread Simulator Pro")
st.caption("Save Your Self / 火災拡大シミュレーション（Gemini 2.5 Flash Ensemble）")

with st.sidebar:
    st.header("基本条件")

    fuel_class = st.selectbox(
        "燃料種",
        options=["grass", "shrub", "timber"],
        index=0,
        help="草地/低木/立木。燃料が重いほど基礎延焼速度は遅めになります。",
    )

    # 予測時間：分・時間・日を選択可能（内部は分に換算）
    duration_unit = st.selectbox(
        "予測時間の単位",
        options=["分", "時間", "日"],
        index=1,  # デフォルト: 時間
    )

    c1, c2 = st.columns(2)
    with c1:
        if duration_unit == "分":
            raw_duration = st.number_input(
                "予測時間[分]（最大 7日 = 10080分）",
                5,
                10080,
                60,
                step=5,
            )
            duration_min = float(raw_duration)
        elif duration_unit == "時間":
            raw_duration = st.number_input(
                "予測時間[時間]",
                1,
                168,
                24,
                step=1,
            )
            duration_min = float(raw_duration * 60)
        else:  # 日
            raw_duration = st.number_input(
                "予測時間[日]",
                1,
                7,
                1,
                step=1,
            )
            duration_min = float(raw_duration * 1440)

        wind_speed_ms = st.slider("風速[m/s]", 0.0, 20.0, 5.0, 0.5)
        slope_percent = st.slider(
            "斜面勾配[%]",
            0.0,
            100.0,
            10.0,
            1.0,
            help="上り勾配で延焼は加速します。",
        )
        init_radius_m = st.number_input("初期半径[m]", 0.0, 200.0, 5.0, step=1.0)
    with c2:
        wind_dir_deg = st.slider("風向[°] (0=北/90=東)", 0, 359, 90, 1)
        rel_humidity = st.slider("相対湿度[%]", 5, 100, 40, 1)
        air_temp_c = st.slider("気温[°C]", -10, 50, 25, 1)

    st.caption(
        f"※内部計算では {duration_min:.0f} 分（約 {duration_min/60:.1f} 時間）として扱います。"
    )

    st.markdown("---")
    st.header("消火設定")
    c3, c4, c5 = st.columns(3)
    with c3:
        default_app_rate = {"grass": 4.0, "shrub": 8.0, "timber": 12.0}[fuel_class]
        app_rate_lpm_per_m = st.number_input(
            "散水比率[L/min/m]",
            0.1,
            50.0,
            float(default_app_rate),
            step=0.1,
            help="単位延長1mあたり1分の散水量。",
        )
    with c4:
        attack_duration_min = st.number_input(
            "初期攻勢[min]",
            1.0,
            180.0,
            15.0,
            step=1.0,
            help="初動で連続散水する推定時間。",
        )
    with c5:
        efficiency = st.slider(
            "散水効率",
            0.10,
            1.00,
            0.60,
            0.05,
            help="散水の実効率。低いほど必要量は増えます。",
        )

    inputs = Inputs(
        duration_min=duration_min,
        wind_speed_ms=wind_speed_ms,
        wind_dir_deg=float(wind_dir_deg),
        rel_humidity=float(rel_humidity),
        air_temp_c=float(air_temp_c),
        slope_percent=float(slope_percent),
        fuel_class=fuel_class,
        init_radius_m=float(init_radius_m),
        attack_duration_min=float(attack_duration_min),
        app_rate_lpm_per_m=float(app_rate_lpm_per_m),
        efficiency=float(efficiency),
    )

# ------------------------------ 発生源 & 気象 ------------------------------
st.subheader("1. 発生源の指定と気象データ")

left_loc, right_loc = st.columns([1.3, 1])

with left_loc:
    method_options = ["住所から検索", "緯度経度を直接入力"]
    if HAS_FOLIUM:
        method_options.insert(0, "地図上で指定")

    method = st.radio(
        "発生源の指定方法",
        method_options,
        index=0,
        horizontal=True,
    )

    cur_lat = st.session_state["origin_lat"]
    cur_lon = st.session_state["origin_lon"]

    if method == "緯度経度を直接入力":
        lat = st.number_input("緯度", -90.0, 90.0, float(cur_lat), step=0.0001, format="%.5f")
        lon = st.number_input("経度", -180.0, 180.0, float(cur_lon), step=0.0001, format="%.5f")
        st.session_state["origin_lat"] = lat
        st.session_state["origin_lon"] = lon

    elif method == "住所から検索":
        addr = st.text_input("住所（例：愛媛県松山市...）", "")
        if st.button("住所から発生源を検索"):
            if addr.strip():
                result = geocode_address_mapbox(addr.strip())
                if result is not None:
                    lat, lon = result
                    st.session_state["origin_lat"] = lat
                    st.session_state["origin_lon"] = lon
                    st.success(f"発生源を設定しました：緯度 {lat:.5f}, 経度 {lon:.5f}")
            else:
                st.warning("住所を入力してください。")

    else:  # 地図上で指定
        st.caption("地図をクリックすると、その地点を発生源として設定できます。")
        m = folium.Map(
            location=[cur_lat, cur_lon],
            zoom_start=16,  # 近めのスケール
            tiles="OpenStreetMap",
        )
        folium.Marker(
            location=[cur_lat, cur_lon],
            popup="現在の発生源",
            icon=folium.Icon(color="red", icon="fire"),
        ).add_to(m)
        m.add_child(folium.LatLngPopup())
        out = st_folium(m, width=650, height=380, returned_objects=[])
        if out and out.get("last_clicked") is not None:
            lat = out["last_clicked"]["lat"]
            lon = out["last_clicked"]["lng"]
            st.session_state["origin_lat"] = lat
            st.session_state["origin_lon"] = lon
            st.info(f"クリックした地点を発生源に設定: 緯度 {lat:.5f}, 経度 {lon:.5f}")

    if not HAS_FOLIUM:
        st.warning(
            "地図上で指定するには `streamlit-folium` が必要です。\n"
            "requirements.txt に `streamlit-folium` を追加してください。",
            icon="ℹ️",
        )

with right_loc:
    st.markdown("**現在の発生源**")
    st.write(
        f"緯度: `{st.session_state['origin_lat']:.5f}`, "
        f"経度: `{st.session_state['origin_lon']:.5f}`"
    )

    if st.button("この位置の気象情報を取得（OpenWeather）"):
        w = fetch_openweather(st.session_state["origin_lat"], st.session_state["origin_lon"])
        if w is not None:
            st.session_state["weather_info"] = w
            st.success("気象情報を取得しました。Gemini 解析に反映されます。")
    weather_info = st.session_state["weather_info"]

    if weather_info is not None:
        st.markdown("**取得した気象情報（参考）**")
        st.write(
            f"- 気温: {weather_info['temp_c']:.1f} ℃\n"
            f"- 相対湿度: {weather_info['humidity']:.0f} %\n"
            f"- 風速: {weather_info['wind_speed']:.1f} m/s\n"
            f"- 風向(deg): {weather_info.get('wind_deg', 'N/A')}\n"
            f"- 天気: {weather_info.get('description', '')}"
        )
        st.caption("※必要に応じてサイドバーの風速・湿度・気温を手動で合わせてください。")

origin_tuple: Optional[Tuple[float, float]] = (
    st.session_state["origin_lat"],
    st.session_state["origin_lon"],
)
weather_ctx: Optional[Dict[str, float]] = st.session_state["weather_info"]

st.markdown("---")

# ------------------------------ 2. 解析実行と結果 ------------------------------
st.subheader("2. 解析結果（Gemini アンサンブル + 物理モデル）")

outputs, ensemble_meta = run_gemini_ensemble(inputs, origin_tuple, weather_ctx)

m1, m2, m3, m4 = st.columns(4)
metric_block(m1, "等価半径", outputs.radius_m, "m")
metric_block(m2, "延焼面積", outputs.area_sqm, "m²")
metric_block(m3, "必要水量(推定)", outputs.water_volume_tons, "ton")
metric_block(m4, "周長(楕円)", outputs.perimeter_m, "m")

if ensemble_meta["mode"] == "gemini_ensemble":
    st.success("Gemini 2.5 Flash による並列アンサンブル解析結果を表示しています。", icon="✅")
else:
    st.warning("Gemini が無効なため、物理モデルのみで計算しています。", icon="⚠️")

st.caption(
    "※本モデルは現場判断の補助を目的とした簡易推定です。"
    " 実際の地形・燃料・気象・活動状況によって結果は大きく変わります。"
)

# ------------------------------ タブ: グラフ / アニメ / データ / 感度 / 解説 / 詳細 ------------------------------
tab_main, tab_anim, tab_data, tab_sens, tab_explain, tab_detail = st.tabs(
    ["📊 グラフ", "🌏 延焼アニメーション", "📁 データ出力", "🧪 感度分析", "🧠 Gemini解析の解説", "🔍 詳細・ヘルプ"]
)

physical_for_plots = run_physical_model(inputs)

# ---- メイングラフ ----
with tab_main:
    st.markdown("#### 延焼形状（物理モデル）")

    fig1, ax1 = plt.subplots(figsize=(5.5, 5.5))
    a = physical_for_plots.ellipse_a_m
    b = physical_for_plots.ellipse_b_m
    t = np.linspace(0, 2 * np.pi, 400)
    x = a * np.cos(t)
    y = b * np.sin(t)
    theta = math.radians(90 - inputs.wind_dir_deg)
    rot = np.array([[math.cos(theta), -math.sin(theta)],
                    [math.sin(theta),  math.cos(theta)]])
    xy = rot @ np.vstack([x, y])
    ax1.plot(xy[0], xy[1], linewidth=2)
    ax1.scatter([0], [0], marker="*", s=120)
    ax1.set_aspect("equal", "box")
    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.set_title("延焼楕円（上から見た図）")
    st.pyplot(fig1)

    st.markdown("#### 時間とともに変化する半径・水量（物理モデル）")

    fig2, ax2 = plt.subplots(figsize=(6.5, 4))
    times = np.linspace(max(1.0, inputs.duration_min / 20), inputs.duration_min, 40)
    radii = []
    waters = []
    for tt in times:
        o = run_physical_model(
            Inputs(
                duration_min=float(tt),
                wind_speed_ms=inputs.wind_speed_ms,
                wind_dir_deg=inputs.wind_dir_deg,
                rel_humidity=inputs.rel_humidity,
                air_temp_c=inputs.air_temp_c,
                slope_percent=inputs.slope_percent,
                fuel_class=inputs.fuel_class,
                init_radius_m=inputs.init_radius_m,
                attack_duration_min=inputs.attack_duration_min,
                app_rate_lpm_per_m=inputs.app_rate_lpm_per_m,
                efficiency=inputs.efficiency,
            )
        )
        radii.append(o.radius_m)
        waters.append(o.water_volume_tons)
    ax2.plot(times, radii, linewidth=2)
    ax2.set_xlabel("時間[min]")
    ax2.set_ylabel("半径[m]")
    ax2.set_title("時間と半径の関係")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(6.5, 4))
    ax3.plot(times, waters, linewidth=2)
    ax3.set_xlabel("時間[min]")
    ax3.set_ylabel("必要水量[ton]")
    ax3.set_title("時間と必要水量の関係")
    st.pyplot(fig3)

# ---- 延焼アニメーション（地図・スライダーのみ） ----
with tab_anim:
    st.markdown("#### 地図上で見る延焼の広がり（時間スライダー）")

    if not HAS_FOLIUM:
        st.warning(
            "延焼アニメーションを表示するには `streamlit-folium` が必要です。\n"
            "requirements.txt に `streamlit-folium` を追加してください。",
            icon="ℹ️",
        )
    else:
        lat0, lon0 = origin_tuple

        # 最大時間（分）とステップ
        max_t = max(5.0, float(inputs.duration_min))
        n_steps = 30  # 表示の滑らかさ
        step_t = max(1.0, max_t / n_steps)

        # 現在時間（スライダー値）をセッションから取得
        current_t = float(st.session_state.get("anim_t_sel", 0.0))
        current_t = clamp(current_t, 0.0, max_t)

        # スライダー（手動操作）: key="anim_t_sel"
        t_sel = st.slider(
            "経過時間[min]",
            0.0,
            max_t,
            value=current_t,
            step=step_t,
            key="anim_t_sel",
        )

        # 選択時間での延焼範囲を計算
        if t_sel <= 0.0:
            tmp_inputs = Inputs(
                duration_min=0.0,
                wind_speed_ms=inputs.wind_speed_ms,
                wind_dir_deg=inputs.wind_dir_deg,
                rel_humidity=inputs.rel_humidity,
                air_temp_c=inputs.air_temp_c,
                slope_percent=inputs.slope_percent,
                fuel_class=inputs.fuel_class,
                init_radius_m=inputs.init_radius_m,
                attack_duration_min=inputs.attack_duration_min,
                app_rate_lpm_per_m=inputs.app_rate_lpm_per_m,
                efficiency=inputs.efficiency,
            )
        else:
            tmp_inputs = Inputs(
                duration_min=float(t_sel),
                wind_speed_ms=inputs.wind_speed_ms,
                wind_dir_deg=inputs.wind_dir_deg,
                rel_humidity=inputs.rel_humidity,
                air_temp_c=inputs.air_temp_c,
                slope_percent=inputs.slope_percent,
                fuel_class=inputs.fuel_class,
                init_radius_m=inputs.init_radius_m,
                attack_duration_min=inputs.attack_duration_min,
                app_rate_lpm_per_m=inputs.app_rate_lpm_per_m,
                efficiency=inputs.efficiency,
            )

        o_t = run_physical_model(tmp_inputs)

        st.caption(
            f"経過時間: {t_sel:.1f} 分 "
            f"(約 {t_sel/60:.1f} 時間 / 約 {t_sel/1440:.2f} 日) / "
            f"等価半径: {o_t.radius_m:.1f} m / "
            f"延焼面積: {o_t.area_sqm:.0f} m²"
        )

        # 風速に応じた長軸比を使って、楕円の中心シフト量を決定
        lb = length_breadth_ratio(inputs.wind_speed_ms)
        center_shift_factor = 0.5 * (1.0 - 1.0 / lb)  # 0〜0.4程度

        poly_latlon = ellipse_polygon_latlon(
            lat0,
            lon0,
            o_t.ellipse_a_m,
            o_t.ellipse_b_m,
            inputs.wind_dir_deg,
            center_shift_factor=center_shift_factor,
            n_points=180,
        )

        m_anim = folium.Map(
            location=[lat0, lon0],
            zoom_start=16,  # 近めのスケール
            tiles="OpenStreetMap",
        )
        folium.Marker(
            location=[lat0, lon0],
            popup="発生源",
            icon=folium.Icon(color="red", icon="fire"),
        ).add_to(m_anim)
        folium.Polygon(
            locations=poly_latlon,
            color="orange",
            fill=True,
            fill_opacity=0.35,
            popup=f"{t_sel:.1f} 分後の推定延焼範囲",
        ).add_to(m_anim)

        st_folium(m_anim, width=800, height=480, returned_objects=[])

# ---- データ出力 ----
with tab_data:
    st.markdown("#### JSON 出力（Gemini アンサンブル結果）")
    json_str = to_json(outputs)
    st.code(json_str, language="json")
    st.download_button(
        "JSON をダウンロード",
        data=json_str.encode("utf-8"),
        file_name="fire_spread_output.json",
        mime="application/json",
    )

    st.markdown("#### CSV 出力（主要指標）")
    csv_lines = [
        "metric,value,unit",
        f"radius_m,{outputs.radius_m:.2f},m",
        f"area_sqm,{outputs.area_sqm:.2f},m2",
        f"water_volume_tons,{outputs.water_volume_tons:.2f},ton",
        f"ellipse_a_m,{outputs.ellipse_a_m:.2f},m",
        f"ellipse_b_m,{outputs.ellipse_b_m:.2f},m",
        f"perimeter_m,{outputs.perimeter_m:.2f},m",
    ]
    csv_data = "\n".join(csv_lines)
    st.download_button(
        "CSV をダウンロード",
        data=csv_data.encode("utf-8"),
        file_name="fire_spread_output.csv",
        mime="text/csv",
    )

# ---- 感度分析（物理モデル） ----
with tab_sens:
    st.markdown("#### 感度分析（物理モデルのみ）")
    st.caption("風速・湿度・勾配・燃料種を変えたときの半径と必要水量の変化をざっくり比較できます。")

    axis = st.selectbox("変更パラメータ", ["風速", "湿度", "斜面勾配", "燃料種"], index=0)

    scenarios: List[Tuple[str, Inputs]] = []

    if axis == "風速":
        winds = [max(0.0, inputs.wind_speed_ms + d) for d in (-3, 0, +3, +6)]
        for w in winds:
            label = f"風速 {w:.1f} m/s"
            scenarios.append((label, Inputs(**{**inputs.__dict__, "wind_speed_ms": w})))
    elif axis == "湿度":
        rhs = [clamp(inputs.rel_humidity + d, 5, 100) for d in (-20, 0, +20, +40)]
        for r in rhs:
            label = f"湿度 {r:.0f}%"
            scenarios.append((label, Inputs(**{**inputs.__dict__, "rel_humidity": r})))
    elif axis == "斜面勾配":
        slopes = [clamp(inputs.slope_percent + d, 0, 100) for d in (-10, 0, +10, +20)]
        for s in slopes:
            label = f"勾配 {s:.0f}%"
            scenarios.append((label, Inputs(**{**inputs.__dict__, "slope_percent": s})))
    else:
        fuels = ["grass", "shrub", "timber"]
        for f in fuels:
            label = f"燃料 {f}"
            scenarios.append((label, Inputs(**{**inputs.__dict__, "fuel_class": f})))

    figS, axS = plt.subplots(figsize=(6.5, 4))
    for label, sc_inp in scenarios:
        o = run_physical_model(sc_inp)
        axS.scatter(o.radius_m, o.water_volume_tons, s=60)
        axS.annotate(
            label,
            (o.radius_m, o.water_volume_tons),
            xytext=(5, 5),
            textcoords="offset points",
        )
    axS.set_xlabel("等価半径[m]")
    axS.set_ylabel("必要水量[ton]")
    axS.set_title("パラメータ変更時の半径と必要水量")
    st.pyplot(figS)

# ---- Gemini解析の解説（消防士トーンでの現状評価・延焼・消火アドバイス） ----
with tab_explain:
    st.markdown("#### Gemini による状況評価と現場アドバイス")

    if ensemble_meta["mode"] != "gemini_ensemble":
        st.warning("Gemini が無効なため、物理モデルのみで計算しています。Gemini の解説は表示できません。")
    else:
        phys = ensemble_meta["physical"]
        details = ensemble_meta["ensemble_details"]

        d_r = pct_diff(outputs.radius_m, phys["radius_m"])
        d_a = pct_diff(outputs.area_sqm, phys["area_sqm"])
        d_w = pct_diff(outputs.water_volume_tons, phys["water_volume_tons"])

        def sign_fmt(x: float) -> str:
            return f"{x:+.1f}%"

        # --- 1. 現在の状況（数値を消防士目線で要約） ---
        st.markdown("##### 1. 現在の状況（延焼規模のイメージ）")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**物理モデル（純粋な計算値）**")
            st.write(
                f"- 等価半径（計算上の広がり）: {phys['radius_m']:.1f} m\n"
                f"- 延焼面積: {phys['area_sqm']:.0f} m²\n"
                f"- 必要水量（計算上）: {phys['water_volume_tons']:.1f} ton"
            )
        with col_b:
            st.markdown("**Gemini アンサンブル（現場を意識した見立て）**")
            st.write(
                f"- 等価半径（警戒すべき広がり）: {outputs.radius_m:.1f} m\n"
                f"- 延焼面積（想定被害範囲）: {outputs.area_sqm:.0f} m²\n"
                f"- 必要水量（準備したい総量）: {outputs.water_volume_tons:.1f} ton"
            )

        st.markdown("**計算値からの補正の方向**")
        st.write(
            f"- 半径: 物理モデル比 {sign_fmt(d_r)}\n"
            f"- 面積: 物理モデル比 {sign_fmt(d_a)}\n"
            f"- 水量: 物理モデル比 {sign_fmt(d_w)}"
        )
        st.caption(
            "※プラス側なら『余裕を持って広め・多めに見ている』、マイナス側なら『資機材制約を意識して絞っている』というイメージです。"
        )

        # --- 2. 延焼の可能性（どこまで広がりうるかの目安） ---
        st.markdown("##### 2. 延焼の可能性（どこまで広がりうるか）")

        # 単純な目安レベル分け
        level_text = ""
        if outputs.radius_m < 100:
            level_text = "建物火災〜小規模な林野火災レベルで、エリアとしては比較的コンパクトです。"
        elif outputs.radius_m < 500:
            level_text = "面的な延焼が見込まれる中規模火災で、周辺の建物・林地への波及を強く意識するレベルです。"
        else:
            level_text = "大規模な延焼ポテンシャルを持つ火災です。面としての延焼に加え、飛び火・スポット火災も強く警戒が必要な規模です。"

        st.write(
            f"- 想定される延焼半径: 約 **{outputs.radius_m:.0f} m**\n"
            f"- 想定される延焼面積: 約 **{outputs.area_sqm/10_000:.1f} ha**（ヘクタール換算）\n\n"
            f"{level_text}"
        )

        st.caption(
            "※実際には地形・道路・防火帯・建物配置によって延焼パターンは大きく変わります。"
            "ここでは『素の燃え広がり方』の目安として捉えてください。"
        )

        # --- 3. 消火・対応のポイント（数値から読み取れる戦略） ---
        st.markdown("##### 3. 消火・対応のポイント（水量・時間からの作戦イメージ）")

        # 仮のポンプ能力（1台 2.4 ton/min）
        if inputs.attack_duration_min > 0:
            ton_per_min_per_pump = 2.4
            total_min = inputs.attack_duration_min
            if total_min <= 0:
                total_min = 1.0
            est_pumps = outputs.water_volume_tons / (ton_per_min_per_pump * total_min)
        else:
            est_pumps = 0.0

        st.write(
            f"- 必要水量の目安: **約 {outputs.water_volume_tons:.1f} ton**\n"
            f"- 初期攻勢時間: 約 **{inputs.attack_duration_min:.0f} 分** を想定\n"
            f"- 仮に1台あたり毎分約 2.4 ton 出せるポンプとすると、\n"
            f"  → 必要ポンプ台数のざっくり目安: **{est_pumps:.1f} 台分の能力**"
        )

        st.markdown("**現場対応の考え方（定性的なアドバイス）**")
        st.markdown(
            """
- **現在の状況**
  - このモデルでは、燃料・風・斜面・湿度から「どの程度の広さまで燃え広がりうるか」を数値化しています。
  - 等価半径と面積は「警戒すべき範囲」の目安、水量は「最低限このくらいは確保しておきたい」というラインです。

- **延焼の可能性**
  - 風下側の楕円が長くなるほど、火頭の移動スピードと到達範囲が大きくなります。
  - 風速が高い・斜面が上り・燃料が重い（森林など）場合は、同じ半径でも火勢が強く、人員の安全距離を広めに取る必要があります。

- **消火・対応のポイント**
  - 数値上の必要水量が大きい場合は、「全面制圧」ではなく、**延焼先行区のカットライン確保** や **重要施設の防御** を優先する判断が重要です。
  - 逆に、必要水量が小さめに出ている場合でも、風向・風速が変わりやすい条件では、**退路の確保** と **水利の二重化** を意識しておくと安全です。
  - いずれの場合も、この結果は「机上の最低ライン」であり、実際の現場では **安全側に上乗せした水量確保と人員配置** を強く推奨します。
            """
        )

        # --- 4. 各ロールの違い（簡潔に） ---
        st.markdown("##### 4. 各ロールごとの見立ての違い")

        for role in details:
            role_id = role["role_id"]
            data = role["data"]
            r = pct_diff(data["radius_m"], phys["radius_m"])
            a_ = pct_diff(data["area_sqm"], phys["area_sqm"])
            w_ = pct_diff(data["water_volume_tons"], phys["water_volume_tons"])

            if role_id == "balanced":
                title = "バランス型（balanced）"
                desc = "安全と資機材効率のバランスを取り、現実的な数字に落とし込んだ見立てです。"
            elif role_id == "safety":
                title = "安全マージン重視（safety）"
                desc = "『万が一』に備えて広め・多めに見積もった、慎重側の見立てです。"
            else:
                title = "資機材効率重視（resource）"
                desc = "水利や車両数に制約がある状況を想定し、効率を重視した見立てです。"

            with st.expander(f"{title} の結果とニュアンス", expanded=(role_id == "balanced")):
                st.markdown(desc)
                st.write(
                    f"- 半径: {data['radius_m']:.1f} m（物理比 {sign_fmt(r)}）\n"
                    f"- 面積: {data['area_sqm']:.0f} m²（物理比 {sign_fmt(a_)}）\n"
                    f"- 水量: {data['water_volume_tons']:.1f} ton（物理比 {sign_fmt(w_)}）"
                )
                st.caption(
                    "現場では、safety を『最悪を見たシナリオ』、resource を『資源制約が厳しいときのシナリオ』として、"
                    "balanced や最終アンサンブル値と合わせて見ると方針を立てやすくなります。"
                )

# ---- 詳細情報・ヘルプ ----
with tab_detail:
    st.markdown("#### モデルの考え方（概要）")
    st.markdown(
        """
- **物理モデル**
  - 延焼速度(ROS) = 基準ROS(燃料別) × 湿度係数 × 風係数 × 斜面係数
  - 風下方向に長い楕円として延焼範囲を近似
  - 等価半径 = 楕円面積と同じ円の半径
  - 必要水量 = 周長×散水比率×散水時間 / 散水効率

- **Gemini 2.5 Flash アンサンブル**
  - 物理モデル結果をベースラインとして提示
  - 「安全マージン重視」「資機材効率重視」「バランス型」の3ロールで並列推定
  - 各ロールは ±30% の範囲で補正された数値を JSON で返し、重み付き平均で最終値を決定
  - 発生源位置と OpenWeather の気象情報を解析コンテキストに含めます

- **延焼アニメーション（時間スライダー）**
  - スライダーを動かすことで、任意の時間における延焼範囲を即座に確認できます。
  - 風速に応じた長軸/短軸比から「中心の風下方向へのシフト量」を計算し、
    発生源がやや後端寄り・火頭が風下に伸びる形状を表現しています。
        """
    )

    st.markdown("#### Gemini アンサンブルの内部データ（必要な場合のみ）")
    with st.expander("詳細を見る（上級者向け）"):
        st.json(ensemble_meta)

# ---- 機械連携用 JSON（コピー用）----
with st.expander("機械連携用 JSON (Gemini アンサンブル結果)"):
    st.code(to_json(outputs), language="json")
