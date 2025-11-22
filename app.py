# -*- coding: utf-8 -*-
"""
Fire Spread Simulator Pro (Streamlit + Gemini 2.5 Flash Ensemble)
----------------------------------------------------------------
- 物理モデル + Gemini 2.5 Flash を組み合わせたハイブリッド火災拡大シミュレーション
- Gemini を複数視点で並列実行し、重み付きアンサンブルで総合判断
- UI は世界標準的なダッシュボード構成（メトリクス / グラフ / エクスポート / 感度分析）

■ 必要ライブラリ
- streamlit
- numpy
- matplotlib
- google-generativeai  (pip install google-generativeai)

■ 起動
streamlit run app.py

■ .streamlit/secrets.toml に以下のような構造で API を定義しておくこと：
[general]
api_key = "（ここにGoogle API Key）"
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
import google.generativeai as genai

# ------------------------- ページ設定 / グローバル -------------------------
st.set_page_config(
    page_title="Fire Spread Simulator Pro",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- 最小限のCSSで可読性向上（ダーク/ライト両対応） ----
CUSTOM_CSS = """
/* タイトルの余白最適化 */
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
/* メトリクスの文字強調 */
div[data-testid="stMetric"] > div {white-space: nowrap;}
/* サブヘッダ視認性 */
h3, h4 { margin-top: 0.6rem; }
/* 小さなヘルプテキスト */
.small { font-size: 0.92rem; opacity: 0.8; }
/* ダウンロードボタンの幅 */
button[kind="secondary"] { min_width: 200px; }
"""
st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

# ------------------------------ ドメインモデル ------------------------------
@dataclass
class Inputs:
    duration_min: float      # 予測時間 [min]
    wind_speed_ms: float     # 風速 [m/s]
    wind_dir_deg: float      # 風向 [deg, 0=北, 90=東]
    rel_humidity: float      # 相対湿度 [%]
    air_temp_c: float        # 気温 [°C]
    slope_percent: float     # 斜面勾配 [%]
    fuel_class: str          # 燃料種: grass/shrub/timber
    init_radius_m: float     # 初期半径 [m]
    attack_duration_min: float  # 初期攻勢継続 [min]
    app_rate_lpm_per_m: float   # 散水比率 [L/min/m]
    efficiency: float           # 散水効率 [0-1]

@dataclass
class Outputs:
    radius_m: float
    area_sqm: float
    water_volume_tons: float
    ellipse_a_m: float      # 風下方向の半径(長軸)
    ellipse_b_m: float      # 横方向の半径(短軸)
    perimeter_m: float

# ------------------------------ 物理モデル用パラメータ ------------------------------
BASE_RATE_BY_FUEL = {
    # 基準: 無風・無斜面・RH=30% でのベース延焼速度 [m/min]
    "grass": 8.0,    # 草地は速い
    "shrub": 3.0,    # 低木
    "timber": 0.6,   # 立木/森林は遅い
}

# 湿度係数: RHが高いほど抑制。RH=30%で1.0、上昇で減衰、低下で増加
HUMIDITY_K = 1.1

# 風係数: U[m/s] に対して (1 + aU + bU^2)
WIND_A = 0.10
WIND_B = 0.010

# 斜面係数: 1 + k * tan(theta), theta ~ atan(slope)
SLOPE_K = 4.0

# 風による長径/短径比(L/B)の近似: 1 + c*U (上限あり)
LB_C = 0.30
LB_MAX = 5.0

EPS = 1e-9

# ------------------------------ 汎用関数 ------------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# ------------------------------ 物理モデル ------------------------------
def humidity_factor(rh: float) -> float:
    # RH=30% →1.0、RH↑で指数減衰、RH↓で増加。極端値はクリップ
    f = math.exp(-HUMIDITY_K * max(0.0, rh - 30.0) / 70.0)
    if rh < 30.0:
        f = 1.0 + 0.02 * (30.0 - rh)  # 乾燥側の増幅(上限1.6)
    return clamp(f, 0.25, 1.6)

def wind_factor(u_ms: float) -> float:
    f = 1.0 + WIND_A * u_ms + WIND_B * (u_ms ** 2)
    return clamp(f, 1.0, 6.0)

def slope_factor(slope_percent: float) -> float:
    tan_th = (slope_percent / 100.0)
    f = 1.0 + SLOPE_K * tan_th
    return clamp(f, 1.0, 5.0)

def base_rate(fuel: str) -> float:
    return BASE_RATE_BY_FUEL.get(fuel, BASE_RATE_BY_FUEL["grass"])  # m/min

def ros_m_per_min(inp: Inputs) -> float:
    r0 = base_rate(inp.fuel_class)
    f_h = humidity_factor(inp.rel_humidity)
    f_w = wind_factor(inp.wind_speed_ms)
    f_s = slope_factor(inp.slope_percent)
    return max(EPS, r0 * f_h * f_w * f_s)

def length_breadth_ratio(u_ms: float) -> float:
    return clamp(1.0 + LB_C * u_ms, 1.0, LB_MAX)

def ellipse_axes(ros: float, t_min: float, init_r: float, u_ms: float) -> Tuple[float, float]:
    """風下方向(長軸A)と横方向(短軸B)の半径[m]を返す。初期半径を加算。"""
    A = ros * t_min + init_r
    lb = length_breadth_ratio(u_ms)
    B = max(EPS, A / lb)
    return A, B

def ellipse_area_perimeter(a: float, b: float) -> Tuple[float, float]:
    area = math.pi * a * b
    # Ramanujan 近似で周長
    h = ((a - b) ** 2) / ((a + b) ** 2 + EPS)
    perimeter = math.pi * (a + b) * (1 + (3*h)/(10 + math.sqrt(4 - 3*h + EPS)))
    return area, perimeter

def water_requirement_ton(perimeter_m: float, app_rate_lpm_per_m: float, duration_min: float, efficiency: float) -> float:
    liters = app_rate_lpm_per_m * perimeter_m * duration_min
    liters_eff = liters / max(efficiency, 0.05)
    return liters_eff / 1000.0  # ton

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

# ------------------------------ Gemini 2.5 Flash 設定 ------------------------------
def get_gemini_model() -> Optional[genai.GenerativeModel]:
    """
    secrets.toml の [general].api_key を利用して Gemini を初期化する。
    [general]
    api_key = "YOUR_GOOGLE_API_KEY"
    """
    try:
        # ユーザーの secrets.toml 構造に合わせる
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

def build_gemini_prompt(inputs: Inputs, physical: Outputs, role_desc: str) -> str:
    """
    各ロール（安全重視・資機材重視・バランス）のためのプロンプト。
    物理モデル結果をベースに ±30% の補正範囲で出力させる。
    """
    return f"""
あなたは火災拡大シミュレーションの専門家です。
あなたの視点: {role_desc}

以下の条件で、火災の拡大と必要水量を評価してください。

[入力条件]
- 燃料種: {inputs.fuel_class}
- 予測時間: {inputs.duration_min:.1f} 分
- 風速: {inputs.wind_speed_ms:.1f} m/s
- 風向: {inputs.wind_dir_deg:.0f} 度 (0=北, 90=東)
- 相対湿度: {inputs.rel_humidity:.0f} %
- 気温: {inputs.air_temp_c:.1f} ℃
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
- あなたのロールに応じて、以下の傾向を持たせてください:
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
    """Gemini からの応答から JSON 部分だけを抽出するユーティリティ。"""
    text = text.strip()
    # ```json ... ``` 対応
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines()
            if not line.strip().startswith("```")
        ).strip()
    # 先頭の { から最後の } まで
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
) -> Dict:
    """各ロールの Gemini 呼び出し。失敗時は物理モデルをそのまま返す。"""
    prompt = build_gemini_prompt(inputs, physical, role_desc)
    try:
        response = model.generate_content(
            prompt,
            generation_config={"temperature": temperature, "max_output_tokens": 256},
        )
        text = _extract_json(response.text or "")
        data = json.loads(text)
        # 必須キーが揃っているか軽くチェック
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
        # フォールバック: 物理モデル値を返す
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

def run_gemini_ensemble(inputs: Inputs) -> Tuple[Outputs, Dict]:
    """
    物理モデル + Gemini アンサンブルによる総合出力。
    - 物理モデル: ベースライン
    - Gemini: 安全重視 / 資機材効率重視 / バランス型 の3ロール
    - 並列実行 + 重み付き平均で最終値を決定
    """
    physical = run_physical_model(inputs)
    model = get_gemini_model()
    if model is None:
        # Gemini 利用不可の場合は物理モデルのみ
        meta = {
            "mode": "physical_only",
            "physical": physical.__dict__,
            "ensemble_details": [],
        }
        return physical, meta

    roles = [
        # role_id, 説明, temperature, weight
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
                )
            )
        for fut in as_completed(futures):
            results.append(fut.result())

    # 重み付き平均
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
    }
    return agg, meta

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

# ------------------------------ メインUI ------------------------------
st.title("Fire Spread Simulator Pro")
st.caption("Save Your Self / 火災拡大シミュレーション（Gemini 2.5 Flash Ensemble）")

with st.sidebar:
    st.header("入力パラメータ")

    fuel_class = st.selectbox(
        "燃料種",
        options=["grass", "shrub", "timber"],
        index=0,
        help="草地/低木/立木。燃料が重いほど基礎延焼速度は遅めになります。",
    )

    c1, c2 = st.columns(2)
    with c1:
        duration_min = st.number_input("予測時間[min]", 5.0, 360.0, 60.0, step=5.0)
        wind_speed_ms = st.slider("風速[m/s]", 0.0, 20.0, 5.0, 0.5)
        slope_percent = st.slider(
            "斜面勾配[%]",
            0.0,
            100.0,
            10.0,
            1.0,
            help="上り勾配で延焼は加速します。% = 垂直/水平×100",
        )
        init_radius_m = st.number_input("初期半径[m]", 0.0, 200.0, 5.0, step=1.0)
    with c2:
        wind_dir_deg = st.slider("風向[°] (0=北/90=東)", 0, 359, 90, 1)
        rel_humidity = st.slider("相対湿度[%]", 5, 100, 40, 1)
        air_temp_c = st.slider("気温[°C]", -10, 50, 25, 1)

    st.divider()
    st.subheader("消火設定")
    c3, c4, c5 = st.columns(3)
    with c3:
        default_app_rate = {"grass": 4.0, "shrub": 8.0, "timber": 12.0}[fuel_class]
        app_rate_lpm_per_m = st.number_input(
            "散水比率[L/min/m]",
            0.1,
            50.0,
            float(default_app_rate),
            step=0.1,
            help="単位延長1mあたり1分間に必要な散水量の目安。燃料が重いほど大きく。",
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
            help="散水の実効率(損失を考慮)。低いほど必要量は増えます。",
        )

    # 入力構造体
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

# ------------------------------ 主要出力エリア（Geminiアンサンブル） ------------------------------
outputs, ensemble_meta = run_gemini_ensemble(inputs)

m1, m2, m3, m4 = st.columns(4)
metric_block(m1, "等価半径 (Gemini ensemble)", outputs.radius_m, "m")
metric_block(m2, "延焼面積", outputs.area_sqm, "m²")
metric_block(m3, "必要水量(推定)", outputs.water_volume_tons, "ton")
metric_block(m4, "周長(楕円)", outputs.perimeter_m, "m")

if ensemble_meta["mode"] == "gemini_ensemble":
    st.success("Gemini 2.5 Flash による並列アンサンブル解析結果を表示しています。", icon="✅")
else:
    st.warning("Gemini が無効なため、物理モデルのみで計算しています。", icon="⚠️")

st.info(
    "本モデルは現場安全判断の補助を目的とした簡易推定です。"
    " 実地の燃料状態・気象・地形・活動状況により大きく変動します。",
    icon="ℹ️",
)

# ------------------------------ タブ: 図/JSON/感度 ------------------------------
tab_fig, tab_json, tab_sensitivity, tab_help = st.tabs(
    ["📈 可視化", "🧾 JSON/エクスポート", "🧪 感度分析", "❓ ヘルプ"]
)

# グラフや感度分析は「高速性」を優先して物理モデルで描画
physical_for_plots = run_physical_model(inputs)

with tab_fig:
    st.subheader("延焼楕円の可視化（物理モデル形状）")
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    a = physical_for_plots.ellipse_a_m
    b = physical_for_plots.ellipse_b_m
    t = np.linspace(0, 2 * np.pi, 400)
    x = a * np.cos(t)
    y = b * np.sin(t)
    # 風向に合わせて回転(0°=北→y+)。北を+Y、東を+Xとして回転。
    theta = np.deg2rad(90 - inputs.wind_dir_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    xy = rot @ np.vstack([x, y])
    ax1.plot(xy[0], xy[1], linewidth=2)
    ax1.scatter([0], [0], marker="*", s=120)  # 火点
    ax1.set_aspect("equal", "box")
    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.grid(True, alpha=0.4)
    st.pyplot(fig1)

    st.subheader("時間に対する半径/水量の推移（物理モデルベース）")
    fig2, ax2 = plt.subplots(figsize=(7, 4))
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
    ax2.plot(times, radii, label="半径[m]")
    ax2.set_xlabel("時間[min]")
    ax2.set_ylabel("半径[m]")
    ax2.grid(True, alpha=0.4)
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.plot(times, waters, label="水量[ton]")
    ax3.set_xlabel("時間[min]")
    ax3.set_ylabel("必要水量[ton]")
    ax3.grid(True, alpha=0.4)
    st.pyplot(fig3)

with tab_json:
    st.subheader("JSON 出力（Gemini ensemble）")
    json_str = to_json(outputs)
    st.code(json_str, language="json")
    st.download_button(
        "JSONをダウンロード",
        data=json_str.encode("utf-8"),
        file_name="fire_spread_output.json",
        mime="application/json",
    )

    st.divider()
    st.subheader("CSV 出力 (主要値)")
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
        "CSVをダウンロード",
        data=csv_data.encode("utf-8"),
        file_name="fire_spread_output.csv",
        mime="text/csv",
    )

with tab_sensitivity:
    st.subheader("感度分析 (シナリオ比較 / 物理モデル)")
    st.caption("任意の軸を変更して、半径・水量の変化を高速に比較")

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
    else:  # 燃料種
        fuels = ["grass", "shrub", "timber"]
        for f in fuels:
            label = f"燃料 {f}"
            scenarios.append((label, Inputs(**{**inputs.__dict__, "fuel_class": f})))

    figS, axS = plt.subplots(figsize=(7, 4))
    for label, sc_inp in scenarios:
        o = run_physical_model(sc_inp)
        axS.scatter(o.radius_m, o.water_volume_tons, s=60, label=label)
        axS.annotate(
            label,
            (o.radius_m, o.water_volume_tons),
            xytext=(5, 5),
            textcoords="offset points",
        )
    axS.set_xlabel("等価半径[m]")
    axS.set_ylabel("必要水量[ton]")
    axS.grid(True, alpha=0.4)
    st.pyplot(figS)

with tab_help:
    st.subheader("モデルの考え方")
    st.markdown(
        """
- **物理モデルコア**
  - 延焼速度(ROS) = 基準ROS(燃料別) × 湿度係数 × 風係数 × 斜面係数
  - 風下方向に長い楕円として延焼形状を近似
  - 等価半径 = 楕円面積と同じ円の半径
  - 必要水量 = 周長×散水比率×散水時間 / 散水効率

- **Gemini 2.5 Flash アンサンブル**
  - 物理モデル結果をベースラインとして提示
  - 「安全マージン重視」「資機材効率重視」「バランス型」の3ロールで並列推定
  - 各ロールは ±30% の範囲で補正された数値を JSON で返す
  - 3つの結果を重み付き平均して、最終的な推奨値を決定
  - ヘッダのメトリクスはこのアンサンブル結果を表示

- **高速性の確保**
  - Gemini 呼び出しは主要出力の1回のみ（3ロールを並列実行）
  - グラフや感度分析は物理モデルで計算し、インタラクティブ操作でも高速に応答
        """
    )

    st.subheader("Gemini アンサンブル詳細（デバッグ・検証用）")
    with st.expander("内部ロールの生データを見る"):
        st.json(ensemble_meta)

# ------------------------------ 機械連携用JSON ------------------------------
with st.expander("機械連携用JSON (コピー用 / Gemini ensemble)"):
    st.code(to_json(outputs), language="json")
