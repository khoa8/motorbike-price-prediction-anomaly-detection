from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

from src.batch import run_batch_inference
from src.features import prepare_vehicle_input
from src.inference import (
    analyze_price_anomaly,
    get_segment_statistics,
    predict_price,
)


st.set_page_config(
    page_title="Motorbike Price & Anomaly Detection",
    page_icon="🏍️",
    layout="wide",
)

ROOT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT_DIR / "artifacts"
REPORT_DIR = ROOT_DIR / "reports"
EXAMPLE_DIR = ROOT_DIR / "examples"
BANNER_CANDIDATES = (
    ROOT_DIR / "assets" / "motorbike_banner.png",
    ROOT_DIR / "assets" / "motorbike_banner.jpg",
    ROOT_DIR / "assets" / "motorbike_banner.jpeg",
    ROOT_DIR / "motorbike_banner.png",
)


@st.cache_resource
def load_models() -> dict[str, Any]:
    return {
        "price_model": joblib.load(ARTIFACT_DIR / "price_model.joblib"),
        "isolation_preprocessor": joblib.load(
            ARTIFACT_DIR / "isolation_preprocessor.joblib"
        ),
        "isolation_forest": joblib.load(
            ARTIFACT_DIR / "isolation_forest.joblib"
        ),
    }


@st.cache_data
def load_support_data() -> dict[str, Any]:
    with open(
        ARTIFACT_DIR / "deployment_config.json",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    with open(
        ARTIFACT_DIR / "segment_rules.json",
        encoding="utf-8",
    ) as file:
        segment_rules = json.load(file)

    with open(
        ARTIFACT_DIR / "input_options.json",
        encoding="utf-8",
    ) as file:
        input_options = json.load(file)

    segment_statistics = pd.read_csv(
        ARTIFACT_DIR / "segment_statistics.csv"
    )

    csv_artifacts: dict[str, pd.DataFrame] = {}
    for key, filename in {
        "deployment_test": "deployment_model_evaluation.csv",
        "deployment_oof": "deployment_oof_evaluation.csv",
        "deployment_decision": "deployment_model_decision.csv",
        "price_bands": "deployment_price_band_metrics.csv",
        "oof_price_bands": "deployment_oof_price_band_metrics.csv",
        "feature_importance": "deployment_feature_importance_grouped.csv",
    }.items():
        path = ARTIFACT_DIR / filename
        if path.exists():
            csv_artifacts[key] = pd.read_csv(path)

    report_tables: dict[str, pd.DataFrame] = {}
    for key, filename in {
        "sklearn_models": "sklearn_model_comparison.csv",
        "spark_models": "spark_model_comparison.csv",
        "sklearn_features": "sklearn_feature_importance.csv",
        "spark_features": "spark_rf_feature_importance.csv",
    }.items():
        path = REPORT_DIR / filename
        if path.exists():
            report_tables[key] = pd.read_csv(path)

    summary: dict[str, Any] = {}
    summary_path = ARTIFACT_DIR / "deployment_results_summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as file:
            summary = json.load(file)

    return {
        "config": config,
        "segment_rules": segment_rules,
        "input_options": input_options,
        "segment_statistics": segment_statistics,
        "artifacts": csv_artifacts,
        "reports": report_tables,
        "summary": summary,
    }


def apply_app_theme(dark_mode: bool) -> None:
    """Apply a session-level light/dark skin without restarting the app."""
    if dark_mode:
        colors = {
            "app": "#0B1120",
            "panel": "#111827",
            "panel_alt": "#172033",
            "sidebar": "#0F172A",
            "text": "#F8FAFC",
            "muted": "#CBD5E1",
            "border": "#334155",
            "input": "#111827",
            "shadow": "rgba(0, 0, 0, 0.30)",
        }
    else:
        colors = {
            "app": "#FFFFFF",
            "panel": "#FFFFFF",
            "panel_alt": "#F3F4F6",
            "sidebar": "#F5F6F8",
            "text": "#111827",
            "muted": "#4B5563",
            "border": "#D1D5DB",
            "input": "#FFFFFF",
            "shadow": "rgba(15, 23, 42, 0.08)",
        }

    st.markdown(
        f"""
<style>
:root {{
    --app-bg: {colors['app']};
    --panel-bg: {colors['panel']};
    --panel-alt: {colors['panel_alt']};
    --sidebar-bg: {colors['sidebar']};
    --app-text: {colors['text']};
    --muted-text: {colors['muted']};
    --app-border: {colors['border']};
    --input-bg: {colors['input']};
    --app-shadow: {colors['shadow']};
}}

[data-testid="stAppViewContainer"],
[data-testid="stMain"] {{
    background: var(--app-bg);
    color: var(--app-text);
}}

[data-testid="stSidebar"] > div:first-child {{
    background: var(--sidebar-bg);
}}

[data-testid="stSidebar"],
[data-testid="stSidebar"] *:not(svg) {{
    color: var(--app-text);
}}

[data-testid="stHeader"] {{
    background: color-mix(in srgb, var(--app-bg) 92%, transparent);
}}

h1, h2, h3, h4, h5, h6, p, label,
[data-testid="stMarkdownContainer"],
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {{
    color: var(--app-text);
}}

[data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"] p {{
    color: var(--muted-text);
}}

[data-testid="stMetric"],
[data-testid="stExpander"],
[data-testid="stFileUploaderDropzone"] {{
    background: var(--panel-bg);
    border-color: var(--app-border);
}}

input, textarea,
div[data-baseweb="select"] > div,
div[data-baseweb="base-input"] {{
    background: var(--input-bg) !important;
    color: var(--app-text) !important;
    border-color: var(--app-border) !important;
}}

[data-testid="stDataFrame"],
[data-testid="stTable"] {{
    border: 1px solid var(--app-border);
    border-radius: 0.75rem;
    overflow: hidden;
}}

[data-testid="stAlert"] {{
    border-radius: 0.85rem;
}}

# [data-testid="stAlert"] {{
#     padding: 1.25rem 1.4rem;
# }}

[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {{
    font-size: 1.15rem;
    line-height: 1.7;
}}

[data-testid="stAlert"] [data-testid="stMarkdownContainer"] strong {{
    font-size: 1.5rem;
    line-height: 1.35;
}}

.block-container {{
    padding-top: 3rem;
    padding-bottom: 3rem;
}}

.app-hero {{
    min-height: 245px;
    border-radius: 22px;
    margin: 0.1rem 0 1.5rem 0;
    padding: 2.1rem 2.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    background-position: center;
    background-size: cover;
    box-shadow: 0 12px 32px var(--app-shadow);
    border: 1px solid rgba(255, 255, 255, 0.18);
}}

.app-hero-content {{
    width: 100%;
    max-width: 1600px;
    color: #FFFFFF;
    text-shadow:
        0 3px 10px rgba(0, 0, 0, 0.85),
        0 1px 3px rgba(0, 0, 0, 0.95);
}}

.app-hero-title {{
    margin: 0;
    padding: 0 1rem;
    font-size: clamp(1.65rem, 3.3vw, 3.4rem);
    line-height: 1.15;
    font-weight: 900;
    letter-spacing: 0.025em;
    text-transform: uppercase;
}}

.score-track {{
    width: 100%;
    height: 14px;
    background: var(--panel-alt);
    border: 1px solid var(--app-border);
    border-radius: 999px;
    overflow: hidden;
    margin: 0.35rem 0 1rem 0;
}}

.score-fill {{
    height: 100%;
    border-radius: 999px;
    transition: width 240ms ease;
}}

.state-note {{
    padding: 0.65rem 0.85rem;
    margin: 0.25rem 0 1rem 0;
    border: 1px solid var(--app-border);
    border-radius: 0.75rem;
    background: var(--panel-alt);
    color: var(--muted-text);
    font-size: 0.9rem;
}}
</style>
""",
        unsafe_allow_html=True,
    )


def banner_background() -> str:
    """Return a CSS background image, or a built-in gradient fallback."""
    for path in BANNER_CANDIDATES:
        if not path.exists():
            continue

        suffix = path.suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"url('data:{mime};base64,{encoded}')"

    return (
        "radial-gradient(circle at 18% 25%, rgba(34, 211, 238, 0.42), "
        "transparent 30%), radial-gradient(circle at 82% 70%, "
        "rgba(239, 68, 68, 0.34), transparent 28%), "
        "linear-gradient(120deg, #071426, #12325A 48%, #0F172A)"
    )


def render_global_banner() -> None:
    background = banner_background()

    st.markdown(
        f"""
<div class="app-hero" style="background-image: {background};">
  <div class="app-hero-content">
    <div class="app-hero-title">
      DỰ ĐOÁN GIÁ VÀ PHÁT HIỆN BẤT THƯỜNG GIÁ XE MÁY
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_session_note() -> None:
    st.markdown(
        """
<div class="state-note">
Kết quả và dữ liệu nhập được giữ khi chuyển trang trong cùng phiên làm việc.
Chúng được đặt lại khi tải lại toàn bộ trang hoặc mở một phiên mới.
</div>
""",
        unsafe_allow_html=True,
    )


def default_index(options: list[str], preferred: str) -> int:
    try:
        return options.index(preferred)
    except ValueError:
        return 0


def render_vehicle_inputs(
    *,
    prefix: str,
    support: dict[str, Any],
) -> dict[str, Any]:
    options = support["input_options"]
    config = support["config"]

    left, right = st.columns(2)

    with left:
        brands = options["brands"]
        brand = st.selectbox(
            "Thương hiệu",
            brands,
            index=default_index(brands, "Honda"),
            key=f"{prefix}_brand",
        )

        available_models = options["models_by_brand"].get(
            brand, ["Không rõ"]
        )
        model_key = f"{prefix}_model"
        if (
            model_key in st.session_state
            and st.session_state[model_key] not in available_models
        ):
            st.session_state[model_key] = available_models[0]
        model = st.selectbox(
            "Dòng xe",
            available_models,
            key=model_key,
        )

        bike_types = options["bike_types"]
        bike_type = st.selectbox(
            "Loại xe",
            bike_types,
            index=default_index(bike_types, "Tay ga"),
            key=f"{prefix}_bike_type",
        )

        capacities = options["capacities"]
        capacity = st.selectbox(
            "Dung tích xe",
            capacities,
            index=default_index(capacities, "100 - 175 cc"),
            key=f"{prefix}_capacity",
        )

        origins = options["origins"]
        origin = st.selectbox(
            "Xuất xứ",
            origins,
            index=default_index(origins, "Việt Nam"),
            key=f"{prefix}_origin",
        )

    with right:
        districts = options["districts"]
        district = st.selectbox(
            "Quận/huyện",
            districts,
            index=default_index(districts, "Quận 1"),
            key=f"{prefix}_district",
        )

        year = st.number_input(
            "Năm đăng ký",
            min_value=1979,
            max_value=int(config["reference_year"]),
            value=2020,
            step=1,
            key=f"{prefix}_year",
        )

        km_unknown = st.checkbox(
            "Không biết số kilomet",
            key=f"{prefix}_km_unknown",
        )

        km: float | None
        if km_unknown:
            st.caption("Model sẽ dùng giá trị kilomet trung vị đã học.")
            km = None
        else:
            km = float(
                st.number_input(
                    "Số kilomet đã đi",
                    min_value=0,
                    max_value=1_000_000,
                    value=20_000,
                    step=1_000,
                    key=f"{prefix}_km",
                )
            )

        title = st.text_input(
            "Tiêu đề tin đăng",
            placeholder="Ví dụ: Honda SH 350i ABS chính chủ",
            key=f"{prefix}_title",
        )

        description = st.text_area(
            "Mô tả chi tiết",
            placeholder=(
                "Tình trạng xe, phiên bản, nguồn gốc, "
                "lịch sử bảo dưỡng..."
            ),
            key=f"{prefix}_description",
        )

    return {
        "brand": brand,
        "model": model,
        "year": int(year),
        "km": km,
        "bike_type": bike_type,
        "capacity": capacity,
        "origin": origin,
        "district": district,
        "title": title,
        "description": description,
    }


def make_vehicle_frame(
    values: dict[str, Any],
    support: dict[str, Any],
) -> pd.DataFrame:
    return prepare_vehicle_input(
        **values,
        config=support["config"],
        segment_rules=support["segment_rules"],
    )


def page_business_problem() -> None:
    st.title("🎯 Bài toán kinh doanh")
    st.markdown(
        """
Ứng dụng phục vụ hai nhóm người dùng:

**Người bán**

- Nhận mức giá tham khảo dựa trên thông tin xe.
- So sánh giá dự định đăng với mặt bằng phân khúc.

**Nền tảng và người kiểm duyệt**

- Phát hiện các tin có dấu hiệu quá rẻ hoặc quá đắt.
- Xem lý do và các tín hiệu tạo nên anomaly score.
- Kiểm tra hàng loạt nhiều tin bằng file CSV.
"""
    )

    st.info(
        "Kết quả chỉ hỗ trợ quyết định. Tin bị gắn cờ không đồng nghĩa "
        "chắc chắn nhập sai hoặc gian lận."
    )

    st.subheader("Phạm vi")
    st.markdown(
        """
- Dữ liệu tin đăng xe máy cũ tại TP.HCM.
- Snapshot trước ngày 01/07/2025.
- Reference year của model: 2025.
- Xe hiếm và xe trên 100 triệu có sai số cao hơn xe phổ thông.
"""
    )


def page_evaluation() -> None:
    st.title("📊 Evaluation & Report")
    support = load_support_data()
    artifacts = support["artifacts"]
    reports = support["reports"]
    summary = support["summary"]

    decision = summary.get("model_decision", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "OOF MAE thay đổi",
        f"{decision.get('overall_mae_change_percent', 0):.2f}%",
    )
    col2.metric(
        "Premium MAE thay đổi",
        f"{decision.get('premium_mae_change_percent', 0):.2f}%",
    )
    col3.metric(
        "Premium RMSE thay đổi",
        f"{decision.get('premium_rmse_change_percent', 0):.2f}%",
    )
    col4.metric(
        "Giữ model mới",
        "Có" if decision.get("keep_premium_model") else "Không",
    )

    if "deployment_test" in artifacts:
        st.subheader("So sánh trên test set")
        st.dataframe(
            artifacts["deployment_test"].round(3),
            use_container_width=True,
            hide_index=True,
        )

    if "deployment_oof" in artifacts:
        st.subheader("So sánh bằng 5-fold OOF")
        st.dataframe(
            artifacts["deployment_oof"].round(3),
            use_container_width=True,
            hide_index=True,
        )

    if "oof_price_bands" in artifacts:
        st.subheader("OOF theo nhóm giá")
        st.dataframe(
            artifacts["oof_price_bands"].round(3),
            use_container_width=True,
            hide_index=True,
        )

    if "feature_importance" in artifacts:
        st.subheader("Feature importance của deployment model")
        chart_data = (
            artifacts["feature_importance"]
            .head(15)
            .set_index("source_feature")["importance"]
        )
        st.bar_chart(chart_data)
        st.dataframe(
            artifacts["feature_importance"].head(20).round(4),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Benchmark scikit-learn và Spark"):
        if "sklearn_models" in reports:
            st.markdown("**scikit-learn**")
            st.dataframe(
                reports["sklearn_models"].round(3),
                use_container_width=True,
                hide_index=True,
            )
        if "spark_models" in reports:
            st.markdown("**Spark ML**")
            st.dataframe(
                reports["spark_models"].round(3),
                use_container_width=True,
                hide_index=True,
            )

    st.warning(
        "Dữ liệu không có ground truth anomaly, nên chưa thể tính "
        "Precision, Recall hoặc F1 thực sự cho anomaly detection."
    )


def render_prediction_result(result: dict[str, Any]) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Giá gợi ý",
        f"{result['predicted_price']:,.1f} triệu đồng",
    )
    col2.metric("P10 phân khúc", f"{result['p10']:,.1f} triệu")
    col3.metric("P90 phân khúc", f"{result['p90']:,.1f} triệu")

    st.write("**Phân khúc:**", result["segment"])

    if result["segment"] == "Nhóm hiếm" or result["predicted_price"] > 100:
        st.warning(
            "Xe thuộc nhóm hiếm hoặc nhóm giá cao; sai số có thể "
            "lớn hơn xe phổ thông."
        )

    st.caption(
        "P10-P90 là dải giá lịch sử phổ biến của phân khúc, "
        "không phải khoảng bảo đảm cho dự đoán."
    )


def page_price_prediction() -> None:
    st.title("💰 Dự đoán giá xe")
    render_session_note()
    models = load_models()
    support = load_support_data()

    values = render_vehicle_inputs(prefix="prediction", support=support)

    if st.button("Dự đoán giá", type="primary", key="prediction_submit"):
        try:
            vehicle = make_vehicle_frame(values, support)
            predicted_price = predict_price(
                vehicle, models["price_model"]
            )
            segment = str(vehicle.iloc[0]["segment"])
            stats = get_segment_statistics(
                segment=segment,
                segment_statistics=support["segment_statistics"],
                config=support["config"],
            )
            st.session_state["prediction_result"] = {
                "predicted_price": predicted_price,
                "segment": segment,
                "p10": float(stats["p10"]),
                "p90": float(stats["p90"]),
            }
        except (ValueError, KeyError) as error:
            st.error(str(error))

    result = st.session_state.get("prediction_result")
    if isinstance(result, dict):
        st.divider()
        render_prediction_result(result)


def yes_no(value: bool) -> str:
    return "Có" if value else "Không"


def render_review_status(result: dict[str, Any]) -> None:
    """Show the three-level business review status."""
    status = result["review_status"]
    explanation = result["explanations"]["status"]

    if status == "Bất thường":
        st.error(f"**Bất thường — {result['anomaly_type']}**\n\n{explanation}")
    elif status == "Cần kiểm tra thủ công":
        st.warning(
            f"**Cần kiểm tra thủ công — {result['anomaly_type']}**"
            f"\n\n{explanation}"
        )
    else:
        st.success(f"**Bình thường**\n\n{explanation}")


def anomaly_status_color(status: str) -> str:
    return {
        "Bình thường": "#16A34A",
        "Cần kiểm tra thủ công": "#F59E0B",
        "Bất thường": "#DC2626",
    }.get(status, "#64748B")


def render_anomaly_progress(result: dict[str, Any]) -> None:
    score = min(max(float(result["anomaly_score"]), 0.0), 100.0)
    color = anomaly_status_color(str(result["review_status"]))
    st.markdown(
        f"""
<div class="score-track" role="progressbar" aria-valuemin="0"
     aria-valuemax="100" aria-valuenow="{score:.2f}">
  <div class="score-fill" style="width: {score:.2f}%; background: {color};"></div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_anomaly_explanations(result: dict[str, Any]) -> None:
    """Explain score, threshold and every component signal."""
    distance = float(result["distance_to_threshold"])
    distance_label = (
        f"Vượt {abs(distance):.2f} điểm"
        if distance <= 0
        else f"Còn {distance:.2f} điểm"
    )

    metric1, metric2, metric3, metric4 = st.columns(
        [1, 1, 1.5, 1],
        gap="medium",
    )
    metric1.metric("Anomaly score", f"{result['anomaly_score']:.2f}/100")
    metric2.metric("Ngưỡng", f"{result['anomaly_threshold']:.2f}/100")
    metric3.metric("Khoảng cách", distance_label)
    metric4.metric(
        "Tín hiệu cảnh báo",
        f"{result['component_flag_count']}/4",
    )

    score_components = pd.DataFrame(
        {
            "Tín hiệu": [
                "Residual-Z",
                "P1-P99",
                "P10-P90",
                "Isolation Forest",
            ],
            "Giá trị của trường hợp": [
                f"z = {result['residual_z']:.2f}",
                (
                    f"Giá {result['listed_price']:.1f}; "
                    f"biên {result['p01']:.1f}-{result['p99']:.1f}"
                ),
                (
                    f"Giá {result['listed_price']:.1f}; "
                    f"dải {result['p10']:.1f}-{result['p90']:.1f}"
                ),
                (
                    f"Mức tách biệt "
                    f"{result['isolation_normalized_score']:.2f}/1.00"
                ),
            ],
            "Điều kiện bật cờ": [
                "|z| ≥ 3",
                "Giá < P1 hoặc > P99",
                "Giá < P10 hoặc > P90",
                "Isolation Forest dự đoán -1",
            ],
            "Cờ": [
                yes_no(result["flag_residual_z"]),
                yes_no(result["flag_minmax"]),
                yes_no(result["flag_common_range"]),
                yes_no(result["flag_isolation"]),
            ],
            "Điểm đóng góp": [
                result["residual_contribution"],
                result["minmax_contribution"],
                result["common_range_contribution"],
                result["isolation_contribution"],
            ],
            "Điểm tối đa": [40.0, 20.0, 20.0, 20.0],
        }
    )

    st.subheader("Cấu thành anomaly score")
    st.dataframe(
        score_components.round({"Điểm đóng góp": 2, "Điểm tối đa": 0}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        'Cờ "Có/Không" cho biết tín hiệu đã vượt điều kiện cảnh báo. '
        "Điểm đóng góp cho biết mức độ mạnh của tín hiệu; vì vậy cờ bật "
        "không nhất thiết nhận đủ điểm tối đa."
    )

    st.write("**Lý do/tín hiệu chính:**", "; ".join(result["reasons"]))

    explanations = result["explanations"]
    with st.expander(
        "📘 Giải thích các chỉ số và lý do bật/tắt từng cờ",
        expanded=False,
    ):
        st.markdown(f"**Anomaly score**  \n{explanations['score']}")
        st.markdown(f"**Ngưỡng**  \n{explanations['threshold']}")
        st.markdown(f"**Residual-Z**  \n{explanations['residual_z']}")
        st.markdown(f"**P1-P99**  \n{explanations['p01_p99']}")
        st.markdown(f"**P10-P90**  \n{explanations['p10_p90']}")
        st.markdown(
            f"**Isolation Forest**  \n{explanations['isolation_forest']}"
        )

        if result["used_global_fallback"]:
            st.warning(
                "Không tìm thấy đủ thống kê cho đúng phân khúc này, nên app "
                "đã dùng thống kê toàn bộ dữ liệu làm giá trị dự phòng. "
                "Kết quả cần được diễn giải thận trọng hơn."
            )


def render_anomaly_result(result: dict[str, Any]) -> None:
    predicted_price = float(result["predicted_price"])
    relative_gap = (
        100.0 * float(result["difference"]) / predicted_price
        if predicted_price > 0
        else 0.0
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Giá dự đoán",
        f"{predicted_price:,.1f} triệu",
    )
    col2.metric(
        "Giá nhập",
        f"{result['listed_price']:,.1f} triệu",
    )
    col3.metric(
        "Chênh lệch",
        f"{result['difference']:+,.1f} triệu",
    )
    col4.metric(
        "Chênh lệch %",
        f"{relative_gap:+,.1f}%",
    )

    st.subheader("Anomaly Result")
    render_anomaly_progress(result)
    render_review_status(result)

    if result["review_status"] == "Bình thường" and abs(relative_gap) >= 50:
        st.info(
            f"Giá lệch {abs(relative_gap):.1f}% so với dự đoán nhưng vẫn chưa "
            "vượt ngưỡng anomaly. Hệ thống chấm theo Residual-Z, phân vị giá "
            "của phân khúc và Isolation Forest, không dùng riêng tỷ lệ % chênh lệch."
        )

    render_anomaly_explanations(result)

    st.caption(
        "Kết quả chỉ hỗ trợ định giá và ưu tiên kiểm duyệt. "
        "Bất thường không đồng nghĩa chắc chắn nhập sai hoặc gian lận."
    )


def page_anomaly_check() -> None:
    st.title("🚨 Kiểm tra giá bất thường")
    render_session_note()
    models = load_models()
    support = load_support_data()

    values = render_vehicle_inputs(prefix="anomaly", support=support)
    listed_price = st.number_input(
        "Giá dự định đăng (triệu đồng)",
        min_value=0.1,
        value=50.0,
        step=1.0,
        key="anomaly_listed_price",
    )

    if st.button("Phân tích giá", type="primary", key="anomaly_submit"):
        try:
            vehicle = make_vehicle_frame(values, support)
            result = analyze_price_anomaly(
                vehicle_data=vehicle,
                listed_price=listed_price,
                price_model=models["price_model"],
                isolation_preprocessor=models["isolation_preprocessor"],
                isolation_forest=models["isolation_forest"],
                segment_statistics=support["segment_statistics"],
                config=support["config"],
            )
            st.session_state["anomaly_result"] = result
        except (ValueError, KeyError) as error:
            st.error(str(error))

    result = st.session_state.get("anomaly_result")
    if isinstance(result, dict):
        st.divider()
        render_anomaly_result(result)


def read_uploaded_csv(uploaded_file: Any) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded_file, encoding="utf-8-sig")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="utf-8")


def page_batch_check() -> None:
    st.title("📁 Kiểm tra hàng loạt")
    render_session_note()
    st.write(
        "Tải lên CSV để dự đoán giá và chấm anomaly cho nhiều xe "
        "trong một lần."
    )

    models = load_models()
    support = load_support_data()

    sample_path = EXAMPLE_DIR / "motorbike_batch_template.csv"
    if sample_path.exists():
        st.download_button(
            "⬇️ Tải file mẫu",
            data=sample_path.read_bytes(),
            file_name=sample_path.name,
            mime="text/csv",
            on_click="ignore",
        )

    with st.expander("Định dạng file CSV"):
        required = support["config"]["batch_schema"]["required_columns"]
        optional = support["config"]["batch_schema"]["optional_columns"]
        st.write("**Cột bắt buộc:**", ", ".join(required))
        st.write("**Cột tùy chọn:**", ", ".join(optional))
        st.write(
            "`listed_price_million` là giá đăng theo đơn vị triệu đồng. "
            "Nếu để trống, hệ thống chỉ dự đoán giá và không chấm anomaly."
        )
        st.write(
            "Dòng thiếu số kilomet vẫn được dự đoán bằng giá trị impute; "
            "dòng thiếu cột phân loại bắt buộc sẽ được đưa vào bảng lỗi."
        )

    uploaded_file = st.file_uploader(
        "Chọn file CSV",
        type=["csv"],
        max_upload_size=10,
        key="batch_uploader",
    )

    if uploaded_file is not None:
        try:
            uploaded_data = read_uploaded_csv(uploaded_file)
        except Exception as error:
            st.error(f"Không đọc được CSV: {error}")
            uploaded_data = None

        if isinstance(uploaded_data, pd.DataFrame):
            st.subheader("Xem trước dữ liệu")
            st.dataframe(uploaded_data.head(20), use_container_width=True)

            if st.button(
                "Chạy kiểm tra hàng loạt",
                type="primary",
                key="batch_submit",
            ):
                try:
                    with st.spinner("Đang xử lý dữ liệu..."):
                        results, errors = run_batch_inference(
                            uploaded_data,
                            price_model=models["price_model"],
                            isolation_preprocessor=models[
                                "isolation_preprocessor"
                            ],
                            isolation_forest=models["isolation_forest"],
                            segment_statistics=support["segment_statistics"],
                            config=support["config"],
                            segment_rules=support["segment_rules"],
                            input_options=support["input_options"],
                        )
                    st.session_state["batch_results"] = results
                    st.session_state["batch_errors"] = errors
                except ValueError as error:
                    st.error(str(error))
                except Exception as error:
                    st.exception(error)

    results = st.session_state.get("batch_results")
    errors = st.session_state.get("batch_errors")

    if isinstance(results, pd.DataFrame) and not results.empty:
        statuses = results["review_status"].fillna("Chỉ dự đoán")
        anomaly_mask = statuses.eq("Bất thường")
        manual_mask = statuses.eq("Cần kiểm tra thủ công")
        normal_mask = statuses.eq("Bình thường")

        metric1, metric2, metric3, metric4, metric5, metric6 = st.columns(6)
        metric1.metric("Dòng hợp lệ", f"{len(results):,}")
        metric2.metric(
            "Có giá đăng",
            f"{results['listed_price_million'].notna().sum():,}",
        )
        metric3.metric("Bất thường", f"{anomaly_mask.sum():,}")
        metric4.metric("Cần kiểm tra", f"{manual_mask.sum():,}")
        metric5.metric("Bình thường", f"{normal_mask.sum():,}")
        metric6.metric(
            "Chỉ dự đoán",
            f"{statuses.eq('Chỉ dự đoán').sum():,}",
        )

        with st.expander("Giải thích trạng thái và các cột kết quả"):
            st.markdown(
                """
- **Bất thường:** anomaly score đã vượt ngưỡng top khoảng 5%.
- **Cần kiểm tra thủ công:** score chưa vượt ngưỡng nhưng gần ngưỡng hoặc có ít nhất 2/4 cờ thành phần bật hoặc chênh lệch giá ít nhất 50% và 10 triệu.
- **Bình thường:** score cách ngưỡng đủ xa và không có nhiều tín hiệu mạnh.
- **Residual-Z:** chênh lệch giá đã chuẩn hóa theo phân khúc; dấu âm là thấp hơn kỳ vọng, dấu dương là cao hơn kỳ vọng.
- **P1-P99:** biên giá rất rộng; nằm ngoài biên là trường hợp cực đoan.
- **P10-P90:** dải giá phổ biến chứa khoảng 80% tin lịch sử của phân khúc.
- **Isolation Forest:** phát hiện mẫu có tổ hợp đặc điểm và giá tách biệt so với dữ liệu huấn luyện.
- **anomaly_score:** tổng điểm của bốn tín hiệu, không phải xác suất gian lận.
"""
            )

        status_filter = st.selectbox(
            "Lọc theo trạng thái",
            [
                "Tất cả",
                "Bất thường",
                "Cần kiểm tra thủ công",
                "Bình thường",
                "Chỉ dự đoán",
            ],
            key="batch_status_filter",
        )
        display_results = results
        if status_filter != "Tất cả":
            display_results = results[statuses.eq(status_filter)]

        st.subheader("Kết quả")
        st.dataframe(
            display_results.sort_values(
                "anomaly_score", ascending=False, na_position="last"
            ),
            use_container_width=True,
            hide_index=True,
        )

        csv_bytes = results.to_csv(
            index=False, encoding="utf-8-sig"
        ).encode("utf-8-sig")
        st.download_button(
            "⬇️ Tải toàn bộ kết quả",
            data=csv_bytes,
            file_name="motorbike_batch_results.csv",
            mime="text/csv",
            on_click="ignore",
        )

        scored = results.dropna(subset=["anomaly_score"])
        if not scored.empty:
            st.subheader("Top anomaly score")
            top_scores = (
                scored.nlargest(15, "anomaly_score")
                .set_index("source_row")["anomaly_score"]
            )
            st.bar_chart(top_scores)

    if isinstance(errors, pd.DataFrame) and not errors.empty:
        st.warning(f"Có {len(errors):,} dòng không hợp lệ.")
        st.dataframe(
            errors[
                [
                    "source_row",
                    "brand",
                    "model",
                    "year",
                    "validation_error",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        error_bytes = errors.to_csv(
            index=False, encoding="utf-8-sig"
        ).encode("utf-8-sig")
        st.download_button(
            "⬇️ Tải danh sách dòng lỗi",
            data=error_bytes,
            file_name="motorbike_batch_errors.csv",
            mime="text/csv",
            on_click="ignore",
        )


def page_team() -> None:
    st.title("👥 Thông tin nhóm")
    st.markdown(
        """
### Nhóm 2

- Nguyễn Minh Khoa
- Nguyễn Hoàng Quỳnh Anh
"""
    )


pages = {
    "Project": [
        st.Page(
            page_business_problem,
            title="Business Problem",
            icon="🎯",
            default=True,
        ),
        st.Page(
            page_evaluation,
            title="Evaluation & Report",
            icon="📊",
        ),
    ],
    "Ứng dụng": [
        st.Page(
            page_price_prediction,
            title="Price Prediction",
            icon="💰",
        ),
        st.Page(
            page_anomaly_check,
            title="Anomaly Check",
            icon="🚨",
        ),
        st.Page(
            page_batch_check,
            title="Batch Check",
            icon="📁",
        ),
    ],
    "Nhóm": [
        st.Page(
            page_team,
            title="Team Information",
            icon="👥",
        )
    ],
}

navigation = st.navigation(pages)

with st.sidebar:
    st.divider()
    st.toggle(
        "🌙 Giao diện tối",
        key="dark_mode",
        help="Chuyển giao diện sáng/tối trong phiên hiện tại.",
    )

apply_app_theme(bool(st.session_state.get("dark_mode", False)))
render_global_banner()
navigation.run()
