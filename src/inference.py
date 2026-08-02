from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


MANUAL_REVIEW_MARGIN = 5.0
MANUAL_REVIEW_MIN_FLAGS = 2
MANUAL_REVIEW_MIN_RELATIVE_GAP = 0.50
MANUAL_REVIEW_MIN_ABSOLUTE_GAP_MILLION = 10.0


def predict_price(
    vehicle_data: pd.DataFrame,
    price_model: Any,
) -> float:
    """Predict one market price in million VND."""
    prediction_log = float(price_model.predict(vehicle_data)[0])
    return max(0.0, float(np.expm1(prediction_log)))


def get_segment_statistics(
    *,
    segment: str,
    segment_statistics: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, float | bool]:
    """Get segment-level statistics or global fallback values."""
    matched = segment_statistics[
        segment_statistics["segment"].eq(segment)
    ]

    if len(matched) == 1:
        row = matched.iloc[0]
        return {
            "residual_median": float(row["residual_median"]),
            "residual_scale": float(row["residual_scale"]),
            "p01": float(row["p01"]),
            "p10": float(row["p10"]),
            "p90": float(row["p90"]),
            "p99": float(row["p99"]),
            "used_global_fallback": False,
        }

    percentiles = config["global_percentiles"]
    return {
        "residual_median": float(config["global_residual_median"]),
        "residual_scale": float(config["global_residual_scale"]),
        "p01": float(percentiles["p01"]),
        "p10": float(percentiles["p10"]),
        "p90": float(percentiles["p90"]),
        "p99": float(percentiles["p99"]),
        "used_global_fallback": True,
    }


def analyze_price_anomaly(
    *,
    vehicle_data: pd.DataFrame,
    listed_price: float,
    price_model: Any,
    isolation_preprocessor: Any,
    isolation_forest: Any,
    segment_statistics: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run the complete single-vehicle anomaly pipeline."""
    listed_price = float(listed_price)
    if listed_price <= 0:
        raise ValueError("Giá dự định đăng phải lớn hơn 0.")

    predicted_price = predict_price(vehicle_data, price_model)
    segment = str(vehicle_data.iloc[0]["segment"])

    stats = get_segment_statistics(
        segment=segment,
        segment_statistics=segment_statistics,
        config=config,
    )

    residual = listed_price - predicted_price
    absolute_gap = abs(residual)
    relative_gap = absolute_gap / max(predicted_price, 1e-9)
    difference_percent = 100.0 * residual / max(predicted_price, 1e-9)
    flag_large_price_gap = bool(
        relative_gap >= MANUAL_REVIEW_MIN_RELATIVE_GAP
        and absolute_gap >= MANUAL_REVIEW_MIN_ABSOLUTE_GAP_MILLION
    )

    residual_scale = max(float(stats["residual_scale"]), 1e-9)
    residual_z = (
        residual - float(stats["residual_median"])
    ) / residual_scale

    flag_residual_z = abs(residual_z) >= 3
    flag_minmax = (
        listed_price < float(stats["p01"])
        or listed_price > float(stats["p99"])
    )
    flag_common_range = (
        listed_price < float(stats["p10"])
        or listed_price > float(stats["p90"])
    )

    isolation_input = vehicle_data.copy()
    isolation_input["deployment_predicted_price"] = predicted_price
    isolation_input["deployment_residual_z"] = residual_z

    isolation_columns = (
        config["isolation_numeric_features"]
        + config["isolation_categorical_features"]
    )
    isolation_matrix = isolation_preprocessor.transform(
        isolation_input[isolation_columns]
    )

    isolation_raw_score = float(
        -isolation_forest.score_samples(isolation_matrix)[0]
    )
    flag_isolation = bool(
        isolation_forest.predict(isolation_matrix)[0] == -1
    )

    # Four normalized signals in the range 0-1.
    s1_residual = float(np.clip(abs(residual_z) / 6, 0, 1))
    s2_minmax = float(flag_minmax)

    common_width = max(float(stats["p90"]) - float(stats["p10"]), 1e-9)
    distance_common_range = max(
        float(stats["p10"]) - listed_price,
        listed_price - float(stats["p90"]),
        0.0,
    )
    s3_common_range = float(
        np.clip(distance_common_range / common_width, 0, 1)
    )

    isolation_min = float(config["isolation_score_min"])
    isolation_max = float(config["isolation_score_max"])
    isolation_range = max(isolation_max - isolation_min, 1e-9)
    s4_isolation = float(
        np.clip(
            (isolation_raw_score - isolation_min) / isolation_range,
            0,
            1,
        )
    )

    weights = config["weights"]

    residual_contribution = float(
        100 * weights["residual"] * s1_residual
    )
    minmax_contribution = float(
        100 * weights["minmax"] * s2_minmax
    )
    common_range_contribution = float(
        100 * weights["common_range"] * s3_common_range
    )
    isolation_contribution = float(
        100 * weights["isolation"] * s4_isolation
    )

    anomaly_score = float(
        residual_contribution
        + minmax_contribution
        + common_range_contribution
        + isolation_contribution
    )

    threshold = float(config["anomaly_threshold"])
    is_anomaly = bool(anomaly_score >= threshold)

    component_flag_count = int(
        bool(flag_residual_z)
        + bool(flag_minmax)
        + bool(flag_common_range)
        + bool(flag_isolation)
    )
    distance_to_threshold = float(threshold - anomaly_score)
    near_threshold = bool(distance_to_threshold <= MANUAL_REVIEW_MARGIN)
    multiple_component_flags = bool(
        component_flag_count >= MANUAL_REVIEW_MIN_FLAGS
    )

    needs_manual_review = bool(
        not is_anomaly
        and (
            near_threshold
            or multiple_component_flags
            or flag_large_price_gap
        )
    )

    if is_anomaly:
        review_status = "Bất thường"
    elif needs_manual_review:
        review_status = "Cần kiểm tra thủ công"
    else:
        review_status = "Bình thường"

    if review_status == "Bình thường":
        anomaly_type = "Bình thường"
    elif residual < 0:
        anomaly_type = "Có xu hướng quá rẻ"
    else:
        anomaly_type = "Có xu hướng quá đắt"

    if residual < 0:
        price_direction = "thấp hơn"
        directional_label = "quá rẻ"
    else:
        price_direction = "cao hơn"
        directional_label = "quá đắt"

    score_position_text = (
        f"cao hơn ngưỡng {abs(distance_to_threshold):.2f} điểm"
        if is_anomaly
        else f"thấp hơn ngưỡng {distance_to_threshold:.2f} điểm"
    )

    score_explanation = (
        f"Anomaly score {anomaly_score:.2f}/100 là điểm tổng hợp từ bốn "
        "tín hiệu: Residual-Z (tối đa 40 điểm), P1-P99 (20 điểm), "
        "P10-P90 (20 điểm) và Isolation Forest (20 điểm). "
        f"Điểm càng cao thì tin càng khác thường. Trường hợp này {score_position_text}. "
        "Đây không phải xác suất gian lận."
    )

    threshold_explanation = (
        f"Ngưỡng {threshold:.2f}/100 là mốc phân vị 95 của anomaly score "
        "trong dữ liệu hiệu chỉnh. Score từ ngưỡng này trở lên thuộc nhóm "
        "khoảng 5% tin có điểm bất thường cao nhất và được gắn nhãn "
        "'Bất thường'. Các tin gần ngưỡng, có nhiều cờ hoặc chênh giá rất lớn "
        "vẫn được đưa vào trạng thái 'Cần kiểm tra thủ công'."
    )

    residual_explanation = (
        f"Giá nhập {listed_price:,.1f} triệu {price_direction} giá dự đoán "
        f"{abs(residual):,.1f} triệu. Sau khi so sánh phần chênh lệch này với "
        f"các sai số thường thấy trong cùng phân khúc, Residual-Z = {residual_z:.2f}. "
        f"Cờ bật khi |Residual-Z| ≥ 3; trường hợp này "
        f"{'bật' if flag_residual_z else 'không bật'} vì |z| = {abs(residual_z):.2f}. "
        "Dấu âm cho thấy giá thấp hơn kỳ vọng; dấu dương cho thấy giá cao hơn kỳ vọng."
    )

    if flag_minmax:
        p01_p99_position = (
            f"thấp hơn P1 {float(stats['p01']):,.1f} triệu"
            if listed_price < float(stats["p01"])
            else f"cao hơn P99 {float(stats['p99']):,.1f} triệu"
        )
        p01_p99_reason = f"Giá hiện tại {p01_p99_position}, nên cờ bật."
    else:
        p01_p99_reason = (
            f"Giá hiện tại nằm trong khoảng {float(stats['p01']):,.1f}-"
            f"{float(stats['p99']):,.1f} triệu, nên cờ không bật."
        )

    p01_p99_explanation = (
        "P1-P99 là biên rất rộng của giá lịch sử trong cùng phân khúc: "
        "P1 là mức chỉ khoảng 1% tin thấp hơn, P99 là mức chỉ khoảng 1% tin "
        f"cao hơn. {p01_p99_reason}"
    )

    if flag_common_range:
        if listed_price < float(stats["p10"]):
            common_position = (
                f"thấp hơn P10 {float(stats['p10']):,.1f} triệu "
                f"{float(stats['p10']) - listed_price:,.1f} triệu"
            )
        else:
            common_position = (
                f"cao hơn P90 {float(stats['p90']):,.1f} triệu "
                f"{listed_price - float(stats['p90']):,.1f} triệu"
            )
        p10_p90_reason = f"Giá hiện tại {common_position}, nên cờ bật."
    else:
        p10_p90_reason = (
            f"Giá hiện tại nằm trong dải {float(stats['p10']):,.1f}-"
            f"{float(stats['p90']):,.1f} triệu, nên cờ không bật."
        )

    p10_p90_explanation = (
        "P10-P90 là dải giá phổ biến chứa khoảng 80% tin lịch sử trong cùng "
        f"phân khúc. {p10_p90_reason} Đây là dải tham khảo thị trường, không "
        "phải khoảng bảo đảm cho dự đoán."
    )

    isolation_explanation = (
        "Isolation Forest xem đồng thời đặc điểm xe, giá dự đoán và Residual-Z "
        "để tìm các mẫu hiếm hoặc tách biệt so với dữ liệu huấn luyện. "
        f"Mức tách biệt chuẩn hóa của trường hợp này là {s4_isolation:.2f}/1.00, "
        f"đóng góp {isolation_contribution:.2f}/20 điểm và cờ "
        f"{'bật' if flag_isolation else 'không bật'}. Cờ này chỉ cho biết mẫu "
        "khác thường, không chứng minh tin sai hoặc gian lận."
    )

    manual_review_triggers: list[str] = []
    if near_threshold:
        manual_review_triggers.append(
            f"score chỉ còn cách ngưỡng {max(distance_to_threshold, 0):.2f} điểm"
        )
    if multiple_component_flags:
        manual_review_triggers.append(
            f"có {component_flag_count}/4 tín hiệu cảnh báo"
        )
    if flag_large_price_gap:
        manual_review_triggers.append(
            f"giá chênh {abs(difference_percent):.1f}% và {absolute_gap:.1f} triệu"
        )

    if review_status == "Bất thường":
        status_explanation = (
            f"Score đã vượt ngưỡng và giá có xu hướng {directional_label}. "
            "Tin nên được ưu tiên kiểm tra."
        )
    elif review_status == "Cần kiểm tra thủ công":
        status_explanation = (
            "Score chưa vượt ngưỡng bất thường, nhưng "
            + "; ".join(manual_review_triggers)
            + f". Giá có xu hướng {directional_label}; nên kiểm tra thủ công."
        )
    else:
        status_explanation = (
            "Score cách ngưỡng đủ xa, không có nhiều tín hiệu cảnh báo mạnh "
            "và chênh lệch giá chưa đồng thời vượt 50% và 10 triệu. "
            "Kết quả vẫn chỉ là tham khảo, không phải bảo đảm giá chắc chắn hợp lý."
        )

    reasons: list[str] = []
    if flag_residual_z:
        reasons.append(f"Residual-Z = {residual_z:.2f} (|z| ≥ 3)")
    if flag_minmax:
        reasons.append(
            f"Giá {listed_price:,.1f} triệu nằm ngoài P1-P99 "
            f"({float(stats['p01']):,.1f}-{float(stats['p99']):,.1f})"
        )
    if flag_common_range:
        reasons.append(
            f"Giá {listed_price:,.1f} triệu nằm ngoài P10-P90 "
            f"({float(stats['p10']):,.1f}-{float(stats['p90']):,.1f})"
        )
    if flag_isolation:
        reasons.append("Isolation Forest đánh dấu mẫu là tách biệt")
    if flag_large_price_gap:
        reasons.append(
            f"Chênh giá {abs(difference_percent):.1f}% và {absolute_gap:.1f} triệu "
            "vượt quy tắc kiểm tra thủ công"
        )
    if not reasons:
        reasons.append("Không có cờ thành phần mạnh")

    return {
        "predicted_price": predicted_price,
        "listed_price": listed_price,
        "difference": residual,
        "difference_percent": difference_percent,
        "absolute_gap_million": absolute_gap,
        "segment": segment,
        "p01": float(stats["p01"]),
        "p10": float(stats["p10"]),
        "p90": float(stats["p90"]),
        "p99": float(stats["p99"]),
        "residual_z": float(residual_z),
        "isolation_raw_score": isolation_raw_score,
        "isolation_normalized_score": s4_isolation,
        "flag_residual_z": bool(flag_residual_z),
        "flag_minmax": bool(flag_minmax),
        "flag_common_range": bool(flag_common_range),
        "flag_isolation": bool(flag_isolation),
        "flag_large_price_gap": flag_large_price_gap,
        "residual_contribution": residual_contribution,
        "minmax_contribution": minmax_contribution,
        "common_range_contribution": common_range_contribution,
        "isolation_contribution": isolation_contribution,
        "anomaly_score": anomaly_score,
        "anomaly_threshold": threshold,
        "distance_to_threshold": distance_to_threshold,
        "component_flag_count": component_flag_count,
        "is_anomaly": is_anomaly,
        "needs_manual_review": needs_manual_review,
        "review_status": review_status,
        "anomaly_type": anomaly_type,
        "reasons": reasons,
        "used_global_fallback": bool(stats["used_global_fallback"]),
        "explanations": {
            "status": status_explanation,
            "score": score_explanation,
            "threshold": threshold_explanation,
            "residual_z": residual_explanation,
            "p01_p99": p01_p99_explanation,
            "p10_p90": p10_p90_explanation,
            "isolation_forest": isolation_explanation,
        },
    }
