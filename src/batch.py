from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .features import prepare_batch_features
from .inference import (
    MANUAL_REVIEW_MARGIN,
    MANUAL_REVIEW_MIN_ABSOLUTE_GAP_MILLION,
    MANUAL_REVIEW_MIN_FLAGS,
    MANUAL_REVIEW_MIN_RELATIVE_GAP,
)


def validate_batch_dataframe(
    data: pd.DataFrame,
    *,
    config: dict[str, Any],
    input_options: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate the CSV schema and return valid rows plus row-level errors."""
    if data.empty:
        raise ValueError("File CSV không có dữ liệu.")

    frame = data.copy()
    frame.columns = [str(column).strip() for column in frame.columns]

    required = list(config["batch_schema"]["required_columns"])
    optional = list(config["batch_schema"]["optional_columns"])

    missing_columns = [
        column for column in required if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            "Thiếu cột bắt buộc: " + ", ".join(missing_columns)
        )

    for column in optional:
        if column not in frame.columns:
            frame[column] = np.nan

    max_rows = int(config["max_batch_rows"])
    if len(frame) > max_rows:
        raise ValueError(
            f"File có {len(frame):,} dòng; giới hạn là {max_rows:,} dòng."
        )

    frame = frame.reset_index(drop=True)
    frame["source_row"] = np.arange(2, len(frame) + 2)

    text_columns = [
        "brand",
        "model",
        "bike_type",
        "capacity",
        "origin",
        "district",
        "title",
        "description",
    ]
    for column in text_columns:
        frame[column] = frame[column].fillna("").astype(str).str.strip()

    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["km"] = pd.to_numeric(frame["km"], errors="coerce")
    frame["listed_price_million"] = pd.to_numeric(
        frame["listed_price_million"], errors="coerce"
    )

    known_brands = set(input_options["brands"])
    known_bike_types = set(input_options["bike_types"])
    known_capacities = set(input_options["capacities"])
    known_origins = set(input_options["origins"])
    known_districts = set(input_options["districts"])
    models_by_brand = input_options["models_by_brand"]

    reference_year = int(config["reference_year"])

    error_messages: list[str] = []

    for _, row in frame.iterrows():
        errors: list[str] = []

        for column in required:
            value = row[column]
            if pd.isna(value) or str(value).strip() == "":
                errors.append(f"thiếu {column}")

        year = row["year"]
        if pd.isna(year):
            errors.append("year không phải số")
        elif year < 1979 or year > reference_year:
            errors.append(f"year phải trong 1979-{reference_year}")

        km = row["km"]
        if pd.notna(km) and km < 0:
            errors.append("km không được âm")

        listed_price = row["listed_price_million"]
        if pd.notna(listed_price) and listed_price <= 0:
            errors.append("listed_price_million phải lớn hơn 0")

        brand = row["brand"]
        model = row["model"]

        if brand and brand not in known_brands:
            errors.append("brand chưa có trong dữ liệu huấn luyện")
        elif brand and model:
            known_models = set(models_by_brand.get(brand, []))
            if model not in known_models:
                errors.append("model không thuộc brand đã chọn")

        if row["bike_type"] and row["bike_type"] not in known_bike_types:
            errors.append("bike_type chưa có trong dữ liệu huấn luyện")
        if row["capacity"] and row["capacity"] not in known_capacities:
            errors.append("capacity chưa có trong dữ liệu huấn luyện")
        if row["origin"] and row["origin"] not in known_origins:
            errors.append("origin chưa có trong dữ liệu huấn luyện")
        if row["district"] and row["district"] not in known_districts:
            errors.append("district chưa có trong dữ liệu huấn luyện")

        error_messages.append("; ".join(dict.fromkeys(errors)))

    frame["validation_error"] = error_messages
    invalid = frame[frame["validation_error"].ne("")].copy()
    valid = frame[frame["validation_error"].eq("")].copy()

    return valid, invalid


def run_batch_inference(
    data: pd.DataFrame,
    *,
    price_model: Any,
    isolation_preprocessor: Any,
    isolation_forest: Any,
    segment_statistics: pd.DataFrame,
    config: dict[str, Any],
    segment_rules: dict[str, Any],
    input_options: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run vectorized price prediction and anomaly scoring."""
    valid, invalid = validate_batch_dataframe(
        data,
        config=config,
        input_options=input_options,
    )

    if valid.empty:
        return pd.DataFrame(), invalid

    prepared = prepare_batch_features(
        valid,
        config=config,
        segment_rules=segment_rules,
    )

    model_features = config["model_features"]
    predicted_log = price_model.predict(prepared[model_features])
    prepared["predicted_price_million"] = np.maximum(
        0, np.expm1(predicted_log)
    )

    stats = segment_statistics.copy()
    prepared = prepared.merge(stats, on="segment", how="left")

    global_percentiles = config["global_percentiles"]
    fallback_values = {
        "residual_median": float(config["global_residual_median"]),
        "residual_scale": float(config["global_residual_scale"]),
        "p01": float(global_percentiles["p01"]),
        "p10": float(global_percentiles["p10"]),
        "p90": float(global_percentiles["p90"]),
        "p99": float(global_percentiles["p99"]),
    }

    prepared["used_global_fallback"] = prepared["residual_median"].isna()
    for column, fallback in fallback_values.items():
        prepared[column] = prepared[column].fillna(fallback)

    prepared["listed_price_million"] = pd.to_numeric(
        prepared["listed_price_million"], errors="coerce"
    )
    has_price = prepared["listed_price_million"].notna()

    numeric_output_columns = [
        "difference_million",
        "difference_percent",
        "absolute_gap_million",
        "residual_z",
        "isolation_raw_score",
        "isolation_normalized_score",
        "residual_contribution",
        "minmax_contribution",
        "common_range_contribution",
        "isolation_contribution",
        "anomaly_score",
        "distance_to_threshold",
        "component_flag_count",
    ]
    boolean_output_columns = [
        "flag_residual_z",
        "flag_p01_p99",
        "flag_p10_p90",
        "flag_isolation",
        "flag_large_price_gap",
        "is_anomaly",
        "needs_manual_review",
    ]
    text_output_columns = [
        "review_status",
        "anomaly_type",
        "reasons",
    ]

    for column in numeric_output_columns:
        prepared[column] = np.nan
    for column in boolean_output_columns:
        prepared[column] = pd.NA
    for column in text_output_columns:
        prepared[column] = None

    if has_price.any():
        priced = prepared.loc[has_price].copy()

        priced["difference_million"] = (
            priced["listed_price_million"]
            - priced["predicted_price_million"]
        )
        priced["absolute_gap_million"] = priced[
            "difference_million"
        ].abs()
        predicted_safe = priced["predicted_price_million"].clip(lower=1e-9)
        priced["difference_percent"] = (
            100.0 * priced["difference_million"] / predicted_safe
        )
        priced["flag_large_price_gap"] = (
            priced["difference_percent"]
            .abs()
            .ge(100.0 * MANUAL_REVIEW_MIN_RELATIVE_GAP)
            & priced["absolute_gap_million"].ge(
                MANUAL_REVIEW_MIN_ABSOLUTE_GAP_MILLION
            )
        )

        priced["residual_z"] = (
            priced["difference_million"] - priced["residual_median"]
        ) / priced["residual_scale"].clip(lower=1e-9)

        priced["flag_residual_z"] = priced["residual_z"].abs().ge(3)
        priced["flag_p01_p99"] = (
            priced["listed_price_million"].lt(priced["p01"])
            | priced["listed_price_million"].gt(priced["p99"])
        )
        priced["flag_p10_p90"] = (
            priced["listed_price_million"].lt(priced["p10"])
            | priced["listed_price_million"].gt(priced["p90"])
        )

        isolation_input = priced.copy()
        isolation_input["deployment_predicted_price"] = priced[
            "predicted_price_million"
        ]
        isolation_input["deployment_residual_z"] = priced["residual_z"]

        isolation_columns = (
            config["isolation_numeric_features"]
            + config["isolation_categorical_features"]
        )
        isolation_matrix = isolation_preprocessor.transform(
            isolation_input[isolation_columns]
        )

        priced["isolation_raw_score"] = (
            -isolation_forest.score_samples(isolation_matrix)
        )
        priced["flag_isolation"] = (
            isolation_forest.predict(isolation_matrix) == -1
        )

        s1 = (priced["residual_z"].abs() / 6).clip(0, 1)
        s2 = priced["flag_p01_p99"].astype(float)

        common_width = (priced["p90"] - priced["p10"]).clip(lower=1e-9)
        distance = np.maximum.reduce(
            [
                (priced["p10"] - priced["listed_price_million"]).to_numpy(),
                (priced["listed_price_million"] - priced["p90"]).to_numpy(),
                np.zeros(len(priced)),
            ]
        )
        s3 = pd.Series(
            np.clip(distance / common_width.to_numpy(), 0, 1),
            index=priced.index,
        )

        isolation_min = float(config["isolation_score_min"])
        isolation_max = float(config["isolation_score_max"])
        isolation_range = max(isolation_max - isolation_min, 1e-9)
        s4 = (
            (priced["isolation_raw_score"] - isolation_min)
            / isolation_range
        ).clip(0, 1)
        priced["isolation_normalized_score"] = s4

        weights = config["weights"]
        priced["residual_contribution"] = (
            100 * weights["residual"] * s1
        )
        priced["minmax_contribution"] = (
            100 * weights["minmax"] * s2
        )
        priced["common_range_contribution"] = (
            100 * weights["common_range"] * s3
        )
        priced["isolation_contribution"] = (
            100 * weights["isolation"] * s4
        )
        priced["anomaly_score"] = (
            priced["residual_contribution"]
            + priced["minmax_contribution"]
            + priced["common_range_contribution"]
            + priced["isolation_contribution"]
        )

        threshold = float(config["anomaly_threshold"])
        priced["is_anomaly"] = priced["anomaly_score"].ge(threshold)
        priced["component_flag_count"] = (
            priced["flag_residual_z"].astype(int)
            + priced["flag_p01_p99"].astype(int)
            + priced["flag_p10_p90"].astype(int)
            + priced["flag_isolation"].astype(int)
        )
        priced["distance_to_threshold"] = (
            threshold - priced["anomaly_score"]
        )
        priced["needs_manual_review"] = (
            ~priced["is_anomaly"]
            & (
                priced["distance_to_threshold"].le(MANUAL_REVIEW_MARGIN)
                | priced["component_flag_count"].ge(MANUAL_REVIEW_MIN_FLAGS)
                | priced["flag_large_price_gap"]
            )
        )

        priced["review_status"] = np.select(
            [
                priced["is_anomaly"],
                priced["needs_manual_review"],
            ],
            [
                "Bất thường",
                "Cần kiểm tra thủ công",
            ],
            default="Bình thường",
        )

        priced["anomaly_type"] = np.select(
            [
                priced["review_status"].eq("Bình thường"),
                priced["difference_million"].lt(0),
            ],
            [
                "Bình thường",
                "Có xu hướng quá rẻ",
            ],
            default="Có xu hướng quá đắt",
        )

        def build_reasons(row: pd.Series) -> str:
            reasons: list[str] = []
            if bool(row["flag_residual_z"]):
                reasons.append(
                    f"Residual-Z={row['residual_z']:.2f} (|z|>=3)"
                )
            if bool(row["flag_p01_p99"]):
                reasons.append(
                    "Giá ngoài P1-P99 "
                    f"({row['p01']:.1f}-{row['p99']:.1f})"
                )
            if bool(row["flag_p10_p90"]):
                reasons.append(
                    "Giá ngoài P10-P90 "
                    f"({row['p10']:.1f}-{row['p90']:.1f})"
                )
            if bool(row["flag_isolation"]):
                reasons.append("Isolation Forest đánh dấu")
            if bool(row["flag_large_price_gap"]):
                reasons.append(
                    f"Chênh giá {abs(row['difference_percent']):.1f}% và "
                    f"{row['absolute_gap_million']:.1f} triệu"
                )
            return "; ".join(reasons) if reasons else "Không có cờ mạnh"

        priced["reasons"] = priced.apply(build_reasons, axis=1)

        assign_columns = (
            numeric_output_columns
            + boolean_output_columns
            + text_output_columns
        )
        prepared.loc[priced.index, assign_columns] = priced[assign_columns]

    prepared.loc[~has_price, "review_status"] = "Chỉ dự đoán"
    prepared.loc[~has_price, "anomaly_type"] = "Chỉ dự đoán"
    prepared.loc[~has_price, "reasons"] = (
        "Không có giá đăng để chấm anomaly"
    )

    result_columns = [
        "source_row",
        "brand",
        "model",
        "year",
        "km",
        "bike_type",
        "capacity",
        "origin",
        "district",
        "title",
        "description",
        "listed_price_million",
        "predicted_price_million",
        "difference_million",
        "difference_percent",
        "absolute_gap_million",
        "segment",
        "p01",
        "p10",
        "p90",
        "p99",
        "residual_z",
        "flag_residual_z",
        "flag_p01_p99",
        "flag_p10_p90",
        "flag_isolation",
        "flag_large_price_gap",
        "isolation_normalized_score",
        "residual_contribution",
        "minmax_contribution",
        "common_range_contribution",
        "isolation_contribution",
        "anomaly_score",
        "distance_to_threshold",
        "component_flag_count",
        "is_anomaly",
        "needs_manual_review",
        "review_status",
        "anomaly_type",
        "reasons",
        "used_global_fallback",
    ]

    return prepared[result_columns].sort_values("source_row"), invalid
