from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


IMPORT_PATTERN = (
    r"nhập\s*(?:ý|khẩu|thái)?"
    r"|hqcn"
    r"|hải\s*quan"
    r"|chính\s*ngạch"
    r"|xe\s*thùng"
    r"|đập\s*thùng"
)

ABS_PATTERN = r"(?<!\w)abs(?!\w)"

COLLECTIBLE_PATTERN = (
    r"limited"
    r"|sưu\s*tầm"
    r"|nguyên\s*zin"
    r"|full\s*zin"
    r"|đập\s*thùng"
    r"|(?<!\w)yaz(?!\w)"
    r"|xipo"
    r"|vespa\s*946"
)


def get_year_band(year: int) -> str:
    """Map registration year to the same bands used during training."""
    if year <= 1999:
        return "Trước 2000"
    if year <= 2009:
        return "2000-2009"
    if year <= 2014:
        return "2010-2014"
    if year <= 2019:
        return "2015-2019"
    return "2020-2025"


def assign_segment(
    *,
    brand: str,
    model: str,
    year: int,
    bike_type: str,
    capacity: str,
    rules: dict[str, Any],
) -> str:
    """Assign one new vehicle to a learned deployment segment."""
    year_band = get_year_band(int(year))

    detail_key = f"{brand} | {model} | {year_band}"
    brand_type_key = f"{brand} | {bike_type}"
    type_capacity_key = f"{bike_type} | {capacity}"

    if detail_key in rules["valid_detail"]:
        return detail_key
    if brand_type_key in rules["valid_brand_type"]:
        return brand_type_key
    if type_capacity_key in rules["valid_type_capacity"]:
        return type_capacity_key
    return "Nhóm hiếm"


def _contains_pattern(text: str, pattern: str) -> int:
    return int(bool(re.search(pattern, text, flags=re.IGNORECASE)))


def prepare_vehicle_input(
    *,
    brand: str,
    model: str,
    year: int,
    km: float | None,
    bike_type: str,
    capacity: str,
    origin: str,
    district: str,
    title: str,
    description: str,
    config: dict[str, Any],
    segment_rules: dict[str, Any],
) -> pd.DataFrame:
    """Create a one-row feature frame with the exact deployment schema."""
    reference_year = int(config["reference_year"])
    km_limit = float(config["km_suspicious_limit"])

    year = int(year)
    if year < 1979 or year > reference_year:
        raise ValueError(
            f"Năm đăng ký phải nằm trong khoảng 1979-{reference_year}."
        )

    age = reference_year - year

    km_is_missing = km is None or pd.isna(km)
    if km_is_missing:
        km_value = np.nan
        km_suspicious = 0
        km_clean = np.nan
    else:
        km_value = float(km)
        if km_value < 0:
            raise ValueError("Số kilomet không được âm.")
        km_suspicious = int(km_value >= km_limit)
        km_clean = np.nan if km_suspicious else km_value

    log_km = np.nan if pd.isna(km_clean) else np.log1p(km_clean)
    km_per_year = (
        np.nan if pd.isna(km_clean) else km_clean / max(age, 1)
    )

    title = str(title or "").strip()
    description = str(description or "").strip()
    listing_text = f"{title} {description}".lower()

    segment = assign_segment(
        brand=brand,
        model=model,
        year=year,
        bike_type=bike_type,
        capacity=capacity,
        rules=segment_rules,
    )

    row = pd.DataFrame(
        {
            "age": [age],
            "age_squared": [age**2],
            "km_clean": [km_clean],
            "log_km": [log_km],
            "km_per_year": [km_per_year],
            "km_suspicious": [km_suspicious],
            "title_length": [len(title)],
            "description_length": [len(description)],
            "is_import_keyword": [
                _contains_pattern(listing_text, IMPORT_PATTERN)
            ],
            "is_abs_keyword": [
                _contains_pattern(listing_text, ABS_PATTERN)
            ],
            "is_collectible_keyword": [
                _contains_pattern(listing_text, COLLECTIBLE_PATTERN)
            ],
            "is_high_capacity": [
                int(capacity == "Trên 175 cc")
            ],
            "brand": [brand],
            "model": [model],
            "brand_model": [f"{brand} | {model}"],
            "bike_type": [bike_type],
            "capacity": [capacity],
            "origin": [origin],
            "district": [district],
            "segment": [segment],
        }
    )

    return row[config["model_features"]]


def prepare_batch_features(
    data: pd.DataFrame,
    *,
    config: dict[str, Any],
    segment_rules: dict[str, Any],
) -> pd.DataFrame:
    """Vectorized feature engineering for valid batch rows."""
    frame = data.copy()
    reference_year = int(config["reference_year"])
    km_limit = float(config["km_suspicious_limit"])

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

    frame["age"] = reference_year - frame["year"]
    frame["age_squared"] = frame["age"] ** 2

    frame["km_suspicious"] = (
        frame["km"].notna() & (frame["km"] >= km_limit)
    ).astype(int)

    frame["km_clean"] = frame["km"].mask(
        frame["km_suspicious"].eq(1)
    )
    frame["log_km"] = np.log1p(frame["km_clean"].clip(lower=0))
    frame["km_per_year"] = (
        frame["km_clean"] / frame["age"].clip(lower=1)
    )

    frame["title_length"] = frame["title"].str.len()
    frame["description_length"] = frame["description"].str.len()

    listing_text = (
        frame["title"] + " " + frame["description"]
    ).str.lower()

    frame["is_import_keyword"] = listing_text.str.contains(
        IMPORT_PATTERN, regex=True, na=False
    ).astype(int)
    frame["is_abs_keyword"] = listing_text.str.contains(
        ABS_PATTERN, regex=True, na=False
    ).astype(int)
    frame["is_collectible_keyword"] = listing_text.str.contains(
        COLLECTIBLE_PATTERN, regex=True, na=False
    ).astype(int)
    frame["is_high_capacity"] = frame["capacity"].eq(
        "Trên 175 cc"
    ).astype(int)

    frame["brand_model"] = frame["brand"] + " | " + frame["model"]

    year = frame["year"]
    frame["year_band"] = np.select(
        [
            year <= 1999,
            year <= 2009,
            year <= 2014,
            year <= 2019,
        ],
        [
            "Trước 2000",
            "2000-2009",
            "2010-2014",
            "2015-2019",
        ],
        default="2020-2025",
    )

    detail_key = (
        frame["brand"] + " | " + frame["model"] + " | " + frame["year_band"]
    )
    brand_type_key = frame["brand"] + " | " + frame["bike_type"]
    type_capacity_key = frame["bike_type"] + " | " + frame["capacity"]

    valid_detail = set(segment_rules["valid_detail"])
    valid_brand_type = set(segment_rules["valid_brand_type"])
    valid_type_capacity = set(segment_rules["valid_type_capacity"])

    frame["segment"] = "Nhóm hiếm"

    detail_mask = detail_key.isin(valid_detail)
    frame.loc[detail_mask, "segment"] = detail_key[detail_mask]

    brand_type_mask = (
        ~detail_mask & brand_type_key.isin(valid_brand_type)
    )
    frame.loc[brand_type_mask, "segment"] = brand_type_key[
        brand_type_mask
    ]

    type_capacity_mask = (
        ~detail_mask
        & ~brand_type_mask
        & type_capacity_key.isin(valid_type_capacity)
    )
    frame.loc[type_capacity_mask, "segment"] = type_capacity_key[
        type_capacity_mask
    ]

    return frame
