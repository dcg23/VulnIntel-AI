# VulnIntel-AI retrospectively classifies current CISA KEV membership.
# The target is 1 when an NVD CVE ID appears in the downloaded KEV snapshot.
# Predictors come only from NVD records, and text is represented with TF-IDF.

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


# Reproducible project settings.
RANDOM_SEED = 42
DEFAULT_FIRST_YEAR = 1999
DEFAULT_LAST_YEAR = datetime.now(timezone.utc).year
NVD_WINDOW_DAYS = 120
NVD_PAGE_SIZE = 2_000
REQUEST_TIMEOUT_SECONDS = 90
MAX_REQUEST_ATTEMPTS = 5
TFIDF_MAX_FEATURES = 10_000
DEFAULT_TUNING_MAX_ROWS = 75_000
DEFAULT_SEARCH_JOBS = 1

# Use one consistent, color-blind-friendly palette across report figures.
REPORT_COLORS = {
    "blue": "#2F6690",
    "orange": "#F28E2B",
    "green": "#59A14F",
    "red": "#E15759",
    "purple": "#8F63B8",
    "gray": "#6B7280",
    "light_gray": "#E5E7EB",
}

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        "axes.edgecolor": "#374151",
        "axes.labelcolor": "#111827",
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "font.size": 10,
        "legend.frameon": False,
        "grid.color": "#D1D5DB",
        "grid.alpha": 0.55,
        "savefig.facecolor": "white",
    }
)

NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
CISA_KEV_CSV_URL = (
    "https://www.cisa.gov/sites/default/files/csv/"
    "known_exploited_vulnerabilities.csv"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
IMPORTANCE_DIR = PROJECT_ROOT / "outputs" / "feature_importance"

CISA_CACHE_FILE = RAW_DATA_DIR / "cisa_kev.csv"
CLEAN_DATA_FILE = PROCESSED_DATA_DIR / "vulnintel_model_dataset.csv"
TUNING_RESULTS_FILE = RESULTS_DIR / "hyperparameter_tuning_results.csv"
TUNING_CANDIDATES_FILE = RESULTS_DIR / "hyperparameter_search_candidates.csv"
MODEL_COMPARISON_FILE = RESULTS_DIR / "model_comparison.csv"
MODEL_SELECTION_FILE = RESULTS_DIR / "model_selection_summary.txt"
TEMPORAL_SPLIT_FILE = RESULTS_DIR / "temporal_split_summary.csv"
TARGET_TEMPORAL_AUDIT_FILE = RESULTS_DIR / "target_temporal_audit.csv"

TEXT_FEATURE = "description_text"
NUMERIC_FEATURES = ["base_score", "exploitability_score", "impact_score"]
CATEGORICAL_FEATURES = [
    "cvss_version",
    "base_severity",
    "attack_vector",
    "attack_complexity",
    "privileges_required",
    "user_interaction",
    "scope",
    "confidentiality_impact",
    "integrity_impact",
    "availability_impact",
    "primary_cwe",
]
MODEL_INPUT_COLUMNS = [TEXT_FEATURE, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
TARGET_COLUMN = "kev_exploited"

NVD_OUTPUT_COLUMNS = [
    "cve_id",
    "description_text",
    "published_date",
    "last_modified_date",
    *NUMERIC_FEATURES,
    *CATEGORICAL_FEATURES,
]


# Create the documented data, result, figure, and importance folders.
def create_output_directories() -> None:
    for output_directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        RESULTS_DIR,
        FIGURES_DIR,
        IMPORTANCE_DIR,
    ):
        output_directory.mkdir(parents=True, exist_ok=True)


# Normalize CVE IDs so lowercase or extra spaces cannot break the KEV join.
def normalise_cve_id(cve_value: Any) -> str | None:
    if pd.isna(cve_value):
        return None
    normalized_id = str(cve_value).strip().upper()
    return normalized_id or None


# Use the English NVD description as the TF-IDF text input.
def select_english_description(description_items: Any) -> str:
    if not isinstance(description_items, list):
        return ""
    for description_item in description_items:
        if description_item.get("lang") == "en":
            return str(description_item.get("value", "")).strip()
    return ""


# Keep the first valid CWE as a simple structured category.
def select_primary_cwe(weakness_items: Any) -> str:
    if not isinstance(weakness_items, list):
        return "UNKNOWN"
    for weakness_group in weakness_items:
        for weakness_description in weakness_group.get("description", []):
            cwe_value = str(weakness_description.get("value", "")).strip().upper()
            if cwe_value.startswith("CWE-"):
                return cwe_value
    return "UNKNOWN"


# Supply explicit missing values when a CVE has no supported CVSS record.
def empty_cvss_record() -> dict[str, Any]:
    return {
        "cvss_version": "UNKNOWN",
        "base_score": np.nan,
        "exploitability_score": np.nan,
        "impact_score": np.nan,
        "base_severity": "UNKNOWN",
        "attack_vector": "UNKNOWN",
        "attack_complexity": "UNKNOWN",
        "privileges_required": "UNKNOWN",
        "user_interaction": "UNKNOWN",
        "scope": "UNKNOWN",
        "confidentiality_impact": "UNKNOWN",
        "integrity_impact": "UNKNOWN",
        "availability_impact": "UNKNOWN",
    }


# Extract one primary CVSS record so every CVE has one consistent feature row.
def extract_cvss_record(cve_object: dict[str, Any]) -> dict[str, Any]:
    metric_groups = cve_object.get("metrics", {})
    candidates = (
        ("cvssMetricV31", "3.1"),
        ("cvssMetricV30", "3.0"),
        ("cvssMetricV40", "4.0"),
        ("cvssMetricV2", "2.0"),
    )
    for metric_key, fallback_version in candidates:
        metric_records = metric_groups.get(metric_key, [])
        if not metric_records:
            continue
        # Prefer an NVD primary record when one is marked as such.
        chosen_metric = next(
            (record for record in metric_records if record.get("type") == "Primary"),
            metric_records[0],
        )
        cvss_values = chosen_metric.get("cvssData", {})
        return {
            "cvss_version": str(cvss_values.get("version", fallback_version)),
            "base_score": cvss_values.get("baseScore"),
            "exploitability_score": chosen_metric.get("exploitabilityScore"),
            "impact_score": chosen_metric.get("impactScore"),
            "base_severity": cvss_values.get(
                "baseSeverity", chosen_metric.get("baseSeverity", "UNKNOWN")
            ),
            "attack_vector": cvss_values.get("attackVector", "UNKNOWN"),
            "attack_complexity": cvss_values.get("attackComplexity", "UNKNOWN"),
            "privileges_required": cvss_values.get("privilegesRequired", "UNKNOWN"),
            "user_interaction": cvss_values.get("userInteraction", "UNKNOWN"),
            "scope": cvss_values.get("scope", "UNKNOWN"),
            "confidentiality_impact": cvss_values.get(
                "confidentialityImpact", "UNKNOWN"
            ),
            "integrity_impact": cvss_values.get("integrityImpact", "UNKNOWN"),
            "availability_impact": cvss_values.get(
                "availabilityImpact", "UNKNOWN"
            ),
        }
    return empty_cvss_record()


# Flatten the nested NVD response into the fields used by cleaning and modelling.
def parse_nvd_vulnerability(vulnerability_item: dict[str, Any]) -> dict[str, Any]:
    cve_object = vulnerability_item.get("cve", {})
    return {
        "cve_id": normalise_cve_id(cve_object.get("id")),
        "description_text": select_english_description(cve_object.get("descriptions")),
        "published_date": cve_object.get("published"),
        "last_modified_date": cve_object.get("lastModified"),
        "primary_cwe": select_primary_cwe(cve_object.get("weaknesses")),
        **extract_cvss_record(cve_object),
    }


# Retry temporary HTTP failures, but stop after a fixed number of attempts.
def request_with_backoff(
    request_url: str,
    *,
    request_parameters: dict[str, Any] | None = None,
    request_headers: dict[str, str] | None = None,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt_number in range(1, MAX_REQUEST_ATTEMPTS + 1):
        try:
            response = requests.get(
                request_url,
                params=request_parameters,
                headers=request_headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return response
        except requests.RequestException as request_error:
            last_error = request_error
            if attempt_number == MAX_REQUEST_ATTEMPTS:
                break
            retry_after = None
            if getattr(request_error, "response", None) is not None:
                retry_after = request_error.response.headers.get("Retry-After")
            try:
                wait_seconds = float(retry_after) if retry_after else 2 ** attempt_number
            except ValueError:
                wait_seconds = 2 ** attempt_number
            wait_seconds = min(wait_seconds, 60.0)
            print(
                f"Request attempt {attempt_number} failed: {request_error}. "
                f"Retrying in {wait_seconds:.1f} seconds."
            )
            time.sleep(wait_seconds)
    raise RuntimeError(
        f"Download failed after {MAX_REQUEST_ATTEMPTS} attempts: {last_error}"
    ) from last_error


# Read the optional API key from the environment instead of hard-coding a secret.
def nvd_request_headers() -> dict[str, str]:
    request_headers = {"User-Agent": "VulnIntel-AI-Academic-Research/1.0"}
    nvd_api_key = os.getenv("NVD_API_KEY", "").strip()
    if nvd_api_key:
        request_headers["apiKey"] = nvd_api_key
    return request_headers


# Download every paginated result inside one NVD date window.
def fetch_nvd_window(
    window_start: datetime,
    window_end: datetime,
    maximum_records: int | None = None,
) -> list[dict[str, Any]]:
    window_rows: list[dict[str, Any]] = []
    page_start_index = 0
    formatted_start = window_start.strftime("%Y-%m-%dT00:00:00.000Z")
    formatted_end = window_end.strftime("%Y-%m-%dT23:59:59.999Z")

    while maximum_records is None or len(window_rows) < maximum_records:
        remaining_records = (
            NVD_PAGE_SIZE
            if maximum_records is None
            else min(NVD_PAGE_SIZE, maximum_records - len(window_rows))
        )
        request_parameters = {
            "pubStartDate": formatted_start,
            "pubEndDate": formatted_end,
            "resultsPerPage": remaining_records,
            "startIndex": page_start_index,
            "noRejected": "",
        }
        response = request_with_backoff(
            NVD_API_URL,
            request_parameters=request_parameters,
            request_headers=nvd_request_headers(),
        )
        try:
            response_payload = response.json()
        except requests.JSONDecodeError as json_error:
            raise RuntimeError("NVD returned a non-JSON response.") from json_error

        vulnerability_items = response_payload.get("vulnerabilities", [])
        total_results = int(response_payload.get("totalResults", 0))
        window_rows.extend(parse_nvd_vulnerability(item) for item in vulnerability_items)
        page_start_index += len(vulnerability_items)
        print(
            f"NVD {window_start.date()} to {window_end.date()}: "
            f"{page_start_index:,}/{total_results:,} records"
        )
        # Apply the delay after every page, including a one-page date window.
        time.sleep(0.7 if os.getenv("NVD_API_KEY", "").strip() else 6.0)
        if not vulnerability_items or page_start_index >= total_results:
            break
    return window_rows


# Split the requested years into API-safe windows and save a reusable CSV cache.
def download_nvd_dataset(
    first_year: int,
    last_year: int,
    cache_file: Path,
    maximum_records: int | None = None,
) -> pd.DataFrame:
    if first_year > last_year:
        raise ValueError("first_year must be less than or equal to last_year.")
    all_nvd_rows: list[dict[str, Any]] = []
    overall_end = min(
        datetime(last_year, 12, 31, tzinfo=timezone.utc),
        datetime.now(timezone.utc),
    )
    window_start = datetime(first_year, 1, 1, tzinfo=timezone.utc)
    while window_start <= overall_end:
        window_end = min(
            window_start + timedelta(days=NVD_WINDOW_DAYS - 1), overall_end
        )
        remaining = (
            None
            if maximum_records is None
            else maximum_records - len(all_nvd_rows)
        )
        if remaining is not None and remaining <= 0:
            break
        all_nvd_rows.extend(fetch_nvd_window(window_start, window_end, remaining))
        window_start = window_end + timedelta(days=1)

    nvd_dataset = pd.DataFrame(all_nvd_rows, columns=NVD_OUTPUT_COLUMNS)
    if nvd_dataset.empty:
        raise RuntimeError("NVD download returned no CVE records for the chosen period.")
    nvd_dataset["cve_id"] = nvd_dataset["cve_id"].map(normalise_cve_id)
    nvd_dataset.to_csv(cache_file, index=False)
    print(f"Saved {len(nvd_dataset):,} NVD records to {cache_file}")
    return nvd_dataset


# Download the official CISA KEV catalog used only to construct the target.
def download_cisa_kev_dataset(cache_file: Path) -> pd.DataFrame:
    response = request_with_backoff(CISA_KEV_CSV_URL)
    cache_file.write_bytes(response.content)
    kev_dataset = pd.read_csv(cache_file)
    if "cveID" not in kev_dataset.columns:
        raise ValueError("CISA KEV CSV does not contain the expected cveID column.")
    kev_dataset["cveID"] = kev_dataset["cveID"].map(normalise_cve_id)
    kev_dataset = kev_dataset.dropna(subset=["cveID"]).drop_duplicates("cveID")
    kev_dataset.to_csv(cache_file, index=False)
    print(f"Saved {len(kev_dataset):,} CISA KEV records to {cache_file}")
    return kev_dataset


# Reuse local CSV files unless the user requests a fresh download.
def load_or_download_data(
    *,
    first_year: int,
    last_year: int,
    refresh: bool,
    development_mode: bool,
    development_max_records: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_suffix = (
        f"_development_{development_max_records}"
        if development_mode
        else ""
    )
    nvd_cache = RAW_DATA_DIR / (
        f"nvd_cves_{first_year}_{last_year}{cache_suffix}.csv"
    )
    if refresh or not nvd_cache.exists():
        nvd_dataset = download_nvd_dataset(
            first_year,
            last_year,
            nvd_cache,
            development_max_records if development_mode else None,
        )
    else:
        # CVSS versions may look numeric or textual, so load them consistently.
        nvd_dataset = pd.read_csv(nvd_cache, dtype={"cvss_version": "string"})
        print(f"Loaded cached NVD data from {nvd_cache}")

    if refresh or not CISA_CACHE_FILE.exists():
        kev_dataset = download_cisa_kev_dataset(CISA_CACHE_FILE)
    else:
        kev_dataset = pd.read_csv(CISA_CACHE_FILE)
        print(f"Loaded cached CISA KEV data from {CISA_CACHE_FILE}")
    return nvd_dataset, kev_dataset


# Create the target only from normalized CVE-ID membership in CISA KEV.
def build_kev_target_dataset(
    nvd_dataset: pd.DataFrame, kev_dataset: pd.DataFrame
) -> pd.DataFrame:
    if "cveID" in kev_dataset.columns:
        kev_id_column = "cveID"
    elif "cve_id" in kev_dataset.columns:
        kev_id_column = "cve_id"
    else:
        raise ValueError("CISA data needs a cveID or cve_id column.")
    labelled_dataset = nvd_dataset.copy()
    labelled_dataset["cve_id"] = labelled_dataset["cve_id"].map(normalise_cve_id)
    kev_ids = set(kev_dataset[kev_id_column].map(normalise_cve_id).dropna())
    labelled_dataset[TARGET_COLUMN] = labelled_dataset["cve_id"].isin(kev_ids).astype(int)
    return labelled_dataset


# Quantify why current-snapshot membership is retrospective rather than an
# as-of-publication prediction target. KEV dateAdded is used only for this audit.
def save_target_temporal_audit(
    labelled_dataset: pd.DataFrame, kev_dataset: pd.DataFrame
) -> None:
    if "dateAdded" not in kev_dataset.columns:
        raise ValueError("CISA data needs dateAdded for the temporal target audit.")
    kev_id_column = "cveID" if "cveID" in kev_dataset.columns else "cve_id"
    cve_dates = labelled_dataset[["cve_id", "published_date", TARGET_COLUMN]].copy()
    cve_dates["published_date"] = pd.to_datetime(
        cve_dates["published_date"], errors="coerce", utc=True
    )
    kev_dates = kev_dataset[[kev_id_column, "dateAdded"]].copy()
    kev_dates["cve_id"] = kev_dates[kev_id_column].map(normalise_cve_id)
    kev_dates["kev_date_added"] = pd.to_datetime(
        kev_dates["dateAdded"], errors="coerce", utc=True
    )
    positive_dates = cve_dates[cve_dates[TARGET_COLUMN].eq(1)].merge(
        kev_dates[["cve_id", "kev_date_added"]], on="cve_id", how="left"
    )
    valid_pairs = positive_dates.dropna(subset=["published_date", "kev_date_added"])
    added_after_publication = valid_pairs["kev_date_added"].gt(
        valid_pairs["published_date"]
    )
    pd.DataFrame(
        [
            {
                "target_definition": "membership_in_downloaded_kev_snapshot",
                "interpretation": "retrospective_classification_not_future_exploitation_prediction",
                "nvd_rows": len(labelled_dataset),
                "kev_positive_rows": len(positive_dates),
                "positive_rows_with_valid_date_pair": len(valid_pairs),
                "positive_rows_added_to_kev_after_cve_publication": int(
                    added_after_publication.sum()
                ),
                "share_of_valid_positive_pairs_added_after_publication": (
                    added_after_publication.mean() if len(valid_pairs) else np.nan
                ),
                "latest_kev_date_added_in_snapshot": kev_dates["kev_date_added"].max(),
            }
        ]
    ).to_csv(TARGET_TEMPORAL_AUDIT_FILE, index=False)


# Save EDA evidence before cleaning changes missing values or duplicate counts.
def save_eda_outputs(labelled_dataset: pd.DataFrame) -> None:
    dimensions = pd.DataFrame(
        [{"row_count": len(labelled_dataset), "column_count": len(labelled_dataset.columns)}]
    )
    dimensions.to_csv(RESULTS_DIR / "dataset_dimensions.csv", index=False)

    data_types = pd.DataFrame(
        {
            "column": labelled_dataset.columns,
            "data_type": labelled_dataset.dtypes.astype(str).values,
        }
    )
    data_types.to_csv(RESULTS_DIR / "data_types.csv", index=False)

    duplicate_rows = labelled_dataset.loc[
        labelled_dataset["cve_id"].duplicated(keep=False), ["cve_id"]
    ].sort_values("cve_id")
    duplicate_rows.to_csv(RESULTS_DIR / "duplicate_cve_ids.csv", index=False)

    missing_counts = labelled_dataset.isna().sum()
    empty_string_counts = pd.Series(0, index=labelled_dataset.columns, dtype=int)
    for column_name in labelled_dataset.select_dtypes(
        include=["object", "string"]
    ).columns:
        empty_string_counts[column_name] = int(
            labelled_dataset[column_name]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
            - missing_counts[column_name]
        )
    missing_or_empty_counts = missing_counts + empty_string_counts
    missing_table = pd.DataFrame(
        {
            "column": missing_counts.index,
            "missing_count": missing_counts.values,
            "missing_percentage": (missing_counts.values / max(len(labelled_dataset), 1))
            * 100,
            "empty_string_count": empty_string_counts.values,
            "missing_or_empty_count": missing_or_empty_counts.values,
            "missing_or_empty_percentage": (
                missing_or_empty_counts.values / max(len(labelled_dataset), 1)
            )
            * 100,
        }
    ).sort_values("missing_or_empty_percentage", ascending=False)
    missing_table.to_csv(RESULTS_DIR / "missing_values.csv", index=False)

    numeric_values = labelled_dataset[NUMERIC_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    numeric_summary = (
        numeric_values.describe()
        .transpose()
        .rename_axis("numeric_feature")
        .reset_index()
    )
    numeric_summary.to_csv(RESULTS_DIR / "cvss_numeric_summary.csv", index=False)

    publication_dates = pd.to_datetime(
        labelled_dataset["published_date"], errors="coerce", utc=True
    )
    publication_coverage = pd.DataFrame(
        [
            {
                "earliest_published_date": publication_dates.min(),
                "latest_published_date": publication_dates.max(),
                "valid_date_count": int(publication_dates.notna().sum()),
                "missing_or_invalid_date_count": int(publication_dates.isna().sum()),
            }
        ]
    )
    publication_coverage.to_csv(
        RESULTS_DIR / "publication_date_coverage.csv", index=False
    )
    print(f"Dataset dimensions: {dimensions.iloc[0]['row_count']:,} rows x "
          f"{dimensions.iloc[0]['column_count']:,} columns")
    print(f"Duplicate CVE IDs before cleaning: {len(duplicate_rows):,}")


# Clean impossible or missing values without deleting legitimate rare KEV cases.
def clean_modelling_dataset(labelled_dataset: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"cve_id", "published_date", TARGET_COLUMN, *MODEL_INPUT_COLUMNS}
    missing_required = sorted(required_columns.difference(labelled_dataset.columns))
    if missing_required:
        raise ValueError(f"Required NVD columns are missing: {missing_required}")

    cleaned_dataset = labelled_dataset.copy()
    original_rows = len(cleaned_dataset)
    cleaned_dataset["cve_id"] = cleaned_dataset["cve_id"].map(normalise_cve_id)
    missing_id_count = int(cleaned_dataset["cve_id"].isna().sum())
    cleaned_dataset = cleaned_dataset.dropna(subset=["cve_id"])
    duplicate_count = int(cleaned_dataset["cve_id"].duplicated().sum())
    cleaned_dataset = cleaned_dataset.drop_duplicates("cve_id", keep="last")

    descriptions = cleaned_dataset[TEXT_FEATURE].fillna("").astype(str).str.strip()
    empty_description_count = int(descriptions.eq("").sum())
    cleaned_dataset[TEXT_FEATURE] = descriptions.mask(
        descriptions.eq(""), "unknown vulnerability description"
    )

    invalid_numeric_counts: dict[str, int] = {}
    for numeric_feature in NUMERIC_FEATURES:
        numeric_values = pd.to_numeric(cleaned_dataset[numeric_feature], errors="coerce")
        invalid_mask = numeric_values.notna() & ~numeric_values.between(0, 10)
        invalid_numeric_counts[numeric_feature] = int(invalid_mask.sum())
        cleaned_dataset[numeric_feature] = numeric_values.mask(invalid_mask)

    for categorical_feature in CATEGORICAL_FEATURES:
        category_values = (
            cleaned_dataset[categorical_feature]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
            .str.upper()
        )
        cleaned_dataset[categorical_feature] = category_values.mask(
            category_values.eq(""), "UNKNOWN"
        )

    cleaned_dataset["published_date"] = pd.to_datetime(
        cleaned_dataset["published_date"], errors="coerce", utc=True
    )
    if "last_modified_date" in cleaned_dataset:
        cleaned_dataset["last_modified_date"] = pd.to_datetime(
            cleaned_dataset["last_modified_date"], errors="coerce", utc=True
        )
    cleaned_dataset[TARGET_COLUMN] = cleaned_dataset[TARGET_COLUMN].astype(int)
    # Establish one canonical chronological row order. Missing dates remain in
    # the cleaned export for auditability but are excluded from temporal splits.
    cleaned_dataset = cleaned_dataset.sort_values(
        ["published_date", "cve_id"], kind="stable", na_position="last"
    ).reset_index(drop=True)

    cleaning_report = pd.DataFrame(
        [
            {"check": "rows_before_cleaning", "count": original_rows},
            {"check": "missing_cve_ids_removed", "count": missing_id_count},
            {"check": "duplicate_cve_ids_removed", "count": duplicate_count},
            {"check": "empty_descriptions_replaced", "count": empty_description_count},
            *[
                {"check": f"invalid_{feature}_set_missing", "count": count}
                for feature, count in invalid_numeric_counts.items()
            ],
            {"check": "rows_after_cleaning", "count": len(cleaned_dataset)},
        ]
    )
    cleaning_report.to_csv(RESULTS_DIR / "cleaning_report.csv", index=False)
    cleaned_dataset.to_csv(CLEAN_DATA_FILE, index=False)
    return cleaned_dataset


# Record counts and percentages so model results can be interpreted under imbalance.
def save_class_balance(cleaned_dataset: pd.DataFrame) -> pd.DataFrame:
    class_counts = cleaned_dataset[TARGET_COLUMN].value_counts().sort_index()
    class_balance = pd.DataFrame(
        {
            "target_value": [0, 1],
            "class_label": ["Non-KEV", "KEV exploited"],
            "count": [int(class_counts.get(0, 0)), int(class_counts.get(1, 0))],
        }
    )
    class_balance["percentage"] = (
        class_balance["count"] / max(class_balance["count"].sum(), 1) * 100
    )
    class_balance.to_csv(RESULTS_DIR / "class_balance.csv", index=False)
    print("\nClass balance")
    print(class_balance.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    minority_count = int(class_balance["count"].min())
    majority_count = int(class_balance["count"].max())
    imbalance_ratio = (
        majority_count / minority_count if minority_count else float("inf")
    )
    balance_description = (
        "imbalanced" if imbalance_ratio >= 1.5 else "relatively balanced"
    )
    imbalance_text = (
        f"The target is {balance_description}: majority/minority ratio = "
        f"{imbalance_ratio:.3f}. Imbalance handling is fitted on training data only."
    )
    (RESULTS_DIR / "class_imbalance_assessment.txt").write_text(
        imbalance_text + "\n", encoding="utf-8"
    )
    print(imbalance_text)

    figure, (count_axis, share_axis) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    class_colors = [REPORT_COLORS["blue"], REPORT_COLORS["red"]]

    # A logarithmic count axis keeps the rare KEV class visible.
    count_bars = count_axis.barh(
        class_balance["class_label"],
        class_balance["count"],
        color=class_colors,
        edgecolor="white",
        height=0.58,
    )
    if minority_count > 0 and imbalance_ratio >= 20:
        count_axis.set_xscale("log")
        count_axis.set_xlim(0.8, majority_count * 2.2)
        count_axis.set_xlabel("CVE count — logarithmic scale")
    else:
        count_axis.set_xlabel("CVE count")
    count_axis.set_title("Class counts")
    count_axis.grid(axis="x")
    count_axis.grid(axis="y", visible=False)
    for bar, count_value, percentage_value in zip(
        count_bars, class_balance["count"], class_balance["percentage"]
    ):
        count_axis.text(
            max(float(count_value) * 1.08, 1.05),
            bar.get_y() + bar.get_height() / 2,
            f"{int(count_value):,}  ({percentage_value:.2f}%)",
            va="center",
            fontweight="bold",
        )

    # The stacked percentage bar makes the imbalance proportion explicit.
    running_left = 0.0
    for class_row, class_color in zip(class_balance.itertuples(), class_colors):
        share_axis.barh(
            ["All CVEs"],
            [class_row.percentage],
            left=running_left,
            color=class_color,
            height=0.42,
            label=class_row.class_label,
        )
        if class_row.percentage >= 4:
            share_axis.text(
                running_left + class_row.percentage / 2,
                0,
                f"{class_row.percentage:.2f}%",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )
        running_left += class_row.percentage
    share_axis.set_xlim(0, 100)
    share_axis.set_xlabel("Share of dataset (%)")
    share_axis.set_title(f"Composition — {imbalance_ratio:.1f}:1 imbalance")
    share_axis.grid(axis="x")
    share_axis.grid(axis="y", visible=False)
    share_axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2)

    figure.suptitle("CISA KEV target distribution", fontsize=17, fontweight="bold")
    figure.text(
        0.5,
        0.01,
        "KEV is the positive class. Log scaling is used only to make both counts visible.",
        ha="center",
        color=REPORT_COLORS["gray"],
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.93))
    figure.savefig(FIGURES_DIR / "class_balance.png", dpi=240, bbox_inches="tight")
    plt.close(figure)
    return class_balance


# Stop the experiment if EPSS, the target, or KEV metadata enters the feature list.
def assert_no_leakage_features(feature_columns: list[str]) -> None:
    blocked_exact_names = {
        "kevexploited",
        "percentile",
        "dateadded",
        "vendorproject",
        "product",
        "vulnerabilityname",
        "shortdescription",
        "requiredaction",
        "knownransomwarecampaignuse",
        "duedate",
        "notes",
        "cwes",
    }
    leaked_features = []
    for feature_name in feature_columns:
        normalized_name = feature_name.lower().replace("_", "")
        if (
            "epss" in normalized_name
            or "kev" in normalized_name
            or normalized_name in blocked_exact_names
        ):
            leaked_features.append(feature_name)
    if leaked_features:
        raise ValueError(
            "Feature leakage detected. Remove EPSS/KEV-derived inputs: "
            f"{sorted(leaked_features)}"
        )
    print("Leakage guard passed: no EPSS or KEV-derived model input is present.")


# Keep all learned transformations inside the pipeline to prevent train-test leakage.
def create_feature_preprocessor() -> ColumnTransformer:
    text_processor = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=TFIDF_MAX_FEATURES,
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
    )
    numeric_processor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_processor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("text", text_processor, TEXT_FEATURE),
            ("numeric", numeric_processor, NUMERIC_FEATURES),
            ("categorical", categorical_processor, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


# Split by publication time so every test CVE is published after every train CVE.
# This evaluates generalisation to a later publication cohort. It does not simulate
# labels or NVD feature values as they were known at the cutoff date.
def split_modelling_dataset(
    cleaned_dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dated_dataset = cleaned_dataset.dropna(subset=["published_date"]).sort_values(
        ["published_date", "cve_id"], kind="stable"
    )
    missing_date_count = len(cleaned_dataset) - len(dated_dataset)
    if len(dated_dataset) < 10:
        raise ValueError(
            "Temporal splitting needs at least 10 CVEs with valid publication dates."
        )
    nominal_test_start = max(int(np.floor(len(dated_dataset) * 0.80)), 1)
    if nominal_test_start >= len(dated_dataset):
        raise ValueError("Temporal split could not allocate a non-empty test set.")

    # Keep identical timestamps on the same side of the boundary.
    test_start_date = dated_dataset.iloc[nominal_test_start]["published_date"]
    training_rows = dated_dataset[dated_dataset["published_date"] < test_start_date]
    testing_rows = dated_dataset[dated_dataset["published_date"] >= test_start_date]
    if training_rows.empty or testing_rows.empty:
        raise ValueError(
            "Publication dates do not provide a usable chronological boundary."
        )
    for partition_name, partition in (("training", training_rows), ("testing", testing_rows)):
        if partition[TARGET_COLUMN].nunique() != 2:
            raise ValueError(
                f"The temporal {partition_name} partition contains only one target "
                "class. Expand the NVD date range or sample size."
            )

    training_features = training_rows[MODEL_INPUT_COLUMNS]
    testing_features = testing_rows[MODEL_INPUT_COLUMNS]
    training_target = training_rows[TARGET_COLUMN]
    testing_target = testing_rows[TARGET_COLUMN]
    pd.DataFrame(
        [
            {
                "split_method": "chronological_by_cve_publication_date",
                "target_interpretation": "retrospective_membership_in_downloaded_kev_snapshot",
                "requested_train_fraction": 0.80,
                "training_rows": len(training_rows),
                "testing_rows": len(testing_rows),
                "excluded_missing_publication_date_rows": missing_date_count,
                "training_start": training_rows["published_date"].min(),
                "training_end": training_rows["published_date"].max(),
                "testing_start": testing_rows["published_date"].min(),
                "testing_end": testing_rows["published_date"].max(),
                "training_kev_rate": training_target.mean(),
                "testing_kev_rate": testing_target.mean(),
                "prospective_forecast": False,
            }
        ]
    ).to_csv(TEMPORAL_SPLIT_FILE, index=False)
    print(
        "Temporal publication-date split: "
        f"train through {training_rows['published_date'].max().isoformat()}; "
        f"test from {testing_rows['published_date'].min().isoformat()}."
    )
    print(
        f"Train/test rows: {len(training_features):,}/{len(testing_features):,}; "
        f"train KEV rate={training_target.mean():.4%}; "
        f"test KEV rate={testing_target.mean():.4%}"
    )
    return training_features, testing_features, training_target, testing_target


# Build expanding-window folds and discard folds whose train or validation side
# lacks either class (average precision is not meaningful in that situation).
def create_temporal_validation_folds(
    tuning_target: pd.Series, requested_splits: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    if not tuning_target.index.is_monotonic_increasing:
        raise ValueError("Tuning rows must remain in publication-date order.")
    splitter = TimeSeriesSplit(n_splits=requested_splits)
    valid_folds = []
    for training_indices, validation_indices in splitter.split(tuning_target):
        training_fold = tuning_target.iloc[training_indices]
        validation_fold = tuning_target.iloc[validation_indices]
        if training_fold.nunique() == 2 and validation_fold.nunique() == 2:
            valid_folds.append((training_indices, validation_indices))
    if len(valid_folds) < 2:
        raise ValueError(
            "Temporal tuning needs at least two expanding-window folds with both "
            "KEV and non-KEV rows. Expand the date range or tuning sample."
        )
    return valid_folds


# Calculate XGBoost imbalance weight from training labels only.
def calculate_xgboost_class_weight(training_target: pd.Series) -> float:
    positive_count = int(training_target.eq(1).sum())
    negative_count = int(training_target.eq(0).sum())
    if positive_count == 0:
        raise ValueError("XGBoost class weight cannot be calculated without KEV rows.")
    return negative_count / positive_count


# Build the majority baseline and exactly the three required main classifiers.
def create_model_pipelines(training_target: pd.Series) -> dict[str, Pipeline]:
    xgboost_weight = calculate_xgboost_class_weight(training_target)
    return {
        "Majority-class Dummy": Pipeline(
            [
                # DummyClassifier ignores feature values, so TF-IDF is unnecessary here.
                ("classifier", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "Logistic Regression": Pipeline(
            [
                ("features", create_feature_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=3_000,
                        solver="liblinear",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("features", create_feature_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=400,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("features", create_feature_preprocessor()),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=400,
                        learning_rate=0.08,
                        max_depth=6,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        scale_pos_weight=xgboost_weight,
                        eval_metric="logloss",
                        tree_method="hist",
                        n_jobs=-1,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
    }


# Define meaningful parameter choices for each required ML model.
def tuning_specifications(training_target: pd.Series) -> dict[str, dict[str, Any]]:
    pipelines = create_model_pipelines(training_target)
    # SearchCV parallelizes candidates, so tree estimators use one worker each.
    pipelines["Random Forest"].set_params(classifier__n_jobs=1)
    pipelines["XGBoost"].set_params(classifier__n_jobs=1)
    return {
        "Logistic Regression": {
            "pipeline": pipelines["Logistic Regression"],
            "parameters": {
                "classifier__C": [0.02, 0.1, 0.5, 1.0, 5.0, 20.0],
                "classifier__penalty": ["l1", "l2"],
            },
            "iterations": 10,
        },
        "Random Forest": {
            "pipeline": pipelines["Random Forest"],
            "parameters": {
                "classifier__n_estimators": [250, 400, 600],
                "classifier__max_depth": [None, 12, 24, 36],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4],
                "classifier__max_features": ["sqrt", "log2", 0.5],
            },
            "iterations": 12,
        },
        "XGBoost": {
            "pipeline": pipelines["XGBoost"],
            "parameters": {
                "classifier__n_estimators": [250, 400, 600],
                "classifier__max_depth": [3, 5, 7],
                "classifier__learning_rate": [0.03, 0.08, 0.15],
                "classifier__subsample": [0.75, 0.9, 1.0],
                "classifier__colsample_bytree": [0.75, 0.9, 1.0],
                "classifier__min_child_weight": [1, 3, 6],
            },
            "iterations": 12,
        },
    }


# Use a stratified subset for parameter search when the full training set is large.
# The winning pipeline is fitted again on all training rows after the search.
def create_tuning_subset(
    training_features: pd.DataFrame,
    training_target: pd.Series,
    maximum_tuning_rows: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if maximum_tuning_rows < 10:
        raise ValueError("maximum_tuning_rows must be at least 10.")
    if len(training_features) <= maximum_tuning_rows:
        return training_features, training_target

    tuning_features, _, tuning_target, _ = train_test_split(
        training_features,
        training_target,
        train_size=maximum_tuning_rows,
        random_state=RANDOM_SEED,
        stratify=training_target,
    )
    # The outer training data is chronological; restoring its index order makes
    # expanding-window validation chronological after the resource-safe sample.
    tuning_features = tuning_features.sort_index()
    tuning_target = tuning_target.loc[tuning_features.index]
    print(
        f"Resource-safe tuning subset: {len(tuning_features):,} of "
        f"{len(training_features):,} training rows."
    )
    return tuning_features, tuning_target


# Tune only on training data with expanding-window temporal validation.
def tune_ml_models(
    training_features: pd.DataFrame,
    training_target: pd.Series,
    maximum_tuning_rows: int = DEFAULT_TUNING_MAX_ROWS,
    search_jobs: int = DEFAULT_SEARCH_JOBS,
) -> dict[str, Pipeline]:
    if search_jobs < 1:
        raise ValueError("search_jobs must be at least 1.")
    tuning_features, tuning_target = create_tuning_subset(
        training_features, training_target, maximum_tuning_rows
    )
    validation_folds = create_temporal_validation_folds(tuning_target)
    tuned_models: dict[str, Pipeline] = {}
    all_tuning_rows: list[pd.DataFrame] = []
    tuning_summary_rows: list[dict[str, Any]] = []
    for model_name, tuning_spec in tuning_specifications(tuning_target).items():
        print(
            f"\nTuning {model_name} with {len(validation_folds)} "
            "training-only expanding-window folds..."
        )
        model_search = RandomizedSearchCV(
            estimator=tuning_spec["pipeline"],
            param_distributions=tuning_spec["parameters"],
            n_iter=tuning_spec["iterations"],
            scoring="average_precision",
            cv=validation_folds,
            n_jobs=search_jobs,
            random_state=RANDOM_SEED,
            refit=True,
            return_train_score=False,
            verbose=1,
        )
        model_search.fit(tuning_features, tuning_target)

        # Refit the chosen configuration on every training row before testing.
        final_training_pipeline = clone(model_search.best_estimator_)
        if model_name == "XGBoost":
            final_training_pipeline.set_params(
                classifier__scale_pos_weight=calculate_xgboost_class_weight(
                    training_target
                )
            )
        print(f"Refitting best {model_name} on all {len(training_features):,} training rows...")
        final_training_pipeline.fit(training_features, training_target)
        tuned_models[model_name] = final_training_pipeline
        candidate_results = pd.DataFrame(model_search.cv_results_)[
            ["params", "mean_test_score", "std_test_score", "rank_test_score"]
        ].copy()
        candidate_results.insert(0, "model", model_name)
        candidate_results["params"] = candidate_results["params"].map(
            lambda parameters: json.dumps(parameters, sort_keys=True)
        )
        candidate_results = candidate_results.rename(
            columns={"mean_test_score": "mean_cv_pr_auc", "std_test_score": "std_cv_pr_auc"}
        )
        all_tuning_rows.append(candidate_results)
        tuning_summary_rows.append(
            {
                "model": model_name,
                "best_cv_pr_auc": model_search.best_score_,
                "best_parameters": json.dumps(
                    model_search.best_params_, sort_keys=True
                ),
                "candidate_count": len(candidate_results),
                "cv_folds": len(validation_folds),
                "cv_method": "expanding_window_by_publication_date",
                "optimization_metric": "average_precision",
                "tuning_row_count": len(tuning_features),
                "full_training_row_count": len(training_features),
                "search_jobs": search_jobs,
            }
        )
        print(f"Best CV PR-AUC: {model_search.best_score_:.4f}")
        print(f"Best parameters: {model_search.best_params_}")
    pd.DataFrame(tuning_summary_rows).to_csv(TUNING_RESULTS_FILE, index=False)
    pd.concat(all_tuning_rows, ignore_index=True).to_csv(
        TUNING_CANDIDATES_FILE, index=False
    )
    return tuned_models


# Fit default models only for quick development checks when tuning is skipped.
def train_default_ml_models(
    training_features: pd.DataFrame, training_target: pd.Series
) -> dict[str, Pipeline]:
    pipelines = create_model_pipelines(training_target)
    trained_models: dict[str, Pipeline] = {}
    for model_name in ("Logistic Regression", "Random Forest", "XGBoost"):
        print(f"Training default {model_name}...")
        trained_models[model_name] = pipelines[model_name].fit(
            training_features, training_target
        )
    return trained_models


# Convert a model display name into a stable file name.
def safe_file_stem(model_name: str) -> str:
    return model_name.lower().replace("-", "_").replace(" ", "_")


# Save a readable matrix whose four cells explicitly show TN, FP, FN, and TP.
def save_confusion_matrix(model_name: str, matrix_values: np.ndarray) -> None:
    row_totals = matrix_values.sum(axis=1, keepdims=True)
    normalized_values = np.divide(
        matrix_values,
        row_totals,
        out=np.zeros_like(matrix_values, dtype=float),
        where=row_totals != 0,
    )

    # Color by row percentage so minority-class errors remain visible.
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    image = axis.imshow(normalized_values, cmap="Blues", vmin=0, vmax=1)
    axis.set(
        title=f"{model_name}\nHeld-out test confusion matrix",
        xlabel="Predicted class",
        ylabel="Actual class",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["0 — Non-KEV", "1 — KEV exploited"],
        yticklabels=["0 — Non-KEV", "1 — KEV exploited"],
    )
    cell_names = np.array([["TN", "FP"], ["FN", "TP"]])
    for row_number in range(2):
        for column_number in range(2):
            value = int(matrix_values[row_number, column_number])
            row_percentage = normalized_values[row_number, column_number] * 100
            axis.text(
                column_number,
                row_number,
                f"{cell_names[row_number, column_number]}\n{value:,}\n{row_percentage:.1f}% of actual class",
                ha="center",
                va="center",
                color="white" if normalized_values[row_number, column_number] > 0.55 else "#111827",
                fontsize=11,
                fontweight="bold",
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Share of each actual class")
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    figure.text(
        0.5,
        0.02,
        "Colors are row-normalized; annotations retain the original CVE counts.",
        ha="center",
        color=REPORT_COLORS["gray"],
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(
        FIGURES_DIR / f"{safe_file_stem(model_name)}_confusion_matrix.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(figure)


# Evaluate class 1 with probabilities so PR-AUC is not calculated from hard labels.
def evaluate_model(
    model_name: str,
    trained_model: Pipeline,
    testing_features: pd.DataFrame,
    testing_target: pd.Series,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    predicted_target = trained_model.predict(testing_features)
    positive_probability = trained_model.predict_proba(testing_features)[:, 1]
    matrix_values = confusion_matrix(testing_target, predicted_target, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix_values.ravel()
    metrics_row = {
        "model": model_name,
        "precision_class_1": precision_score(
            testing_target, predicted_target, pos_label=1, zero_division=0
        ),
        "recall_class_1": recall_score(
            testing_target, predicted_target, pos_label=1, zero_division=0
        ),
        "f1_class_1": f1_score(
            testing_target, predicted_target, pos_label=1, zero_division=0
        ),
        "pr_auc_average_precision": average_precision_score(
            testing_target, positive_probability
        ),
        "accuracy_additional": accuracy_score(testing_target, predicted_target),
        "tn": int(true_negative),
        "fp": int(false_positive),
        "fn": int(false_negative),
        "tp": int(true_positive),
        "confusion_matrix": json.dumps(matrix_values.tolist()),
    }
    save_confusion_matrix(model_name, matrix_values)
    print(f"\n{model_name}")
    print(pd.DataFrame([metrics_row]).drop(columns="confusion_matrix").to_string(index=False))
    print(
        classification_report(
            testing_target,
            predicted_target,
            labels=[0, 1],
            target_names=["Non-KEV", "KEV exploited"],
            zero_division=0,
        )
    )
    return metrics_row, predicted_target, positive_probability


# Save separate and combined precision-recall curves for the three ML models.
def save_precision_recall_curves(
    probability_results: dict[str, np.ndarray], testing_target: pd.Series
) -> None:
    positive_prevalence = float(testing_target.mean())
    model_colors = {
        "Logistic Regression": REPORT_COLORS["blue"],
        "Random Forest": REPORT_COLORS["orange"],
        "XGBoost": REPORT_COLORS["green"],
    }
    curve_results = {}
    informative_precision_values = []

    for model_name, positive_probability in probability_results.items():
        precision_values, recall_values, _ = precision_recall_curve(
            testing_target, positive_probability
        )
        average_precision = average_precision_score(testing_target, positive_probability)
        curve_results[model_name] = (
            precision_values,
            recall_values,
            average_precision,
        )
        informative_precision_values.extend(precision_values[recall_values > 0])

    # A second panel zooms into the low-precision region common in rare-event data.
    precision_quantile = (
        float(np.quantile(informative_precision_values, 0.95))
        if informative_precision_values
        else positive_prevalence
    )
    zoom_limit = min(
        1.0,
        max(
            0.05,
            positive_prevalence * 6,
            precision_quantile * 1.25,
            max(curve_value[2] for curve_value in curve_results.values()) * 4,
        ),
    )

    for model_name, curve_values in curve_results.items():
        precision_values, recall_values, average_precision = curve_values
        model_color = model_colors.get(model_name, REPORT_COLORS["purple"])
        ap_lift = average_precision / positive_prevalence if positive_prevalence else np.nan
        figure, (full_axis, zoom_axis) = plt.subplots(1, 2, figsize=(12, 5.2))
        for axis, y_limit, panel_title in (
            (full_axis, 1.02, "Full precision range"),
            (zoom_axis, zoom_limit, "Minority-class detail"),
        ):
            axis.step(
                recall_values,
                precision_values,
                where="post",
                color=model_color,
                linewidth=2.4,
                label=f"{model_name} — AP {average_precision:.4f} ({ap_lift:.1f}× baseline)",
            )
            axis.axhline(
                positive_prevalence,
                color=REPORT_COLORS["red"],
                linestyle="--",
                linewidth=1.7,
                label=f"No-skill prevalence — {positive_prevalence:.3%}",
            )
            axis.set(
                title=panel_title,
                xlabel="Recall — share of KEVs found",
                ylabel="Precision — share of alerts that are KEV",
                xlim=(0, 1),
                ylim=(0, y_limit),
            )
            axis.grid(True)
        full_axis.legend(loc="upper right", fontsize=9)
        figure.suptitle(
            f"{model_name} precision–recall performance",
            fontsize=17,
            fontweight="bold",
        )
        figure.tight_layout(rect=(0, 0, 1, 0.93))
        figure.savefig(
            FIGURES_DIR / f"{safe_file_stem(model_name)}_pr_curve.png",
            dpi=240,
            bbox_inches="tight",
        )
        plt.close(figure)

    combined_figure, (combined_full_axis, combined_zoom_axis) = plt.subplots(
        1, 2, figsize=(13, 5.4)
    )
    for axis, y_limit, panel_title in (
        (combined_full_axis, 1.02, "Full precision range"),
        (combined_zoom_axis, zoom_limit, "Minority-class detail"),
    ):
        for model_name, curve_values in curve_results.items():
            precision_values, recall_values, average_precision = curve_values
            axis.step(
                recall_values,
                precision_values,
                where="post",
                color=model_colors.get(model_name, REPORT_COLORS["purple"]),
                linewidth=2.2,
                label=f"{model_name} — AP {average_precision:.4f}",
            )
        axis.axhline(
            positive_prevalence,
            color=REPORT_COLORS["red"],
            linestyle="--",
            linewidth=1.7,
            label=f"No-skill prevalence — {positive_prevalence:.3%}",
        )
        axis.set(
            title=panel_title,
            xlabel="Recall — share of KEVs found",
            ylabel="Precision — share of alerts that are KEV",
            xlim=(0, 1),
            ylim=(0, y_limit),
        )
        axis.grid(True)
    combined_full_axis.legend(loc="upper right", fontsize=8.5)
    combined_figure.suptitle(
        "Required ML models — precision–recall comparison",
        fontsize=17,
        fontweight="bold",
    )
    combined_figure.tight_layout(rect=(0, 0, 1, 0.93))
    combined_figure.savefig(
        FIGURES_DIR / "combined_ml_precision_recall_curves.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(combined_figure)


# Export exploited KEV CVEs that were incorrectly predicted as non-KEV.
def export_false_negatives(
    model_name: str,
    cleaned_dataset: pd.DataFrame,
    testing_features: pd.DataFrame,
    testing_target: pd.Series,
    predicted_target: np.ndarray,
    positive_probability: np.ndarray,
) -> pd.DataFrame:
    inspection_columns = [
        "cve_id",
        "published_date",
        "description_text",
        "base_score",
        "base_severity",
        "primary_cwe",
    ]
    test_details = cleaned_dataset.loc[testing_features.index, inspection_columns].copy()
    test_details["actual_kev_exploited"] = testing_target.to_numpy()
    test_details["predicted_kev_exploited"] = predicted_target
    test_details["predicted_kev_probability"] = positive_probability
    false_negatives = test_details.loc[
        test_details["actual_kev_exploited"].eq(1)
        & test_details["predicted_kev_exploited"].eq(0)
    ].sort_values("predicted_kev_probability")
    false_negatives.to_csv(
        RESULTS_DIR / f"{safe_file_stem(model_name)}_false_negatives.csv", index=False
    )
    print(f"{model_name} false negatives exported: {len(false_negatives):,}")
    return false_negatives


# Convert pipeline names into short report labels while retaining feature families.
def readable_feature_name(feature_name: str) -> tuple[str, str]:
    feature_prefix, _, raw_feature_name = feature_name.partition("__")
    family_names = {
        "text": "Text TF-IDF",
        "numeric": "Numeric CVSS",
        "categorical": "Categorical NVD",
    }
    feature_family = family_names.get(feature_prefix, "Other")
    readable_name = raw_feature_name.replace("_", " ")
    return f"[{feature_family}] {readable_name}", feature_family


# Map native tree importance values back to readable transformed feature names.
def export_tree_feature_importance(
    model_name: str, trained_model: Pipeline, top_feature_count: int = 25
) -> pd.DataFrame:
    feature_names = trained_model.named_steps["features"].get_feature_names_out()
    importance_values = trained_model.named_steps["classifier"].feature_importances_
    if len(feature_names) != len(importance_values):
        raise RuntimeError(
            f"{model_name} feature-name and importance lengths do not match."
        )
    importance_table = pd.DataFrame(
        {"feature": feature_names, "importance": importance_values}
    ).sort_values("importance", ascending=False, ignore_index=True)
    readable_features = importance_table["feature"].map(readable_feature_name)
    importance_table["readable_feature"] = readable_features.map(lambda value: value[0])
    importance_table["feature_family"] = readable_features.map(lambda value: value[1])
    file_stem = safe_file_stem(model_name)
    importance_table.to_csv(
        IMPORTANCE_DIR / f"{file_stem}_feature_importance.csv", index=False
    )
    nonzero_features = importance_table.loc[importance_table["importance"].gt(0)]
    top_features = nonzero_features.head(top_feature_count).sort_values("importance")
    if top_features.empty:
        top_features = importance_table.head(top_feature_count).sort_values("importance")

    family_colors = {
        "Text TF-IDF": REPORT_COLORS["blue"],
        "Numeric CVSS": REPORT_COLORS["orange"],
        "Categorical NVD": REPORT_COLORS["green"],
        "Other": REPORT_COLORS["gray"],
    }
    bar_colors = top_features["feature_family"].map(family_colors)
    figure_height = max(7.5, len(top_features) * 0.34)
    figure, (feature_axis, family_axis) = plt.subplots(
        1,
        2,
        figsize=(14, figure_height),
        gridspec_kw={"width_ratios": [3.3, 1]},
    )
    feature_axis.barh(
        top_features["readable_feature"],
        top_features["importance"],
        color=bar_colors,
        edgecolor="white",
    )
    feature_axis.set(
        title=f"Top {len(top_features)} non-zero features",
        xlabel="Native feature importance",
        ylabel="Transformed feature",
    )
    feature_axis.grid(axis="x")
    feature_axis.grid(axis="y", visible=False)

    family_importance = (
        importance_table.groupby("feature_family", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=True)
        .reset_index(drop=True)
    )
    family_axis.barh(
        family_importance["feature_family"],
        family_importance["importance"],
        color=family_importance["feature_family"].map(family_colors),
    )
    family_axis.set(
        title="Importance by feature family",
        xlabel="Total native importance",
        ylabel="",
    )
    family_axis.grid(axis="x")
    family_axis.grid(axis="y", visible=False)
    for row_value in family_importance.itertuples():
        family_axis.text(
            row_value.importance,
            row_value.Index,
            f"  {row_value.importance:.1%}",
            va="center",
            fontweight="bold",
        )

    legend_handles = [
        Patch(color=color_value, label=family_name)
        for family_name, color_value in family_colors.items()
        if family_name in set(top_features["feature_family"])
    ]
    feature_axis.legend(handles=legend_handles, loc="lower right")
    figure.suptitle(
        f"{model_name} feature importance",
        fontsize=17,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "Native importance describes this fitted model; it does not prove causality.",
        ha="center",
        color=REPORT_COLORS["gray"],
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    figure.savefig(
        IMPORTANCE_DIR / f"{file_stem}_feature_importance.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(figure)
    return importance_table


# Select from measured results and record security and operational trade-offs.
def save_model_selection_summary(comparison_table: pd.DataFrame) -> str:
    main_model_results = comparison_table.loc[
        comparison_table["model"].ne("Majority-class Dummy")
    ].copy()
    best_pr_auc_row = main_model_results.loc[
        main_model_results["pr_auc_average_precision"].idxmax()
    ]
    fewest_false_negative_count = int(main_model_results["fn"].min())
    fewest_miss_models = main_model_results.loc[
        main_model_results["fn"].eq(fewest_false_negative_count), "model"
    ].tolist()
    fewest_miss_text = ", ".join(fewest_miss_models)
    critical_recall_note = ""
    if main_model_results["recall_class_1"].max() == 0:
        critical_recall_note = (
            "Critical threshold finding: every main model had zero KEV recall and "
            "missed every exploited CVE in the held-out test set. High accuracy is "
            "therefore misleading for this run.\n"
        )
    summary_text = (
        "Model selection is based on actual held-out test results from this run.\n"
        f"Primary PR-AUC model: {best_pr_auc_row['model']} "
        f"(PR-AUC={best_pr_auc_row['pr_auc_average_precision']:.6f}, "
        f"precision={best_pr_auc_row['precision_class_1']:.6f}, "
        f"recall={best_pr_auc_row['recall_class_1']:.6f}, "
        f"FN={int(best_pr_auc_row['fn'])}, FP={int(best_pr_auc_row['fp'])}).\n"
        f"Fewest-false-negative model(s): {fewest_miss_text} "
        f"(FN={fewest_false_negative_count}; ties are reported).\n"
        f"{critical_recall_note}"
        "PR-AUC is the primary ranking metric because KEV is the minority class. "
        "Operational selection should also consider recall and false negatives: "
        "missing an exploited CVE can cause dangerous under-prioritization. "
        "False positives have a different cost, including unnecessary patching "
        "urgency, analyst effort, and remediation resources.\n"
    )
    MODEL_SELECTION_FILE.write_text(summary_text, encoding="utf-8")
    print("\n" + summary_text)
    return summary_text


# Summarize model quality and operational errors in one report-ready dashboard.
def save_model_comparison_figure(comparison_table: pd.DataFrame) -> None:
    display_table = comparison_table.copy()
    display_table["display_model"] = display_table["model"].replace(
        {"Majority-class Dummy": "Dummy baseline"}
    )
    model_colors = {
        "Dummy baseline": REPORT_COLORS["gray"],
        "Logistic Regression": REPORT_COLORS["blue"],
        "Random Forest": REPORT_COLORS["orange"],
        "XGBoost": REPORT_COLORS["green"],
    }
    metric_columns = [
        "precision_class_1",
        "recall_class_1",
        "f1_class_1",
        "pr_auc_average_precision",
    ]
    metric_labels = ["Precision", "Recall", "F1", "PR-AUC"]
    metric_positions = np.arange(len(metric_labels))
    figure, (metric_axis, error_axis) = plt.subplots(
        1, 2, figsize=(14, 5.8), gridspec_kw={"width_ratios": [1.4, 1]}
    )

    bar_width = 0.19
    for model_number, model_row in enumerate(display_table.itertuples()):
        metric_values = [getattr(model_row, column) for column in metric_columns]
        bar_positions = metric_positions + (model_number - 1.5) * bar_width
        bars = metric_axis.bar(
            bar_positions,
            metric_values,
            width=bar_width,
            color=model_colors[model_row.display_model],
            label=model_row.display_model,
        )
        for bar, metric_value in zip(bars, metric_values):
            if metric_value > 0:
                metric_axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    metric_value,
                    f"{metric_value:.3f}",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=7.5,
                )
    maximum_metric = float(display_table[metric_columns].to_numpy().max())
    metric_axis.set_ylim(0, max(0.05, maximum_metric * 1.4))
    metric_axis.set_xticks(metric_positions, metric_labels)
    metric_axis.set_ylabel("Score for KEV class 1")
    metric_axis.set_title("Positive-class performance")
    metric_axis.grid(axis="y")
    metric_axis.grid(axis="x", visible=False)
    metric_axis.legend(fontsize=8, ncol=2)

    model_positions = np.arange(len(display_table))
    false_positive_bars = error_axis.bar(
        model_positions - 0.18,
        display_table["fp"],
        width=0.36,
        color=REPORT_COLORS["orange"],
        label="False positives — analyst cost",
    )
    false_negative_bars = error_axis.bar(
        model_positions + 0.18,
        display_table["fn"],
        width=0.36,
        color=REPORT_COLORS["red"],
        label="False negatives — security risk",
    )
    error_axis.bar_label(false_positive_bars, fmt="%d", padding=3, fontsize=9)
    error_axis.bar_label(false_negative_bars, fmt="%d", padding=3, fontsize=9)
    error_axis.set_xticks(
        model_positions,
        display_table["display_model"],
        rotation=22,
        ha="right",
    )
    error_axis.set_ylabel("Held-out test CVE count")
    error_axis.set_title("Operational error trade-off")
    error_axis.grid(axis="y")
    error_axis.grid(axis="x", visible=False)
    error_axis.legend(fontsize=8)

    figure.suptitle(
        "Model comparison dashboard",
        fontsize=17,
        fontweight="bold",
    )
    main_model_rows = display_table.loc[
        display_table["model"].ne("Majority-class Dummy")
    ]
    if main_model_rows["recall_class_1"].max() == 0:
        dashboard_note = (
            "Critical finding: all three ML models missed every KEV at the default "
            "threshold; accuracy is misleading under this imbalance."
        )
        dashboard_note_color = REPORT_COLORS["red"]
    else:
        dashboard_note = (
            "PR-AUC is the primary ranking metric; false negatives represent missed "
            "exploited CVEs."
        )
        dashboard_note_color = REPORT_COLORS["gray"]
    figure.text(
        0.5,
        0.01,
        dashboard_note,
        ha="center",
        color=dashboard_note_color,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    figure.savefig(
        FIGURES_DIR / "model_comparison_dashboard.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(figure)


# Run data collection, EDA, cleaning, modelling, evaluation, and output creation.
def run_complete_experiment(
    *,
    first_year: int = DEFAULT_FIRST_YEAR,
    last_year: int = DEFAULT_LAST_YEAR,
    refresh: bool = False,
    development_mode: bool = False,
    development_max_records: int = 10_000,
    tune_models: bool = True,
    maximum_tuning_rows: int = DEFAULT_TUNING_MAX_ROWS,
    search_jobs: int = DEFAULT_SEARCH_JOBS,
) -> dict[str, Any]:
    create_output_directories()
    assert_no_leakage_features(MODEL_INPUT_COLUMNS)
    nvd_dataset, kev_dataset = load_or_download_data(
        first_year=first_year,
        last_year=last_year,
        refresh=refresh,
        development_mode=development_mode,
        development_max_records=development_max_records,
    )
    labelled_dataset = build_kev_target_dataset(nvd_dataset, kev_dataset)
    save_target_temporal_audit(labelled_dataset, kev_dataset)

    # The labelled copy now contains everything needed from both raw tables.
    del nvd_dataset, kev_dataset
    gc.collect()

    save_eda_outputs(labelled_dataset)
    cleaned_dataset = clean_modelling_dataset(labelled_dataset)

    # Release the uncleaned copy before TF-IDF and model training begin.
    del labelled_dataset
    gc.collect()

    class_balance = save_class_balance(cleaned_dataset)
    training_features, testing_features, training_target, testing_target = (
        split_modelling_dataset(cleaned_dataset)
    )

    baseline = create_model_pipelines(training_target)["Majority-class Dummy"]
    baseline.fit(training_features, training_target)
    ml_models = (
        tune_ml_models(
            training_features,
            training_target,
            maximum_tuning_rows=maximum_tuning_rows,
            search_jobs=search_jobs,
        )
        if tune_models
        else train_default_ml_models(training_features, training_target)
    )
    all_models = {"Majority-class Dummy": baseline, **ml_models}

    comparison_rows: list[dict[str, Any]] = []
    ml_probability_results: dict[str, np.ndarray] = {}
    false_negative_tables: dict[str, pd.DataFrame] = {}
    for model_name, trained_model in all_models.items():
        metrics_row, predicted_target, positive_probability = evaluate_model(
            model_name, trained_model, testing_features, testing_target
        )
        comparison_rows.append(metrics_row)
        if model_name != "Majority-class Dummy":
            ml_probability_results[model_name] = positive_probability
            false_negative_tables[model_name] = export_false_negatives(
                model_name,
                cleaned_dataset,
                testing_features,
                testing_target,
                predicted_target,
                positive_probability,
            )

    comparison_table = pd.DataFrame(comparison_rows).sort_values(
        "pr_auc_average_precision", ascending=False, ignore_index=True
    )
    comparison_table.to_csv(MODEL_COMPARISON_FILE, index=False)
    print("\nModel comparison (ranked by PR-AUC / Average Precision)")
    print(comparison_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    save_model_selection_summary(comparison_table)
    save_model_comparison_figure(comparison_table)
    save_precision_recall_curves(ml_probability_results, testing_target)
    export_tree_feature_importance("Random Forest", ml_models["Random Forest"])
    export_tree_feature_importance("XGBoost", ml_models["XGBoost"])

    print(
        "\nFalse negative meaning: an actually exploited KEV vulnerability "
        "(actual kev_exploited = 1) was predicted as non-KEV (prediction = 0). "
        "This can cause dangerous under-prioritization in a cybersecurity "
        "remediation workflow."
    )
    print(
        "False positive meaning: a non-KEV CVE (actual = 0) was predicted as "
        "KEV (prediction = 1). This may create unnecessary patching urgency, "
        "waste analyst time, and use remediation resources inefficiently. "
        "That operational cost is distinct from the security risk of a false negative."
    )
    print(f"Cleaned dataset: {CLEAN_DATA_FILE}")
    print(f"Results: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    print(f"Feature importance: {IMPORTANCE_DIR}")
    return {
        "cleaned_dataset": cleaned_dataset,
        "class_balance": class_balance,
        "comparison": comparison_table,
        "models": all_models,
        "false_negatives": false_negative_tables,
    }


# Define full research mode and a clearly labelled development mode.
def parse_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrospectively classify membership in the downloaded CISA KEV "
            "snapshot with TF-IDF and classical ML."
        )
    )
    parser.add_argument("--first-year", type=int, default=DEFAULT_FIRST_YEAR)
    parser.add_argument("--last-year", type=int, default=DEFAULT_LAST_YEAR)
    parser.add_argument(
        "--refresh", action="store_true", help="Ignore local CSV caches and download again."
    )
    parser.add_argument(
        "--development",
        action="store_true",
        help="Use a capped NVD download for code checks; not a final research run.",
    )
    parser.add_argument(
        "--development-max-records", type=int, default=10_000
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Fit default parameters for a quick check; final research runs should tune.",
    )
    parser.add_argument(
        "--tuning-max-rows",
        type=int,
        default=DEFAULT_TUNING_MAX_ROWS,
        help="Training-only rows used during search before full-training refit.",
    )
    parser.add_argument(
        "--search-jobs",
        type=int,
        default=DEFAULT_SEARCH_JOBS,
        help="Parallel search workers. Keep 1 on memory-limited computers.",
    )
    return parser.parse_args()


# Read command-line settings and start the requested experiment.
def main() -> None:
    # Show progress immediately when the script runs inside an IDE terminal.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    command_line = parse_command_line()
    first_year = command_line.first_year
    if command_line.development and first_year == DEFAULT_FIRST_YEAR:
        first_year = max(DEFAULT_LAST_YEAR - 3, 1999)
        print(
            f"DEVELOPMENT MODE: capped at {command_line.development_max_records:,} "
            f"NVD records from {first_year}; outputs are not final research results."
        )
    run_complete_experiment(
        first_year=first_year,
        last_year=command_line.last_year,
        refresh=command_line.refresh,
        development_mode=command_line.development,
        development_max_records=command_line.development_max_records,
        tune_models=not command_line.skip_tuning,
        maximum_tuning_rows=command_line.tuning_max_rows,
        search_jobs=command_line.search_jobs,
    )


if __name__ == "__main__":
    main()
