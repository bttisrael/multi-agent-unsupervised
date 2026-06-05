"""
Auto Segmenter AI v2.1 — Multi-Agent Unsupervised Learning + GitOps Pipeline
=================================================================

Goal
----
Turn a transactional/customer dataset into a portfolio-ready unsupervised
learning project:
  - ingestion from Kaggle or local file
  - automatic column detection
  - customer/entity-level feature engineering, with RFM when possible
  - clustering model competition
  - statistical cluster evaluation
  - business segment interpretation, optionally powered by Claude
  - Streamlit app, README and Jupyter notebook generation

Default project
---------------
Customer Segmentation & Revenue Growth Intelligence using the Online Retail
Kaggle dataset. This is designed to strengthen a portfolio for roles asking for
unsupervised learning, RGM, customer profiling, product/data solutions and GenAI.

Install
-------
pip install -r requirements_unsupervised.txt

Environment variables (.env optional)
-------------------------------------
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key_here
ANTHROPIC_API_KEY=sk-ant-...   # optional. Pipeline works without it.

Run examples
------------
python multi_agent_ds_unsupervised_v1.py
python multi_agent_ds_unsupervised_v1.py --local-path data.csv
python multi_agent_ds_unsupervised_v1.py --dataset-slug hellbuoy/online-retail-customer-clustering
streamlit run streamlit_segment_app.py
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import math
import os
import re
import sys
import textwrap
import subprocess
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:  # fallback if seaborn is not installed
    sns = None

from dotenv import load_dotenv
from scipy import stats as ss
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler

load_dotenv()

# =============================================================================
# Logging
# =============================================================================

logger = logging.getLogger("AutoSegmenterAI")
logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    project_name: str = "Customer Segmentation & Revenue Growth Intelligence"
    dataset_slug: str = "hellbuoy/online-retail-customer-clustering"
    dataset_url: str = "https://www.kaggle.com/datasets/hellbuoy/online-retail-customer-clustering"
    local_path: Optional[str] = None
    output_dir: str = "unsupervised_outputs"
    random_state: int = 42
    max_rows: int = 1_000_000
    clustering_sample_size: int = 80_000
    silhouette_sample_size: int = 20_000
    max_k: int = 10
    min_k: int = 2
    min_cluster_pct: float = 0.02
    use_power_transform: bool = True
    ai_enabled: bool = True
    anthropic_model: str = "claude-sonnet-4-5"
    max_ai_tokens: int = 2000

    # GitOps agent
    git_enabled: bool = True
    git_push: bool = True
    git_remote: str = "origin"
    git_branch: str = "main"
    git_commit_message: Optional[str] = None

    @property
    def out(self) -> Path:
        return Path(self.output_dir)


CONFIG = Config()

ARTIFACTS = {
    "silver": "df1_silver.parquet",
    "profile": "Data_Profile.md",
    "entity_config": "entity_config.json",
    "features": "df2_customer_features.parquet",
    "feature_report": "Feature_Engineering_Report.md",
    "cluster_matrix": "df3_cluster_matrix.parquet",
    "clustered": "df4_clustered_customers.parquet",
    "metrics": "cluster_metrics.json",
    "segment_profiles": "segment_profiles.json",
    "segment_md": "Segment_Profiles.md",
    "strategy_md": "Business_Strategy.md",
    "hypotheses_json": "segment_hypotheses.json",
    "hypotheses_md": "Segment_Hypothesis_Validation.md",
    "pca_png": "cluster_pca_map.png",
    "sizes_png": "cluster_sizes.png",
    "rfm_png": "cluster_rfm_profile.png",
    "heatmap_png": "cluster_feature_heatmap.png",
    "app": "streamlit_segment_app.py",
    "notebook": "analysis_notebook_unsupervised.ipynb",
    "readme": "README_unsupervised.md",
    "requirements": "requirements_unsupervised.txt",
}


# =============================================================================
# Utilities
# =============================================================================

def out_path(key: str) -> Path:
    return CONFIG.out / ARTIFACTS[key]


def ensure_output_dir() -> None:
    CONFIG.out.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=json_safe), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_safe(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new_cols = []
    for col in df.columns:
        c = str(col).strip().lower()
        c = re.sub(r"[^a-z0-9]+", "_", c)
        c = re.sub(r"_+", "_", c).strip("_")
        new_cols.append(c)
    df.columns = new_cols
    return df


def safe_markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    try:
        return df.head(max_rows).to_markdown(index=False)
    except Exception:
        return df.head(max_rows).to_string(index=False)


def read_business_context() -> str:
    p = Path("business_context.txt")
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return (
        "Use unsupervised learning to segment customers/entities, understand revenue behavior, "
        "prioritize commercial actions and create an executive RGM-style analytics product."
    )


def ask_claude(prompt: str, max_tokens: Optional[int] = None) -> str:
    """Optional AI reasoning. Deterministic fallback when Anthropic is unavailable."""
    if not CONFIG.ai_enabled or not os.getenv("ANTHROPIC_API_KEY"):
        return "AI_DISABLED"
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        msg = client.messages.create(
            model=CONFIG.anthropic_model,
            max_tokens=max_tokens or CONFIG.max_ai_tokens,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as exc:
        logger.warning("Claude call failed. Using deterministic fallback. Error: %s", exc)
        return "AI_ERROR"


def parse_json_from_text(text: str) -> Optional[dict]:
    if not text or text in {"AI_DISABLED", "AI_ERROR"}:
        return None
    cleaned = text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0]
    try:
        return json.loads(cleaned)
    except Exception:
        return None


# =============================================================================
# Agent 1 — Ingestor
# =============================================================================

def download_and_save_silver() -> str:
    """Load dataset from local file or Kaggle, standardize columns, save silver parquet."""
    ensure_output_dir()
    try:
        if CONFIG.local_path:
            raw_path = Path(CONFIG.local_path)
            if not raw_path.exists():
                return f"INGESTION_ERROR: local file not found: {raw_path}"
            logger.info("[Ingestor] Reading local file: %s", raw_path)
            df = read_any_table(raw_path)
        else:
            logger.info("[Ingestor] Downloading Kaggle dataset: %s", CONFIG.dataset_slug)
            if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
                return (
                    "INGESTION_ERROR: KAGGLE_USERNAME/KAGGLE_KEY not found in .env. "
                    "Either add Kaggle credentials or run with --local-path your_file.csv"
                )
            import kagglehub
            path = Path(kagglehub.dataset_download(CONFIG.dataset_slug))
            candidates = sorted(
                [p for p in path.rglob("*") if p.suffix.lower() in {".csv", ".xlsx", ".xls", ".parquet"}],
                key=lambda p: p.stat().st_size,
                reverse=True,
            )
            if not candidates:
                return f"INGESTION_ERROR: no readable table found in {path}"
            logger.info("[Ingestor] Reading largest table found: %s", candidates[0])
            df = read_any_table(candidates[0])

        df = standardize_columns(df)
        if len(df) > CONFIG.max_rows:
            logger.info("[Ingestor] Sampling %s rows from %s", CONFIG.max_rows, len(df))
            df = df.sample(CONFIG.max_rows, random_state=CONFIG.random_state).reset_index(drop=True)
        df.to_parquet(out_path("silver"), index=False)
        return f"INGESTION_SUCCESS: shape={df.shape}, file={out_path('silver')}"
    except Exception as exc:
        return f"INGESTION_ERROR: {exc}\n{traceback.format_exc()}"


def read_any_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    # CSV fallback with encoding attempts
    encodings = ["utf-8", "latin1", "iso-8859-1", "cp1252"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as exc:
            last_error = exc
    # Try semicolon separator
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, sep=";", low_memory=False)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not read {path}: {last_error}")


# =============================================================================
# Agent 2 — Data Profiler + Entity Detector
# =============================================================================

def profile_and_detect_entity() -> str:
    try:
        df = pd.read_parquet(out_path("silver"))
        profile = build_profile(df)
        entity_config = detect_entity_config(df, profile)
        save_json(out_path("entity_config"), entity_config)

        profile_md = render_profile_md(df, profile, entity_config)
        out_path("profile").write_text(profile_md, encoding="utf-8")
        return (
            "PROFILE_SUCCESS: "
            f"entity_col={entity_config.get('entity_col')}, "
            f"date_col={entity_config.get('date_col')}, "
            f"monetary_cols={entity_config.get('monetary_cols')}, "
            f"file={out_path('profile')}"
        )
    except Exception as exc:
        return f"PROFILE_ERROR: {exc}\n{traceback.format_exc()}"


def build_profile(df: pd.DataFrame) -> Dict[str, Any]:
    profile: Dict[str, Any] = {
        "shape": list(df.shape),
        "columns": {},
        "duplicates": int(df.duplicated().sum()),
    }
    for col in df.columns:
        s = df[col]
        item = {
            "dtype": str(s.dtype),
            "nunique": int(s.nunique(dropna=True)),
            "null_pct": round(float(s.isna().mean() * 100), 3),
            "sample": [str(x) for x in s.dropna().head(5).tolist()],
        }
        if pd.api.types.is_numeric_dtype(s):
            item.update({
                "mean": round(float(s.mean()), 4) if s.notna().any() else None,
                "std": round(float(s.std()), 4) if s.notna().any() else None,
                "min": round(float(s.min()), 4) if s.notna().any() else None,
                "max": round(float(s.max()), 4) if s.notna().any() else None,
                "skew": round(float(s.skew()), 4) if s.notna().sum() > 2 else None,
            })
        profile["columns"][col] = item
    return profile


def detect_entity_config(df: pd.DataFrame, profile: Dict[str, Any]) -> Dict[str, Any]:
    cols = list(df.columns)

    def find_by_patterns(patterns: List[str], max_cardinality_ratio: Optional[float] = None) -> Optional[str]:
        for pat in patterns:
            for c in cols:
                if re.search(pat, c, flags=re.I):
                    if max_cardinality_ratio is not None:
                        ratio = df[c].nunique(dropna=True) / max(len(df), 1)
                        if ratio > max_cardinality_ratio:
                            continue
                    return c
        return None

    entity_col = find_by_patterns([
        r"customer.*id", r"client.*id", r"user.*id", r"account.*id", r"consumer.*id",
        r"customer", r"client", r"user_id", r"member.*id",
    ], max_cardinality_ratio=0.95)

    invoice_col = find_by_patterns([r"invoice", r"order.*id", r"transaction.*id", r"basket", r"cart"])
    date_col = find_by_patterns([r"invoice.*date", r"order.*date", r"event.*time", r"timestamp", r"date", r"datetime", r"time"])
    quantity_col = find_by_patterns([r"quantity", r"qty", r"units", r"amount"])
    price_col = find_by_patterns([r"unit.*price", r"price", r"unitprice", r"valor", r"value"])
    product_col = find_by_patterns([r"stock.*code", r"product.*id", r"sku", r"item.*id", r"product", r"description"])
    country_col = find_by_patterns([r"country", r"region", r"state", r"city", r"location"])

    # Generic fallback: choose a high-cardinality ID-like column, but not unique row ID.
    if entity_col is None:
        candidates = []
        for c in cols:
            nunique = df[c].nunique(dropna=True)
            ratio = nunique / max(len(df), 1)
            if 20 <= nunique <= max(100_000, len(df) * 0.5) and ratio < 0.95:
                score = 0
                if "id" in c.lower(): score += 3
                if any(x in c.lower() for x in ["customer", "client", "user", "account"]): score += 5
                if df[c].dtype == "object": score += 1
                candidates.append((score, nunique, c))
        if candidates:
            candidates.sort(reverse=True)
            entity_col = candidates[0][2]

    monetary_cols = []
    if quantity_col and price_col:
        monetary_cols = [quantity_col, price_col]
    else:
        for c in df.select_dtypes(include="number").columns:
            if any(k in c.lower() for k in ["revenue", "sales", "amount", "value", "price", "total", "monetary"]):
                monetary_cols.append(c)

    ai_config = ai_refine_entity_config(df, profile, {
        "entity_col": entity_col,
        "invoice_col": invoice_col,
        "date_col": date_col,
        "quantity_col": quantity_col,
        "price_col": price_col,
        "product_col": product_col,
        "country_col": country_col,
        "monetary_cols": monetary_cols,
        "clustering_unit": "customer" if entity_col else "row",
    })

    return ai_config or {
        "entity_col": entity_col,
        "invoice_col": invoice_col,
        "date_col": date_col,
        "quantity_col": quantity_col,
        "price_col": price_col,
        "product_col": product_col,
        "country_col": country_col,
        "monetary_cols": monetary_cols,
        "clustering_unit": "customer" if entity_col else "row",
        "strategy": "RFM/entity aggregation" if entity_col else "row-level numeric clustering",
    }


def ai_refine_entity_config(df: pd.DataFrame, profile: Dict[str, Any], detected: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    prompt = f"""
You are a senior data scientist designing an unsupervised learning pipeline.

Business context:
{read_business_context()}

Dataset shape: {df.shape}
Detected configuration by heuristics:
{json.dumps(detected, indent=2, default=json_safe)}

Column profile sample:
{json.dumps({k: profile['columns'][k] for k in list(profile['columns'])[:30]}, indent=2, default=json_safe)}

Task:
Return the best entity configuration for customer/entity segmentation.
Use only existing column names or null.

JSON schema:
{{
  "entity_col": "column_or_null",
  "invoice_col": "column_or_null",
  "date_col": "column_or_null",
  "quantity_col": "column_or_null",
  "price_col": "column_or_null",
  "product_col": "column_or_null",
  "country_col": "column_or_null",
  "monetary_cols": ["col1", "col2"],
  "clustering_unit": "customer" or "row" or "order" or "product",
  "strategy": "short explanation"
}}
Respond ONLY with JSON.
"""
    raw = ask_claude(prompt, max_tokens=1200)
    parsed = parse_json_from_text(raw)
    if not parsed:
        return None
    valid_cols = set(df.columns)
    for key in ["entity_col", "invoice_col", "date_col", "quantity_col", "price_col", "product_col", "country_col"]:
        if parsed.get(key) not in valid_cols:
            parsed[key] = None
    parsed["monetary_cols"] = [c for c in parsed.get("monetary_cols", []) if c in valid_cols]
    return parsed


def render_profile_md(df: pd.DataFrame, profile: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    rows = []
    for col, item in profile["columns"].items():
        rows.append({
            "column": col,
            "dtype": item["dtype"],
            "nunique": item["nunique"],
            "null_pct": item["null_pct"],
            "sample": ", ".join(item["sample"][:3]),
        })
    summary_df = pd.DataFrame(rows)
    return f"""# Data Profile — Auto Segmenter AI

## Dataset
- Shape: **{df.shape[0]:,} rows × {df.shape[1]:,} columns**
- Duplicates: **{profile['duplicates']:,}**

## Detected Segmentation Configuration
```json
{json.dumps(cfg, indent=2, ensure_ascii=False, default=json_safe)}
```

## Column Summary
{safe_markdown_table(summary_df, max_rows=80)}
"""


# =============================================================================
# Agent 3 — Feature Engineering
# =============================================================================

def build_entity_features() -> str:
    try:
        df = pd.read_parquet(out_path("silver"))
        cfg = load_json(out_path("entity_config"))
        features, report = create_features(df, cfg)
        features.to_parquet(out_path("features"), index=False)
        out_path("feature_report").write_text(report, encoding="utf-8")
        return f"FEATURES_SUCCESS: shape={features.shape}, file={out_path('features')}"
    except Exception as exc:
        return f"FEATURES_ERROR: {exc}\n{traceback.format_exc()}"


def create_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    entity_col = cfg.get("entity_col")
    if entity_col and entity_col in df.columns:
        features = create_entity_aggregates(df, cfg)
        strategy = "entity-level aggregation with RFM-style features"
    else:
        features = create_row_level_features(df, cfg)
        strategy = "row-level numeric clustering fallback because no entity column was detected"

    # Drop rows with all null feature values except entity id
    id_cols = [c for c in [entity_col, "segment_key"] if c in features.columns]
    value_cols = [c for c in features.columns if c not in id_cols]
    features = features.dropna(subset=value_cols, how="all").reset_index(drop=True)

    # Remove infinite values
    features = features.replace([np.inf, -np.inf], np.nan)

    report = f"""# Feature Engineering Report

## Strategy
{strategy}

## Result
- Rows: **{features.shape[0]:,}**
- Columns: **{features.shape[1]:,}**

## Generated Features
{', '.join(features.columns)}

## Sample
{safe_markdown_table(features.head(10), max_rows=10)}
"""
    return features, report


def create_entity_aggregates(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    entity_col = cfg["entity_col"]
    invoice_col = cfg.get("invoice_col")
    date_col = cfg.get("date_col")
    quantity_col = cfg.get("quantity_col")
    price_col = cfg.get("price_col")
    product_col = cfg.get("product_col")
    country_col = cfg.get("country_col")

    # Monetary feature
    if quantity_col in df.columns and price_col in df.columns:
        df[quantity_col] = pd.to_numeric(df[quantity_col], errors="coerce")
        df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
        df["_line_revenue"] = df[quantity_col] * df[price_col]
    else:
        monetary_cols = cfg.get("monetary_cols", [])
        if monetary_cols:
            df["_line_revenue"] = pd.to_numeric(df[monetary_cols[0]], errors="coerce")
        else:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            df["_line_revenue"] = pd.to_numeric(df[numeric_cols[0]], errors="coerce") if numeric_cols else 1.0

    # Clean cancellations/returns but keep rates as features
    if quantity_col in df.columns:
        df["_is_return"] = (pd.to_numeric(df[quantity_col], errors="coerce") < 0).astype(int)
    elif invoice_col in df.columns:
        df["_is_return"] = df[invoice_col].astype(str).str.startswith("C").astype(int)
    else:
        df["_is_return"] = 0

    df["_positive_revenue"] = df["_line_revenue"].clip(lower=0)

    # Date handling
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        reference_date = df[date_col].max() + pd.Timedelta(days=1)
        has_date = True
    else:
        reference_date = None
        has_date = False

    group = df.groupby(entity_col, dropna=True)
    agg = pd.DataFrame({entity_col: group.size().index})
    agg["n_transactions"] = group.size().values
    agg["total_revenue"] = group["_positive_revenue"].sum().values
    agg["avg_revenue_per_line"] = group["_positive_revenue"].mean().values
    agg["median_revenue_per_line"] = group["_positive_revenue"].median().values
    agg["std_revenue_per_line"] = group["_positive_revenue"].std().fillna(0).values
    agg["return_rate"] = group["_is_return"].mean().values

    if invoice_col in df.columns:
        agg["n_orders"] = group[invoice_col].nunique().values
        agg["avg_lines_per_order"] = agg["n_transactions"] / agg["n_orders"].replace(0, np.nan)
        order_revenue = df.groupby([entity_col, invoice_col])["_positive_revenue"].sum().reset_index()
        basket = order_revenue.groupby(entity_col)["_positive_revenue"].agg(["mean", "median", "max"]).reset_index()
        basket.columns = [entity_col, "avg_order_value", "median_order_value", "max_order_value"]
        agg = agg.merge(basket, on=entity_col, how="left")
    else:
        agg["n_orders"] = agg["n_transactions"]
        agg["avg_lines_per_order"] = 1.0
        agg["avg_order_value"] = agg["avg_revenue_per_line"]
        agg["median_order_value"] = agg["median_revenue_per_line"]
        agg["max_order_value"] = group["_positive_revenue"].max().values

    if quantity_col in df.columns:
        q = pd.to_numeric(df[quantity_col], errors="coerce")
        df["_positive_qty"] = q.clip(lower=0)
        qty_group = df.groupby(entity_col)["_positive_qty"]
        agg["total_quantity"] = qty_group.sum().values
        agg["avg_quantity"] = qty_group.mean().values
    else:
        agg["total_quantity"] = np.nan
        agg["avg_quantity"] = np.nan

    if product_col in df.columns:
        agg["product_diversity"] = group[product_col].nunique().values
        agg["revenue_per_product"] = agg["total_revenue"] / agg["product_diversity"].replace(0, np.nan)
    else:
        agg["product_diversity"] = np.nan
        agg["revenue_per_product"] = np.nan

    if country_col in df.columns:
        agg["country_diversity"] = group[country_col].nunique().values
    else:
        agg["country_diversity"] = np.nan

    if has_date:
        last_date = group[date_col].max()
        first_date = group[date_col].min()
        active_days = (last_date - first_date).dt.days + 1
        recency = (reference_date - last_date).dt.days
        agg = agg.merge(pd.DataFrame({entity_col: last_date.index, "recency_days": recency.values}), on=entity_col)
        agg = agg.merge(pd.DataFrame({entity_col: active_days.index, "customer_lifetime_days": active_days.values}), on=entity_col)
        agg["frequency_per_month"] = agg["n_orders"] / (agg["customer_lifetime_days"].clip(lower=1) / 30)
    else:
        agg["recency_days"] = np.nan
        agg["customer_lifetime_days"] = np.nan
        agg["frequency_per_month"] = np.nan

    # RFM-friendly transforms
    agg["monetary_log"] = np.log1p(agg["total_revenue"].clip(lower=0))
    agg["frequency_log"] = np.log1p(agg["n_orders"].clip(lower=0))
    if "recency_days" in agg.columns:
        agg["recency_log"] = np.log1p(agg["recency_days"].clip(lower=0))

    return agg


def create_row_level_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").copy()
    if numeric.empty:
        # Encode low-cardinality categoricals as counts if there are no numeric cols.
        for col in df.columns[:10]:
            numeric[f"freq_{col}"] = df[col].map(df[col].value_counts())
    numeric = numeric.reset_index(drop=False).rename(columns={"index": "segment_key"})
    return numeric


# =============================================================================
# Agent 4 — Clustering Scientist
# =============================================================================

def run_clustering_models() -> str:
    try:
        features = pd.read_parquet(out_path("features"))
        matrix, prep_info = prepare_cluster_matrix(features)
        matrix.to_parquet(out_path("cluster_matrix"), index=False)

        cluster_results, labels_by_model = compete_clustering_models(matrix)
        save_json(out_path("metrics"), cluster_results)

        best = cluster_results["best_model"]
        labels = labels_by_model[best["model_key"]]
        clustered = features.copy()
        clustered["cluster"] = labels.astype(int)
        clustered.to_parquet(out_path("clustered"), index=False)

        generate_cluster_charts(clustered, matrix, labels)
        return (
            "CLUSTERING_SUCCESS: "
            f"best={best['model_name']}, clusters={best['n_clusters']}, "
            f"silhouette={best.get('silhouette')}, file={out_path('clustered')}"
        )
    except Exception as exc:
        return f"CLUSTERING_ERROR: {exc}\n{traceback.format_exc()}"


def prepare_cluster_matrix(features: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    id_like = {"cluster", "segment_key"}
    cfg = load_json(out_path("entity_config")) if out_path("entity_config").exists() else {}
    if cfg.get("entity_col"):
        id_like.add(cfg["entity_col"])
    feature_cols = [c for c in features.columns if c not in id_like and pd.api.types.is_numeric_dtype(features[c])]
    if not feature_cols:
        raise RuntimeError("No numeric features available for clustering.")

    X = features[feature_cols].copy().replace([np.inf, -np.inf], np.nan)

    # Winsorize extreme outliers for clustering stability
    for c in X.columns:
        if X[c].notna().sum() > 10:
            low, high = X[c].quantile([0.01, 0.99])
            X[c] = X[c].clip(low, high)

    steps = [("imputer", SimpleImputer(strategy="median"))]
    if CONFIG.use_power_transform:
        steps.append(("power", PowerTransformer(method="yeo-johnson", standardize=False)))
    steps.append(("scaler", RobustScaler()))
    pipe = Pipeline(steps)

    arr = pipe.fit_transform(X)
    matrix = pd.DataFrame(arr, columns=feature_cols)
    return matrix, {"feature_cols": feature_cols, "steps": [s[0] for s in steps]}


def compete_clustering_models(matrix: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    X = matrix.values
    X_fit = sample_array(X, CONFIG.clustering_sample_size)
    labels_by_model: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, Any]] = []

    # KMeans
    for k in range(CONFIG.min_k, CONFIG.max_k + 1):
        model = KMeans(n_clusters=k, n_init=20, random_state=CONFIG.random_state)
        labels = model.fit_predict(X)
        rows.append(evaluate_labels("kmeans", f"KMeans k={k}", labels, X))
        labels_by_model[f"kmeans_{k}"] = labels

    # Gaussian Mixture
    for k in range(CONFIG.min_k, CONFIG.max_k + 1):
        try:
            gm = GaussianMixture(n_components=k, covariance_type="full", random_state=CONFIG.random_state)
            labels = gm.fit_predict(X)
            rows.append(evaluate_labels("gmm", f"GaussianMixture k={k}", labels, X))
            labels_by_model[f"gmm_{k}"] = labels
        except Exception as exc:
            logger.warning("GMM k=%s failed: %s", k, exc)

    # Agglomerative on smaller datasets or sample labels nearest-centroid fallback
    if len(X) <= 30_000:
        for k in range(CONFIG.min_k, min(CONFIG.max_k, 8) + 1):
            try:
                ag = AgglomerativeClustering(n_clusters=k)
                labels = ag.fit_predict(X)
                rows.append(evaluate_labels("agg", f"Agglomerative k={k}", labels, X))
                labels_by_model[f"agg_{k}"] = labels
            except Exception as exc:
                logger.warning("Agglomerative k=%s failed: %s", k, exc)

    # DBSCAN candidates. Only useful if structure is density-based.
    if len(X_fit) >= 100:
        for eps in [0.6, 0.9, 1.2, 1.8, 2.5]:
            try:
                db = DBSCAN(eps=eps, min_samples=max(5, int(math.log(len(X_fit)))))
                labels_sample = db.fit_predict(X_fit)
                n_clusters = len(set(labels_sample)) - (1 if -1 in labels_sample else 0)
                if n_clusters >= 2:
                    # Use DBSCAN only on full set when not too large.
                    if len(X) <= 50_000:
                        labels = DBSCAN(eps=eps, min_samples=max(5, int(math.log(len(X))))).fit_predict(X)
                        rows.append(evaluate_labels("dbscan", f"DBSCAN eps={eps}", labels, X))
                        labels_by_model[f"dbscan_{str(eps).replace('.', '_')}"] = labels
            except Exception as exc:
                logger.warning("DBSCAN eps=%s failed: %s", eps, exc)

    if not rows:
        raise RuntimeError("No clustering model succeeded.")

    # Add model keys after evaluation
    for row in rows:
        if row["algorithm"] == "kmeans":
            row["model_key"] = f"kmeans_{row['n_clusters']}"
        elif row["algorithm"] == "gmm":
            row["model_key"] = f"gmm_{row['n_clusters']}"
        elif row["algorithm"] == "agg":
            row["model_key"] = f"agg_{row['n_clusters']}"
        elif row["algorithm"] == "dbscan":
            eps = row["model_name"].split("eps=")[-1].replace(".", "_")
            row["model_key"] = f"dbscan_{eps}"

    valid_rows = [r for r in rows if r.get("is_valid", False)]
    if not valid_rows:
        valid_rows = rows

    # Rank: high silhouette & CH, low DB, no tiny cluster, interpretable cluster count.
    scored = rank_cluster_results(valid_rows)
    best = scored[0]
    result = {
        "best_model": best,
        "all_models": scored,
        "selection_logic": (
            "Weighted ranking using Silhouette, Davies-Bouldin, Calinski-Harabasz, "
            "minimum cluster size and interpretability."
        ),
    }
    return result, labels_by_model


def sample_array(X: np.ndarray, n: int) -> np.ndarray:
    if len(X) <= n:
        return X
    rng = np.random.RandomState(CONFIG.random_state)
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx]


def evaluate_labels(algorithm: str, model_name: str, labels: np.ndarray, X: np.ndarray) -> Dict[str, Any]:
    labels = np.asarray(labels)
    unique = sorted([x for x in set(labels) if x != -1])
    n_clusters = len(unique)
    noise_pct = float((labels == -1).mean()) if -1 in labels else 0.0
    cluster_counts = pd.Series(labels).value_counts(normalize=True).to_dict()
    min_cluster_pct = min([v for k, v in cluster_counts.items() if k != -1], default=0.0)
    valid = n_clusters >= 2 and min_cluster_pct >= CONFIG.min_cluster_pct and noise_pct <= 0.4

    metrics = {
        "algorithm": algorithm,
        "model_name": model_name,
        "n_clusters": int(n_clusters),
        "noise_pct": round(noise_pct, 4),
        "min_cluster_pct": round(float(min_cluster_pct), 4),
        "is_valid": bool(valid),
        "cluster_distribution": {str(k): round(float(v), 4) for k, v in cluster_counts.items()},
    }

    if n_clusters >= 2:
        X_eval, labels_eval = sample_for_metrics(X, labels, CONFIG.silhouette_sample_size)
        try:
            metrics["silhouette"] = round(float(silhouette_score(X_eval, labels_eval)), 4)
        except Exception:
            metrics["silhouette"] = None
        try:
            metrics["davies_bouldin"] = round(float(davies_bouldin_score(X_eval, labels_eval)), 4)
        except Exception:
            metrics["davies_bouldin"] = None
        try:
            metrics["calinski_harabasz"] = round(float(calinski_harabasz_score(X_eval, labels_eval)), 4)
        except Exception:
            metrics["calinski_harabasz"] = None
    else:
        metrics["silhouette"] = None
        metrics["davies_bouldin"] = None
        metrics["calinski_harabasz"] = None
    return metrics


def sample_for_metrics(X: np.ndarray, labels: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    mask = labels != -1
    X2, y2 = X[mask], labels[mask]
    if len(X2) <= n:
        return X2, y2
    rng = np.random.RandomState(CONFIG.random_state)
    idx = rng.choice(len(X2), size=n, replace=False)
    return X2[idx], y2[idx]


def rank_cluster_results(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    df = pd.DataFrame(rows).copy()

    def norm_high(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        if s.max() == s.min():
            return pd.Series(0.5, index=s.index)
        return (s - s.min()) / (s.max() - s.min())

    def norm_low(s: pd.Series) -> pd.Series:
        return 1 - norm_high(s)

    df["sil_score"] = norm_high(df["silhouette"].fillna(df["silhouette"].min()))
    df["db_score"] = norm_low(df["davies_bouldin"].fillna(df["davies_bouldin"].max()))
    df["ch_score"] = norm_high(np.log1p(df["calinski_harabasz"].fillna(0)))
    df["size_score"] = df["min_cluster_pct"].clip(0, 0.15) / 0.15
    df["k_score"] = df["n_clusters"].apply(lambda k: 1.0 if 3 <= k <= 6 else 0.7 if 2 <= k <= 8 else 0.4)
    df["business_score"] = (
        0.35 * df["sil_score"] +
        0.20 * df["db_score"] +
        0.15 * df["ch_score"] +
        0.15 * df["size_score"] +
        0.15 * df["k_score"]
    )
    df = df.sort_values("business_score", ascending=False)
    return df.to_dict(orient="records")


# =============================================================================
# Agent 5 — Segment Interpreter
# =============================================================================

def interpret_segments() -> str:
    try:
        clustered = pd.read_parquet(out_path("clustered"))
        profiles = build_segment_profiles(clustered)
        profiles = enrich_profiles_with_ai(profiles, clustered)
        save_json(out_path("segment_profiles"), profiles)
        out_path("segment_md").write_text(render_segment_profiles_md(profiles), encoding="utf-8")
        out_path("strategy_md").write_text(render_business_strategy_md(profiles), encoding="utf-8")
        return f"SEGMENT_SUCCESS: segments={len(profiles['segments'])}, file={out_path('segment_md')}"
    except Exception as exc:
        return f"SEGMENT_ERROR: {exc}\n{traceback.format_exc()}"


def build_segment_profiles(clustered: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols = [c for c in clustered.select_dtypes(include="number").columns if c != "cluster"]
    overall = clustered[numeric_cols].median(numeric_only=True).to_dict()
    segments = []
    for cluster_id, g in clustered.groupby("cluster"):
        seg = {
            "cluster": int(cluster_id),
            "size": int(len(g)),
            "size_pct": round(float(len(g) / len(clustered)), 4),
            "metrics": {},
            "drivers_high": [],
            "drivers_low": [],
        }
        med = g[numeric_cols].median(numeric_only=True).to_dict()
        for col in numeric_cols:
            val = med.get(col)
            base = overall.get(col)
            if base is None or pd.isna(base) or abs(base) < 1e-9:
                ratio = None
            else:
                ratio = float(val / base)
            seg["metrics"][col] = {
                "median": round(float(val), 4) if pd.notna(val) else None,
                "overall_median": round(float(base), 4) if pd.notna(base) else None,
                "ratio_vs_overall": round(ratio, 3) if ratio is not None and np.isfinite(ratio) else None,
            }
        scored = []
        for col, item in seg["metrics"].items():
            ratio = item["ratio_vs_overall"]
            if ratio is not None:
                scored.append((abs(math.log(max(ratio, 1e-9))), ratio, col))
        scored.sort(reverse=True)
        seg["drivers_high"] = [col for _, ratio, col in scored if ratio > 1.15][:5]
        seg["drivers_low"] = [col for _, ratio, col in scored if ratio < 0.85][:5]
        name, action = deterministic_segment_name(seg)
        seg["segment_name"] = name
        seg["recommended_action"] = action
        segments.append(seg)
    return {
        "project": CONFIG.project_name,
        "n_entities": int(len(clustered)),
        "segments": segments,
    }


def deterministic_segment_name(seg: Dict[str, Any]) -> Tuple[str, str]:
    high = set(seg.get("drivers_high", []))
    low = set(seg.get("drivers_low", []))
    metrics = seg.get("metrics", {})

    def ratio(col: str) -> float:
        return metrics.get(col, {}).get("ratio_vs_overall") or 1.0

    if ratio("total_revenue") > 1.5 or ratio("monetary_log") > 1.25:
        return "High-Value Customers", "Protect and expand with premium retention, cross-sell and priority service."
    if ratio("recency_days") > 1.5 and (ratio("total_revenue") > 1.0 or ratio("n_orders") > 1.0):
        return "At-Risk Valuable Customers", "Launch win-back campaigns and personalized incentives before churn risk increases."
    if ratio("n_orders") > 1.4 or ratio("frequency_log") > 1.2:
        return "Frequent Buyers", "Increase basket size through bundles, recommendations and loyalty mechanics."
    if ratio("product_diversity") > 1.4:
        return "Diverse Explorers", "Use assortment-based recommendations and category expansion campaigns."
    if ratio("total_revenue") < 0.75 and ratio("n_orders") < 0.85:
        return "Low Engagement Customers", "Use low-cost nurturing campaigns and test onboarding offers."
    return "Balanced Mainstream Customers", "Maintain engagement and test incremental offers with controlled discounting."


def enrich_profiles_with_ai(profiles: Dict[str, Any], clustered: pd.DataFrame) -> Dict[str, Any]:
    compact = []
    for s in profiles["segments"]:
        compact.append({
            "cluster": s["cluster"],
            "size_pct": s["size_pct"],
            "drivers_high": s["drivers_high"],
            "drivers_low": s["drivers_low"],
            "default_name": s["segment_name"],
            "default_action": s["recommended_action"],
        })

    prompt = f"""
You are a Revenue Growth Management data scientist.

Business context:
{read_business_context()}

Segments discovered by unsupervised learning:
{json.dumps(compact, indent=2, default=json_safe)}

Task:
Rename each segment with a business-friendly name and recommend one action.
Actions should be practical: retention, cross-sell, discount policy, lifecycle, communication, sales prioritization.

Return ONLY JSON:
{{
  "segments": [
    {{"cluster": 0, "segment_name": "...", "persona": "...", "recommended_action": "...", "business_risk": "...", "business_opportunity": "..."}}
  ],
  "executive_summary": "..."
}}
"""
    parsed = parse_json_from_text(ask_claude(prompt, max_tokens=2500))
    if not parsed or "segments" not in parsed:
        profiles["executive_summary"] = (
            "The clustering pipeline identified behavior-based customer segments with different revenue, "
            "frequency, recency and product diversity patterns. Each segment can be translated into targeted RGM actions."
        )
        return profiles

    ai_by_cluster = {int(s["cluster"]): s for s in parsed.get("segments", []) if "cluster" in s}
    for seg in profiles["segments"]:
        ai = ai_by_cluster.get(seg["cluster"])
        if ai:
            for key in ["segment_name", "persona", "recommended_action", "business_risk", "business_opportunity"]:
                if ai.get(key):
                    seg[key] = ai[key]
    profiles["executive_summary"] = parsed.get("executive_summary", "")
    return profiles


def render_segment_profiles_md(profiles: Dict[str, Any]) -> str:
    lines = ["# Segment Profiles — Unsupervised Learning\n"]
    lines.append(f"**Project:** {profiles['project']}\n")
    lines.append(f"**Entities clustered:** {profiles['n_entities']:,}\n")
    if profiles.get("executive_summary"):
        lines.append(f"## Executive Summary\n{profiles['executive_summary']}\n")
    for seg in sorted(profiles["segments"], key=lambda x: x["cluster"]):
        lines.append(f"## Cluster {seg['cluster']} — {seg.get('segment_name','Segment')}\n")
        lines.append(f"- Size: **{seg['size']:,}** ({seg['size_pct']:.1%})\n")
        if seg.get("persona"):
            lines.append(f"- Persona: {seg['persona']}\n")
        lines.append(f"- High drivers: {', '.join(seg.get('drivers_high', [])) or 'N/A'}\n")
        lines.append(f"- Low drivers: {', '.join(seg.get('drivers_low', [])) or 'N/A'}\n")
        lines.append(f"- Recommended action: **{seg.get('recommended_action','N/A')}**\n")
        if seg.get("business_risk"):
            lines.append(f"- Business risk: {seg['business_risk']}\n")
        if seg.get("business_opportunity"):
            lines.append(f"- Business opportunity: {seg['business_opportunity']}\n")
        top_metrics = []
        for col, item in seg["metrics"].items():
            r = item.get("ratio_vs_overall")
            if r is not None and (r >= 1.25 or r <= 0.75):
                top_metrics.append({"feature": col, "median": item["median"], "ratio_vs_overall": r})
        if top_metrics:
            lines.append("\nKey metrics:\n")
            lines.append(safe_markdown_table(pd.DataFrame(top_metrics).head(12), 12))
            lines.append("\n")
    return "\n".join(lines)


def render_business_strategy_md(profiles: Dict[str, Any]) -> str:
    rows = []
    for seg in profiles["segments"]:
        rows.append({
            "cluster": seg["cluster"],
            "segment": seg.get("segment_name"),
            "size_pct": f"{seg['size_pct']:.1%}",
            "action": seg.get("recommended_action"),
            "opportunity": seg.get("business_opportunity", ""),
            "risk": seg.get("business_risk", ""),
        })
    return f"""# Business Strategy — RGM Actions by Segment

This file translates unsupervised learning output into practical business actions.

{safe_markdown_table(pd.DataFrame(rows), max_rows=50)}

## Suggested Next Steps
1. Validate segment definitions with business stakeholders.
2. Run controlled campaigns by segment and measure uplift.
3. Monitor segment migration monthly.
4. Add margin, channel and promotion variables to improve RGM decisions.
5. Connect the segmentation table to CRM/BI tools for operational use.
"""


# =============================================================================
# Agent 6 — Hypothesis Validator by Segment
# =============================================================================

def validate_segment_hypotheses() -> str:
    try:
        clustered = pd.read_parquet(out_path("clustered"))
        tests = []
        numeric_cols = [c for c in clustered.select_dtypes(include="number").columns if c != "cluster"]
        priority = [
            "total_revenue", "n_orders", "recency_days", "avg_order_value", "product_diversity",
            "return_rate", "frequency_per_month", "total_quantity",
        ]
        ordered = [c for c in priority if c in numeric_cols] + [c for c in numeric_cols if c not in priority]
        for col in ordered[:12]:
            groups = [g[col].dropna().values for _, g in clustered.groupby("cluster") if g[col].dropna().size > 5]
            if len(groups) >= 2:
                try:
                    stat, p = ss.kruskal(*groups)
                    medians = clustered.groupby("cluster")[col].median().to_dict()
                    top_cluster = max(medians, key=medians.get)
                    tests.append({
                        "hypothesis": f"Segments differ significantly in {col}.",
                        "feature": col,
                        "test": "Kruskal-Wallis",
                        "p_value": float(p),
                        "verdict": "TRUE" if p < 0.05 else "FALSE",
                        "business_insight": (
                            f"Cluster {top_cluster} has the highest median {col}; this feature helps explain segment behavior."
                            if p < 0.05 else f"{col} does not strongly differentiate the current segments."
                        ),
                    })
                except Exception:
                    pass
        save_json(out_path("hypotheses_json"), tests)
        out_path("hypotheses_md").write_text(render_hypotheses_md(tests), encoding="utf-8")
        return f"HYPOTHESIS_SUCCESS: tests={len(tests)}, file={out_path('hypotheses_md')}"
    except Exception as exc:
        return f"HYPOTHESIS_ERROR: {exc}\n{traceback.format_exc()}"


def render_hypotheses_md(tests: List[Dict[str, Any]]) -> str:
    df = pd.DataFrame(tests)
    if not df.empty:
        df["p_value"] = df["p_value"].map(lambda x: f"{x:.4g}")
    return f"""# Segment Hypothesis Validation

These statistical tests check whether the discovered clusters differ significantly on key business variables.

{safe_markdown_table(df, max_rows=50) if not df.empty else 'No tests were generated.'}
"""


# =============================================================================
# Charts
# =============================================================================

def generate_cluster_charts(clustered: pd.DataFrame, matrix: pd.DataFrame, labels: np.ndarray) -> None:
    X = matrix.values
    if len(X) > 50_000:
        rng = np.random.RandomState(CONFIG.random_state)
        idx = rng.choice(len(X), size=50_000, replace=False)
        X_plot = X[idx]
        labels_plot = labels[idx]
    else:
        X_plot = X
        labels_plot = labels

    # PCA map
    pca = PCA(n_components=2, random_state=CONFIG.random_state)
    coords = pca.fit_transform(X_plot)
    plt.figure(figsize=(10, 7))
    plt.scatter(coords[:, 0], coords[:, 1], c=labels_plot, s=8, alpha=0.65)
    plt.title("Cluster Map — PCA 2D Projection")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_path("pca_png"), dpi=160)
    plt.close()

    # Cluster sizes
    counts = clustered["cluster"].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title("Cluster Sizes")
    plt.xlabel("Cluster")
    plt.ylabel("Number of entities")
    plt.tight_layout()
    plt.savefig(out_path("sizes_png"), dpi=160)
    plt.close()

    # RFM profile if available
    rfm_cols = [c for c in ["recency_days", "n_orders", "total_revenue", "avg_order_value", "product_diversity"] if c in clustered.columns]
    if rfm_cols:
        prof = clustered.groupby("cluster")[rfm_cols].median()
        prof_norm = (prof - prof.min()) / (prof.max() - prof.min()).replace(0, 1)
        plt.figure(figsize=(10, 5))
        if sns is not None:
            sns.heatmap(prof_norm, annot=True, fmt=".2f", linewidths=0.4)
        else:
            plt.imshow(prof_norm.values, aspect="auto")
            plt.xticks(range(len(prof_norm.columns)), prof_norm.columns, rotation=45, ha="right")
            plt.yticks(range(len(prof_norm.index)), prof_norm.index)
            plt.colorbar()
        plt.title("Normalized Segment Profile — Key RFM Features")
        plt.tight_layout()
        plt.savefig(out_path("rfm_png"), dpi=160)
        plt.close()

    # Feature heatmap top numeric
    numeric_cols = [c for c in clustered.select_dtypes(include="number").columns if c != "cluster"][:15]
    if numeric_cols:
        prof = clustered.groupby("cluster")[numeric_cols].median()
        prof_norm = (prof - prof.min()) / (prof.max() - prof.min()).replace(0, 1)
        plt.figure(figsize=(max(10, len(numeric_cols) * 0.7), 6))
        if sns is not None:
            sns.heatmap(prof_norm, cmap="viridis", linewidths=0.4)
        else:
            plt.imshow(prof_norm.values, aspect="auto")
            plt.xticks(range(len(prof_norm.columns)), prof_norm.columns, rotation=45, ha="right")
            plt.yticks(range(len(prof_norm.index)), prof_norm.index)
            plt.colorbar()
        plt.title("Cluster Feature Heatmap")
        plt.tight_layout()
        plt.savefig(out_path("heatmap_png"), dpi=160)
        plt.close()


# =============================================================================
# Agent 7 — App, README, Notebook
# =============================================================================

def generate_deliverables() -> str:
    try:
        generate_streamlit_app()
        generate_requirements()
        generate_notebook()
        generate_readme()
        return (
            "DELIVERABLES_SUCCESS: "
            f"app={out_path('app')}, notebook={out_path('notebook')}, readme={out_path('readme')}"
        )
    except Exception as exc:
        return f"DELIVERABLES_ERROR: {exc}\n{traceback.format_exc()}"


def generate_streamlit_app() -> None:
    app_code = f'''import json
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

BASE = Path(__file__).parent

st.set_page_config(page_title="Auto Segmenter AI", layout="wide")
st.title("Auto Segmenter AI — Customer Segmentation")
st.caption("Multi-agent unsupervised learning pipeline for RGM-style segmentation")

clustered_path = BASE / "{ARTIFACTS['clustered']}"
profiles_path = BASE / "{ARTIFACTS['segment_profiles']}"
metrics_path = BASE / "{ARTIFACTS['metrics']}"

if not clustered_path.exists():
    st.error("df4_clustered_customers.parquet not found. Run the pipeline first.")
    st.stop()

clustered = pd.read_parquet(clustered_path)
profiles = json.loads(profiles_path.read_text(encoding="utf-8")) if profiles_path.exists() else {{"segments": []}}
metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {{}}

segments = {{s["cluster"]: s for s in profiles.get("segments", [])}}

st.sidebar.header("Filters")
cluster_options = sorted(clustered["cluster"].unique().tolist())
selected_clusters = st.sidebar.multiselect("Clusters", cluster_options, default=cluster_options)
filtered = clustered[clustered["cluster"].isin(selected_clusters)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Entities", f"{{len(filtered):,}}")
c2.metric("Clusters", filtered["cluster"].nunique())
if "total_revenue" in filtered.columns:
    c3.metric("Total Revenue", f"{{filtered['total_revenue'].sum():,.0f}}")
if "avg_order_value" in filtered.columns:
    c4.metric("Avg Order Value", f"{{filtered['avg_order_value'].median():,.2f}}")

st.subheader("Cluster Summary")
summary_rows = []
for cid in cluster_options:
    s = segments.get(cid, {{}})
    summary_rows.append({{
        "cluster": cid,
        "segment": s.get("segment_name", f"Cluster {{cid}}"),
        "size": int((clustered["cluster"] == cid).sum()),
        "size_pct": f"{{(clustered['cluster'] == cid).mean():.1%}}",
        "recommended_action": s.get("recommended_action", ""),
    }})
st.dataframe(pd.DataFrame(summary_rows), width="stretch")

left, right = st.columns(2)
with left:
    counts = filtered["cluster"].value_counts().reset_index()
    counts.columns = ["cluster", "count"]
    st.plotly_chart(px.bar(counts, x="cluster", y="count", title="Cluster Sizes"), width="stretch")
with right:
    numeric_cols = [c for c in filtered.select_dtypes(include="number").columns if c != "cluster"]
    y_col = "total_revenue" if "total_revenue" in numeric_cols else numeric_cols[0] if numeric_cols else None
    if y_col:
        st.plotly_chart(px.box(filtered, x="cluster", y=y_col, title=f"{{y_col}} by Cluster"), width="stretch")

st.subheader("Segment Deep Dive")
selected = st.selectbox("Choose a cluster", cluster_options)
seg = segments.get(selected, {{}})
st.markdown(f"### Cluster {{selected}} — {{seg.get('segment_name', 'Segment')}}")
st.write(seg.get("persona", ""))
st.info(seg.get("recommended_action", "No recommendation available."))
if seg.get("business_opportunity"):
    st.success("Opportunity: " + seg["business_opportunity"])
if seg.get("business_risk"):
    st.warning("Risk: " + seg["business_risk"])

cluster_data = clustered[clustered["cluster"] == selected]
st.dataframe(cluster_data.head(100), width="stretch")

st.subheader("Model Selection")
if metrics.get("best_model"):
    st.json(metrics["best_model"])

st.subheader("Download")
st.download_button(
    "Download clustered customers as CSV",
    data=clustered.to_csv(index=False).encode("utf-8"),
    file_name="clustered_customers.csv",
    mime="text/csv",
)
'''
    out_path("app").write_text(app_code, encoding="utf-8")


def generate_requirements() -> None:
    req = """pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.4.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyarrow>=14.0.0
python-dotenv>=1.0.0
kagglehub>=0.2.0
anthropic>=0.25.0
streamlit>=1.32.0
plotly>=5.18.0
nbformat>=5.9.0
openpyxl>=3.1.0
"""
    out_path("requirements").write_text(req, encoding="utf-8")


def generate_notebook() -> None:
    try:
        import nbformat
        from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
    except Exception:
        logger.warning("nbformat not installed; skipping notebook generation.")
        return

    cells = [
        new_markdown_cell(f"# {CONFIG.project_name}\n\nMulti-agent unsupervised learning pipeline for customer/entity segmentation."),
        new_code_cell("import pandas as pd\nfrom pathlib import Path\nBASE = Path('.')"),
        new_markdown_cell("## 1. Load clustered customers"),
        new_code_cell(f"df = pd.read_parquet('{ARTIFACTS['clustered']}')\ndf.head()"),
        new_markdown_cell("## 2. Segment sizes"),
        new_code_cell("df['cluster'].value_counts(normalize=True).sort_index()"),
        new_markdown_cell("## 3. Segment profile"),
        new_code_cell("numeric_cols = [c for c in df.select_dtypes('number').columns if c != 'cluster']\ndf.groupby('cluster')[numeric_cols].median().T.head(30)"),
        new_markdown_cell("## 4. Business interpretation\nSee `Segment_Profiles.md` and `Business_Strategy.md`."),
        new_code_cell("from IPython.display import Markdown, display\ndisplay(Markdown(Path('Segment_Profiles.md').read_text(encoding='utf-8')))"),
    ]
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    with open(out_path("notebook"), "w", encoding="utf-8") as f:
        nbformat.write(nb, f)


def generate_readme() -> None:
    """Generate both output README and root README.md for GitHub portfolio display."""
    output_readme = build_portfolio_readme(asset_prefix="")
    root_readme = build_portfolio_readme(asset_prefix=f"{CONFIG.output_dir}/")

    out_path("readme").write_text(output_readme, encoding="utf-8")
    Path("README.md").write_text(root_readme, encoding="utf-8")


def build_portfolio_readme(asset_prefix: str = "") -> str:
    metrics = load_json(out_path("metrics")) if out_path("metrics").exists() else {}
    best = metrics.get("best_model", {})
    cfg = load_json(out_path("entity_config")) if out_path("entity_config").exists() else {}
    profiles = load_json(out_path("segment_profiles")) if out_path("segment_profiles").exists() else {"segments": []}

    clustered_shape = "N/A"
    raw_shape = "N/A"
    feature_shape = "N/A"
    try:
        if out_path("silver").exists():
            raw_shape = f"{pd.read_parquet(out_path('silver')).shape[0]:,} rows"
        if out_path("features").exists():
            fdf = pd.read_parquet(out_path("features"))
            feature_shape = f"{fdf.shape[0]:,} entities × {fdf.shape[1]:,} features"
        if out_path("clustered").exists():
            cdf = pd.read_parquet(out_path("clustered"))
            clustered_shape = f"{cdf.shape[0]:,} entities"
    except Exception:
        pass

    segment_rows = []
    for s in profiles.get("segments", []):
        segment_rows.append({
            "Cluster": s.get("cluster"),
            "Segment": s.get("segment_name", f"Cluster {s.get('cluster')}"),
            "Size": f"{s.get('size', 0):,}",
            "Size %": f"{s.get('size_pct', 0):.1%}",
            "Recommended Action": s.get("recommended_action", ""),
        })
    segment_table = safe_markdown_table(pd.DataFrame(segment_rows), max_rows=20) if segment_rows else "Segments are generated after running the pipeline."

    all_models = metrics.get("all_models", [])
    model_rows = []
    for m in all_models[:8]:
        model_rows.append({
            "Model": m.get("model_name"),
            "Clusters": m.get("n_clusters"),
            "Silhouette": m.get("silhouette"),
            "Davies-Bouldin": m.get("davies_bouldin"),
            "Calinski-Harabasz": m.get("calinski_harabasz"),
            "Business Score": round(float(m.get("business_score", 0)), 4) if m.get("business_score") is not None else None,
        })
    model_table = safe_markdown_table(pd.DataFrame(model_rows), max_rows=10) if model_rows else "Model metrics are generated after running the pipeline."

    best_model = best.get("model_name", "N/A")
    best_clusters = best.get("n_clusters", "N/A")
    best_silhouette = best.get("silhouette", "N/A")
    best_db = best.get("davies_bouldin", "N/A")
    best_ch = best.get("calinski_harabasz", "N/A")

    return f"""# Auto Segmenter AI — Multi-Agent Unsupervised Learning Pipeline

> **Portfolio Project:** {CONFIG.project_name}

## Executive Summary

Retail and e-commerce businesses generate thousands of transactions, but most behavioral data remains underused for customer strategy. This project solves that gap by building a **multi-agent unsupervised learning pipeline** that transforms raw transactional data into customer-level intelligence.

The pipeline automatically ingests data, detects the segmentation entity, engineers RFM and behavioral features, compares multiple clustering algorithms, selects the best segmentation strategy, interprets each segment in business language, validates segment hypotheses, and generates a portfolio-ready Streamlit app, notebook and README.

The final output is not only a clustering model. It is a complete decision-support product for **Revenue Growth Management**, supporting actions such as retention, cross-sell, discount control, lifecycle campaigns and customer prioritization.

---

## 1. Project Result

| Item | Result |
|---|---|
| Learning type | Unsupervised Learning |
| Business objective | Customer segmentation for Revenue Growth Intelligence |
| Dataset | `{CONFIG.dataset_slug}` |
| Raw data processed | {raw_shape} |
| Feature table | {feature_shape} |
| Clustered entities | {clustered_shape} |
| Best model | {best_model} |
| Number of clusters | {best_clusters} |
| Silhouette Score | {best_silhouette} |
| Davies-Bouldin | {best_db} |
| Calinski-Harabasz | {best_ch} |
| Output | Segment profiles, business strategy, hypothesis validation, Streamlit app and notebook |

---

## 2. Pipeline Architecture

```text
Kaggle / Local Dataset
        │
        ▼
┌──────────────────┐
│ 1. Ingestor      │  Standardizes raw data and saves the silver layer
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 2. Data Profiler │  Detects entity, date, monetary and product columns
└────────┬─────────┘
         ▼
┌──────────────────────┐
│ 3. Feature Engineer  │  Builds RFM and behavioral customer features
└────────┬─────────────┘
         ▼
┌──────────────────────┐
│ 4. Clustering Agent  │  Tests KMeans, GMM, Agglomerative and DBSCAN
└────────┬─────────────┘
         ▼
┌────────────────────────┐
│ 5. Segment Interpreter │  Converts clusters into business personas/actions
└────────┬───────────────┘
         ▼
┌────────────────────────┐
│ 6. Hypothesis Agent    │  Validates if segments differ statistically
└────────┬───────────────┘
         ▼
┌────────────────────────┐
│ 7. Deliverable Agent   │  Generates app, notebook, README and reports
└────────┬───────────────┘
         ▼
┌────────────────────────┐
│ 8. GitOps Agent        │  Cleans repo, prepares README and optionally pushes
└────────────────────────┘
```

---

## 3. Dataset

- **Kaggle slug:** `{CONFIG.dataset_slug}`
- **URL:** {CONFIG.dataset_url}
- **Local path:** `{CONFIG.local_path}`
- **Detected entity configuration:**

```json
{json.dumps(cfg, indent=2, ensure_ascii=False, default=json_safe)}
```

---

## 4. Customer-Level Feature Engineering

The pipeline converts transaction-level data into an entity-level analytical table.

Examples of generated features:

- `n_transactions`
- `n_orders`
- `total_revenue`
- `avg_order_value`
- `recency_days`
- `customer_lifetime_days`
- `frequency_per_month`
- `product_diversity`
- `return_rate`
- `monetary_log`
- `frequency_log`
- `recency_log`

This makes the clustering more meaningful because the model is not grouping raw rows; it is grouping customer behavior.

---

## 5. Clustering Strategy

The pipeline compares multiple unsupervised algorithms:

- **KMeans**
- **Gaussian Mixture Models**
- **Agglomerative Clustering**
- **DBSCAN**

Model selection is based on a weighted ranking using:

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score
- Minimum cluster size
- Business interpretability of the number of clusters

### Top Model Candidates

{model_table}

---

## 6. Segment Interpretation

The model output is translated into business-friendly segments and actions.

{segment_table}

The objective is to move beyond “Cluster 0, Cluster 1, Cluster 2” and generate usable segment intelligence for business teams.

---

## 7. Business Impact

This project converts unsupervised machine learning into actions:

- Identify high-value customers for loyalty and VIP campaigns.
- Detect low-frequency or at-risk customers for retention strategies.
- Separate low-engagement customers from high-potential buyers.
- Prioritize cross-sell and product recommendation strategies by segment.
- Support Revenue Growth Management decisions with explainable customer groups.
- Create a reusable segmentation engine that can be rerun when the dataset changes.

---

## 8. Visual Outputs

### Cluster Map — PCA Projection

![Cluster PCA Map]({asset_prefix}cluster_pca_map.png)

### RFM Segment Profile

![Cluster RFM Profile]({asset_prefix}cluster_rfm_profile.png)

### Cluster Size Distribution

![Cluster Sizes]({asset_prefix}cluster_sizes.png)

### Feature Heatmap

![Cluster Feature Heatmap]({asset_prefix}cluster_feature_heatmap.png)

---

## 9. Streamlit App

The pipeline generates an interactive Streamlit app for segment exploration.

Run locally:

```bash
streamlit run {asset_prefix}streamlit_segment_app.py
```

The app allows users to:

- Filter clusters
- Compare segment sizes
- Inspect segment-level metrics
- Read business recommendations
- Download the final clustered customer table

---

## 10. Output Files

| File | Description |
|---|---|
| `df1_silver.parquet` | Clean standardized raw table |
| `df2_customer_features.parquet` | Customer/entity-level features |
| `df3_cluster_matrix.parquet` | Scaled numeric matrix used for clustering |
| `df4_clustered_customers.parquet` | Final segmentation table |
| `cluster_metrics.json` | Model comparison and chosen model |
| `Segment_Profiles.md` | Business explanation of each segment |
| `Business_Strategy.md` | Recommended actions by segment |
| `Segment_Hypothesis_Validation.md` | Statistical validation of segment differences |
| `streamlit_segment_app.py` | Interactive Streamlit app |
| `analysis_notebook_unsupervised.ipynb` | Reproducible notebook |
| `README_unsupervised.md` | Generated project documentation |

---

## 11. How to Reproduce

Install dependencies:

```bash
pip install -r requirements_unsupervised.txt
```

Run with the default Kaggle dataset:

```bash
python multi_agent_ds_unsupervised_v1.py
```

Run with another Kaggle dataset:

```bash
python multi_agent_ds_unsupervised_v1.py --dataset-slug "your/kaggle-dataset"
```

Run with a local file:

```bash
python multi_agent_ds_unsupervised_v1.py --local-path "data.csv"
```

Run without Anthropic/Claude:

```bash
python multi_agent_ds_unsupervised_v1.py --no-ai
```

Run and automatically push the updated project to GitHub:

```bash
python multi_agent_ds_unsupervised_v2_1.py --dataset-slug "your/kaggle-dataset"
```

Run without pushing:

```bash
python multi_agent_ds_unsupervised_v2_1.py --dataset-slug "your/kaggle-dataset" --no-git-push
```

---

## 12. GitOps Agent

The GitOps Agent prepares the repository after every pipeline run:

- Creates/updates `.gitignore`
- Copies the generated README to root `README.md`
- Prevents `.venv`, `.env`, parquet files and local artifacts from being tracked
- Runs `git add`, `git commit` and optionally `git push`

This allows the project to be refreshed whenever the dataset changes.

---

## 13. Limitations and Next Steps

Current limitations:

- Clustering quality depends strongly on the available behavioral features.
- Silhouette Score may be modest in real customer data because behavioral segments often overlap.
- Business validation still requires campaign testing and stakeholder review.
- Margin, promotion, channel and customer demographic data would improve RGM recommendations.

Next improvements:

- Add UMAP visualization.
- Add customer migration tracking over time.
- Add margin-aware segmentation.
- Add automated campaign simulation.
- Add CRM-ready output tables.
- Add deployment to Streamlit Cloud.

---

## Why This Project Matters

Most unsupervised learning projects stop at KMeans. This project goes further by combining:

- automated feature engineering,
- clustering model competition,
- statistical validation,
- GenAI-powered segment interpretation,
- business strategy generation,
- Streamlit delivery,
- GitOps automation.

The result is a reusable **multi-agent customer intelligence system** instead of a one-off clustering notebook.
"""



# =============================================================================
# Agent 8 — GitOps Agent
# =============================================================================

def gitops_agent() -> str:
    """
    Full GitOps agent for portfolio publishing.

    Default behavior:
    - update .gitignore
    - update root README.md
    - remove unsafe tracked files from the Git index
    - stage source code, docs, app, notebook and visual outputs
    - commit when there are changes
    - push automatically to GitHub when a remote exists

    Safety rules:
    - Never track .env
    - Never track .venv
    - Avoid parquet/csv/pkl/xlsx/zip outputs in Git
    """
    try:
        if not getattr(CONFIG, "git_enabled", True):
            return "GITOPS_SKIPPED: disabled."

        ensure_gitignore()
        sync_root_readme()

        if not is_git_repository():
            run_cmd(["git", "init"], check=False)

        untrack_unsafe_files()

        current_script = Path(sys.argv[0]).name if sys.argv and sys.argv[0] else "multi_agent_ds_unsupervised_v2_1.py"

        paths_to_add = [
            ".gitignore",
            "README.md",
            current_script,
            "requirements_unsupervised.txt",
            str(CONFIG.out / "README_unsupervised.md"),
            str(CONFIG.out / "Data_Profile.md"),
            str(CONFIG.out / "Feature_Engineering_Report.md"),
            str(CONFIG.out / "Segment_Profiles.md"),
            str(CONFIG.out / "Business_Strategy.md"),
            str(CONFIG.out / "Segment_Hypothesis_Validation.md"),
            str(CONFIG.out / "cluster_metrics.json"),
            str(CONFIG.out / "segment_hypotheses.json"),
            str(CONFIG.out / "segment_profiles.json"),
            str(CONFIG.out / "entity_config.json"),
            str(CONFIG.out / "pipeline_run_results.json"),
            str(CONFIG.out / "cluster_pca_map.png"),
            str(CONFIG.out / "cluster_sizes.png"),
            str(CONFIG.out / "cluster_rfm_profile.png"),
            str(CONFIG.out / "cluster_feature_heatmap.png"),
            str(CONFIG.out / "streamlit_segment_app.py"),
            str(CONFIG.out / "analysis_notebook_unsupervised.ipynb"),
        ]

        existing = [p for p in paths_to_add if Path(p).exists()]
        if existing:
            run_cmd(["git", "add", *existing], check=False)

        status = run_cmd(["git", "status", "--porcelain"], check=False)

        commit_result = "No local changes to commit."
        if status.strip():
            msg = CONFIG.git_commit_message or build_git_commit_message()
            commit_result = run_cmd(["git", "commit", "-m", msg], check=False)

        if not CONFIG.git_push:
            return f"GITOPS_SUCCESS: local repository updated. Push disabled.\n{commit_result[-1000:]}"

        remotes = run_cmd(["git", "remote"], check=False).split()
        if CONFIG.git_remote not in remotes:
            return (
                "GITOPS_SUCCESS_WITH_WARNING: committed locally, but push was skipped because "
                f"remote '{CONFIG.git_remote}' was not found. Add it once with:\n"
                f"git remote add {CONFIG.git_remote} https://github.com/YOUR_USER/YOUR_REPO.git\n"
                f"{commit_result[-1000:]}"
            )

        run_cmd(["git", "branch", "-M", CONFIG.git_branch], check=False)
        push_out = run_cmd(["git", "push", "-u", CONFIG.git_remote, CONFIG.git_branch], check=False)

        return (
            f"GITOPS_SUCCESS: repository updated and pushed to "
            f"{CONFIG.git_remote}/{CONFIG.git_branch}.\n"
            f"{commit_result[-800:]}\n{push_out[-1200:]}"
        )

    except Exception as exc:
        return f"GITOPS_ERROR: {exc}\n{traceback.format_exc()}"


def ensure_gitignore() -> None:
    lines = [
        ".venv/",
        "venv/",
        "__pycache__/",
        ".env",
        "*.log",
        "*.pkl",
        "*.parquet",
        "*.csv",
        "*.xlsx",
        "*.zip",
        ".kaggle/",
        ".ipynb_checkpoints/",
        "unsupervised_outputs/*.parquet",
        "unsupervised_outputs/*.csv",
        "unsupervised_outputs/*.pkl",
        "unsupervised_outputs/*.xlsx",
    ]
    path = Path(".gitignore")
    current = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    merged = list(dict.fromkeys(current + lines))
    path.write_text("\n".join(merged) + "\n", encoding="utf-8")


def sync_root_readme() -> None:
    if out_path("readme").exists():
        Path("README.md").write_text(out_path("readme").read_text(encoding="utf-8"), encoding="utf-8")


def is_git_repository() -> bool:
    out = run_cmd(["git", "rev-parse", "--is-inside-work-tree"], check=False)
    return "true" in out.lower()


def untrack_unsafe_files() -> None:
    tracked = run_cmd(["git", "ls-files"], check=False)
    if not tracked.strip():
        return

    unsafe = []
    for raw in tracked.splitlines():
        p = raw.strip().replace("\\", "/")
        if (
            p.startswith(".venv/")
            or p.startswith("venv/")
            or p == ".env"
            or p.endswith(".parquet")
            or p.endswith(".csv")
            or p.endswith(".pkl")
            or p.endswith(".xlsx")
            or p.endswith(".zip")
            or p.endswith(".log")
        ):
            unsafe.append(raw.strip())

    for chunk in chunked(unsafe, 50):
        run_cmd(["git", "rm", "--cached", "--ignore-unmatch", *chunk], check=False)


def build_git_commit_message() -> str:
    dataset = CONFIG.local_path if CONFIG.local_path else CONFIG.dataset_slug
    return f"Auto-refresh unsupervised segmentation project for {dataset}"


def run_cmd(cmd: List[str], check: bool = False) -> str:
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{output}")
    return output


def chunked(items: List[str], n: int) -> List[List[str]]:
    return [items[i:i+n] for i in range(0, len(items), n)]


# =============================================================================
# Orchestration
# =============================================================================

PIPELINE_STEPS = [
    ("Ingestor", download_and_save_silver),
    ("Data Profiler", profile_and_detect_entity),
    ("Feature Engineer", build_entity_features),
    ("Clustering Scientist", run_clustering_models),
    ("Segment Interpreter", interpret_segments),
    ("Hypothesis Validator", validate_segment_hypotheses),
    ("Deliverable Generator", generate_deliverables),
    ("GitOps Agent", gitops_agent),
]


def run_pipeline() -> None:
    ensure_output_dir()
    logger.info("Starting Auto Segmenter AI v2.1.1")
    logger.info("Output directory: %s", CONFIG.out.resolve())
    results = []
    for name, fn in PIPELINE_STEPS:
        logger.info("========== %s ==========" , name)
        result = fn()
        logger.info(result)
        results.append({"step": name, "result": result})
        if "ERROR" in result:
            save_json(CONFIG.out / "pipeline_run_results.json", results)
            raise RuntimeError(f"Pipeline stopped at step {name}: {result}")
    save_json(CONFIG.out / "pipeline_run_results.json", results)
    logger.info("Pipeline completed successfully.")
    logger.info("Open app with: streamlit run %s", out_path("app"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto Segmenter AI — Unsupervised Learning Pipeline")
    parser.add_argument("--dataset-slug", default=CONFIG.dataset_slug, help="Kaggle dataset slug")
    parser.add_argument("--local-path", default=None, help="Local CSV/XLSX/Parquet path. Bypasses Kaggle.")
    parser.add_argument("--output-dir", default=CONFIG.output_dir, help="Output directory")
    parser.add_argument("--max-rows", type=int, default=CONFIG.max_rows, help="Maximum rows to process")
    parser.add_argument("--max-k", type=int, default=CONFIG.max_k, help="Maximum number of clusters to test")
    parser.add_argument("--no-ai", action="store_true", help="Disable Claude/Anthropic interpretation")
    parser.add_argument("--no-git-push", action="store_true", help="Commit locally but do not push to GitHub")
    parser.add_argument("--no-git", action="store_true", help="Disable GitOps agent")
    parser.add_argument("--git-remote", default=CONFIG.git_remote, help="Git remote name, usually origin")
    parser.add_argument("--git-branch", default=CONFIG.git_branch, help="Git branch name")
    parser.add_argument("--git-message", default=None, help="Custom Git commit message")
    return parser.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    CONFIG.dataset_slug = args.dataset_slug
    CONFIG.local_path = args.local_path
    CONFIG.output_dir = args.output_dir
    CONFIG.max_rows = args.max_rows
    CONFIG.max_k = args.max_k
    CONFIG.ai_enabled = not args.no_ai
    CONFIG.git_enabled = not args.no_git
    CONFIG.git_push = not args.no_git_push
    CONFIG.git_remote = args.git_remote
    CONFIG.git_branch = args.git_branch
    CONFIG.git_commit_message = args.git_message


if __name__ == "__main__":
    args = parse_args()
    apply_args(args)
    run_pipeline()
