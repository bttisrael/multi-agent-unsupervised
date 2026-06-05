# Auto Segmenter AI — Multi-Agent Unsupervised Learning

> Portfolio project: **Customer Segmentation & Revenue Growth Intelligence**

This project adapts the previous Auto Data Scientist idea from supervised AutoML into an unsupervised learning product.
Instead of predicting a target, the pipeline discovers natural customer/entity segments and translates them into business actions.

## Business Goal
Identify behavior-based segments that can support RGM, retention, cross-sell, discount strategy and campaign prioritization.

## Dataset
- Kaggle slug: `hellbuoy/online-retail-customer-clustering`
- URL: https://www.kaggle.com/datasets/hellbuoy/online-retail-customer-clustering
- Local path: `None`

## Detected Entity Configuration
```json
{
  "entity_col": "customerid",
  "invoice_col": "invoiceno",
  "date_col": "invoicedate",
  "quantity_col": "quantity",
  "price_col": "unitprice",
  "product_col": "stockcode",
  "country_col": "country",
  "monetary_cols": [
    "quantity",
    "unitprice"
  ],
  "clustering_unit": "customer",
  "strategy": "Customer-level segmentation for RGM analytics. With 4372 unique customers and transactional data spanning invoices, products, and geography, we can derive RFM features (recency, frequency, monetary), basket composition, product diversity, geographic concentration, and return behavior (negative quantities indicate returns). This enables actionable customer segments for prioritized commercial actions and revenue optimization."
}
```

## Best Clustering Model
```json
{
  "algorithm": "kmeans",
  "model_name": "KMeans k=3",
  "n_clusters": 3,
  "noise_pct": 0.0,
  "min_cluster_pct": 0.2171,
  "is_valid": true,
  "cluster_distribution": {
    "1": 0.3975,
    "0": 0.3854,
    "2": 0.2171
  },
  "silhouette": 0.2197,
  "davies_bouldin": 1.5012,
  "calinski_harabasz": 1294.7239,
  "model_key": "kmeans_3",
  "sil_score": 0.9944961896697714,
  "db_score": 0.9644376298348006,
  "ch_score": 0.9735679501856048,
  "size_score": 1.0,
  "k_score": 1.0,
  "business_score": 0.9869963848792208
}
```

## Pipeline Architecture
1. **Ingestor** — downloads/loads data and saves `df1_silver.parquet`.
2. **Data Profiler** — detects entity, date, order, price, quantity and product columns.
3. **Feature Engineer** — creates RFM/entity features.
4. **Clustering Scientist** — tests KMeans, Gaussian Mixture, Agglomerative and DBSCAN.
5. **Segment Interpreter** — profiles segments and optionally uses Claude to create business names/actions.
6. **Hypothesis Validator** — tests whether clusters differ statistically on business variables.
7. **Deliverable Generator** — creates Streamlit app, notebook and README.

## Output Files
| File | Description |
|---|---|
| `df1_silver.parquet` | Clean standardized raw table |
| `df2_customer_features.parquet` | Customer/entity-level features |
| `df3_cluster_matrix.parquet` | Scaled numeric matrix used for clustering |
| `df4_clustered_customers.parquet` | Final segmentation table |
| `cluster_metrics.json` | Model comparison and chosen model |
| `Segment_Profiles.md` | Business explanation of each segment |
| `Business_Strategy.md` | Recommended actions by segment |
| `streamlit_segment_app.py` | Interactive app |
| `analysis_notebook_unsupervised.ipynb` | Notebook for GitHub/portfolio |

## How to Run
```bash
pip install -r requirements_unsupervised.txt
python multi_agent_ds_unsupervised_v1.py
streamlit run streamlit_segment_app.py
```

## Why This Matters
Most unsupervised projects stop at K-Means. This project goes further by combining feature engineering, clustering competition, statistical validation, GenAI interpretation and a deployable app.
