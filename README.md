# Auto Segmenter AI — Multi-Agent Unsupervised Learning Pipeline

> **Portfolio Project:** Customer Segmentation & Revenue Growth Intelligence

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
| Dataset | `hellbuoy/online-retail-customer-clustering` |
| Raw data processed | 541,909 rows |
| Feature table | 4,372 entities × 23 features |
| Clustered entities | 4,372 entities |
| Best model | KMeans k=3 |
| Number of clusters | 3 |
| Silhouette Score | 0.2197 |
| Davies-Bouldin | 1.5012 |
| Calinski-Harabasz | 1294.7239 |
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

- **Kaggle slug:** `hellbuoy/online-retail-customer-clustering`
- **URL:** https://www.kaggle.com/datasets/hellbuoy/online-retail-customer-clustering
- **Local path:** `None`
- **Detected entity configuration:**

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
  "strategy": "Customer-level segmentation for RGM analytics. 4372 unique customers provide sufficient granularity for behavioral segmentation. Will aggregate transaction data to create RFM features (recency, frequency, monetary), product diversity, geography, and purchase patterns per customer. 25% null customerid requires handling (exclude or separate segment). Negative quantities indicate returns - important feature for churn/satisfaction analysis."
}
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

            Model  Clusters  Silhouette  Davies-Bouldin  Calinski-Harabasz  Business Score
       KMeans k=3         3      0.2197          1.5012          1294.7239          0.9870
       KMeans k=4         4      0.2210          1.4293          1178.0069          0.9372
Agglomerative k=2         2      0.2194          1.5199          1123.9470          0.9256
       KMeans k=2         2      0.2076          1.6717          1347.1366          0.9112
Agglomerative k=3         3      0.1851          1.7274          1084.2550          0.8956
       KMeans k=5         5      0.1879          1.4486          1078.6873          0.8667
Agglomerative k=4         4      0.1799          1.6702           934.0477          0.8213
Agglomerative k=5         5      0.1323          1.7062           844.5064          0.7371

---

## 6. Segment Interpretation

The model output is translated into business-friendly segments and actions.

 Cluster                     Segment  Size Size %                                                                                                                                                                                                         Recommended Action
       0       Established Champions 1,685  38.5%                          Implement VIP retention program with dedicated account management, exclusive early access to new products, and quarterly business reviews to identify cross-sell opportunities and prevent churn.
       1 Dormant & Occasional Buyers 1,738  39.8%                              Deploy automated win-back campaigns with time-limited incentives (15-20% discount on next purchase) and personalized onboarding sequences to drive second and third purchases within 60 days.
       2      Active Variety Seekers   949  21.7% Launch bundle and upsell campaigns targeting higher-margin SKUs within their preferred categories. Use personalized recommendations to trade customers up from low to mid-tier products with 'complete the set' messaging.

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

![Cluster PCA Map](cluster_pca_map.png)

### RFM Segment Profile

![Cluster RFM Profile](cluster_rfm_profile.png)

### Cluster Size Distribution

![Cluster Sizes](cluster_sizes.png)

### Feature Heatmap

![Cluster Feature Heatmap](cluster_feature_heatmap.png)

---

## 9. Streamlit App

The pipeline generates an interactive Streamlit app for segment exploration.

Run locally:

```bash
streamlit run streamlit_segment_app.py
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
