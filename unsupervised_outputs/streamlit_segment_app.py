import json
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

BASE = Path(__file__).parent

st.set_page_config(page_title="Auto Segmenter AI", layout="wide")
st.title("Auto Segmenter AI — Customer Segmentation")
st.caption("Multi-agent unsupervised learning pipeline for RGM-style segmentation")

clustered_path = BASE / "df4_clustered_customers.parquet"
profiles_path = BASE / "segment_profiles.json"
metrics_path = BASE / "cluster_metrics.json"

if not clustered_path.exists():
    st.error("df4_clustered_customers.parquet not found. Run the pipeline first.")
    st.stop()

clustered = pd.read_parquet(clustered_path)
profiles = json.loads(profiles_path.read_text(encoding="utf-8")) if profiles_path.exists() else {"segments": []}
metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

segments = {s["cluster"]: s for s in profiles.get("segments", [])}

st.sidebar.header("Filters")
cluster_options = sorted(clustered["cluster"].unique().tolist())
selected_clusters = st.sidebar.multiselect("Clusters", cluster_options, default=cluster_options)
filtered = clustered[clustered["cluster"].isin(selected_clusters)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Entities", f"{len(filtered):,}")
c2.metric("Clusters", filtered["cluster"].nunique())
if "total_revenue" in filtered.columns:
    c3.metric("Total Revenue", f"{filtered['total_revenue'].sum():,.0f}")
if "avg_order_value" in filtered.columns:
    c4.metric("Avg Order Value", f"{filtered['avg_order_value'].median():,.2f}")

st.subheader("Cluster Summary")
summary_rows = []
for cid in cluster_options:
    s = segments.get(cid, {})
    summary_rows.append({
        "cluster": cid,
        "segment": s.get("segment_name", f"Cluster {cid}"),
        "size": int((clustered["cluster"] == cid).sum()),
        "size_pct": f"{(clustered['cluster'] == cid).mean():.1%}",
        "recommended_action": s.get("recommended_action", ""),
    })
st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

left, right = st.columns(2)
with left:
    counts = filtered["cluster"].value_counts().reset_index()
    counts.columns = ["cluster", "count"]
    st.plotly_chart(px.bar(counts, x="cluster", y="count", title="Cluster Sizes"), use_container_width=True)
with right:
    numeric_cols = [c for c in filtered.select_dtypes(include="number").columns if c != "cluster"]
    y_col = "total_revenue" if "total_revenue" in numeric_cols else numeric_cols[0] if numeric_cols else None
    if y_col:
        st.plotly_chart(px.box(filtered, x="cluster", y=y_col, title=f"{y_col} by Cluster"), use_container_width=True)

st.subheader("Segment Deep Dive")
selected = st.selectbox("Choose a cluster", cluster_options)
seg = segments.get(selected, {})
st.markdown(f"### Cluster {selected} — {seg.get('segment_name', 'Segment')}")
st.write(seg.get("persona", ""))
st.info(seg.get("recommended_action", "No recommendation available."))
if seg.get("business_opportunity"):
    st.success("Opportunity: " + seg["business_opportunity"])
if seg.get("business_risk"):
    st.warning("Risk: " + seg["business_risk"])

cluster_data = clustered[clustered["cluster"] == selected]
st.dataframe(cluster_data.head(100), use_container_width=True)

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
