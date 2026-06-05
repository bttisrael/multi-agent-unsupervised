# Segment Hypothesis Validation

These statistical tests check whether the discovered clusters differ significantly on key business variables.

                                               hypothesis                 feature           test    p_value verdict                                                                                       business_insight
          Segments differ significantly in total_revenue.           total_revenue Kruskal-Wallis          0    TRUE           Cluster 0 has the highest median total_revenue; this feature helps explain segment behavior.
               Segments differ significantly in n_orders.                n_orders Kruskal-Wallis          0    TRUE                Cluster 0 has the highest median n_orders; this feature helps explain segment behavior.
           Segments differ significantly in recency_days.            recency_days Kruskal-Wallis 5.216e-138    TRUE            Cluster 1 has the highest median recency_days; this feature helps explain segment behavior.
        Segments differ significantly in avg_order_value.         avg_order_value Kruskal-Wallis 8.421e-267    TRUE         Cluster 0 has the highest median avg_order_value; this feature helps explain segment behavior.
      Segments differ significantly in product_diversity.       product_diversity Kruskal-Wallis          0    TRUE       Cluster 0 has the highest median product_diversity; this feature helps explain segment behavior.
            Segments differ significantly in return_rate.             return_rate Kruskal-Wallis 1.947e-101    TRUE             Cluster 0 has the highest median return_rate; this feature helps explain segment behavior.
    Segments differ significantly in frequency_per_month.     frequency_per_month Kruskal-Wallis  1.516e-66    TRUE     Cluster 1 has the highest median frequency_per_month; this feature helps explain segment behavior.
         Segments differ significantly in total_quantity.          total_quantity Kruskal-Wallis          0    TRUE          Cluster 0 has the highest median total_quantity; this feature helps explain segment behavior.
             Segments differ significantly in customerid.              customerid Kruskal-Wallis  4.689e-29    TRUE              Cluster 2 has the highest median customerid; this feature helps explain segment behavior.
         Segments differ significantly in n_transactions.          n_transactions Kruskal-Wallis          0    TRUE          Cluster 0 has the highest median n_transactions; this feature helps explain segment behavior.
   Segments differ significantly in avg_revenue_per_line.    avg_revenue_per_line Kruskal-Wallis          0    TRUE    Cluster 0 has the highest median avg_revenue_per_line; this feature helps explain segment behavior.
Segments differ significantly in median_revenue_per_line. median_revenue_per_line Kruskal-Wallis          0    TRUE Cluster 0 has the highest median median_revenue_per_line; this feature helps explain segment behavior.
