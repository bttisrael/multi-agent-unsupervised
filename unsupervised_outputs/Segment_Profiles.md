# Segment Profiles — Unsupervised Learning

**Project:** Customer Segmentation & Revenue Growth Intelligence

**Entities clustered:** 4,372

## Executive Summary
Customer base shows clear three-tier structure: 38.5% are high-value veterans requiring retention focus, 39.8% are low-engagement prospects needing cost-effective activation, and 21.7% are active explorers ready for margin optimization. Priority actions: (1) Protect Champion revenue with VIP programs, (2) Activate Dormant Prospects through automated low-cost campaigns, (3) Upgrade Variety Seekers to premium products via personalized bundling. Combined approach addresses 100% of base with segment-specific ROI targets.

## Cluster 0 — Established Champions

- Size: **1,685** (38.5%)

- Persona: Long-tenured customers with high cumulative revenue, large order volumes, and consistent transaction history. They represent the revenue backbone but show lower recent activity frequency.

- High drivers: customer_lifetime_days, total_quantity, total_revenue, n_transactions, n_orders

- Low drivers: frequency_per_month, recency_days

- Recommended action: **Implement VIP retention program with dedicated account management, exclusive early access to new products, and quarterly business reviews to identify cross-sell opportunities and prevent churn.**

- Business risk: Complacency leading to gradual disengagement. Lower frequency signals potential vulnerability to competitive offers despite strong historical performance.

- Business opportunity: 38.5% of customer base with proven spending capacity. High potential for wallet share expansion through strategic upselling and premium tier migration.


Key metrics:

               feature    median  ratio_vs_overall
        n_transactions   90.0000             2.143
         total_revenue 1993.7000             2.997
  std_revenue_per_line   16.4713             1.690
              n_orders    6.0000             2.000
       avg_order_value  368.7550             1.524
    median_order_value  342.4600             1.450
       max_order_value  709.6300             1.939
        total_quantity 1176.0000             3.140
     product_diversity   66.0000             1.886
   revenue_per_product   27.9974             1.421
          recency_days   62.0000             0.508
customer_lifetime_days  186.0000             5.812


## Cluster 1 — Dormant Prospects

- Size: **1,738** (39.8%)

- Persona: Recently acquired or inactive customers with minimal transaction history, low product engagement, and limited revenue contribution. High recency indicates recent but infrequent touchpoints.

- High drivers: frequency_per_month, recency_days

- Low drivers: customer_lifetime_days, n_transactions, product_diversity, total_quantity, total_revenue

- Recommended action: **Deploy automated win-back campaigns with time-limited activation offers (15-20% discount on next purchase). Use email and digital channels to minimize cost while testing product-market fit.**

- Business risk: High churn probability with minimal revenue impact. Risk of wasting resources on customers with low lifetime value potential.

- Business opportunity: Largest segment at 39.8% - successful activation of even 10-15% could significantly impact revenue growth. Low-cost testing ground for onboarding optimization.


Key metrics:

               feature   median  ratio_vs_overall
        n_transactions  16.0000             0.381
         total_revenue 329.7500             0.496
              n_orders   2.0000             0.667
   avg_lines_per_order   9.3333             0.639
       max_order_value 259.3100             0.709
        total_quantity 175.5000             0.469
     product_diversity  15.0000             0.429
          recency_days 184.0000             1.508
customer_lifetime_days   1.0000             0.031
   frequency_per_month  30.0000             6.400


## Cluster 2 — Active Variety Seekers

- Size: **949** (21.7%)

- Persona: Highly engaged customers who frequently purchase across multiple product categories with diverse basket composition, but favor lower-priced items. Recent activity indicates strong current engagement.

- High drivers: frequency_per_month, avg_lines_per_order, product_diversity, n_transactions, recency_days

- Low drivers: customer_lifetime_days, median_revenue_per_line, avg_quantity, avg_revenue_per_line, revenue_per_product

- Recommended action: **Launch targeted bundle campaigns pairing their preferred categories with higher-margin products. Use personalized recommendations to drive premium product trial and increase average order value.**

- Business risk: Price-sensitive behavior may limit margin expansion. High activity with low per-item spend suggests deal-seeking mentality.

- Business opportunity: 21.7% segment with proven engagement and category exploration behavior. Strong candidates for premium product education and value-based upselling to improve revenue quality.


Key metrics:

                feature   median  ratio_vs_overall
         n_transactions  72.0000             1.714
          total_revenue 423.0400             0.636
   avg_revenue_per_line   6.0350             0.350
median_revenue_per_line   4.0500             0.265
   std_revenue_per_line   5.5517             0.570
               n_orders   2.0000             0.667
    avg_lines_per_order  31.6250             2.166
        max_order_value 260.9200             0.713
         total_quantity 262.0000             0.700
           avg_quantity   3.3605             0.349
      product_diversity  62.0000             1.771
    revenue_per_product   7.2667             0.369

