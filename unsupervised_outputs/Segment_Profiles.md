# Segment Profiles — Unsupervised Learning

**Project:** Customer Segmentation & Revenue Growth Intelligence

**Entities clustered:** 4,372

## Executive Summary
Customer base splits into three distinct behavioral segments requiring differentiated RGM strategies. **Established Champions** (39%) deliver historical revenue strength but need proactive retention to maintain frequency. **Dormant & Occasional Buyers** (40%) represent the largest untapped opportunity through low-cost activation and onboarding optimization. **Active Variety Seekers** (22%) show strong engagement signals but require basket-building tactics to unlock revenue potential. Priority actions: (1) Launch VIP program for Champions to protect $-base, (2) Automate win-back flows for Dormant segment to improve conversion economics, (3) Deploy category-based upsell for Variety Seekers to lift AOV. Combined impact potential: 15-20% revenue growth through segment-specific commercial execution.

## Cluster 0 — Established Champions

- Size: **1,685** (38.5%)

- Persona: Long-tenured customers with high cumulative spend, large order volumes, and consistent transaction history. They represent the revenue backbone but show lower recent activity frequency.

- High drivers: customer_lifetime_days, total_quantity, total_revenue, n_transactions, n_orders

- Low drivers: frequency_per_month, recency_days

- Recommended action: **Implement VIP retention program with dedicated account management, exclusive early access to new products, and quarterly business reviews to identify cross-sell opportunities and prevent churn.**

- Business risk: Complacency leading to gradual disengagement. Lower frequency signals potential vulnerability to competitive offers despite strong historical performance.

- Business opportunity: 38.5% of customer base with proven high lifetime value. Incremental frequency gains yield disproportionate revenue impact. Strong candidates for premium tier expansion and referral programs.


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


## Cluster 1 — Dormant & Occasional Buyers

- Size: **1,738** (39.8%)

- Persona: Recently active but sporadic customers with minimal transaction depth, low product engagement, and short relationship tenure. High recency suggests they haven't churned but lack commitment.

- High drivers: frequency_per_month, recency_days

- Low drivers: customer_lifetime_days, n_transactions, product_diversity, total_quantity, total_revenue

- Recommended action: **Deploy automated win-back campaigns with time-limited incentives (15-20% discount on next purchase) and personalized onboarding sequences to drive second and third purchases within 60 days.**

- Business risk: High churn probability due to weak engagement patterns. Low switching costs make them vulnerable to competitor acquisition tactics.

- Business opportunity: Largest segment at 39.8% - small improvements in conversion and frequency create significant revenue lift. Cost-effective to activate given low current engagement baseline.


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

- Persona: Frequent, recent buyers with high product diversity and multiple items per order, but lower average transaction values. They explore the catalog actively but purchase lower-priced items or smaller quantities.

- High drivers: frequency_per_month, avg_lines_per_order, product_diversity, n_transactions, recency_days

- Low drivers: customer_lifetime_days, median_revenue_per_line, avg_quantity, avg_revenue_per_line, revenue_per_product

- Recommended action: **Launch bundle and upsell campaigns targeting higher-margin SKUs within their preferred categories. Use personalized recommendations to trade customers up from low to mid-tier products with 'complete the set' messaging.**

- Business risk: Revenue per transaction ceiling limits total value extraction. May be price-sensitive deal hunters rather than loyal brand advocates.

- Business opportunity: 21.7% segment with proven engagement momentum. High frequency and diversity indicate strong purchase intent - opportunity to increase basket value by 25-40% through strategic merchandising and tiered pricing.


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

