# Segment Profiles — Unsupervised Learning

**Project:** Customer Segmentation & Revenue Growth Intelligence

**Entities clustered:** 4,372

## Executive Summary
Customer base shows clear three-tier structure requiring differentiated RGM strategies. **Established Champions (38.5%)** are revenue foundation but showing frequency warning signs—prioritize retention and relationship deepening. **Dormant Prospects (39.8%)** represent the largest segment with lowest engagement—focus on cost-efficient activation and churn prevention. **Active Explorers (21.7%)** demonstrate strong behavioral loyalty but low wallet share per transaction—optimize for margin expansion through strategic upselling. Immediate priority: prevent Champion erosion while activating the large Dormant base. Quick win: bundle campaigns for Explorers can drive near-term AOV lift.

## Cluster 0 — Established Champions

- Size: **1,685** (38.5%)

- Persona: Long-tenured customers with high cumulative spend and transaction volume, but lower recent activity frequency. These are your legacy revenue pillars.

- High drivers: customer_lifetime_days, total_quantity, total_revenue, n_transactions, n_orders

- Low drivers: frequency_per_month, recency_days

- Recommended action: **Launch quarterly executive business reviews and dedicated account management to prevent attrition. Implement early warning system for frequency decline and create VIP reactivation offers for those showing 30+ day gaps.**

- Business risk: Complacency leading to silent churn of high-LTV customers. Low frequency signals potential disengagement despite strong history.

- Business opportunity: 38.5% of base represents proven buyers with established trust. High potential for premium tier upgrades, annual contracts, and referral programs.


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

- Persona: Recent but infrequent buyers with minimal transaction history and low product engagement. High recency suggests they haven't churned yet but never fully activated.

- High drivers: frequency_per_month, recency_days

- Low drivers: customer_lifetime_days, n_transactions, product_diversity, total_quantity, total_revenue

- Recommended action: **Deploy automated 3-touch onboarding sequence with educational content and first-purchase incentives. Implement win-back campaigns for those 60+ days inactive. Avoid heavy discounting—focus on value demonstration.**

- Business risk: Largest segment (39.8%) with weakest engagement metrics. High probability of permanent churn without intervention.

- Business opportunity: Untapped volume play. Even modest conversion improvements yield significant revenue given segment size. Low acquisition cost already sunk.


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


## Cluster 2 — Active Explorers

- Size: **949** (21.7%)

- Persona: Frequent, recent buyers with high product diversity and basket complexity, but lower per-item spend. They shop often across categories but favor volume over premium items.

- High drivers: frequency_per_month, avg_lines_per_order, product_diversity, n_transactions, recency_days

- Low drivers: customer_lifetime_days, median_revenue_per_line, avg_quantity, avg_revenue_per_line, revenue_per_product

- Recommended action: **Prioritize for cross-sell and bundle campaigns targeting premium SKU migration. Use personalized recommendations based on browsing behavior. Test 'complete the collection' and tiered loyalty rewards to increase AOV.**

- Business risk: Price-sensitive segment may be vulnerable to competitive promotions. Low revenue per line suggests margin pressure.

- Business opportunity: 21.7% of customers showing strong engagement signals. High visit frequency creates multiple conversion moments. Upsell potential to shift mix toward higher-margin products.


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

