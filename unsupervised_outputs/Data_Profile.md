# Data Profile — Auto Segmenter AI

## Dataset
- Shape: **541,909 rows × 8 columns**
- Duplicates: **5,268**

## Detected Segmentation Configuration
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
  "strategy": "Customer-level segmentation for RGM analytics. With 4372 unique customers and transactional data spanning invoices, products, and geography, we can derive RFM features (recency, frequency, monetary), basket composition, geographic concentration, and product diversity metrics. The 25% null customerid rate requires handling (exclude or separate segment). This enables actionable commercial prioritization by identifying high-value, at-risk, and growth-potential customer segments."
}
```

## Column Summary
     column   dtype  nunique  null_pct                                                                                  sample
  invoiceno     str    25900     0.000                                                                  536365, 536365, 536365
  stockcode     str     4070     0.000                                                                   85123A, 71053, 84406B
description     str     4223     0.268 WHITE HANGING HEART T-LIGHT HOLDER, WHITE METAL LANTERN, CREAM CUPID HEARTS COAT HANGER
   quantity   int64      722     0.000                                                                                 6, 6, 8
invoicedate     str    23260     0.000                                    01-12-2010 08:26, 01-12-2010 08:26, 01-12-2010 08:26
  unitprice float64     1630     0.000                                                                        2.55, 3.39, 2.75
 customerid float64     4372    24.927                                                               17850.0, 17850.0, 17850.0
    country     str       38     0.000                                          United Kingdom, United Kingdom, United Kingdom
