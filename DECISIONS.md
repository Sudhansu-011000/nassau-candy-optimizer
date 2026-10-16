# Decisions log: Nassau Candy Factory Reallocation & Shipping Optimization

This file records every assumption, design choice, and data finding behind the project, in the order the work was done. Items marked `<fill in>` are numbers to copy from your own notebook output before publishing.

---

## 1. Project setup and configuration (Step 1)

- All paths, constants, column names, and assumptions live in `src/config.py`. Paths are built from the file's own location (`Path(__file__)`), so the code runs the same from notebooks and from the app.
- Random seed is 42 everywhere (splits, models, clustering) so results are reproducible. Test size is 20%, cross-validation uses 5 folds.
- Raw data in `data/raw/` is never edited. All cleaning happens in code and is saved to `data/processed/`.
- Reference files (`factories.csv`, `product_factory.csv`, `state_centroids.csv`) are read with normalised headers (BOM and spaces stripped, lowercase, underscores) so header formatting cannot break merges.
- Output folders (`processed/`, `models/`, `reports/figures/`) are created automatically, since Git does not store empty folders.

## 2. Data preparation (Step 2)

- `lead_time` = Ship Date − Order Date, in days. The raw data has no lead-time column.
- Origin factory is derived from the product-to-factory table in the project brief (it is not in the raw data).
- Destination = centroid of the customer's state/province (approximate; no live geocoding, so the project stays offline and reproducible).
- Distance = haversine (great-circle) km between factory and destination, not road distance.
- Cleaning: dropped duplicates, rows with missing lead time, factory, or coordinates, negative lead times, and non-positive units or sales. Lead times above the 99th percentile are treated as extreme outliers and removed (fixed, documented cap). Final dataset: 4,090 rows.
- `validate()` hard-checks the cleaned data: no nulls in model columns, no negative distances or lead times, no more than 5 factories.

## 3. Data findings that shape the whole project

- Lead times average about 1,360 days (range 613 to 1,932, std 298.6). That is roughly 3.7 years from order to ship, which is not realistic shipping time.
- Ship mode ordering is illogical: Same Day averages 1,405 days and Standard Class 1,346.
- Lead time correlates only 0.036 with distance and barely differs by region (1,343 to 1,377 days).
- Order year drives most of the variation: 2024 orders average 1,131 days (n = 1,668) and 2025 orders average 1,518 days (n = 2,422).
- Conclusion: the date fields appear synthetic or offset. All results are interpreted as **relative comparisons**, not real-world transit times. This limitation is stated in the paper and in the app.

## 4. Predictive modeling (Step 3)

**First attempt (logistics features only).** Features: distance_km, units, sales, ship_mode, region, division. Product and factory identity are deliberately **not** features: each product maps to exactly one factory, so including them would make reassignment scenarios impossible to evaluate. A reassignment changes `distance_km` only.

| Model | CV RMSE | Test RMSE | Test R² |
|---|---|---|---|
| Baseline (mean) | 297.5 | 303.0 | -0.006 |
| Linear Regression | 297.0 | 303.3 | -0.008 |
| Random Forest | 304.1 | 310.7 | -0.058 |
| Gradient Boosting | 298.3 | 303.5 | -0.010 |

No model beat the baseline, so logistics features alone carry no usable signal.

**Signal analysis (feature groups, gradient boosting, 5-fold CV RMSE):**

| Feature set | CV RMSE |
|---|---|
| Baseline (target std) | 298.6 |
| Order year only | 230.4 |
| Time features only | 183.8 |
| Logistics features only | 298.7 |
| Time + logistics | 187.0 |
| Time + logistics + product, factory, state | 188.8 |

- Almost all predictable variation comes from order date. Logistics features, product, factory, and state add nothing.
- Decision: include `order_ordinal` (days since 2020-01-01) and `order_month` as **control variables**, so the distance effect is measured net of time. In simulations the order date is held fixed, so the time artifact cancels out of every comparison.
- Estimated distance effect (linear, time-controlled): about 0.0096 days per km, roughly 10 days per 1,000 km (about 0.7% of mean lead time). Practically small.
- Final model features: distance_km, units, sales, order_ordinal, order_month, ship_mode, region, division.
- Selection rule: lowest 5-fold CV RMSE; a simpler model is preferred if within 2% (interpretability).
- Preprocessing (scaling, one-hot encoding) is fitted inside the scikit-learn pipeline, so there is no leakage and the saved model accepts raw dataframes.
- **Selected model: Random Forest. Test RMSE 162.4 vs baseline 303.0, test R² 0.711.** The gain comes almost entirely from the time controls, not from logistics.
- Distance sensitivity check (predicted lead time at 200 / 800 / 1,500 / 2,500 km): 1,334.9 / 1,338.6 / 1,342.5 / 1,342.5 days. Never decreasing, but small (about 0.6% across the range).
- Limitation: the model estimates association, not the causal effect of changing factories.

## 5. Route and product clustering (Step 4)

- Route = factory × destination state (routes with fewer than 10 orders excluded). Region-product combinations are analysed separately for congestion.
- Clustering uses **time-adjusted lead time**: `lead_adj` = actual lead time minus the lead time expected from order date alone (out-of-fold predictions), so routes are not penalised for having more late-dated orders. Positive means slower than expected.
- Cluster features: mean adjusted lead, adjusted lead standard deviation, average distance, profit margin, log order count. KMeans, with k chosen by silhouette score over 2 to 6. Clusters are renamed so 0 = fastest.
- Slow route = top-quartile adjusted lead time AND at least 50% of orders slower than expected. Congested = above-median order volume AND top-quartile adjusted lead time.
- Results: 62 routes, 4 clusters, 16 slow routes, 3 congested region-product combinations. Silhouette score for chosen k: `<fill in>`.
- ANOVA on route differences in adjusted lead time: p = `<fill in>` (factory × state) and p = `<fill in>` (factory × region). If p > 0.05, the slow and congested flags are descriptive, not statistically significant, and the paper says so.

## 6. Scenario simulation (Step 5)

- For each product and candidate factory, only `distance_km` changes. Order dates, units, sales, ship mode, and region are held fixed. Lead-time change = model prediction at the new distance minus model prediction at the current distance (both from the same model).
- **Freight cost is an assumption** (the dataset has none): distance_km × units × `COST_PER_UNIT_KM` = `<fill in value from config.py>` dollars per unit-km. Calibration check: average freight is `<fill in>`% of gross profit.
- Profit impact = −(change in freight) ÷ gross profit of the same orders. Stress-tested at 0.5× and 2× the freight rate. A recommendation counts as profit-stable if its worst case stays within the warning threshold.
- Profit warning threshold: 5% of gross profit. High-risk threshold: 10%.
- Risk reduction = drop in the share of orders travelling on routes flagged slow in Step 4 (routes are descriptive; see the ANOVA result).
- Confidence score (0 to 100) = 70% signal-to-noise (Random Forest tree disagreement on the predicted change) + 30% data support (orders ÷ 100, capped at 1). Below 50 is flagged low confidence.
- Ranking score = 80% × (priority × speed + (1 − priority) × profit) + 20% × risk reduction, with each part scaled 0 to 1 across all scenarios. Only alternatives that shorten predicted lead time are recommended.
- KPIs: lead-time reduction (average of top picks), profit impact stability (share of top picks that stay within the warning threshold under the freight stress test), scenario confidence (average of top picks), recommendation coverage (share of products with at least one beneficial option). Values at priority 0.5: `<fill in>`.
- Limitation: speed and profit both depend only on distance here, so the priority slider has limited effect. Factory production costs and capacities are not in the data.

## 7. Streamlit app (Step 6)

- The app does no analysis of its own. It loads saved files and calls `simulate_product`, `score`, and `add_flags`, so the app and the notebooks always agree.
- Sidebar filters (product, destination region, ship mode, optimisation priority, number of recommendations) apply to every tab, including the KPI cards.
- Scenarios are cached per filter combination, and the model is cached as a resource, so only the first load is slow.
- Four modules match the brief: Factory Optimization Simulator, What-If Scenario Analysis, Recommendation Dashboard, Risk & Impact Panel.
- The app shows a clear error and stops if any file from Steps 1 to 5 is missing.
- Version-specific Streamlit arguments (such as `use_container_width`) are deliberately not used, so the app runs across Streamlit versions.
- The Risk & Impact tab states the main assumptions and data limitations (synthetic-looking dates, assumed freight cost, weak distance effect, straight-line distances).

## 8. Known limitations (summary for the paper and README)

1. Dates appear synthetic, so lead times are not real transit times. Results are relative.
2. Distance explains very little of lead time; predicted improvements are small and confidence is modest.
3. Freight cost is assumed, not observed.
4. Destinations use state/province centroids and straight-line distance.
5. The model estimates association, not causal effects of moving production.
6. Factory production costs and capacities are unknown, so recommendations ignore them.
7. Slow and congested route flags may be descriptive rather than statistically significant (see the ANOVA result).
