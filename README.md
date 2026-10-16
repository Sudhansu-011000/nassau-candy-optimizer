# Nassau Candy: Factory Reallocation & Shipping Optimization

A decision-intelligence system for Nassau Candy Distributor that predicts shipping lead time, simulates reassigning products to different factories, and recommends changes that balance shipping speed and profitability. It was built as a Unified Mentor project and ships with an interactive Streamlit dashboard.

**Live demo:** `<add your Streamlit Community Cloud link here>`

---

## What it does

Nassau Candy assigns products to factories with static rules. This project answers the question leadership actually cares about: *what should we change to improve performance?*

1. **Prepares the data:** derives lead time, origin factory, destination coordinates, and factory-to-customer distance.
2. **Predicts lead time** with Linear Regression, Random Forest, and Gradient Boosting, and selects the best model by accuracy and interpretability.
3. **Clusters routes** by performance and flags consistently slow routes and congested region-product combinations.
4. **Simulates reassignments:** for every product and every alternative factory, it estimates the change in lead time, freight cost, profit, risk, and confidence.
5. **Recommends** ranked reassignment options, controlled by an optimisation-priority slider (speed vs profit).
6. **Serves it all** in a Streamlit app with four modules: Factory Optimization Simulator, What-If Scenario Analysis, Recommendation Dashboard, and Risk & Impact Panel.

## Screenshots

| Simulator | What-If |
|---|---|
| ![Simulator](reports/figures/app_tab1.png) | ![What-If](reports/figures/app_tab2.png) |

| Recommendations | Risk & Impact |
|---|---|
| ![Recommendations](reports/figures/app_tab3.png) | ![Risk](reports/figures/app_tab4.png) |

## Key findings

These results come from an honest reading of the data, including the parts that limit what it can support.

- **The date fields look synthetic.** Lead times average about 1,360 days, Same Day shipping is not faster than Standard Class, and distance correlates only 0.036 with lead time. All results are therefore **relative comparisons, not real transit times**.
- **Order date drives most of the predictable variation** (2024 orders average 1,131 days, 2025 orders 1,518). Logistics features alone do not beat a "predict the average" baseline.
- Order-date features are included as **control variables**, so the distance effect is measured net of time. The final model (Random Forest) reaches a test RMSE of **162.4 days vs 303.0 for the baseline** (R² = 0.711), driven mostly by the time controls.
- **The distance effect is small:** roughly 10 days per 1,000 km (about 0.7% of mean lead time). Predicted improvements from reassignment are correspondingly modest, and the app reports a confidence score for every recommendation.
- Clustering found **62 routes in 4 clusters**, with 16 flagged as consistently slow and 3 congested region-product combinations. See `DECISIONS.md` for whether these differences are statistically significant.

| KPI (priority 0.5) | Value |
|---|---|
| Lead Time Reduction (average of top picks) | `<fill in>` |
| Profit Impact Stability | `<fill in>` |
| Scenario Confidence | `<fill in>` |
| Recommendation Coverage | `<fill in>` |

## Project structure

```
nassau-candy-optimizer/
├── data/
│   ├── raw/                 # original dataset, never edited
│   ├── processed/           # cleaned data, route clusters, scenarios
│   └── reference/           # factories, product-factory map, state centroids
├── notebooks/
│   ├── 01_data_prep.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_clustering.ipynb
│   └── 05_simulation.ipynb
├── src/
│   ├── config.py            # paths, constants, assumptions
│   ├── features.py          # cleaning, geocoding, distance
│   ├── modeling.py          # training, evaluation, selection
│   ├── clustering.py        # route and region-product clustering
│   └── simulate.py          # scenario engine, scoring, KPIs
├── models/                  # saved model (.joblib)
├── app/
│   └── app.py               # Streamlit dashboard
├── reports/
│   ├── figures/
│   ├── research_paper.docx
│   └── executive_summary.docx
├── DECISIONS.md             # every assumption and design choice
├── requirements.txt
└── README.md
```

## Getting started

### 1. Clone and set up the environment

```bash
git clone https://github.com/<your-username>/nassau-candy-optimizer.git
cd nassau-candy-optimizer

python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# Mac/Linux:            source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Add the dataset

Place the Nassau Candy Distributor CSV in `data/raw/`, using the filename set in `RAW_FILE` inside `src/config.py`. The reference files in `data/reference/` are already included.

> The raw dataset is not redistributed here. `<Adjust this note to match whether you included the file.>`

### 3. Run the notebooks in order

Open each notebook with a kernel that uses your virtual environment, and run all cells top to bottom:

1. `01_data_prep.ipynb`: builds `data/processed/clean.csv`
2. `02_eda.ipynb`: exploratory analysis and figures
3. `03_modeling.ipynb`: trains models, saves `models/best_model.joblib`
4. `04_clustering.ipynb`: route clusters and flags
5. `05_simulation.ipynb`: scenario table and KPIs

Each of the last three ends with a verification cell that prints a `... CHECKS PASSED` message.

### 4. Launch the app

From the project root:

```bash
streamlit run app/app.py
```

The app opens at `http://localhost:8501`. If a required file is missing, it tells you which notebook to run.

## Using the dashboard

- **Sidebar controls:** product, destination region, ship mode, optimisation priority (0 = favour profit, 1 = favour speed), and number of recommendations.
- **Factory Optimization Simulator:** predicted performance of every factory for the selected product.
- **What-If Scenario Analysis:** compare the current assignment against any alternative, with deltas in distance, lead time, and freight cost.
- **Recommendation Dashboard:** ranked suggestions across all products, downloadable as CSV.
- **Risk & Impact Panel:** profit-impact alerts, low-confidence warnings, and historical slow and congested routes.

## Method in brief

| Stage | Approach |
|---|---|
| Lead time | Ship Date minus Order Date, in days |
| Distance | Haversine distance from factory to the customer's state centroid |
| Model | Scikit-learn pipelines (scaling and one-hot encoding inside the pipeline, no leakage); 5-fold CV; RMSE, MAE, and R² |
| Clustering | KMeans on time-adjusted route statistics; k chosen by silhouette score |
| Simulation | Only distance changes per scenario; order date, units, sales, ship mode, and region held fixed |
| Profit impact | Assumed freight cost (distance × units × rate), stress-tested at 0.5× and 2× the rate |
| Confidence | 70% signal-to-noise (Random Forest tree disagreement) + 30% data support |
| Ranking | Weighted score of speed, profit, and risk reduction, controlled by the priority slider |

Full reasoning for every choice is in [`DECISIONS.md`](DECISIONS.md).

## Limitations

1. Dates appear synthetic, so lead times are not real transit times. Results are relative.
2. Distance explains very little of lead time, so predicted improvements are small and confidence is modest.
3. Freight cost is assumed, not observed; the rate is set in `src/config.py`.
4. Destinations use state or province centroids, and distances are straight-line.
5. The model estimates association, not the causal effect of moving production.
6. Factory production costs and capacities are not in the data, so recommendations ignore them. Speed and profit therefore both depend only on distance, which limits the effect of the priority slider.

## Tech stack

Python, pandas, NumPy, scikit-learn, SciPy, Plotly, Streamlit, joblib, Jupyter.

## Deliverables

- Research paper (`reports/research_paper.docx`): EDA, insights, and recommendations
- Streamlit dashboard (`app/app.py`): live analytics
- Executive summary (`reports/executive_summary.docx`): for stakeholders

## License

`<Choose a license, e.g. MIT, and add a LICENSE file.>`

## Acknowledgements

Built as a project for [Unified Mentor](https://www.unifiedmentor.com/) using the Nassau Candy Distributor dataset.
