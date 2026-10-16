"""Nassau Candy: Factory Reallocation & Shipping Optimization (Streamlit app)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import *
from src.modeling import load_model
from src.simulate import add_flags, load_bounds, score, simulate_product

st.set_page_config(page_title="Nassau Candy Optimizer", layout="wide")

# ---------- Guard: make sure Steps 1-5 produced their files ----------
REQUIRED = [CLEAN_FILE, MODEL_FILE, SCENARIOS_FILE, ROUTE_CLUSTERS_FILE, REGION_PRODUCT_FILE]
missing = [f.name for f in REQUIRED if not f.exists()]
if missing:
    st.error("Missing files: " + ", ".join(missing) +
             ". Run notebooks 01 to 05 first (see README).")
    st.stop()


# ---------- Cached loaders ----------
@st.cache_data
def load_data():
    return pd.read_csv(CLEAN_FILE)


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_bounds():
    return load_bounds()


@st.cache_data
def get_routes():
    return pd.read_csv(ROUTE_CLUSTERS_FILE)


@st.cache_data
def get_congestion():
    return pd.read_csv(REGION_PRODUCT_FILE)


@st.cache_data(show_spinner=False)
def run_sim(product, region, ship_mode):
    return simulate_product(load_data(), get_model(), product,
                            None if region == "All" else region,
                            None if ship_mode == "All" else ship_mode)


@st.cache_data(show_spinner=False)
def run_all(region, ship_mode):
    parts = [run_sim(p, region, ship_mode) for p in sorted(load_data()["product"].unique())]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------- Helpers ----------
NICE = {
    "product": "Product", "current_factory": "Current factory",
    "candidate_factory": "Candidate factory", "avg_distance_km": "Avg distance (km)",
    "lead_new": "Predicted lead time (days)", "lead_delta_days": "Change vs current (days)",
    "lead_reduction_pct": "Lead-time reduction (%)", "profit_change_pct": "Profit impact (%)",
    "risk_reduction_pp": "Risk reduction (pp)", "confidence": "Confidence (0-100)",
    "score": "Score", "risk_level": "Profit risk", "low_confidence": "Low confidence",
    "profit_stable": "Profit stable", "n_orders": "Orders",
}


def pretty(df, cols):
    return df[cols].rename(columns=NICE).round(2)


def beneficial(s):
    """Alternatives that shorten predicted lead time, best score first."""
    return s[(~s["is_current"]) & (s["lead_reduction_pct"] > 0)].sort_values("score", ascending=False)


def top_per_product(s):
    return beneficial(s).groupby("product").head(1).reset_index(drop=True)


def compute_kpis(s):
    n_products = s["product"].nunique()
    top = top_per_product(s)
    if top.empty:
        return dict(lead=0.0, stable=0.0, conf=0.0, coverage=0.0)
    return dict(lead=top["lead_reduction_pct"].mean(),
                stable=top["profit_stable"].mean() * 100,
                conf=top["confidence"].mean(),
                coverage=len(top) / n_products * 100)


# ---------- Sidebar ----------
df = load_data()
bounds = get_bounds()

st.sidebar.header("Controls")
product = st.sidebar.selectbox("Product", sorted(df["product"].unique()))
region = st.sidebar.selectbox("Destination region", ["All"] + sorted(df["region"].unique()))
ship_mode = st.sidebar.selectbox("Ship mode", ["All"] + sorted(df["ship_mode"].unique()))
priority = st.sidebar.slider("Optimization priority", 0.0, 1.0, 0.5, 0.05,
                             help="0 = favour profit, 1 = favour speed (lead-time reduction)")
st.sidebar.caption("0 = maximise profit  |  1 = maximise speed")
top_n = st.sidebar.slider("Recommendations to show", 1, 4, TOP_N)

# ---------- Run scenarios (cached) ----------
with st.spinner("Running scenarios..."):
    all_scen = run_all(region, ship_mode)

st.title("Nassau Candy: Factory Reallocation Optimizer")
st.caption("Predicts how reassigning a product to another factory would change shipping lead time, "
           "freight cost and profit. Freight cost is an assumption; see the notes in the Risk & Impact tab.")

if all_scen.empty:
    st.warning("No orders match these filters. Try 'All' for region or ship mode.")
    st.stop()

all_scored = add_flags(score(all_scen, bounds, priority))
prod_scored = all_scored[all_scored["product"] == product]

# ---------- KPI cards ----------
k = compute_kpis(all_scored)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Lead Time Reduction (avg of top picks)", f"{k['lead']:.2f}%")
c2.metric("Profit Impact Stability", f"{k['stable']:.0f}%")
c3.metric("Scenario Confidence", f"{k['conf']:.0f} / 100")
c4.metric("Recommendation Coverage", f"{k['coverage']:.0f}%")

tab1, tab2, tab3, tab4 = st.tabs(["Factory Optimization Simulator", "What-If Scenario Analysis",
                                  "Recommendation Dashboard", "Risk & Impact Panel"])

# ---------- Tab 1: simulator ----------
with tab1:
    st.subheader(f"Predicted performance by factory: {product}")
    if prod_scored.empty:
        st.info("This product has no orders under the current region / ship-mode filters.")
    else:
        view = prod_scored.copy()
        view["Type"] = np.where(view["is_current"], "Current", "Alternative")
        fig = px.bar(view, x="candidate_factory", y="lead_delta_days", color="Type",
                     color_discrete_map={"Current": "#7f7f7f", "Alternative": "#1f77b4"},
                     labels={"candidate_factory": "Factory",
                             "lead_delta_days": "Change in predicted lead time vs current (days)"})
        st.plotly_chart(fig)
        st.caption("Negative = faster than the current factory. The current factory is the zero baseline.")
        st.dataframe(pretty(view, ["candidate_factory", "avg_distance_km", "lead_new",
                                   "lead_delta_days", "lead_reduction_pct", "profit_change_pct",
                                   "confidence", "n_orders"]))

# ---------- Tab 2: what-if ----------
with tab2:
    st.subheader("Current vs recommended assignment")
    if prod_scored.empty:
        st.info("This product has no orders under the current filters.")
    else:
        cur = prod_scored[prod_scored["is_current"]].iloc[0]
        alts = prod_scored[~prod_scored["is_current"]].sort_values("score", ascending=False)
        st.write(f"**Current factory:** {cur['current_factory']}")
        choice = st.selectbox("Compare with (best-scoring option first)", list(alts["candidate_factory"]))
        alt = alts[alts["candidate_factory"] == choice].iloc[0]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg distance (km)", f"{alt['avg_distance_km']:.0f}",
                  f"{alt['avg_distance_km'] - cur['avg_distance_km']:+.0f} km", delta_color="inverse")
        m2.metric("Predicted lead time (days)", f"{alt['lead_new']:.1f}",
                  f"{alt['lead_delta_days']:+.2f} days", delta_color="inverse")
        m3.metric("Freight cost (assumed, $)", f"{alt['freight_new']:,.0f}",
                  f"{alt['freight_new'] - alt['freight_cur']:+,.0f}", delta_color="inverse")
        m4.metric("Confidence", f"{alt['confidence']:.0f} / 100")

        fig2 = px.bar(alts, x="candidate_factory", y="lead_reduction_pct", color="risk_level",
                      color_discrete_map={"OK": "#2ca02c", "Warning": "#ff7f0e", "High": "#d62728"},
                      labels={"candidate_factory": "Candidate factory",
                              "lead_reduction_pct": "Lead-time reduction (%)",
                              "risk_level": "Profit risk"})
        st.plotly_chart(fig2)
        st.caption("Positive = predicted improvement. Differences are small because distance explains "
                   "little of lead time in this dataset.")

# ---------- Tab 3: recommendations ----------
with tab3:
    st.subheader("Ranked reassignment suggestions (best option per product)")
    top = top_per_product(all_scored).sort_values("score", ascending=False)
    if top.empty:
        st.info("No product has an alternative factory predicted to shorten lead time under these filters.")
    else:
        cols = ["product", "current_factory", "candidate_factory", "lead_reduction_pct",
                "profit_change_pct", "risk_reduction_pp", "confidence", "score", "risk_level"]
        table = pretty(top, cols)
        st.dataframe(table)
        st.download_button("Download recommendations (CSV)", table.to_csv(index=False).encode("utf-8"),
                           file_name="recommendations.csv", mime="text/csv")
        fig3 = px.bar(top.sort_values("lead_reduction_pct"), x="lead_reduction_pct", y="product",
                      orientation="h", labels={"lead_reduction_pct": "Lead-time reduction (%)",
                                               "product": ""})
        st.plotly_chart(fig3)

    st.subheader(f"Top {top_n} options for {product}")
    sel = beneficial(prod_scored).head(top_n)
    if sel.empty:
        st.info("No alternative is predicted to improve on the current assignment for this product.")
    else:
        st.dataframe(pretty(sel, ["candidate_factory", "lead_reduction_pct", "profit_change_pct",
                                  "risk_reduction_pp", "confidence", "score", "risk_level"]))

# ---------- Tab 4: risk & impact ----------
with tab4:
    st.subheader(f"Alerts for {product}")
    sel = beneficial(prod_scored).head(top_n)
    if sel.empty:
        st.success("No reassignment is suggested for this product, so there are no new risks.")
    for _, r in sel.iterrows():
        name = r["candidate_factory"]
        shown = False
        if r["risk_level"] == "High":
            st.error(f"**{name}:** high profit risk ({r['profit_change_pct']:.1f}% of gross profit).")
            shown = True
        elif r["risk_level"] == "Warning":
            st.warning(f"**{name}:** profit warning ({r['profit_change_pct']:.1f}% of gross profit).")
            shown = True
        if not r["profit_stable"]:
            st.warning(f"**{name}:** profit impact is not stable if the freight rate changes "
                       f"({r['profit_change_pct_low']:.1f}% to {r['profit_change_pct_high']:.1f}%).")
            shown = True
        if r["low_confidence"]:
            st.warning(f"**{name}:** low confidence ({r['confidence']:.0f}/100). "
                       "Treat the predicted improvement as indicative only.")
            shown = True
        if not shown:
            st.success(f"**{name}:** no profit or confidence alerts.")

    st.subheader("Overview across all products (top picks)")
    top_all = top_per_product(all_scored)
    o1, o2, o3 = st.columns(3)
    o1.metric("High profit risk", int((top_all["risk_level"] == "High").sum()) if len(top_all) else 0)
    o2.metric("Profit warnings", int((top_all["risk_level"] == "Warning").sum()) if len(top_all) else 0)
    o3.metric("Low-confidence picks", int(top_all["low_confidence"].sum()) if len(top_all) else 0)

    st.subheader("Historical route risk (from clustering)")
    routes = get_routes()
    cur_f = prod_scored["current_factory"].iloc[0] if not prod_scored.empty else None
    slow = routes[routes["slow_flag"]]
    if cur_f:
        slow = slow[slow["factory"] == cur_f]
    st.write("Routes flagged consistently slow" + (f" from {cur_f}:" if cur_f else ":"))
    if slow.empty:
        st.write("None.")
    else:
        st.dataframe(slow[["factory", "state", "orders", "mean_lead_adj", "share_slow", "cluster"]].round(2))
    cong = get_congestion()
    cong = cong[cong["congested_flag"]]
    st.write("Congested region-product combinations:")
    st.dataframe(cong[["region", "product", "orders", "mean_lead_adj", "congestion_score"]].round(2)
                 if not cong.empty else pd.DataFrame({"note": ["None flagged"]}))

    with st.expander("Assumptions and data limitations"):
        st.markdown(f"""
- **Freight cost is assumed**: distance x units x ${COST_PER_UNIT_KM} per unit-km. The dataset has no freight data.
- **Lead times in this dataset are on the order of years** and depend mostly on order date, so the dates appear
  synthetic. Results are relative comparisons, not real transit times.
- **Distance explains very little of lead time**, so predicted improvements are small and confidence is modest.
- Destinations use state/province centres; distances are straight-line, not road distances.
- Speed and profit both depend only on distance here, so the priority slider has a limited effect.
- Profit warning threshold: {PROFIT_WARNING_PCT}% | high-risk threshold: {PROFIT_HIGH_RISK_PCT}% |
  low-confidence below {MIN_CONFIDENCE}/100.
""")