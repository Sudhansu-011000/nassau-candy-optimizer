"""Scenario simulation engine: what happens if a product is made at another factory?"""
import numpy as np
import pandas as pd

from src.config import *
from src.features import haversine_km, read_reference
from src.modeling import FEATURES, load_model


def load_factories() -> pd.DataFrame:
    f = read_reference(FACTORIES_FILE)
    f["factory"] = f["factory"].str.strip()
    return f.set_index("factory")[["latitude", "longitude"]]


def load_slow_routes() -> set:
    r = pd.read_csv(ROUTE_CLUSTERS_FILE)
    return set(map(tuple, r.loc[r["slow_flag"], ["factory", "state"]].values))


def _tree_preds(model, X):
    """Per-tree predictions (n_trees x n_orders). Only for Random Forest; else None."""
    est = model.named_steps["model"]
    if type(est).__name__ != "RandomForestRegressor":
        return None
    Xt = model.named_steps["prep"].transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    return np.vstack([t.predict(Xt) for t in est.estimators_])


def simulate_product(df, model, product, region=None, ship_mode=None,
                     factories=None, slow_routes=None) -> pd.DataFrame:
    """One row per candidate factory (including the current one as the baseline)."""
    d = df[df["product"] == product]
    if region:
        d = d[d["region"] == region]
    if ship_mode:
        d = d[d["ship_mode"] == ship_mode]
    if d.empty:
        return pd.DataFrame()

    fac = load_factories() if factories is None else factories
    slow = load_slow_routes() if slow_routes is None else slow_routes
    cur = d["factory"].iloc[0]

    X_cur = d[FEATURES].copy()
    pred_cur = model.predict(X_cur)
    trees_cur = _tree_preds(model, X_cur)
    lead_cur = pred_cur.mean()
    cost_cur = (d["distance_km"] * d["units"] * COST_PER_UNIT_KM).sum()
    profit_base = d["gross_profit"].sum()
    slow_cur = np.mean([(cur, s) in slow for s in d["state"]])
    n = len(d)

    rows = []
    for f in fac.index:
        dist_new = haversine_km(fac.loc[f, "latitude"], fac.loc[f, "longitude"],
                                d["dest_lat"], d["dest_lon"]).to_numpy()
        X_new = X_cur.copy()
        X_new["distance_km"] = dist_new
        lead_new = model.predict(X_new).mean()
        cost_new = (dist_new * d["units"].to_numpy() * COST_PER_UNIT_KM).sum()
        slow_new = np.mean([(f, s) in slow for s in d["state"]])
        is_cur = (f == cur)

        # confidence
        snr, conf = np.nan, np.nan
        if not is_cur:
            trees_new = _tree_preds(model, X_new)
            if trees_cur is not None and trees_new is not None:
                delta_t = (trees_new - trees_cur).mean(axis=1)
                snr = abs(delta_t.mean()) / (delta_t.std() + 1e-9)
                snr_part = min(1.0, snr / 3)
            else:
                snr_part = 0.5   # neutral when the model has no tree spread
            conf = 100 * (0.7 * snr_part + 0.3 * min(1.0, n / CONF_REF_ORDERS))

        def pchg(mult=1.0):
            return -(cost_new - cost_cur) * mult / abs(profit_base) * 100 if profit_base else np.nan

        rows.append({
            "product": product, "current_factory": cur, "candidate_factory": f,
            "is_current": is_cur, "n_orders": n,
            "avg_distance_km": dist_new.mean(),
            "lead_cur": lead_cur, "lead_new": lead_new,
            "lead_delta_days": lead_new - lead_cur,
            "lead_reduction_pct": (lead_cur - lead_new) / lead_cur * 100,
            "freight_cur": cost_cur, "freight_new": cost_new,
            "profit_base": profit_base,
            "profit_change_pct": pchg(1.0),
            "profit_change_pct_low": pchg(COST_SENSITIVITY[0]),
            "profit_change_pct_high": pchg(COST_SENSITIVITY[1]),
            "slow_share_cur": slow_cur, "slow_share_new": slow_new,
            "risk_reduction_pp": (slow_cur - slow_new) * 100,
            "snr": snr, "confidence": conf,
        })
    return pd.DataFrame(rows)


def simulate_all(df, model) -> pd.DataFrame:
    fac, slow = load_factories(), load_slow_routes()
    parts = [simulate_product(df, model, p, factories=fac, slow_routes=slow)
             for p in sorted(df["product"].unique())]
    return pd.concat([p for p in parts if not p.empty], ignore_index=True)


def make_bounds(scen: pd.DataFrame) -> dict:
    c = scen[~scen["is_current"]]
    return {k: (float(c[k].min()), float(c[k].max()))
            for k in ["lead_reduction_pct", "profit_change_pct", "risk_reduction_pp"]}


def load_bounds() -> dict:
    return make_bounds(pd.read_csv(SCENARIOS_FILE))


def _scale(x, lo, hi):
    return np.clip((x - lo) / (hi - lo + 1e-12), 0, 1)


def score(scen: pd.DataFrame, bounds: dict, priority: float = 0.5) -> pd.DataFrame:
    """priority: 1.0 = pure speed, 0.0 = pure profit."""
    s = scen.copy()
    sp = _scale(s["lead_reduction_pct"], *bounds["lead_reduction_pct"])
    pr = _scale(s["profit_change_pct"], *bounds["profit_change_pct"])
    rk = _scale(s["risk_reduction_pp"], *bounds["risk_reduction_pp"])
    s["score"] = 100 * ((1 - RISK_WEIGHT) * (priority * sp + (1 - priority) * pr) + RISK_WEIGHT * rk)
    return s


def add_flags(s: pd.DataFrame) -> pd.DataFrame:
    s = s.copy()
    s["risk_level"] = np.where(s["profit_change_pct"] < -PROFIT_HIGH_RISK_PCT, "High",
                        np.where(s["profit_change_pct"] < -PROFIT_WARNING_PCT, "Warning", "OK"))
    worst = s[["profit_change_pct", "profit_change_pct_low", "profit_change_pct_high"]].min(axis=1)
    s["profit_stable"] = worst >= -PROFIT_WARNING_PCT
    s["low_confidence"] = s["confidence"] < MIN_CONFIDENCE
    return s


def recommend(df, model, bounds, product, region=None, ship_mode=None,
              priority=0.5, top_n=TOP_N) -> pd.DataFrame:
    """Ranked reassignment suggestions (only options that shorten predicted lead time)."""
    scen = simulate_product(df, model, product, region, ship_mode)
    if scen.empty:
        return scen
    scen = add_flags(score(scen, bounds, priority))
    cand = scen[(~scen["is_current"]) & (scen["lead_reduction_pct"] > 0)]
    return cand.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)


def compute_kpis(df, model, bounds, priority=0.5) -> dict:
    products = sorted(df["product"].unique())
    tops = []
    for p in products:
        r = recommend(df, model, bounds, p, priority=priority, top_n=1)
        if len(r):
            tops.append(r.iloc[0])
    if not tops:
        return {"lead_time_reduction_pct": 0.0, "profit_impact_stability_pct": 0.0,
                "scenario_confidence": 0.0, "recommendation_coverage_pct": 0.0}
    top = pd.DataFrame(tops)
    return {
        "lead_time_reduction_pct": float(top["lead_reduction_pct"].mean()),
        "profit_impact_stability_pct": float(top["profit_stable"].mean() * 100),
        "scenario_confidence": float(top["confidence"].mean()),
        "recommendation_coverage_pct": len(tops) / len(products) * 100,
    }