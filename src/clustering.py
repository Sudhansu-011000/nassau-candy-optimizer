"""Route and region-product clustering."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from src.config import *

TIME_FEATURES = ["order_ordinal", "order_month"]
CLUSTER_FEATURES = ["mean_lead_adj", "std_lead_adj", "avg_distance_km",
                    "profit_margin", "log_orders"]


def add_adjusted_lead(df: pd.DataFrame) -> pd.DataFrame:
    """lead_adj = actual lead time minus the lead time expected from order date alone
    (out-of-fold predictions, so no row is scored by a model that saw it)."""
    df = df.copy()
    kf = KFold(5, shuffle=True, random_state=RANDOM_SEED)
    gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                   max_depth=3, random_state=RANDOM_SEED)
    time_pred = cross_val_predict(gb, df[TIME_FEATURES], df[TARGET], cv=kf)
    df["lead_adj"] = df[TARGET] - time_pred
    return df


def build_table(df: pd.DataFrame, keys: list, min_orders: int = MIN_ROUTE_ORDERS) -> pd.DataFrame:
    t = (df.groupby(keys)
           .agg(orders=(TARGET, "size"),
                mean_lead=(TARGET, "mean"),
                mean_lead_adj=("lead_adj", "mean"),
                std_lead_adj=("lead_adj", "std"),
                share_slow=("lead_adj", lambda s: (s > 0).mean()),
                avg_distance_km=("distance_km", "mean"),
                sales=("sales", "sum"),
                profit=("gross_profit", "sum"))
           .reset_index())
    t["profit_margin"] = t["profit"] / t["sales"]
    t["log_orders"] = np.log(t["orders"])
    t = t[t["orders"] >= min_orders].reset_index(drop=True)
    return t


def choose_k(table: pd.DataFrame, k_max: int = 6):
    X = StandardScaler().fit_transform(table[CLUSTER_FEATURES])
    rows = []
    for k in range(2, min(k_max, len(table) - 1) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED).fit(X)
        rows.append({"k": k, "silhouette": silhouette_score(X, km.labels_), "inertia": km.inertia_})
    scores = pd.DataFrame(rows).round(4)
    best_k = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    return best_k, scores


def cluster_routes(table: pd.DataFrame, k: int) -> pd.DataFrame:
    X = StandardScaler().fit_transform(table[CLUSTER_FEATURES])
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED).fit(X)
    out = table.copy()
    out["cluster"] = km.labels_
    # Rename clusters by speed: 0 = fastest ... k-1 = slowest (by adjusted lead time)
    order = out.groupby("cluster")["mean_lead_adj"].mean().sort_values().index
    out["cluster"] = out["cluster"].map({old: new for new, old in enumerate(order)})
    return out


def flag_slow_routes(t: pd.DataFrame) -> pd.DataFrame:
    """Consistently slow = top-quartile adjusted lead AND more than half of orders slower than expected."""
    t = t.copy()
    q75 = t["mean_lead_adj"].quantile(0.75)
    t["slow_flag"] = (t["mean_lead_adj"] >= q75) & (t["share_slow"] >= 0.5)
    return t


def flag_congested(t: pd.DataFrame) -> pd.DataFrame:
    """Congested = above-median order volume AND top-quartile adjusted lead time."""
    t = t.copy()
    z = lambda s: (s - s.mean()) / s.std(ddof=0)
    t["congestion_score"] = z(t["orders"]) + z(t["mean_lead_adj"])
    t["congested_flag"] = ((t["orders"] >= t["orders"].median())
                           & (t["mean_lead_adj"] >= t["mean_lead_adj"].quantile(0.75)))
    return t