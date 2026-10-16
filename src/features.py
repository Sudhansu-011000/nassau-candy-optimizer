"""Data loading, cleaning, geocoding, and distance features."""
import numpy as np
import pandas as pd

from src.config import *


def read_reference(path) -> pd.DataFrame:
    """Read a reference CSV and normalize headers: strip BOM/spaces, lowercase, underscores."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = (df.columns.str.strip()
                            .str.lower()
                            .str.replace(r"[\s/]+", "_", regex=True))
    return df


def load_raw() -> pd.DataFrame:
    try:
        df = pd.read_csv(RAW_FILE)
    except UnicodeDecodeError:
        df = pd.read_csv(RAW_FILE, encoding="latin-1")
    df.columns = df.columns.str.strip()
    return df


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Works on scalars or numpy/pandas arrays."""
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def add_lead_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[COL_ORDER_DATE] = pd.to_datetime(df[COL_ORDER_DATE], dayfirst=DATE_DAYFIRST, errors="coerce")
    df[COL_SHIP_DATE] = pd.to_datetime(df[COL_SHIP_DATE], dayfirst=DATE_DAYFIRST, errors="coerce")
    n_bad = df[[COL_ORDER_DATE, COL_SHIP_DATE]].isna().any(axis=1).sum()
    print(f"Rows with unparseable dates: {n_bad}")
    df[TARGET] = (df[COL_SHIP_DATE] - df[COL_ORDER_DATE]).dt.days
    return df


def attach_factory(df: pd.DataFrame) -> pd.DataFrame:
    pf = read_reference(PRODUCT_FACTORY_FILE)
    pf["product_name"] = pf["product_name"].str.strip()
    fac = read_reference(FACTORIES_FILE).rename(
        columns={"latitude": "factory_lat", "longitude": "factory_lon"}
    )
    fac["factory"] = fac["factory"].str.strip()
    pf["factory"] = pf["factory"].str.strip()

    df = df.copy()
    df[COL_PRODUCT] = df[COL_PRODUCT].str.strip()
    df = df.merge(pf[["product_name", "factory"]], left_on=COL_PRODUCT,
                  right_on="product_name", how="left").drop(columns="product_name")
    df = df.merge(fac, on="factory", how="left")
    missing = df.loc[df["factory"].isna(), COL_PRODUCT].unique()
    if len(missing):
        print("WARNING: products with no factory match:", list(missing))
    return df


def attach_destination(df: pd.DataFrame) -> pd.DataFrame:
    cent = read_reference(STATE_CENTROIDS_FILE).rename(
        columns={"state": COL_STATE, "latitude": "dest_lat", "longitude": "dest_lon"}
    )
    cent[COL_STATE] = cent[COL_STATE].str.strip()
    df = df.copy()
    df[COL_STATE] = df[COL_STATE].str.strip()
    df = df.merge(cent, on=COL_STATE, how="left")
    missing = df.loc[df["dest_lat"].isna(), COL_STATE].unique()
    if len(missing):
        print("WARNING: states with no coordinates:", list(missing))
    return df


def add_distance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["distance_km"] = haversine_km(df["factory_lat"], df["factory_lon"],
                                     df["dest_lat"], df["dest_lon"])
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop bad rows and extreme lead-time outliers."""
    n0 = len(df)
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET, "factory", "dest_lat", "distance_km"])
    df = df[df[TARGET] >= 0]
    df = df[(df[COL_UNITS] > 0) & (df[COL_SALES] > 0)]
    cap = df[TARGET].quantile(OUTLIER_PERCENTILE / 100)
    df = df[df[TARGET] <= cap]
    print(f"Lead-time cap ({OUTLIER_PERCENTILE}th pct): {cap:.0f} days")
    print(f"Rows: {n0} -> {len(df)}")
    return df.reset_index(drop=True)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add snake_case copies and time-control features used by the model and app."""
    df = df.copy()
    df["ship_mode"] = df[COL_SHIP_MODE]
    df["region"] = df[COL_REGION]
    df["division"] = df[COL_DIVISION]
    df["units"] = df[COL_UNITS]
    df["sales"] = df[COL_SALES]
    df["gross_profit"] = df[COL_PROFIT]
    df["product"] = df[COL_PRODUCT]
    df["state"] = df[COL_STATE]
    # NEW: time controls
    od = pd.to_datetime(df[COL_ORDER_DATE])
    df["order_ordinal"] = (od - pd.Timestamp(ORDINAL_ORIGIN)).dt.days
    df["order_month"] = od.dt.month
    return df


def build_clean() -> pd.DataFrame:
    df = load_raw()
    df = add_lead_time(df)
    df = attach_factory(df)
    df = attach_destination(df)
    df = add_distance(df)
    df = clean(df)
    df = standardize_columns(df)
    df.to_csv(CLEAN_FILE, index=False)
    return df


def validate(df: pd.DataFrame) -> None:
    """Hard checks. Raise if something is wrong."""
    needed = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET, "factory", "product"]
    assert not df[needed].isna().any().any(), "Nulls found in model columns"
    assert (df["distance_km"] >= 0).all(), "Negative distance"
    assert (df[TARGET] >= 0).all(), "Negative lead time"
    assert df["factory"].nunique() <= 5, "More than 5 factories?"
    print("All validation checks passed.")