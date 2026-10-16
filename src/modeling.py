"""Training, evaluation, and selection of lead-time models."""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import *

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
# Simplest to most complex. Used for the interpretability tie-break.
COMPLEXITY_ORDER = ["Linear Regression", "Random Forest", "Gradient Boosting"]


def load_clean() -> pd.DataFrame:
    return pd.read_csv(CLEAN_FILE)


def split(df: pd.DataFrame):
    X, y = df[FEATURES], df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def get_models() -> dict:
    return {
        "Baseline (mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_SEED),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_SEED),
    }


def make_pipeline(model) -> Pipeline:
    return Pipeline([("prep", make_preprocessor()), ("model", model)])


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def run_experiments(X_train, X_test, y_train, y_test):
    """Train every model; return (results DataFrame, dict of fitted pipelines)."""
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    rows, fitted = [], {}
    for name, model in get_models().items():
        pipe = make_pipeline(model)
        cv = -cross_val_score(pipe, X_train, y_train, cv=kf,
                              scoring="neg_root_mean_squared_error", n_jobs=1)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append({
            "model": name,
            "cv_rmse": cv.mean(),
            "cv_rmse_std": cv.std(),
            "test_rmse": rmse(y_test, pred),
            "test_mae": mean_absolute_error(y_test, pred),
            "test_r2": r2_score(y_test, pred),
        })
        fitted[name] = pipe
    results = pd.DataFrame(rows).round(4)
    return results, fitted


def select_best(results: pd.DataFrame, tol: float = 0.02) -> str:
    """Lowest CV RMSE wins; if a simpler model is within `tol` (2%), prefer the simpler one."""
    cand = results[results["model"] != "Baseline (mean)"].set_index("model")
    best = cand["cv_rmse"].min()
    for name in COMPLEXITY_ORDER:
        if name in cand.index and cand.loc[name, "cv_rmse"] <= best * (1 + tol):
            return name
    return cand["cv_rmse"].idxmin()


def save_model(pipe) -> None:
    joblib.dump(pipe, MODEL_FILE)


def load_model():
    return joblib.load(MODEL_FILE)

