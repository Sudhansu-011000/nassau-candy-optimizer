"""Central configuration: paths, constants, and assumptions."""
from pathlib import Path

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ---------- Files ----------
RAW_FILE = RAW_DIR / "Nassau Candy Distributor.csv"  # <- match your filename

FACTORIES_FILE = REFERENCE_DIR / "factories.csv"
PRODUCT_FACTORY_FILE = REFERENCE_DIR / "product_factory.csv"

CLEAN_FILE = PROCESSED_DIR / "clean.csv"
FEATURES_FILE = PROCESSED_DIR / "feature_matrix.csv"
ROUTE_CLUSTERS_FILE = PROCESSED_DIR / "route_clusters.csv"

MODEL_FILE = MODELS_DIR / "best_model.joblib"

# ---------- Reproducibility ----------
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ---------- Data cleaning ----------
OUTLIER_PERCENTILE = 99

COL_ORDER_DATE = "Order Date"
COL_SHIP_DATE = "Ship Date"
COL_SHIP_MODE = "Ship Mode"
COL_PRODUCT = "Product Name"
COL_REGION = "Region"
COL_STATE = "State/Province"
COL_DIVISION = "Division"
COL_SALES = "Sales"
COL_UNITS = "Units"
COL_PROFIT = "Gross Profit"
COL_COST = "Cost"

TARGET = "lead_time"

# Date parsing: set True if your dates look like 25-12-2024 (day first)
DATE_DAYFIRST = False

STATE_CENTROIDS_FILE = REFERENCE_DIR / "state_centroids.csv"

# Features the model will use (Step 3)
ORDINAL_ORIGIN = "2020-01-01"   # reference date for the order_ordinal feature

NUMERIC_FEATURES = ["distance_km", "units", "sales", "order_ordinal", "order_month"]
CATEGORICAL_FEATURES = ["ship_mode", "region", "division"]

# ---------- Simulation assumptions ----------
COST_PER_UNIT_KM = 0.0005   # ASSUMPTION: no freight data in the dataset
TOP_N = 3
PROFIT_WARNING_PCT = 5.0
PROFIT_HIGH_RISK_PCT = 10.0
MIN_CONFIDENCE = 50

# ---------- Make sure output folders exist ----------
for _folder in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
    _folder.mkdir(parents=True, exist_ok=True)

METRICS_FILE = PROCESSED_DIR / "model_metrics.csv"


MIN_ROUTE_ORDERS = 10    # routes with fewer orders are ignored (too noisy)
REGION_PRODUCT_FILE = PROCESSED_DIR / "region_product_congestion.csv"


SCENARIOS_FILE = PROCESSED_DIR / "scenarios.csv"
RISK_WEIGHT = 0.2                # share of the ranking score given to risk reduction
CONF_REF_ORDERS = 100            # orders needed for full data-support confidence
COST_SENSITIVITY = (0.5, 2.0)    # freight-rate multipliers for the profit stress test