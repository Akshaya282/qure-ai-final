"""
data_preprocessing.py
======================
STEP 1 of the pipeline: Data Ingestion & Preprocessing.

WHAT THIS STEP DOES AND WHY
-----------------------------
1. Ingestion: pulls the disease's dataset straight from the UCI ML Repository via the official
   `ucimlrepo` package (this requires internet access on whichever machine actually runs
   training - it will NOT work inside a fully offline sandbox, which is expected and fine for a
   local/hackathon dev machine).
2. Cleaning: UCI biomedical datasets are messy in realistic ways - missing values encoded as
   '?' strings, categorical text labels, inconsistent capitalization. We standardize all of that.
3. Target binarization: every disease's raw label format is different (severity score 0-4,
   'Positive'/'Negative', 'ckd'/'notckd', 'M'/'B'). We normalize every one of them to {0, 1}
   so the exact same downstream training code works for every disease.
4. Scaling: quantum angle encoding (ZZFeatureMap) expects inputs roughly in [0, pi]. We scale
   with MinMaxScaler to [0, 1] then multiply by pi at the point features enter the quantum layer
   (done in quantum_model.py, not here, so scaler parameters stay in the classical domain and are
   the same numbers a classical model would use).

MODELS/ALGORITHMS USED IN THIS STEP
-------------------------------------
- SimpleImputer (scikit-learn): median imputation for numeric columns, most-frequent for
  categorical columns. Chosen over mean imputation because clinical lab values (e.g. serum
  creatinine) are frequently right-skewed with outliers, and median is robust to that.
- LabelEncoder / OrdinalEncoder (scikit-learn): converts categorical text (e.g. 'normal'/
  'abnormal', 'yes'/'no') into integers so they can be fed into scalers and models.
- MinMaxScaler (scikit-learn): scales every feature to [0, 1] range - required both for fair
  comparison across classical models and for quantum angle encoding downstream.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

from config import DiseaseConfig, get_disease

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class PreparedDataset:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list
    scaler: MinMaxScaler
    disease_id: str


def _binarize_target(y: pd.Series, disease_id: str) -> pd.Series:
    """Map every disease's idiosyncratic label format to {0, 1}, 1 = disease present."""
    y = y.copy()
    # NOTE: pandas >= 2.x can give text columns its dedicated 'str' dtype instead of the legacy
    # numpy 'object' dtype, so `y.dtype == object` alone silently misses those columns (this bit
    # us during development: kidney's target had literal "ckd\t" values sneak through unmapped
    # into NaN because the strip/lower step below was skipped). Checking "not numeric" instead
    # catches both dtypes.
    if not pd.api.types.is_numeric_dtype(y):
        y = y.astype(str).str.strip().str.lower()

    if disease_id in ("heart", "cardiovascular"):
        # target is already binary (0/1) in the bundled CSV; pd.to_numeric + >0 handles both
        # this pre-binarized format AND the original UCI 0-4 severity scale transparently,
        # in case you swap in the raw UCI CSV later.
        return (pd.to_numeric(y, errors="coerce") > 0).astype(int)

    if disease_id == "diabetes":
        # Pima Indians 'Outcome' column is already 0/1 numeric
        return pd.to_numeric(y, errors="coerce").astype(int)

    if disease_id == "kidney":
        # raw values have trailing whitespace/tabs in the source file (e.g. "ckd\t"),
        # hence the strip() applied to all object-dtype targets above.
        return y.map({"ckd": 1, "notckd": 0}).astype(int)

    if disease_id == "cancer":
        return y.map({"m": 1, "b": 0}).astype(int)

    raise ValueError(f"No target-binarization rule defined for disease_id='{disease_id}'")


def _clean_features(X: pd.DataFrame) -> pd.DataFrame:
    """Standardize missing-value markers and strip whitespace before imputation."""
    X = X.copy()
    X = X.replace(["?", "", " ", "NA", "N/A", "nan", "NaN"], np.nan)
    for col in X.columns:
        # Same pandas string/object dtype gotcha as in _binarize_target above.
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype(str).str.strip()
            X[col] = X[col].replace(["?", "", " ", "nan", "NaN", "None"], np.nan)
    return X


def load_raw_dataset(disease_id: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Loads the raw feature/target dataframes for a disease from its bundled CSV under
    backend/data/ (see config.py's module docstring for exactly which real dataset backs each
    disease and where it came from). No network access required.
    """
    cfg = get_disease(disease_id)
    if cfg.modality != "tabular":
        raise ValueError(f"'{disease_id}' is an imaging dataset - use cnn_encoder.py instead")

    logger.info(f"Loading '{cfg.display_name}' from bundled dataset: {cfg.csv_path}")
    df = pd.read_csv(cfg.csv_path)
    df.columns = [c.strip() for c in df.columns]  # kidney_disease.csv has trailing spaces on some headers

    y = df[cfg.target_column]
    X = df.drop(columns=[cfg.target_column])

    # kidney_disease.csv also carries a leading 'id' column that is not a clinical feature
    if "id" in X.columns:
        X = X.drop(columns=["id"])

    logger.info(f"Loaded {X.shape[0]} rows, {X.shape[1]} raw features.")
    return X, y


def _finish_preprocessing(
    X_raw: pd.DataFrame,
    y: pd.Series,
    disease_id: str,
    test_size: float,
    val_size: float,
    random_state: int,
) -> "PreparedDataset":
    """
    Shared tail end of preprocessing, used by BOTH prepare_dataset() (config-registered
    diseases) and prepare_dataset_generic() (arbitrary user-uploaded CSVs, see the General
    module): clean -> impute -> encode -> scale -> stratified split. Pulling this out avoids
    duplicating the same five steps twice for what is otherwise identical logic.
    """
    X_raw = _clean_features(X_raw)

    numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_raw.columns if c not in numeric_cols]

    if numeric_cols:
        num_imputer = SimpleImputer(strategy="median")
        X_raw[numeric_cols] = num_imputer.fit_transform(X_raw[numeric_cols])

    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_raw[categorical_cols] = cat_imputer.fit_transform(X_raw[categorical_cols])
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_raw[categorical_cols] = encoder.fit_transform(X_raw[categorical_cols])

    feature_names = X_raw.columns.tolist()
    X = X_raw.values.astype(np.float64)
    y = y.values.astype(np.int64) if not isinstance(y, np.ndarray) else y.astype(np.int64)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_ratio_of_remaining = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_of_remaining, stratify=y_temp, random_state=random_state
    )

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    logger.info(
        f"[{disease_id}] Split sizes -> train={len(y_train)}, val={len(y_val)}, test={len(y_test)} "
        f"| class balance (train) -> positive={y_train.mean():.2%}"
    )

    return PreparedDataset(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        feature_names=feature_names, scaler=scaler, disease_id=disease_id,
    )


def prepare_dataset(
    disease_id: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> PreparedDataset:
    """
    Full Stage-1 pipeline for a disease already registered in config.py:
    load -> clean -> impute -> encode -> binarize target -> scale -> split.

    Split strategy: stratified 70/15/15 train/val/test. Stratification matters here because most
    of these biomedical datasets are class-imbalanced (e.g. kidney: 250 ckd / 150 notckd), and an
    unstratified split can accidentally starve the validation or test set of positive cases.
    """
    X_raw, y_raw = load_raw_dataset(disease_id)
    y = _binarize_target(y_raw, disease_id)
    return _finish_preprocessing(X_raw, y, disease_id, test_size, val_size, random_state)


def prepare_dataset_generic(
    csv_path: str,
    target_column: str,
    positive_label,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    disease_id: str = "general",
) -> PreparedDataset:
    """
    The General module's entry point: runs the SAME Stage-1 pipeline as every config-registered
    disease, but on an arbitrary user-uploaded CSV with a user-specified target column and a
    user-specified "which value means disease present" label — since we cannot hardcode a
    binarization rule for a dataset we've never seen (unlike _binarize_target above, which only
    knows the five diseases in config.py).

    `positive_label` is whatever the frontend collected from the user (a string, since it comes
    off an HTML form/JSON field) - e.g. "1", "yes", "positive", "M". We compare it against the
    target column AS STRINGS on both sides, so numeric target columns (1/0) and text target
    columns ("yes"/"no") are handled by the exact same comparison instead of two code paths.
    """
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in the uploaded CSV. "
            f"Available columns: {df.columns.tolist()}"
        )

    y_raw = df[target_column]
    X_raw = df.drop(columns=[target_column])

    # Drop an obvious non-feature identifier column if present, same as the bundled kidney CSV
    for id_col in ("id", "ID", "Id", "patient_id", "PatientID"):
        if id_col in X_raw.columns:
            X_raw = X_raw.drop(columns=[id_col])

    y = (y_raw.astype(str).str.strip() == str(positive_label).strip()).astype(int)

    if y.nunique() < 2:
        raise ValueError(
            f"After mapping '{positive_label}' as the positive label, the target column has "
            f"only one class ({y.unique().tolist()}). Check that the positive_label value "
            f"exactly matches one of the target column's actual values: "
            f"{y_raw.unique().tolist()[:20]}"
        )

    return _finish_preprocessing(X_raw, y, disease_id, test_size, val_size, random_state)


def inspect_csv(csv_path: str, max_unique_values: int = 12) -> dict:
    """
    Used by the General module's upload step (before training): reads the CSV and reports its
    shape and, for every column with a manageable number of distinct values, what those values
    are - so the frontend can render a target-column dropdown and, once picked, a "which value
    means disease present" dropdown, without the user having to already know their own data's
    exact string formatting (e.g. whether it's "Yes"/"No" or "yes"/"no" or 1/0).
    """
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    columns_info = []
    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        info = {"name": col, "nunique": int(nunique), "dtype": str(df[col].dtype)}
        if nunique <= max_unique_values:
            info["unique_values"] = [str(v) for v in df[col].dropna().unique().tolist()]
        columns_info.append(info)

    return {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": columns_info,
        # Heuristic suggestion only - the user makes the real choice in the UI
        "likely_target_columns": [c["name"] for c in columns_info if c["nunique"] == 2],
    }


if __name__ == "__main__":
    # Quick manual check: run `python data_preprocessing.py heart` from backend/src/
    import sys
    disease = sys.argv[1] if len(sys.argv) > 1 else "heart"
    ds = prepare_dataset(disease)
    print(f"Features ({len(ds.feature_names)}): {ds.feature_names}")
    print(f"X_train shape: {ds.X_train.shape}, y_train shape: {ds.y_train.shape}")
