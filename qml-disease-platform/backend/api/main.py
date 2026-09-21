"""
api/main.py
=============
STEP 8: Deployment - FastAPI backend for Qure AI.

Serves prediction endpoints for every config-registered disease, PLUS the General module, which
lets a user upload an arbitrary patient CSV, pick a target column, and train the exact same
hybrid quantum-classical pipeline on it with no code changes - see the /general/* endpoints.

WHY FASTAPI (over Flask/Django) FOR THIS PROJECT
------------------------------------------------------
- Automatic request/response validation via Pydantic models - a malformed patient record from
  the React form is rejected with a clear 422 error before it ever reaches the trained model.
- Automatic interactive API docs at /docs, useful for your project demo/viva.
- Native async support, which matters if you later add a queue for slower quantum-circuit
  inference under concurrent load.

RUN WITH:
    cd backend/api
    uvicorn main:app --reload --port 8000
Then the React frontend (configured for http://localhost:8000) can call it directly.
"""

import sys
import os
import json
import io
import joblib
import torch
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from config import DISEASE_REGISTRY, get_disease
from quantum_model import HybridQuantumClassifier
from data_preprocessing import inspect_csv as _inspect_csv
from train import run_pipeline_generic

app = FastAPI(
    title="Qure AI",
    description="Hybrid Quantum-Classical Machine Learning Platform for Early Disease Detection (SIH26139)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
GENERAL_UPLOADS_DIR = os.path.join(DATA_DIR, "general_uploads")
os.makedirs(GENERAL_UPLOADS_DIR, exist_ok=True)

_loaded_models: dict = {}  # cache key -> {"model", "scaler", "rf", "explainer", "metadata"}


# ---------------------------------------------------------------------------
# Shared response models
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(..., description="Feature name -> raw value")


class FeatureContribution(BaseModel):
    feature: str
    value: float
    contribution: float  # signed: positive pushes toward "disease present"


class PredictionResponse(BaseModel):
    disease_id: str
    probability: float
    prediction: str
    confidence_band: str
    model_used: str
    explanation: list[FeatureContribution] = []
    explanation_note: str = ""


# ---------------------------------------------------------------------------
# Core artifact loading (shared by config-registered diseases AND the General module)
# ---------------------------------------------------------------------------

def _resolve_model_dir(disease_id: str) -> tuple[str, str]:
    """
    Returns (cache_key, on_disk_model_dir) for a disease, resolving `alias_of` (see
    config.py: cardiovascular has no trained model of its own, it deliberately serves
    heart's). The cache key stays the REQUESTED disease_id even when aliased, so
    "cardiovascular" and "heart" are cached as separate dict entries even though they load
    the same files from disk - this keeps each response's `disease_id` field accurate without
    extra bookkeeping.
    """
    cfg = get_disease(disease_id)
    source_id = cfg.alias_of or disease_id
    return disease_id, os.path.join(MODELS_DIR, source_id)


def _load_tabular_artifacts(cache_key: str, out_dir: str) -> dict:
    """
    Generic artifact loader used for EVERY tabular model directory - a config-registered
    disease's folder (backend/models/heart/) or the General module's folder
    (backend/models/general/). Both are produced by the same train.py code path and have an
    identical file layout, so one loader function serves both.
    """
    if cache_key in _loaded_models:
        return _loaded_models[cache_key]

    metadata_path = os.path.join(out_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise HTTPException(
            status_code=404,
            detail=f"No trained model found at '{out_dir}'. Train it first (see README).",
        )

    with open(metadata_path) as f:
        metadata = json.load(f)

    model = HybridQuantumClassifier(n_raw_features=metadata["n_qubits"], n_qubits=metadata["n_qubits"])
    model.load_state_dict(torch.load(os.path.join(out_dir, "hybrid_vqc.pt")))
    model.eval()

    scaler = joblib.load(os.path.join(out_dir, "scaler.joblib"))
    rf_model = joblib.load(os.path.join(out_dir, "random_forest.joblib"))
    # TreeExplainer is exact and near-instant for a Random Forest, unlike KernelExplainer on the
    # quantum model itself (which needs ~100 circuit evaluations per explained sample - far too
    # slow for a live HTTP request). Explaining the RF surrogate is the same honest trade-off
    # documented in explainability.py: the RF was trained on the identical selected features, so
    # its feature attributions are a fast, faithful-enough proxy for "why did this look positive
    # or negative", even though the actual served prediction comes from the quantum model.
    rf_explainer = shap.TreeExplainer(rf_model)

    artifacts = {"model": model, "scaler": scaler, "rf_model": rf_model,
                 "rf_explainer": rf_explainer, "metadata": metadata}
    _loaded_models[cache_key] = artifacts
    return artifacts


def _scale_selected_features(artifacts: dict, raw_values: list) -> np.ndarray:
    """Places raw feature values into a full-width vector, scales it, and slices back down to
    just the selected features - because the saved MinMaxScaler was fit on ALL raw features
    before selection, not just the ones the quantum model actually uses."""
    selected_idx = artifacts["metadata"]["selected_feature_indices"]
    scaled_full = np.zeros((1, len(artifacts["metadata"]["all_feature_names"])))
    for pos, feat_idx in enumerate(selected_idx):
        scaled_full[0, feat_idx] = raw_values[pos]
    scaled_full = artifacts["scaler"].transform(scaled_full)
    return scaled_full[0, selected_idx].reshape(1, -1)


def _predict_with_explanation(artifacts: dict, features: dict) -> dict:
    """
    Core single-row prediction logic, shared by the per-disease /predict endpoint, the
    per-disease /predict-batch endpoint (called once per CSV row), and the General module's
    /general/predict - so there is exactly one place that implements "how do we turn a feature
    dict into a probability and an explanation", not three slightly-different copies of it.
    """
    required = artifacts["metadata"]["selected_feature_names"]
    missing = [f for f in required if f not in features]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    raw_values = [features[f] for f in required]
    scaled_selected = _scale_selected_features(artifacts, raw_values)

    with torch.no_grad():
        prob = artifacts["model"](torch.tensor(scaled_selected, dtype=torch.float32)).item()

    if prob >= 0.7 or prob <= 0.3:
        confidence_band = "high"
    elif prob >= 0.6 or prob <= 0.4:
        confidence_band = "moderate"
    else:
        confidence_band = "low"

    # Explainable AI: SHAP contributions from the Random Forest surrogate for THIS specific
    # patient's values (a local explanation, not a global feature-importance ranking).
    shap_values = artifacts["rf_explainer"].shap_values(scaled_selected)
    # shap_values shape varies by scikit-learn/shap version for binary classifiers: either a
    # single (1, n_features) array for the positive class, or a list of two such arrays
    # ([negative_class, positive_class]) - handle both so this doesn't silently break on a
    # different shap version than the one this was built against.
    if isinstance(shap_values, list):
        contributions = shap_values[1][0]
    elif shap_values.ndim == 3:
        contributions = shap_values[0, :, 1]
    else:
        contributions = shap_values[0]

    explanation = sorted(
        [
            {"feature": name, "value": round(float(val), 4), "contribution": round(float(contrib), 4)}
            for name, val, contrib in zip(required, raw_values, contributions)
        ],
        key=lambda e: abs(e["contribution"]),
        reverse=True,
    )

    return {
        "probability": round(prob, 4),
        "prediction": "Disease Likely" if prob >= 0.5 else "Disease Unlikely",
        "confidence_band": confidence_band,
        "explanation": explanation,
    }


EXPLANATION_NOTE = (
    "Feature contributions are computed with SHAP on a Random Forest trained on the same "
    "selected features as the quantum model (a fast, faithful surrogate) - explaining the "
    "quantum circuit itself with SHAP directly would take many seconds per prediction, too slow "
    "for a live request. Positive values push the prediction toward 'disease present'."
)


@app.exception_handler(KeyError)
async def key_error_handler(request, exc: KeyError):
    return JSONResponse(status_code=404, content={"detail": f"Unknown disease: {exc}"})


@app.get("/")
def root():
    return {"status": "ok", "message": "Qure AI - Hybrid Quantum ML Disease Detection Platform API"}


@app.get("/diseases")
def list_diseases():
    """Returns every disease module for the React navbar."""
    return [
        {
            "disease_id": d.disease_id,
            "display_name": d.display_name,
            "modality": d.modality,
            "source": d.source,
        }
        for d in DISEASE_REGISTRY.values()
    ]


@app.get("/diseases/{disease_id}/performance")
def disease_performance(disease_id: str):
    """
    Returns the full benchmarking record for this disease: accuracy, sensitivity, specificity,
    precision, F1, AUC-ROC, confusion matrix, and ROC curve points for EVERY model trained on
    it. Resolves `alias_of` (cardiovascular -> heart) so the performance page works for aliased
    diseases too, without a separate trained model existing on disk for them.
    """
    cfg = get_disease(disease_id)
    source_id = cfg.alias_of or disease_id
    performance_path = os.path.join(MODELS_DIR, source_id, "performance.json")
    if not os.path.exists(performance_path):
        raise HTTPException(
            status_code=404,
            detail=f"No performance data for '{disease_id}' yet. Run `python train.py {source_id}` first.",
        )
    with open(performance_path) as f:
        data = json.load(f)
    # Report under the REQUESTED disease's identity, not the aliased source's
    data["disease_id"] = disease_id
    data["display_name"] = cfg.display_name
    return data


@app.get("/diseases/{disease_id}/metadata")
def disease_metadata(disease_id: str):
    """
    Returns the trained model's selected feature names, so the React frontend can dynamically
    render the correct input form fields for this specific disease.
    """
    cfg = get_disease(disease_id)
    if cfg.modality != "tabular":
        raise HTTPException(status_code=400, detail=f"'{disease_id}' is an imaging module - see /diseases/{disease_id}/predict-image")

    cache_key, out_dir = _resolve_model_dir(disease_id)
    artifacts = _load_tabular_artifacts(cache_key, out_dir)
    return {
        "disease_id": disease_id,
        "display_name": cfg.display_name,
        "required_features": artifacts["metadata"]["selected_feature_names"],
        "n_qubits": artifacts["metadata"]["n_qubits"],
    }


@app.post("/diseases/{disease_id}/predict", response_model=PredictionResponse)
def predict(disease_id: str, request: PredictionRequest):
    cfg = get_disease(disease_id)
    if cfg.modality != "tabular":
        raise HTTPException(status_code=400, detail=f"'{disease_id}' requires image upload, not this endpoint")

    cache_key, out_dir = _resolve_model_dir(disease_id)
    artifacts = _load_tabular_artifacts(cache_key, out_dir)

    try:
        result = _predict_with_explanation(artifacts, request.features)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return PredictionResponse(
        disease_id=disease_id,
        model_used="Hybrid Quantum-Classical VQC (Qiskit)",
        explanation_note=EXPLANATION_NOTE,
        **result,
    )


@app.post("/diseases/{disease_id}/predict-batch")
async def predict_batch(disease_id: str, file: UploadFile = File(...)):
    """
    Upload a CSV of multiple patients at once (one row per patient, columns matching the
    model's required feature names - extra columns are ignored, so you can keep a patient-ID
    column in your file). Returns one prediction per row; a row with missing/invalid values
    reports its own error instead of failing the whole batch.
    """
    cfg = get_disease(disease_id)
    if cfg.modality != "tabular":
        raise HTTPException(status_code=400, detail=f"'{disease_id}' does not support CSV batch prediction")

    cache_key, out_dir = _resolve_model_dir(disease_id)
    artifacts = _load_tabular_artifacts(cache_key, out_dir)

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {e}")
    df.columns = [str(c).strip() for c in df.columns]

    required = artifacts["metadata"]["selected_feature_names"]
    missing_cols = [f for f in required if f not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=422,
            detail=f"CSV is missing required columns: {missing_cols}. "
                   f"Required columns are: {required}",
        )

    results = []
    for idx, row in df.iterrows():
        try:
            features = {f: float(row[f]) for f in required}
            r = _predict_with_explanation(artifacts, features)
            results.append({"row_index": int(idx), **r})
        except Exception as e:
            results.append({"row_index": int(idx), "error": str(e)})

    return {
        "disease_id": disease_id,
        "n_rows": len(df),
        "n_succeeded": sum(1 for r in results if "error" not in r),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Brain Tumor (imaging) module
# ---------------------------------------------------------------------------

def _load_brain_tumor_artifacts() -> dict:
    disease_id = "brain_tumor"
    if disease_id in _loaded_models:
        return _loaded_models[disease_id]

    out_dir = os.path.join(MODELS_DIR, disease_id)
    metadata_path = os.path.join(out_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise HTTPException(
            status_code=404,
            detail="No trained Brain Tumor model found. Train the CNN encoder with "
                   "`cnn_encoder.train_cnn_bottleneck()` and the quantum classifier on top of "
                   "its embeddings, then save both to backend/models/brain_tumor/ before using "
                   "this endpoint. See backend/src/cnn_encoder.py for the full procedure.",
        )

    from cnn_encoder import CNNBottleneckEncoder, IMAGE_TRANSFORM

    with open(metadata_path) as f:
        metadata = json.load(f)

    cnn = CNNBottleneckEncoder(n_qubits=metadata["n_qubits"])
    cnn.load_state_dict(torch.load(os.path.join(out_dir, "cnn_encoder.pt")))
    cnn.eval()

    quantum_model = HybridQuantumClassifier(n_raw_features=metadata["n_qubits"], n_qubits=metadata["n_qubits"])
    quantum_model.load_state_dict(torch.load(os.path.join(out_dir, "hybrid_vqc.pt")))
    quantum_model.eval()

    artifacts = {
        "cnn": cnn, "quantum_model": quantum_model,
        "classes": metadata["classes"], "transform": IMAGE_TRANSFORM,
    }
    _loaded_models[disease_id] = artifacts
    return artifacts


@app.post("/diseases/brain_tumor/predict-image", response_model=PredictionResponse)
async def predict_brain_tumor(file: UploadFile = File(...)):
    artifacts = _load_brain_tumor_artifacts()

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    tensor = artifacts["transform"](image).unsqueeze(0)

    with torch.no_grad():
        embedding = artifacts["cnn"](tensor, return_embedding=True)
        prob = artifacts["quantum_model"].quantum_layer(embedding)
        prob = torch.sigmoid(artifacts["quantum_model"].post_net(prob)).item()

    confidence_band = "high" if (prob >= 0.7 or prob <= 0.3) else "moderate" if (prob >= 0.6 or prob <= 0.4) else "low"

    return PredictionResponse(
        disease_id="brain_tumor",
        probability=round(prob, 4),
        prediction="Disease Likely" if prob >= 0.5 else "Disease Unlikely",
        confidence_band=confidence_band,
        model_used="CNN Bottleneck (ResNet-18) + Hybrid Quantum VQC",
        explanation=[],
        explanation_note="Feature-level explanation is not available for the imaging pipeline "
                          "in this build (see explainability.py for planned Grad-CAM support).",
    )


# ---------------------------------------------------------------------------
# General module: upload ANY tabular dataset, pick a target column, train on it
# ---------------------------------------------------------------------------

@app.post("/general/inspect")
async def general_inspect(file: UploadFile = File(...)):
    """
    Step 1 of the General module: upload a CSV and get back its columns, row count, and (for
    low-cardinality columns) their actual distinct values - so the frontend can render a target
    column dropdown, and once the user picks one, a "which value means disease present"
    dropdown, without guessing your data's exact label formatting.
    """
    content = await file.read()
    tmp_path = os.path.join(GENERAL_UPLOADS_DIR, "_inspect_tmp.csv")
    with open(tmp_path, "wb") as f:
        f.write(content)
    try:
        info = _inspect_csv(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read CSV: {e}")
    return info


@app.post("/general/train")
async def general_train(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    positive_label: str = Form(...),
    display_name: str = Form("Custom Dataset"),
    n_qubits: int = Form(6),
    feature_strategy: str = Form("mutual_info"),
    epochs: int = Form(10),
    max_vqc_train_samples: int = Form(50),
):
    """
    Step 2: trains the full pipeline (classical baselines + Hybrid VQC + QSVM) on the uploaded
    dataset, synchronously. Runs INSIDE this HTTP request - defaults are kept small (10 epochs,
    50-sample VQC cap) specifically so this finishes in roughly a minute rather than several,
    since there is no background job queue in this build. Raise `epochs`/`max_vqc_train_samples`
    via the form if you're calling this from a script and can tolerate a longer wait.
    """
    content = await file.read()
    csv_path = os.path.join(GENERAL_UPLOADS_DIR, "current_dataset.csv")
    with open(csv_path, "wb") as f:
        f.write(content)

    if n_qubits < 2 or n_qubits > 12:
        raise HTTPException(status_code=422, detail="n_qubits must be between 2 and 12 (simulator limits)")

    try:
        result = run_pipeline_generic(
            csv_path, target_column=target_column, positive_label=positive_label,
            display_name=display_name, disease_id="general",
            n_qubits_target=n_qubits, feature_strategy=feature_strategy,
            vqc_epochs=epochs, max_vqc_train_samples=max_vqc_train_samples,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    _loaded_models.pop("general", None)  # invalidate cache so the next predict reloads fresh weights
    return result["performance"]


@app.get("/general/performance")
def general_performance():
    performance_path = os.path.join(MODELS_DIR, "general", "performance.json")
    if not os.path.exists(performance_path):
        raise HTTPException(status_code=404, detail="No custom model trained yet. Use /general/train first.")
    with open(performance_path) as f:
        return json.load(f)


@app.get("/general/metadata")
def general_metadata():
    out_dir = os.path.join(MODELS_DIR, "general")
    metadata_path = os.path.join(out_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise HTTPException(status_code=404, detail="No custom model trained yet. Use /general/train first.")
    artifacts = _load_tabular_artifacts("general", out_dir)
    return {
        "disease_id": "general",
        "display_name": artifacts["metadata"].get("display_name", "Custom Dataset"),
        "required_features": artifacts["metadata"]["selected_feature_names"],
        "n_qubits": artifacts["metadata"]["n_qubits"],
    }


@app.post("/general/predict", response_model=PredictionResponse)
def general_predict(request: PredictionRequest):
    out_dir = os.path.join(MODELS_DIR, "general")
    artifacts = _load_tabular_artifacts("general", out_dir)
    try:
        result = _predict_with_explanation(artifacts, request.features)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return PredictionResponse(
        disease_id="general",
        model_used="Hybrid Quantum-Classical VQC (Qiskit)",
        explanation_note=EXPLANATION_NOTE,
        **result,
    )


@app.post("/general/predict-batch")
async def general_predict_batch(file: UploadFile = File(...)):
    out_dir = os.path.join(MODELS_DIR, "general")
    artifacts = _load_tabular_artifacts("general", out_dir)

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {e}")
    df.columns = [str(c).strip() for c in df.columns]

    required = artifacts["metadata"]["selected_feature_names"]
    missing_cols = [f for f in required if f not in df.columns]
    if missing_cols:
        raise HTTPException(status_code=422, detail=f"CSV is missing required columns: {missing_cols}")

    results = []
    for idx, row in df.iterrows():
        try:
            features = {f: float(row[f]) for f in required}
            r = _predict_with_explanation(artifacts, features)
            results.append({"row_index": int(idx), **r})
        except Exception as e:
            results.append({"row_index": int(idx), "error": str(e)})

    return {
        "disease_id": "general", "n_rows": len(df),
        "n_succeeded": sum(1 for r in results if "error" not in r),
        "results": results,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
