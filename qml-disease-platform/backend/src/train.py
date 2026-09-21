"""
train.py
=========
Orchestrates the FULL pipeline end-to-end for one disease:
  data_preprocessing -> feature_selection -> classical_baselines -> quantum_model (VQC + QSVM)
  -> evaluate -> explainability -> save all artifacts to backend/models/<disease_id>/

USAGE
------
    cd backend/src
    python train.py heart
    python train.py diabetes --feature-strategy pso
    python train.py kidney --feature-strategy mutual_info --epochs 80

This is the script Optuna's hyperparameter search (Step 7 of the project) will call repeatedly
with different circuit depths / learning rates - see the `run_pipeline()` function signature,
which is deliberately kept Optuna-trial-friendly (all tunable knobs are plain function arguments).
"""

import argparse
import json
import logging
import os
import joblib
import torch
import numpy as np

from config import get_disease
from data_preprocessing import prepare_dataset, prepare_dataset_generic
from feature_selection import select_features
from classical_baselines import train_all_baselines, predict_baseline
from quantum_model import train_hybrid_quantum_model, train_qsvm, predict_hybrid, predict_qsvm
from evaluate import build_comparison_table, compute_metrics, compute_roc_curve
from explainability import quantum_parameter_sensitivity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def _run_pipeline_on_prepared_data(
    ds,
    disease_id: str,
    display_name: str,
    n_qubits_target: int,
    feature_strategy: str = "mutual_info",
    vqc_epochs: int = 60,
    vqc_lr: float = 0.05,
    ansatz_reps: int = 2,
    max_vqc_train_samples: int = None,
    save_artifacts: bool = True,
    extra_metadata: dict = None,
) -> dict:
    """
    The actual pipeline body (feature selection -> classical baselines -> quantum models ->
    evaluation -> saving), factored out so BOTH a config-registered disease (run_pipeline below)
    AND an arbitrary user-uploaded dataset (run_pipeline_generic below, backing the General
    module) go through identical training and evaluation code - the only difference between them
    is how `ds` (a PreparedDataset) got built.
    """
    # ---- Stage 2: Feature selection (reduce to n_qubits_target features) ----
    selected_idx = select_features(
        feature_strategy, ds.X_train, ds.y_train, n_qubits_target, ds.feature_names
    )
    X_train = ds.X_train[:, selected_idx]
    X_val = ds.X_val[:, selected_idx]
    X_test = ds.X_test[:, selected_idx]
    selected_names = [ds.feature_names[i] for i in selected_idx]
    logger.info(f"Reduced to {len(selected_idx)} features for quantum encoding: {selected_names}")

    # ---- Stage 3: Classical baselines ----
    baselines = train_all_baselines(X_train, ds.y_train, X_val, ds.y_val)

    # ---- Stage 4: Hybrid quantum model + QSVM ----
    # VQC training cost is dominated by state-vector simulation of the circuit once PER SAMPLE
    # PER EPOCH (full-batch gradient descent here), which is far slower than classical matrix
    # ops. On a laptop/simulator, training on the full split (hundreds of samples) for dozens of
    # epochs can take many minutes - real, measured behavior during development, not a
    # theoretical concern. `max_vqc_train_samples` optionally subsamples (stratified) JUST the
    # VQC's training data to keep wall-clock time reasonable for a hackathon demo; classical
    # baselines above already trained on the FULL split, so report this subsampling explicitly
    # in your results write-up rather than silently comparing unequal training set sizes.
    X_train_vqc, y_train_vqc = X_train, ds.y_train
    if max_vqc_train_samples and len(y_train_vqc) > max_vqc_train_samples:
        from sklearn.model_selection import train_test_split as _tts
        X_train_vqc, _, y_train_vqc, _ = _tts(
            X_train, ds.y_train, train_size=max_vqc_train_samples,
            stratify=ds.y_train, random_state=42,
        )
        logger.warning(
            f"[HybridVQC] Subsampled training set to {max_vqc_train_samples} rows (from "
            f"{len(ds.y_train)}) to keep quantum simulation wall-clock time tractable."
        )

    vqc_result = train_hybrid_quantum_model(
        X_train_vqc, y_train_vqc, X_val, ds.y_val,
        n_qubits=len(selected_idx), epochs=vqc_epochs, lr=vqc_lr, ansatz_reps=ansatz_reps,
    )
    qsvm_result = train_qsvm(X_train, ds.y_train, n_qubits=len(selected_idx))

    # ---- Stage 5: Evaluation on the held-out test set ----
    all_results = {}
    for name, entry in baselines.items():
        all_results[name] = {
            "y_pred_proba": predict_baseline(entry, X_test),
            "train_time_sec": entry["train_time_sec"],
        }
    all_results["HybridVQC"] = {
        "y_pred_proba": predict_hybrid(vqc_result, X_test),
        "train_time_sec": vqc_result["train_time_sec"],
    }
    all_results["QSVM"] = {
        "y_pred_proba": predict_qsvm(qsvm_result, X_test),
        "train_time_sec": qsvm_result["train_time_sec"],
    }
    comparison_df = build_comparison_table(all_results, ds.y_test)

    # ---- Stage 6: Explainability (quantum parameter sensitivity, fast + always run) ----
    sensitivity = quantum_parameter_sensitivity(vqc_result["model"], X_test[:20])

    # ---- Build the detailed per-model performance record the React "Model Performance" page
    #      reads from (confusion matrix + ROC curve for every model, not just the summary table)
    performance = {"disease_id": disease_id, "display_name": display_name, "models": {}}
    for name, r in all_results.items():
        m = compute_metrics(ds.y_test, r["y_pred_proba"])
        roc = compute_roc_curve(ds.y_test, r["y_pred_proba"])
        performance["models"][name] = {
            "accuracy": round(m["accuracy"], 4),
            "sensitivity": round(m["sensitivity_recall"], 4),
            "specificity": round(m["specificity"], 4),
            "precision": round(m["precision"], 4),
            "f1_score": round(m["f1_score"], 4),
            "auc_roc": None if np.isnan(m["auc_roc"]) else round(m["auc_roc"], 4),
            "confusion_matrix": m["confusion_matrix"],
            "roc_curve": roc,
            "train_time_sec": round(r["train_time_sec"], 4),
        }
    performance["test_set_size"] = int(len(ds.y_test))
    performance["test_set_positive_rate"] = round(float(ds.y_test.mean()), 4)
    performance["best_model_by_accuracy"] = comparison_df.iloc[0]["Model"]

    if save_artifacts:
        out_dir = os.path.join(MODELS_DIR, disease_id)
        os.makedirs(out_dir, exist_ok=True)

        torch.save(vqc_result["model"].state_dict(), os.path.join(out_dir, "hybrid_vqc.pt"))
        joblib.dump(ds.scaler, os.path.join(out_dir, "scaler.joblib"))
        joblib.dump(baselines["RandomForest"]["model"], os.path.join(out_dir, "random_forest.joblib"))

        metadata = {
            "disease_id": disease_id,
            "display_name": display_name,
            "all_feature_names": ds.feature_names,
            "selected_feature_indices": selected_idx,
            "selected_feature_names": selected_names,
            "n_qubits": len(selected_idx),
            "ansatz_reps": ansatz_reps,
            "feature_strategy": feature_strategy,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        with open(os.path.join(out_dir, "performance.json"), "w") as f:
            json.dump(performance, f, indent=2)

        comparison_df.to_csv(os.path.join(out_dir, "comparison_table.csv"), index=False)
        logger.info(f"Artifacts saved to {out_dir}")

    return {
        "comparison_table": comparison_df,
        "performance": performance,
        "quantum_sensitivity": sensitivity,
        "selected_features": selected_names,
    }


def run_pipeline(
    disease_id: str,
    feature_strategy: str = "mutual_info",
    vqc_epochs: int = 60,
    vqc_lr: float = 0.05,
    ansatz_reps: int = 2,
    max_vqc_train_samples: int = None,
    save_artifacts: bool = True,
) -> dict:
    cfg = get_disease(disease_id)
    logger.info(f"===== Running pipeline for '{cfg.display_name}' ({disease_id}) =====")
    ds = prepare_dataset(disease_id)
    return _run_pipeline_on_prepared_data(
        ds, disease_id, cfg.display_name, cfg.n_qubits_target,
        feature_strategy, vqc_epochs, vqc_lr, ansatz_reps, max_vqc_train_samples, save_artifacts,
    )


def run_pipeline_generic(
    csv_path: str,
    target_column: str,
    positive_label,
    display_name: str = "Custom Dataset",
    disease_id: str = "general",
    n_qubits_target: int = 6,
    feature_strategy: str = "mutual_info",
    vqc_epochs: int = 10,
    vqc_lr: float = 0.05,
    ansatz_reps: int = 2,
    max_vqc_train_samples: int = 50,
    save_artifacts: bool = True,
) -> dict:
    """
    The General module's training entry point (backs POST /general/train in api/main.py): same
    pipeline as run_pipeline() above, but for a dataset that isn't in config.py at all - the
    user supplied the CSV, the target column name, and which value in that column means
    "disease present" through the upload UI. Defaults (vqc_epochs=10, max_vqc_train_samples=50)
    are set lower than the config-based diseases' defaults since this runs synchronously inside
    an HTTP request from the browser - see api/main.py's /general/train handler for the timeout
    reasoning.
    """
    logger.info(f"===== Running GENERAL pipeline on uploaded dataset '{display_name}' =====")
    ds = prepare_dataset_generic(csv_path, target_column, positive_label, disease_id=disease_id)
    return _run_pipeline_on_prepared_data(
        ds, disease_id, display_name, n_qubits_target,
        feature_strategy, vqc_epochs, vqc_lr, ansatz_reps, max_vqc_train_samples, save_artifacts,
        extra_metadata={
            "source": "user_upload", "target_column": target_column,
            "positive_label": str(positive_label),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the hybrid QML pipeline for one disease.")
    parser.add_argument("disease_id", choices=[
        "heart", "cardiovascular", "diabetes", "kidney", "cancer",
    ], help="Which disease module to train (tabular diseases only - see cnn_encoder.py for brain_tumor).")
    parser.add_argument("--feature-strategy", default="mutual_info", choices=["mutual_info", "rfe", "pso"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--ansatz-reps", type=int, default=2)
    parser.add_argument("--max-vqc-train-samples", type=int, default=None,
                         help="Subsample the VQC's training data to this many rows (stratified) "
                              "to bound quantum-simulation wall-clock time. Recommended on a "
                              "laptop CPU simulator; omit for a full run on stronger hardware.")
    args = parser.parse_args()

    run_pipeline(
        args.disease_id,
        feature_strategy=args.feature_strategy,
        vqc_epochs=args.epochs,
        vqc_lr=args.lr,
        ansatz_reps=args.ansatz_reps,
        max_vqc_train_samples=args.max_vqc_train_samples,
    )
