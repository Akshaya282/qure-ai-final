"""
explainability.py
====================
STEP 6: Explainable AI (XAI).

WHAT THIS STEP DOES AND AN IMPORTANT HONEST LIMITATION
-----------------------------------------------------------
For the CLASSICAL baselines, this is straightforward: SHAP (SHapley Additive exPlanations)
gives per-feature, per-prediction attributions with solid theoretical grounding (Shapley values
from cooperative game theory).

For the QUANTUM circuit, native SHAP does not apply cleanly - SHAP's TreeExplainer/LinearExplainer
assume a classical model structure, and KernelExplainer treats any model as a black box but
requires many hundreds of forward passes per explanation; each forward pass through a quantum
circuit is itself a state-vector simulation, so this is computationally expensive but NOT
impossible - we provide a KernelSHAP wrapper below for the quantum model too, just be aware it
is slower and should be run on a small explanation sample (e.g. 20-30 test points), not the
full test set. We ALSO provide a parameter-sensitivity analysis specific to the variational
circuit (perturb each trained weight parameter, measure output shift) as a faster, quantum-
native complement. Presenting both - and being upfront in your report that quantum XAI tooling
is less mature than classical XAI - reads as rigor to reviewers, not as a weakness to hide.

MODELS/ALGORITHMS USED IN THIS STEP
--------------------------------------
- shap.TreeExplainer   : exact, fast SHAP values for RandomForest / XGBoost.
- shap.LinearExplainer : exact SHAP values for LogisticRegression.
- shap.KernelExplainer : model-agnostic (slower) SHAP, used for SVM, the MLP, and the quantum
                          hybrid model, since none of them have a specialized fast explainer.
- Custom parameter-perturbation sensitivity analysis for the trained VQC ansatz weights.
"""

import logging
import numpy as np
import shap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def explain_classical_model(model_entry: dict, X_background: np.ndarray, X_explain: np.ndarray, feature_names: list):
    """
    Returns a shap.Explanation object. `X_background` (~50-100 samples) sets the reference
    distribution SHAP measures deviation from; `X_explain` are the specific samples you want
    per-feature attributions for (e.g. the test set, or one patient's record for the demo UI).
    """
    model_type = model_entry["type"]

    if model_type == "sklearn" and hasattr(model_entry["model"], "feature_importances_"):
        explainer = shap.TreeExplainer(model_entry["model"])
        shap_values = explainer.shap_values(X_explain)
    elif model_type == "sklearn" and hasattr(model_entry["model"], "coef_"):
        explainer = shap.LinearExplainer(model_entry["model"], X_background)
        shap_values = explainer.shap_values(X_explain)
    else:
        # Generic fallback: works for SVM, MLP, and (slowly) the quantum hybrid model
        def predict_fn(X):
            if model_type == "pytorch" or model_type == "hybrid_quantum":
                import torch
                model_entry["model"].eval()
                with torch.no_grad():
                    return model_entry["model"](torch.tensor(X, dtype=torch.float32)).numpy().ravel()
            return model_entry["model"].predict_proba(X)[:, 1]

        explainer = shap.KernelExplainer(predict_fn, shap.sample(X_background, min(50, len(X_background))))
        shap_values = explainer.shap_values(X_explain, nsamples=100)

    return shap_values


def explain_quantum_model_via_shap(hybrid_model, X_background: np.ndarray, X_explain: np.ndarray):
    """
    KernelSHAP applied to the full hybrid quantum model treated as a black box.
    WARNING: each SHAP sample requires a full forward pass through the quantum circuit
    simulation. Keep X_explain small (<=30 rows) and nsamples modest, or this will be slow.
    """
    import torch

    def predict_fn(X):
        hybrid_model.eval()
        with torch.no_grad():
            return hybrid_model(torch.tensor(X, dtype=torch.float32)).numpy().ravel()

    background_sample = shap.sample(X_background, min(30, len(X_background)))
    explainer = shap.KernelExplainer(predict_fn, background_sample)
    logger.info(f"Computing quantum-model SHAP values for {len(X_explain)} samples "
                f"(this involves {len(X_explain)} x ~2*n_features quantum circuit evaluations)...")
    shap_values = explainer.shap_values(X_explain, nsamples=64)
    return shap_values


def quantum_parameter_sensitivity(hybrid_model, X_sample: np.ndarray, perturbation: float = 0.1) -> dict:
    """
    Quantum-native explainability: for each trainable ansatz parameter, perturb it by
    `perturbation` radians and measure the resulting shift in the model's average output over
    X_sample. Parameters causing the largest output shift are the ones the circuit "relies on"
    most for its decision - a rough analogue to feature importance, but for the quantum weights
    themselves rather than the input features.
    """
    import torch

    hybrid_model.eval()
    X_tensor = torch.tensor(X_sample, dtype=torch.float32)
    with torch.no_grad():
        baseline_output = hybrid_model(X_tensor).mean().item()

    sensitivities = {}
    quantum_weights = hybrid_model.quantum_layer.weight.data
    for i in range(len(quantum_weights)):
        original_val = quantum_weights[i].item()
        quantum_weights[i] = original_val + perturbation
        with torch.no_grad():
            perturbed_output = hybrid_model(X_tensor).mean().item()
        sensitivities[f"ansatz_weight_{i}"] = abs(perturbed_output - baseline_output)
        quantum_weights[i] = original_val  # restore

    sorted_sensitivities = dict(sorted(sensitivities.items(), key=lambda kv: kv[1], reverse=True))
    logger.info(f"Top-5 most sensitive quantum parameters: {list(sorted_sensitivities.items())[:5]}")
    return sorted_sensitivities
