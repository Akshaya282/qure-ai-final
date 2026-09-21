"""
classical_baselines.py
========================
STEP 3 of the pipeline: Classical ML Baselines.

WHY THIS STEP EXISTS
----------------------
The problem statement explicitly requires benchmarking the hybrid quantum model "against
classical machine learning baselines" for accuracy, computational efficiency, and generalization.
Without this step, any quantum-vs-classical comparison in your final report is unsubstantiated.
These baselines are trained on the EXACT SAME feature subset (after feature_selection.py) as the
quantum model, so the comparison is fair - neither side gets an unfair feature-count advantage.

MODELS USED IN THIS STEP
--------------------------
- LogisticRegression   : simple linear baseline; establishes the "floor" performance.
- RandomForestClassifier: strong non-linear tabular baseline, also gives free feature importances.
- SVC (RBF kernel)     : the closest classical analogue to the quantum kernel method (QSVM),
                          since both project data into a higher-dimensional feature space
                          (RBF kernel trick vs. quantum Hilbert space) - this is the most
                          meaningful one-to-one comparison point for your report.
- XGBClassifier        : gradient-boosted trees, typically the strongest tabular baseline and a
                          realistic "state of the art classical" comparison point.
- Shallow MLP (PyTorch): a small feed-forward neural net matching the classical pre/post layers
                          used around the quantum layer in quantum_model.py, so you can isolate
                          "does the quantum layer itself add anything beyond an equivalent-sized
                          classical layer would."
"""

import time
import logging
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    logger.warning("xgboost not installed - skipping XGBoost baseline. `pip install xgboost` to enable.")


class ShallowMLP(nn.Module):
    """
    Classical counterpart to the hybrid quantum model's classical layers, with a comparable
    parameter budget. Used to isolate whether the quantum layer contributes beyond what an
    equivalently-sized classical layer already would.
    """
    def __init__(self, n_features: int, hidden_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp_baseline(X_train, y_train, X_val, y_val, epochs: int = 100, lr: float = 0.01) -> dict:
    """Trains the shallow MLP baseline and returns the fitted model plus timing info."""
    n_features = X_train.shape[1]
    model = ShallowMLP(n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    start = time.time()
    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(Xt)
        loss = loss_fn(preds, yt)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(Xv), yv).item()
            best_val_loss = min(best_val_loss, val_loss)
            logger.info(f"[MLP] epoch {epoch:3d} | train_loss={loss.item():.4f} | val_loss={val_loss:.4f}")
    train_time = time.time() - start
    return {"model": model, "train_time_sec": train_time, "type": "pytorch"}


def train_all_baselines(X_train, y_train, X_val, y_val) -> dict:
    """
    Trains every classical baseline and returns a dict of {name: {model, train_time_sec, type}}.
    `train.py` calls this once per disease and feeds the results straight into evaluate.py
    alongside the quantum model's results for a unified comparison table.
    """
    results = {}

    sklearn_models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
        "SVM_RBF": SVC(kernel="rbf", probability=True, random_state=42),
    }
    if _HAS_XGB:
        sklearn_models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss", random_state=42,
        )

    for name, model in sklearn_models.items():
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start
        results[name] = {"model": model, "train_time_sec": train_time, "type": "sklearn"}
        logger.info(f"[{name}] trained in {train_time:.3f}s")

    results["ShallowMLP"] = train_mlp_baseline(X_train, y_train, X_val, y_val)
    return results


def predict_baseline(entry: dict, X: np.ndarray) -> np.ndarray:
    """Unified predict-probability interface across sklearn and PyTorch baseline models."""
    if entry["type"] == "sklearn":
        return entry["model"].predict_proba(X)[:, 1]
    else:  # pytorch
        entry["model"].eval()
        with torch.no_grad():
            return entry["model"](torch.tensor(X, dtype=torch.float32)).numpy().ravel()
