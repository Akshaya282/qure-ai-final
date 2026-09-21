"""
hyperparameter_search.py
===========================
STEP 7 (rubric: "Hyperparameter Optimization"): Optuna search over the hybrid quantum model's
tunable knobs.

WHAT GETS TUNED AND WHY EACH ONE MATTERS
---------------------------------------------
- ansatz_reps (1-3)   : more repetitions of the variational layer = more expressive circuit, but
                         also more trainable parameters and a higher risk of "barren plateaus"
                         (vanishing gradients in deep quantum circuits) - a real, disease-specific
                         trade-off worth searching rather than guessing.
- learning_rate        : quantum loss landscapes are typically flatter/noisier than classical
                          ones (measurement outputs are bounded in [-1,1] and gradients computed
                          via parameter-shift can be noisier), so the optimal LR is often
                          different from what you'd use for an equivalent classical MLP.
- feature_strategy     : which classical feature-selection method (mutual_info / rfe / pso) feeds
                          the quantum layer - this can matter as much as the circuit design itself.
- vqc_epochs           : traded off against wall-clock time; Optuna will naturally push toward
                          "just enough" epochs to reach good validation loss.

Optuna's TPE (Tree-structured Parzen Estimator) sampler is used by default, which is well-suited
here because circuit training is expensive (each trial requires simulating the quantum circuit
`epochs` times) - TPE needs far fewer trials than grid or random search to converge on good
regions of the search space.

USAGE
------
    python hyperparameter_search.py heart --n-trials 20
"""

import argparse
import logging
import optuna

from config import get_disease
from data_preprocessing import prepare_dataset
from feature_selection import select_features
from quantum_model import train_hybrid_quantum_model, predict_hybrid
from evaluate import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def make_objective(disease_id: str, max_train_samples: int = 30):
    cfg = get_disease(disease_id)
    ds = prepare_dataset(disease_id)

    def objective(trial: optuna.Trial) -> float:
        feature_strategy = trial.suggest_categorical("feature_strategy", ["mutual_info", "rfe"])
        # NOTE: 'pso' is deliberately excluded from the automated Optuna search - running a full
        # PSO feature-selection swarm inside every Optuna trial would be prohibitively slow
        # (nested metaheuristic search inside metaheuristic search). Run PSO once manually via
        # train.py --feature-strategy pso and compare its result to the best Optuna trial instead.
        ansatz_reps = trial.suggest_int("ansatz_reps", 1, 3)
        lr = trial.suggest_float("lr", 0.005, 0.2, log=True)
        epochs = trial.suggest_int("epochs", 5, 20, step=5)

        selected_idx = select_features(feature_strategy, ds.X_train, ds.y_train, cfg.n_qubits_target, ds.feature_names)
        X_train = ds.X_train[:, selected_idx]
        X_val = ds.X_val[:, selected_idx]
        y_train = ds.y_train

        # Same reasoning as train.py's max_vqc_train_samples: quantum circuit simulation cost is
        # dominated by per-sample, per-epoch state-vector evaluation, so running a full Optuna
        # search (many trials, each training from scratch) on the full split would make even a
        # handful of trials take longer than a full train.py run. Subsampling here trades some
        # per-trial accuracy precision for being able to search the space at all on a simulator.
        if max_train_samples and len(y_train) > max_train_samples:
            from sklearn.model_selection import train_test_split as _tts
            X_train, _, y_train, _ = _tts(
                X_train, y_train, train_size=max_train_samples, stratify=y_train, random_state=42,
            )

        result = train_hybrid_quantum_model(
            X_train, y_train, X_val, ds.y_val,
            n_qubits=len(selected_idx), epochs=epochs, lr=lr, ansatz_reps=ansatz_reps,
        )
        val_preds = predict_hybrid(result, X_val)
        val_metrics = compute_metrics(ds.y_val, val_preds)

        # Optimize for accuracy, but report AUC too so you can inspect trade-offs in the dashboard
        trial.set_user_attr("val_auc", val_metrics["auc_roc"])
        trial.set_user_attr("train_time_sec", result["train_time_sec"])
        return val_metrics["accuracy"]

    return objective


def run_search(disease_id: str, n_trials: int = 20, max_train_samples: int = 30, storage: str = None):
    """
    `storage` (e.g. "sqlite:///optuna_heart.db") makes the study resumable across separate
    invocations - useful since quantum-circuit trials are slow enough that you may want to run
    a search in several shorter batches rather than one long-running process.
    """
    study = optuna.create_study(
        study_name=f"{disease_id}_vqc", storage=storage, load_if_exists=bool(storage),
        direction="maximize",
        # A fixed seed only makes sense for a single, non-resumed run (full reproducibility).
        # When resuming an existing study from `storage`, a fresh process re-creates the
        # TPESampler from scratch each time - with a FIXED seed, that means the sampler's early
        # exploration draws are identical across resumed sessions, so you can end up re-testing
        # near-duplicate parameter combinations instead of genuinely continuing the search
        # (observed during development: two 4-trial batches on the same study produced two
        # near-identical pairs of early trials). Omitting the seed on resume lets each session
        # explore fresh points while TPE still conditions its later draws on ALL trials already
        # recorded in storage, which is the part that actually matters for search quality.
        sampler=optuna.samplers.TPESampler(seed=42 if not storage else None),
    )
    study.optimize(make_objective(disease_id, max_train_samples), n_trials=n_trials)

    logger.info(f"Best trial for '{disease_id}': accuracy={study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("disease_id", choices=["heart", "cardiovascular", "diabetes", "kidney", "cancer"])
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--max-train-samples", type=int, default=30,
                         help="Subsample the VQC's training data per trial to keep search "
                              "wall-clock time tractable on a simulator.")
    parser.add_argument("--storage", type=str, default=None,
                         help='Optuna storage URL for a resumable study, e.g. "sqlite:///optuna_heart.db"')
    args = parser.parse_args()
    run_search(args.disease_id, n_trials=args.n_trials,
               max_train_samples=args.max_train_samples, storage=args.storage)
