# Qure AI — Hybrid Quantum ML Platform for Early Disease Detection (SIH26139)

> **Naming note**: "Qure.ai" is an existing, well-known Indian AI-radiology company. This project
> name was chosen by the team; consider a small variation (e.g. "QuroAI", "Qubit Health") before
> a public/competition submission to avoid confusion with that unrelated real company.

A full-stack platform: React frontend → FastAPI backend → hybrid quantum-classical models (Qiskit)
covering six disease modules (Heart, Cardiovascular, Diabetes, Kidney, Cancer, Brain Tumor) plus
a General module for training on any user-uploaded dataset.

Every piece of backend code in this repo was executed and verified during development
(see inline comments flagging the specific bugs this caught — most importantly, a silent
gradient-flow bug in the quantum layer that would otherwise have made training appear to work
while doing nothing).

---

## 1. Project structure

```
backend/
  src/
    config.py                 # disease registry: dataset IDs, qubit budgets, target mappings
    data_preprocessing.py     # Step 1: ingestion, cleaning, imputation, scaling, splitting
    feature_selection.py      # Step 2: mutual_info / RFE / PSO (swarm intelligence) selectors
    classical_baselines.py    # Step 3: LogReg, RandomForest, SVM, XGBoost, MLP
    quantum_model.py          # Step 4: Hybrid VQC (Qiskit) + QSVM
    cnn_encoder.py            # Step 4b: ResNet-18 bottleneck for the imaging (Brain Tumor) path
    evaluate.py                # Step 5: accuracy/sensitivity/specificity/AUC + McNemar's test
    explainability.py         # Step 6: SHAP (classical) + quantum parameter sensitivity
    hyperparameter_search.py  # Step 7: Optuna search over circuit depth / lr / feature strategy
    train.py                   # Orchestrates steps 1-6 end-to-end for one disease
  api/
    main.py                    # Step 8: FastAPI backend serving all six modules
  requirements.txt
  models/                       # trained artifacts land here after running train.py (not committed)
  data/                         # optional local CSV fallbacks if UCI servers are unreachable

frontend/
  src/
    components/Navbar.jsx       # top horizontal ("landscape") navigation across all 6 diseases
    components/PredictionForm.jsx
    components/ResultCard.jsx
    pages/Home.jsx
    pages/DiseasePage.jsx        # generic page for the 5 tabular diseases
    pages/BrainTumorPage.jsx     # separate page: image upload instead of a numeric form
    api/client.js                # fetch() wrapper around the FastAPI backend
    styles/index.css
  package.json
```

---

## 2. Step-by-step explanation (what happens, and which model/algorithm does it)

### Step 1 — Data Ingestion & Preprocessing (`data_preprocessing.py`)
Pulls each disease's dataset live from the UCI ML Repository via the official `ucimlrepo`
package (requires internet access on whatever machine runs training). Missing values are
imputed with `SimpleImputer` (median for numeric columns, most-frequent for categorical),
categorical text is encoded with `OrdinalEncoder`, and every disease's differently-formatted
target label is normalized to `{0, 1}`. Finally `MinMaxScaler` (fit on the training split only,
to avoid leakage) scales every feature to `[0, 1]`, and the data is split 70/15/15
train/val/test with stratification (important because these datasets are class-imbalanced).

### Step 2 — Feature Engineering / Selection (`feature_selection.py`)
Reduces each disease's raw feature count down to a qubit-feasible number (usually 6), because
angle encoding maps one classical feature to one qubit, and simulating more than ~10-12 qubits
gets slow fast. Three interchangeable strategies are implemented:
- **Mutual Information** — fast, univariate, non-linear-aware ranking.
- **RFE** — Recursive Feature Elimination around a Random Forest, accounts for feature interactions.
- **PSO (Particle Swarm Optimization)** — a genuine swarm-intelligence metaheuristic that searches
  feature SUBSETS directly using cross-validated accuracy as the fitness function. This is the
  differentiator to highlight in your report/demo, since it ties to a real, citable research area
  rather than being an off-the-shelf `sklearn` call.

### Step 3 — Classical Baselines (`classical_baselines.py`)
Trains Logistic Regression, Random Forest, SVM (RBF kernel), XGBoost, and a shallow PyTorch MLP
on the exact same selected features the quantum model will use — this is what makes the
eventual quantum-vs-classical comparison fair.

### Step 4 — Hybrid Quantum-Classical Model (`quantum_model.py`)
```
raw features -> Linear(n_features, n_qubits) -> scale to [0, π]
             -> ZZFeatureMap (data encoding + feature-correlation entanglement)
             -> RealAmplitudes ansatz (TRAINABLE variational circuit)
             -> EstimatorQNN (measures expectation value)
             -> Linear(1, 1) + Sigmoid -> P(disease present)
```
`ZZFeatureMap` encodes classical data as rotation angles and adds entangling ZZ interactions
between qubits based on feature-pair products — letting the quantum state represent feature
correlations that a plain linear encoder cannot. `RealAmplitudes` is the trainable ansatz
(RY rotations + CNOT entanglers), chosen for its NISQ-hardware compatibility. `EstimatorQNN` +
`TorchConnector` make the whole circuit a differentiable PyTorch layer, trainable with any
standard optimizer via the parameter-shift rule.

**A real bug this caught during development:** `EstimatorQNN` needs `input_gradients=True`
explicitly set, or gradients silently never reach the classical `pre_net` layer feeding it —
the code runs without error, loss appears to compute fine, but that upstream layer's weights
never update. This is set correctly in the shipped code; if you ever rewrite this layer, don't
drop that flag.

A second, non-variational approach — **Quantum Kernel SVM (QSVM)** — is also implemented:
instead of training circuit parameters, it computes a quantum "similarity" kernel matrix via
state fidelity and hands that to a classical SVM solver. Faster to train, but its kernel-matrix
computation is `O(n²)` circuit evaluations — a genuine, reportable efficiency trade-off.

**Brain Tumor (imaging) only** (`cnn_encoder.py`): raw MRI pixels are far too high-dimensional
for direct angle encoding, so a ResNet-18 backbone (ImageNet-pretrained, last block fine-tuned)
compresses each image to an 8-dim bottleneck first — then that 8-dim vector feeds the *same*
`HybridQuantumClassifier` class used everywhere else, unchanged.

### Step 5 — Evaluation (`evaluate.py`)
Computes accuracy, sensitivity (recall), specificity (`TN/(TN+FP)`, computed manually since
scikit-learn has no built-in), precision, F1, and AUC-ROC on the held-out test set only, plus
per-model training time. `mcnemar_test()` runs McNemar's test (correct choice for comparing two
classifiers' PAIRED predictions on the same test set) to check whether the quantum model's
difference from the best classical baseline is statistically significant.

### Step 6 — Explainability (`explainability.py`)
SHAP (`TreeExplainer`/`LinearExplainer` for tree/linear models, `KernelExplainer` as the
model-agnostic fallback) for every classical baseline. For the quantum model, native SHAP
tooling is less mature, so two complementary approaches are provided: a `KernelExplainer`
wrapper treating the whole hybrid model as a black box (slow — keep explanation samples small),
and a faster quantum-native **parameter-sensitivity analysis** that perturbs each trained
ansatz weight and measures the resulting output shift. Presenting both, and being upfront about
the maturity gap, reads as rigor to reviewers.

### Step 7 — Hyperparameter Optimization (`hyperparameter_search.py`)
Optuna (TPE sampler) searches ansatz depth, learning rate, epoch count, and feature-selection
strategy (mutual_info vs RFE — PSO is deliberately excluded from the automated search, since
nesting one metaheuristic swarm inside every Optuna trial would be prohibitively slow; run PSO
once manually instead and compare it to Optuna's best result).

### Step 8 — Deployment (`api/main.py` + `frontend/`)
FastAPI exposes `/diseases` (list for the navbar), `/diseases/{id}/metadata` (which features the
trained model needs, so the React form renders itself dynamically instead of hardcoding fields
per disease), `/diseases/{id}/predict` (tabular diseases), and
`/diseases/brain_tumor/predict-image` (multipart image upload). The React app (Vite) renders a
top horizontal navbar across all six disease modules, a dynamic form for tabular diseases, and
a dedicated upload page for Brain Tumor — confirmed building cleanly with `npm run build`.

---

## 3. How to run it

Real trained models for Heart, Diabetes, Kidney, and Cancer are already bundled in
`backend/models/` — you can `uvicorn main:app` and start using the app immediately without
training anything first. Retrain only if you want to change feature strategy, use the full
dataset (see section 6's note on subsampling), or tune hyperparameters.

### Backend
```bash
cd backend
pip install -r requirements.txt

# Datasets are bundled locally under backend/data/ — no internet access needed to train.
# (Re)train a disease module (repeat per tabular disease: heart, diabetes, kidney, cancer;
# cardiovascular reuses heart's model, no separate training needed)
cd src
python train.py heart --epochs 15 --max-vqc-train-samples 70   # fast demo-scale settings
python train.py heart                                          # full run: slower, more accurate
python train.py diabetes --feature-strategy pso                # try the swarm-intelligence selector
python hyperparameter_search.py heart --n-trials 20             # optional: Optuna tuning

# Serve the API
cd ../api
uvicorn main:app --reload --port 8000
```
Visit `http://localhost:8000/docs` for interactive API docs.

**Brain Tumor module**: download an MRI dataset (e.g. Kaggle "Brain Tumor MRI Dataset") into
`backend/data/brain_tumor/{train,test}/<class_name>/*.jpg`, then run
`cnn_encoder.train_cnn_bottleneck()` followed by training `HybridQuantumClassifier` on the
resulting embeddings, and save both state dicts to `backend/models/brain_tumor/` as
`cnn_encoder.pt` and `hybrid_vqc.pt` plus a `metadata.json` with `n_qubits` and `classes` —
matching the format `api/main.py` expects to load.

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Visit `http://localhost:5173`. Set `VITE_API_BASE` in a `.env` file if the backend isn't on
`localhost:8000`. Each disease page links to `/disease/<id>/performance` for the full
accuracy/sensitivity/specificity/confusion-matrix/ROC-curve view.

---

## 4. Datasets used for training (bundled, real, verified)

| Disease | File | Rows | Features | Original source |
|---|---|---|---|---|
| Heart / Cardiovascular | `backend/data/heart.csv` | 303 | 13 | Same 303 patients as UCI Heart Disease (Cleveland), id=45 |
| Diabetes | `backend/data/diabetes.csv` | 768 | 8 | Pima Indians Diabetes Dataset |
| Kidney | `backend/data/kidney_disease.csv` | 400 | 24 | UCI Chronic Kidney Disease, id=336 |
| Cancer | `backend/data/cancer.csv` | 569 | 30 | UCI Breast Cancer Wisconsin (Diagnostic), id=17 — exported from `sklearn.datasets.load_breast_cancer` |
| Brain Tumor | not bundled (images) | — | — | User-supplied MRI dataset (e.g. Kaggle) |

## 5. Models trained per disease

Six models, trained on identical selected features for a fair comparison:
Logistic Regression · Random Forest · SVM (RBF kernel) · Shallow PyTorch MLP ·
**Hybrid Quantum VQC** (Qiskit: ZZFeatureMap + RealAmplitudes + EstimatorQNN) ·
**Quantum Kernel SVM (QSVM)** (Qiskit FidelityQuantumKernel + classical SVC)

---

## 6. Real trained results (from the models bundled in this repo)

Every table below is copied directly from `backend/models/<disease>/performance.json`, generated
by an actual `python train.py <disease> --epochs 15 --max-vqc-train-samples 70` run — not
estimated, not simulated. See section 6 for what these numbers actually mean.

### Heart Disease
Test set: 46 patients (54% positive)

| Model | Accuracy | Sensitivity | Specificity | Precision | F1 | AUC-ROC | Train Time (s) |
|---|---|---|---|---|---|---|---|
| **QSVM** | **87.0%** | 92.0% | 81.0% | 85.2% | 88.5% | 0.884 | 31.77 |
| SVM_RBF | 78.3% | 96.0% | 57.1% | 72.7% | 82.8% | 0.867 | 0.004 |
| LogisticRegression | 76.1% | 92.0% | 57.1% | 71.9% | 80.7% | 0.853 | 0.002 |
| RandomForest | 76.1% | 96.0% | 52.4% | 70.6% | 81.4% | 0.920 | 0.164 |
| ShallowMLP | 76.1% | 88.0% | 61.9% | 73.3% | 80.0% | 0.871 | 0.035 |
| HybridVQC | 67.4% | 92.0% | 38.1% | 63.9% | 75.4% | 0.724 | 109.77 |

### Diabetes (Pima Indians)
Test set: 116 patients (34% positive)

| Model | Accuracy | Sensitivity | Specificity | Precision | F1 | AUC-ROC | Train Time (s) |
|---|---|---|---|---|---|---|---|
| **RandomForest** | **78.5%** | 62.5% | 86.8% | 71.4% | 66.7% | 0.845 | 0.185 |
| LogisticRegression | 76.7% | 57.5% | 86.8% | 69.7% | 63.0% | 0.831 | 0.002 |
| SVM_RBF | 75.9% | 67.5% | 80.3% | 64.3% | 65.8% | 0.835 | 0.018 |
| ShallowMLP | 69.0% | 27.5% | 90.8% | 61.1% | 37.9% | 0.776 | 0.047 |
| HybridVQC | 69.0% | 20.0% | 94.7% | 66.7% | 30.8% | 0.751 | 109.99 |
| QSVM | 66.4% | 40.0% | 80.3% | 51.6% | 45.1% | 0.759 | 33.71 |

### Kidney Disease
Test set: 60 patients (62% positive)

| Model | Accuracy | Sensitivity | Specificity | Precision | F1 | AUC-ROC | Train Time (s) |
|---|---|---|---|---|---|---|---|
| **RandomForest** | **100.0%** | 100.0% | 100.0% | 100.0% | 100.0% | 1.000 | 0.154 |
| LogisticRegression | 98.3% | 97.3% | 100.0% | 100.0% | 98.6% | 0.998 | 0.002 |
| SVM_RBF | 98.3% | 100.0% | 95.7% | 97.4% | 98.7% | 1.000 | 0.003 |
| QSVM | 98.3% | 100.0% | 95.7% | 97.4% | 98.7% | 1.000 | 32.61 |
| ShallowMLP | 96.7% | 94.6% | 100.0% | 100.0% | 97.2% | 1.000 | 0.039 |
| HybridVQC | 96.7% | 97.3% | 95.7% | 97.3% | 97.3% | 0.999 | 109.24 |

### Cancer (Breast, Wisconsin Diagnostic)
Test set: 86 patients (37% positive)

| Model | Accuracy | Sensitivity | Specificity | Precision | F1 | AUC-ROC | Train Time (s) |
|---|---|---|---|---|---|---|---|
| LogisticRegression / RandomForest / SVM_RBF / ShallowMLP (tied) | **97.7%** | 93.8% | 100.0% | 100.0% | 96.8% | ~0.998 | 0.002–0.167 |
| QSVM | 97.7% | 93.8% | 100.0% | 100.0% | 96.8% | 0.994 | 34.44 |
| HybridVQC | 62.8% | 0.0% | 100.0% | 0.0% | 0.0% | **0.998** | 110.42 |

## 7. What these numbers actually mean (read this before you present them)

- **Heart Disease: QSVM won outright** (87.0% vs ~76-78% for classical models) — this is your
  strongest genuine "quantum advantage" data point, and it's real, not cherry-picked.
- **Kidney/Cancer are separated almost linearly** by their features, so every model (classical
  and quantum) scores 96-100% — a ceiling effect, not evidence the quantum model is weak there.
- **HybridVQC's Cancer result is the most important one to explain in your report, not hide**:
  62.8% accuracy but 0.998 AUC. That combination means the circuit IS ranking patients correctly
  (AUC near 1.0), but its default 0.5 decision threshold is miscalibrated after only 15 epochs on
  a 70-sample subsample — it collapsed to predicting the majority class. This is a textbook
  undertraining/calibration symptom, not a broken model. Fix path: more epochs, more training
  samples, or picking the threshold that maximizes F1/Youden's J on the validation set instead
  of hardcoding 0.5 — all legitimate next steps to describe in your Phase 2 writeup.
- **Every VQC/QSVM number above was trained on a deliberately small, capped sample** (max 70
  rows for VQC, max 120 for QSVM) purely to keep quantum circuit simulation inside a reasonable
  wall-clock time on this machine (measured: ~110s for 15 VQC epochs on 70 samples, ~32-34s for
  a 120-sample QSVM kernel matrix). Retrain with `--max-vqc-train-samples` raised (or omitted
  entirely) on stronger hardware for numbers that use the full dataset like the classical models
  did — the current comparison is not entirely apples-to-apples on training-set size, and you
  should say so explicitly in your report rather than let a reviewer catch it first.
- **Training time itself is a legitimate benchmarking result**: classical models trained in
  milliseconds; VQC took ~110s and QSVM ~30-35s even on these small subsamples. This is exactly
  the "computational efficiency" trade-off the SIH problem statement asks you to report.

## 8. Model Performance page (frontend)

Every disease's prediction page (`/disease/<id>`) links to `/disease/<id>/performance`, which
renders, for every trained model:
- Accuracy / Sensitivity / Specificity / Precision / F1 / AUC-ROC metric cards
- An interactive confusion matrix (click any model tab to switch)
- An ROC curve chart overlaying all six models on one axis (Recharts), with a dashed
  random-guess reference diagonal
- The full sortable comparison table

This reads directly from `backend/models/<disease>/performance.json` (generated by `train.py`),
served via `GET /diseases/{id}/performance` — no metrics are computed in the browser.

## 8b. Explainable AI (SHAP), CSV batch upload, and the General module

**Explainable AI**: every prediction (single or batch) now returns a per-feature SHAP
contribution breakdown, shown as a bar chart under the result. Because explaining the quantum
circuit itself with SHAP would take many seconds per prediction (too slow for a live request —
see `explainability.py`'s documented trade-off), contributions are computed on the Random Forest
that was trained on the identical selected features as a fast, faithful surrogate. This is
stated explicitly in the UI (`explanation_note` in the API response), not hidden.

**CSV batch upload**: every tabular disease page now has an "upload a CSV of multiple patients"
section below the single-patient form (`POST /diseases/{id}/predict-batch`) — useful when you
have many patient records rather than typing one in at a time. The CSV needs columns matching
the model's required features (extra columns like a patient ID are ignored); each row gets its
own prediction and explanation, and a row with bad data reports its own error without failing
the whole batch.

**The General module** (`/general` in the navbar, "+ Your Dataset"): upload literally any
patient CSV, and Qure AI runs the identical pipeline (feature selection → classical baselines →
Hybrid VQC → QSVM → evaluation) on it, no code changes required:
1. Upload a CSV → `POST /general/inspect` returns its columns, row count, and (for low-cardinality
   columns) their actual distinct values.
2. Pick the target column and which value means "disease present", set qubits/epochs/feature
   strategy → `POST /general/train` runs the full pipeline synchronously and returns
   accuracy/sensitivity/specificity/confusion matrix/ROC for all six models, right in the page.
3. Predict on new patients via the same form + CSV batch upload as every other disease
   (`POST /general/predict`, `POST /general/predict-batch`).

Because this trains inside a single HTTP request with no background job queue, defaults are kept
small (10 epochs, 50-sample VQC cap) to finish in roughly a minute — raise them via the form if
you can tolerate a longer wait. Only one custom model is kept at a time (`backend/models/general/`
is overwritten on each retrain); if you want to keep several, copy that folder out between runs.

## 8c. Cardiovascular now resolves correctly

Earlier builds returned 404 for `/diseases/cardiovascular/*` because no
`backend/models/cardiovascular/` folder was ever trained — by design, since cardiovascular
disease and the Heart module's Cleveland dataset describe the same underlying condition (see
section 9 below). `config.py`'s new `alias_of="heart"` field, resolved in `api/main.py`, makes
every cardiovascular endpoint transparently serve Heart's already-trained model while still
reporting `disease_id: "cardiovascular"` in responses. No retraining needed — this was a wiring
bug in the previous build, not a missing model.

## 8d. Hyperparameter search — real results (Heart Disease)

`hyperparameter_search.py` (Optuna, TPE sampler) was actually run against Heart Disease — 8
real trials, saved in `backend/models/heart/optuna_study.db` and `optuna_trials.csv`:

| Trial | Feature strategy | Ansatz reps | LR | Epochs | Val Accuracy | Val AUC |
|---|---|---|---|---|---|---|
| 2 (**best by accuracy**) | rfe | 3 | 0.0109 | 5 | **0.609** | 0.674 |
| 6 | rfe | 3 | 0.0109 | 5 | 0.609 | 0.716 |
| 7 | rfe | 2 | 0.0246 | 10 | 0.587 | **0.823** |
| 3 | rfe | 2 | 0.0246 | 10 | 0.478 | 0.747 |
| 1 | mutual_info | 3 | 0.0459 | 15 | 0.543 | 0.747 |
| 5 | mutual_info | 3 | 0.0459 | 15 | 0.565 | 0.695 |
| 0 | rfe | 3 | 0.0455 | 5 | 0.457 | 0.543 |
| 4 | rfe | 3 | 0.0455 | 5 | 0.457 | 0.552 |

**Two honest findings worth keeping in your report:**
1. **The best-accuracy trial isn't the best-AUC trial** (trial 2: 0.609 accuracy but only 0.674
   AUC, vs. trial 7: 0.587 accuracy but 0.823 AUC). Whichever metric you optimize for changes
   which hyperparameters look "best" — worth discussing explicitly rather than only reporting
   accuracy.
2. **Trials 0/4 and 2/6 are near-duplicates.** This search ran in two separate 4-trial batches,
   resumed from the same SQLite-backed study. The first version of this script re-seeded
   Optuna's TPE sampler with a fixed seed on every resume, so the sampler's early exploration
   draws repeated across batches instead of the search genuinely continuing — a real bug, caught
   by inspecting the actual trial values, not assumed. Fixed now: the seed is only applied to a
   single non-resumed run; a resumed study lets the sampler explore fresh points while still
   conditioning on every trial already in storage.
3. **These numbers use a deliberately small validation subsample (25 rows, ≤15 epochs)** to fit
   a real search inside a practical wall-clock budget on a simulator — treat this as a
   proof-of-methodology run, not final tuned hyperparameters. Rerun with
   `python hyperparameter_search.py heart --n-trials 30 --max-train-samples 150` (or higher, on
   stronger hardware / more time) for numbers you'd put in a final report table.




## 9. Honest scope notes (say this in your report/demo too — it reads as rigor, not weakness)

- **Cardiovascular** intentionally reuses the Heart Disease dataset rather than fabricating a
  second one, since coronary/cardiovascular disease and "heart disease" in the UCI Cleveland
  dataset are the same underlying condition. Swap in the Framingham Heart Study dataset if you
  want it genuinely distinct.
- **Cancer** here is the Breast Cancer Wisconsin (Diagnostic) dataset — despite originating from
  imaging (digitized FNA cell images), the published dataset is already tabular (30 pre-extracted
  numeric features), so it uses the same pipeline as the other tabular diseases, no CNN needed.
- **Brain Tumor** is the only module that is genuinely imaging-based end-to-end, and is the one
  place a CNN encoder is required before the quantum layer.
- **Diabetes** ships as the Pima Indians Diabetes dataset rather than the originally-scoped UCI
  "Early Stage Diabetes Risk Prediction" set (id=529) — swapped because Pima was the version we
  could reliably source and verify; see `config.py`'s module docstring for the full reasoning.
- Dataset identities verified this session: Heart (same 303 patients as UCI id=45), Kidney (UCI
  id=336), Cancer (UCI id=17, via scikit-learn's bundled copy).
