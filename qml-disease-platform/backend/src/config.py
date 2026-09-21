"""
config.py
=========
Central registry for every disease module on the platform.

WHY THIS FILE EXISTS
---------------------
Every disease (Heart, Cardiovascular, Diabetes, Kidney, Cancer, Brain Tumor) reuses the exact
same hybrid quantum-classical pipeline (data_preprocessing -> feature_selection ->
classical_baselines -> quantum_model -> train -> evaluate -> explainability). The only things
that change per disease are: which dataset to load, which columns are features/target, and how
many qubits the reduced feature set maps to. Keeping all of that in one registry means adding a
7th disease later is a config entry, not a rewrite.

DATA SOURCING: BUNDLED REAL CSVS, NOT A LIVE NETWORK FETCH
---------------------------------------------------------------
Earlier versions of this pipeline pulled straight from the UCI ML Repository via `ucimlrepo` at
training time. That's fine on a machine with normal internet access, but it made the whole
project fragile in restricted/offline environments, and version drift on the UCI servers
becomes a reproducibility risk for anyone re-running your experiments later. Instead, every
tabular disease's real dataset is bundled directly under `backend/data/` as a CSV, verified
against its original source:

    - heart.csv      : 303 rows, 14 columns - the widely-used processed Cleveland Heart Disease
                        dataset (same 303 patients as UCI id=45; this copy has the target
                        pre-binarized to 0/1 rather than UCI's raw 0-4 severity score).
    - diabetes.csv    : 768 rows, 9 columns - the Pima Indians Diabetes dataset (National
                        Institute of Diabetes and Digestive and Kidney Diseases). NOTE: this
                        REPLACES the originally-planned UCI "Early Stage Diabetes Risk
                        Prediction" dataset (id=529, symptom questionnaire, 520 rows) because
                        Pima is the dataset we could reliably source and verify - it is at least
                        as widely cited in diabetes-ML literature. If your SIH submission
                        specifically needs the symptom-questionnaire version instead, swap the
                        CSV and update `target_column`/`n_qubits_target` below accordingly.
    - kidney_disease.csv : 400 rows, 25 columns - UCI Chronic Kidney Disease (id=336), sourced
                        from a GitHub mirror of the original UCI release.
    - cancer.csv      : 569 rows, 31 columns - UCI Breast Cancer Wisconsin (Diagnostic) (id=17),
                        exported directly from scikit-learn's bundled copy
                        (`sklearn.datasets.load_breast_cancer`), which ships the identical data.

QUBIT BUDGET RULE OF THUMB
---------------------------
Each classical feature -> 1 qubit under angle encoding (ZZFeatureMap). Qiskit's local simulator
comfortably handles 4-10 qubits for a hackathon timeline; beyond ~12 qubits, simulation time
explodes. That is why every tabular disease below selects down to n_qubits_target features
(not all raw features) in feature_selection.py before the quantum layer ever sees the data.
"""

import os
from dataclasses import dataclass
from typing import Optional

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


@dataclass
class DiseaseConfig:
    disease_id: str
    display_name: str
    modality: str                      # "tabular" or "imaging"
    source: str                         # human-readable citation of the dataset's origin
    csv_path: Optional[str] = None      # bundled real dataset, relative to backend/data/
    target_column: Optional[str] = None
    n_qubits_target: int = 6
    positive_label_note: str = ""
    notes: str = ""
    alias_of: Optional[str] = None      # if set, this disease serves another disease's trained
                                          # model artifacts instead of training its own (see
                                          # "cardiovascular" below) - api/main.py resolves this


DISEASE_REGISTRY: dict[str, DiseaseConfig] = {

    "heart": DiseaseConfig(
        disease_id="heart",
        display_name="Heart Disease",
        modality="tabular",
        source="Processed Cleveland Heart Disease dataset (same 303 patients as UCI id=45)",
        csv_path=os.path.join(DATA_DIR, "heart.csv"),
        target_column="target",
        n_qubits_target=6,
        positive_label_note="target is already binary: 0 = absent, 1 = present",
        notes="13 raw clinical features (age, sex, cp, trestbps, chol, fbs, restecg, thalach, "
              "exang, oldpeak, slope, ca, thal). Reduced to 6 via feature selection.",
    ),

    "cardiovascular": DiseaseConfig(
        disease_id="cardiovascular",
        display_name="Cardiovascular Risk",
        modality="tabular",
        source="Same clinical dataset as Heart Disease",
        csv_path=os.path.join(DATA_DIR, "heart.csv"),
        target_column="target",
        n_qubits_target=6,
        alias_of="heart",
        positive_label_note="Deliberately aliased to the Heart module: coronary/cardiovascular "
                             "disease and 'heart disease' in this dataset are the same "
                             "underlying condition. We do NOT fabricate a second dataset just to "
                             "fill a menu slot - that would misrepresent the data to reviewers. "
                             "`alias_of='heart'` means this module serves Heart's already-trained "
                             "model directly rather than needing (or having) its own "
                             "backend/models/cardiovascular/ folder - there is nothing separate "
                             "to train here, by design.",
        notes="Intentionally shares heart's model and trained artifacts. If you want a "
              "genuinely distinct cardiovascular dataset, swap in the Framingham Heart Study "
              "CSV here, remove alias_of, and train it separately with its own model folder.",
    ),

    "diabetes": DiseaseConfig(
        disease_id="diabetes",
        display_name="Diabetes",
        modality="tabular",
        source="Pima Indians Diabetes Dataset (National Institute of Diabetes and Digestive "
               "and Kidney Diseases)",
        csv_path=os.path.join(DATA_DIR, "diabetes.csv"),
        target_column="Outcome",
        n_qubits_target=6,
        positive_label_note="Outcome is already binary: 0 = no diabetes, 1 = diabetes",
        notes="8 numeric features (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, "
              "BMI, DiabetesPedigreeFunction, Age), 768 patients. Reduced to 6 via feature "
              "selection. See module docstring above re: swapping to the symptom-questionnaire "
              "UCI dataset if your submission requires it specifically.",
    ),

    "kidney": DiseaseConfig(
        disease_id="kidney",
        display_name="Kidney Disease",
        modality="tabular",
        source="UCI Chronic Kidney Disease (id=336)",
        csv_path=os.path.join(DATA_DIR, "kidney_disease.csv"),
        target_column="classification",
        n_qubits_target=6,
        positive_label_note="raw 'classification' is 'ckd'/'notckd' (sometimes with trailing "
                             "whitespace in the source file - stripped during cleaning) -> "
                             "mapped to 1/0",
        notes="24 mixed numeric/categorical features (serum creatinine, hemoglobin, blood "
              "pressure, etc.), 400 patients, ~38% missing values -> needs imputation.",
    ),

    "cancer": DiseaseConfig(
        disease_id="cancer",
        display_name="Cancer (Breast)",
        modality="tabular",
        source="UCI Breast Cancer Wisconsin (Diagnostic) (id=17), via sklearn.datasets.load_breast_cancer",
        csv_path=os.path.join(DATA_DIR, "cancer.csv"),
        target_column="diagnosis",
        n_qubits_target=6,
        positive_label_note="raw diagnosis is 'M'(alignant)/'B'(enign) -> mapped to 1/0",
        notes="30 numeric features computed from digitized FNA cell-nuclei images (radius, "
              "texture, perimeter, area, smoothness...), 569 patients. NOTE: despite "
              "originating from imaging, the published dataset is already tabular (features "
              "were pre-extracted) - no CNN needed here, unlike Brain Tumor.",
    ),

    "brain_tumor": DiseaseConfig(
        disease_id="brain_tumor",
        display_name="Brain Tumor",
        modality="imaging",
        source="MRI brain tumor classification dataset (e.g. Kaggle 'Brain Tumor MRI Dataset') "
               "- not bundled here (images, not a CSV); user must supply the image folder",
        n_qubits_target=8,
        positive_label_note="Multi-class (glioma/meningioma/pituitary/no-tumor) collapsed to "
                             "binary tumor-present/absent for this pipeline, or kept multi-class "
                             "with a softmax head - see cnn_encoder.py",
        notes="THIS is the one module that genuinely needs imaging: raw MRI slices are far too "
              "high-dimensional for direct quantum angle encoding. Pipeline: CNN encoder "
              "(ResNet-18 backbone) -> 8-dim bottleneck -> quantum layer, identical "
              "quantum_model.py reused with n_qubits_target=8.",
    ),
}


def get_disease(disease_id: str) -> DiseaseConfig:
    if disease_id not in DISEASE_REGISTRY:
        raise KeyError(
            f"Unknown disease_id '{disease_id}'. Valid options: {list(DISEASE_REGISTRY.keys())}"
        )
    return DISEASE_REGISTRY[disease_id]


TABULAR_DISEASES = [d for d, cfg in DISEASE_REGISTRY.items() if cfg.modality == "tabular"]
IMAGING_DISEASES = [d for d, cfg in DISEASE_REGISTRY.items() if cfg.modality == "imaging"]
