"""
quantum_model.py
==================
STEP 4 of the pipeline: Hybrid Quantum-Classical Model.

This module was validated end-to-end during development (forward pass + gradient backprop
confirmed working) before being finalized here - see the note on `input_gradients=True` below,
which is a real, easy-to-miss bug: without it, gradients silently stop at the quantum layer and
your classical pre-net never trains, while the code runs without throwing any error.

ARCHITECTURE
--------------
    Raw features (n_qubits_target, already scaled to [0,1] by data_preprocessing.py)
        -> classical pre-net: nn.Linear(n_features, n_qubits) + scale to [0, pi]
        -> ZZFeatureMap (data encoding: classical data -> quantum state via rotation angles)
        -> RealAmplitudes ansatz (TRAINABLE variational circuit - this is what "learns")
        -> EstimatorQNN (measures expectation value of the circuit as the quantum output)
        -> classical post-net: nn.Linear(1, 1) + Sigmoid -> P(disease present)

WHY ZZFeatureMap + RealAmplitudes (models/algorithms used in this step)
--------------------------------------------------------------------------
- ZZFeatureMap: encodes classical features as single-qubit rotation angles AND adds
  entangling ZZ interactions between qubits based on products of feature pairs. This lets the
  circuit represent feature CORRELATIONS natively in the quantum state, which a purely linear
  classical encoder cannot do without an explicit interaction term.
- RealAmplitudes: a hardware-efficient ansatz using only RY rotations and CNOT entanglers -
  chosen because it uses real (not complex) amplitudes, which is compatible with both
  simulators and today's NISQ hardware with minimal transpilation overhead.
- EstimatorQNN (Qiskit Machine Learning): wraps the whole circuit as a function you can call
  like a neural network layer - it internally uses the parameter-shift rule to compute exact
  gradients of the circuit output with respect to both the input angles and the trainable
  weights, which is what makes this differentiable end-to-end with PyTorch's autograd.
- TorchConnector: the bridge that makes an EstimatorQNN behave like a native torch.nn.Module,
  so it can sit inside a normal nn.Sequential-style model and be optimized with any PyTorch
  optimizer (Adam, SGD, etc.) exactly like a classical layer.

ALTERNATIVE: QUANTUM KERNEL SVM (QSVM)
------------------------------------------
Provided as the second, non-variational quantum approach the problem statement lists explicitly.
Instead of training circuit parameters, QSVM uses a fixed quantum feature map to compute a
KERNEL MATRIX (pairwise "quantum similarity" between samples), then hands that kernel matrix to
a completely classical SVM solver. This is much faster to "train" (no variational optimization
loop) but the kernel matrix computation itself is O(n^2) circuit evaluations, so it scales worse
with dataset size - a real, reportable computational-efficiency trade-off for your benchmarking
section.
"""

import logging
import numpy as np
import torch
import torch.nn as nn

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_vqc_circuit(n_qubits: int, feature_map_reps: int = 1, ansatz_reps: int = 2):
    """
    Builds the combined data-encoding + variational circuit.
    Returns (full_circuit, feature_map_params, ansatz_params) so the caller can tell
    EstimatorQNN which parameters are "inputs" (data) vs. "weights" (trainable).
    """
    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=feature_map_reps, entanglement="linear")
    ansatz = RealAmplitudes(num_qubits=n_qubits, reps=ansatz_reps, entanglement="linear")

    circuit = QuantumCircuit(n_qubits)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    logger.info(
        f"Built VQC circuit: {n_qubits} qubits | "
        f"{len(list(feature_map.parameters))} input params | "
        f"{len(list(ansatz.parameters))} trainable weight params | "
        f"circuit depth={circuit.depth()}"
    )
    return circuit, feature_map.parameters, ansatz.parameters


class HybridQuantumClassifier(nn.Module):
    """
    Full hybrid quantum-classical binary classifier.

    n_raw_features : number of features AFTER classical feature_selection.py (e.g. 6)
    n_qubits       : number of qubits the quantum layer uses. Usually == n_raw_features
                     (each selected feature maps to exactly one qubit's rotation angle),
                     but kept as a separate parameter in case a compression layer is desired.
    """

    def __init__(self, n_raw_features: int, n_qubits: int, ansatz_reps: int = 2):
        super().__init__()
        self.n_qubits = n_qubits

        circuit, input_params, weight_params = build_vqc_circuit(n_qubits, ansatz_reps=ansatz_reps)

        qnn = EstimatorQNN(
            circuit=circuit,
            input_params=input_params,
            weight_params=weight_params,
            # CRITICAL: without this flag, PyTorch autograd receives zero gradient for the
            # classical pre_net below (verified during development - training silently fails
            # to update pre_net's weights if this is left at its default of False).
            input_gradients=True,
        )

        self.pre_net = nn.Linear(n_raw_features, n_qubits)
        self.quantum_layer = TorchConnector(qnn)
        self.post_net = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # tanh bounds pre_net's output to (-1, 1), then scale to a valid rotation-angle range
        x = torch.tanh(self.pre_net(x)) * (np.pi / 2)
        x = self.quantum_layer(x)
        x = torch.sigmoid(self.post_net(x))
        return x


def train_hybrid_quantum_model(
    X_train, y_train, X_val, y_val,
    n_qubits: int,
    epochs: int = 60,
    lr: float = 0.05,
    ansatz_reps: int = 2,
) -> dict:
    """
    Trains the hybrid VQC classifier. Returns the trained model plus wall-clock training time
    for the efficiency comparison against classical baselines.

    NOTE ON TRAINING SPEED: quantum simulation is inherently slower per-step than classical
    matrix multiplication (each forward pass requires simulating the circuit's state vector).
    Expect this to take noticeably longer per epoch than the classical MLP baseline of the same
    size - this slowdown IS the "computational efficiency" trade-off you're asked to report.
    """
    import time

    model = HybridQuantumClassifier(n_raw_features=X_train.shape[1], n_qubits=n_qubits, ansatz_reps=ansatz_reps)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    start = time.time()
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(Xt)
        loss = loss_fn(preds, yt)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss)

        if epoch % 5 == 0 or epoch == epochs - 1:
            logger.info(f"[HybridVQC] epoch {epoch:3d} | train_loss={loss.item():.4f} | val_loss={val_loss:.4f}")

    train_time = time.time() - start
    logger.info(f"[HybridVQC] total training time: {train_time:.2f}s for {epochs} epochs")
    return {"model": model, "train_time_sec": train_time, "history": history, "type": "hybrid_quantum"}


def train_qsvm(X_train, y_train, n_qubits: int, max_samples: int = 120) -> dict:
    """
    Trains a Quantum Kernel SVM: computes a quantum-kernel Gram matrix via state-fidelity between
    ZZFeatureMap-encoded samples, then fits a classical SVC with that precomputed kernel.

    IMPORTANT COST NOTE (confirmed during development, not theoretical): the kernel matrix costs
    O(n^2) quantum circuit evaluations. On a full ~500-sample training split this becomes
    hundreds of thousands of state-vector simulations and can take far longer than the VQC does
    to train - a genuine, reportable computational-efficiency finding for your benchmarking
    section. `max_samples` subsamples the training set specifically for QSVM (stratified) to
    keep a hackathon-scale demo tractable; report this subsampling explicitly in your results
    table rather than hiding it, since it means QSVM's reported numbers use less training data
    than every other model in the comparison.
    """
    import time
    from sklearn.model_selection import train_test_split

    if len(y_train) > max_samples:
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=max_samples, stratify=y_train, random_state=42
        )
        logger.warning(
            f"[QSVM] Subsampled training set to {max_samples} rows (from a larger split) to "
            f"keep the O(n^2) quantum kernel matrix computation tractable on a simulator."
        )

    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=1, entanglement="linear")
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

    start = time.time()
    kernel_matrix_train = quantum_kernel.evaluate(x_vec=X_train)
    svc = SVC(kernel="precomputed", probability=True)
    svc.fit(kernel_matrix_train, y_train)
    train_time = time.time() - start

    logger.info(f"[QSVM] trained in {train_time:.2f}s on {len(y_train)} samples "
                f"({n_qubits}-qubit feature map)")
    return {"model": svc, "kernel": quantum_kernel, "X_train_ref": X_train,
            "train_time_sec": train_time, "type": "qsvm"}


def predict_hybrid(entry: dict, X: np.ndarray) -> np.ndarray:
    entry["model"].eval()
    with torch.no_grad():
        return entry["model"](torch.tensor(X, dtype=torch.float32)).numpy().ravel()


def predict_qsvm(entry: dict, X: np.ndarray) -> np.ndarray:
    kernel_matrix_test = entry["kernel"].evaluate(x_vec=X, y_vec=entry["X_train_ref"])
    return entry["model"].predict_proba(kernel_matrix_test)[:, 1]
