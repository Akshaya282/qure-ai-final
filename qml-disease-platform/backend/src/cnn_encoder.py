"""
cnn_encoder.py
================
Brain Tumor module: the ONE disease on this platform that is genuinely imaging-based (MRI scans),
so it needs a step the tabular diseases don't - compressing a high-dimensional image down to a
qubit-feasible number of values BEFORE it can reach quantum_model.py.

WHY A RAW MRI SLICE CANNOT GO DIRECTLY INTO THE QUANTUM LAYER
------------------------------------------------------------------
A single 224x224 grayscale MRI slice is 50,176 raw pixel values. Angle-encoding that directly
would need 50,176 qubits - many orders of magnitude beyond what any simulator or NISQ device can
handle. So we need a classical CNN to first learn a compact representation of the image, and only
feed that compact representation (n_qubits_target = 8, per config.py) into the quantum layer.

ARCHITECTURE
--------------
    MRI image (3 x 224 x 224, replicated grayscale-to-RGB for pretrained-weight compatibility)
        -> ResNet-18 backbone (ImageNet-pretrained, final FC layer removed)
        -> 512-dim feature vector
        -> Bottleneck: nn.Linear(512, 8) + Tanh  <- THIS is the 8-dim vector that becomes
                                                      the quantum layer's input angles
        -> [from here, reuse quantum_model.HybridQuantumClassifier's quantum_layer + post_net
            exactly as-is, with n_qubits=8]

MODELS/ALGORITHMS USED IN THIS STEP
--------------------------------------
- torchvision.models.resnet18(weights=IMAGENET1K_V1): transfer learning - reusing features
  learned from 1.2M natural images gives far better results than training a CNN from scratch on
  a brain-tumor dataset of only a few thousand images, which is realistic for a hackathon.
- Fine-tuning strategy: freeze all ResNet-18 conv layers except the last residual block ("layer4")
  and the new bottleneck head - this balances training speed (frozen early layers = fewer
  gradients to compute) against adaptation to MRI-specific texture patterns that differ from
  natural photographs.

HOW TO USE THIS WITH THE REST OF THE PIPELINE
-------------------------------------------------
1. Organize your downloaded MRI dataset as:
       data/brain_tumor/train/{glioma,meningioma,pituitary,notumor}/*.jpg
       data/brain_tumor/test/{glioma,meningioma,pituitary,notumor}/*.jpg
   (this is exactly the folder structure of the common Kaggle "Brain Tumor MRI Dataset")
2. Run `train_cnn_bottleneck()` below to fine-tune the encoder and produce 8-dim embeddings.
3. Feed those embeddings into `quantum_model.HybridQuantumClassifier(n_raw_features=8, n_qubits=8)`
   exactly like a tabular disease's selected features - the quantum layer code does not change
   at all between tabular and imaging diseases, only what feeds it does.
"""

import logging
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # MRI is grayscale; ResNet expects 3 channels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
])


class CNNBottleneckEncoder(nn.Module):
    """ResNet-18 backbone + trainable bottleneck head producing an n_qubits-dim embedding."""

    def __init__(self, n_qubits: int = 8, freeze_backbone_until: str = "layer4"):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # drop final FC layer

        # Freeze everything before `freeze_backbone_until` to speed up training and avoid
        # overfitting the small MRI dataset with a fully-unfrozen 11M-parameter network.
        freezing = True
        for name, child in backbone.named_children():
            if name == freeze_backbone_until:
                freezing = False
            if freezing:
                for param in child.parameters():
                    param.requires_grad = False

        self.bottleneck = nn.Linear(512, n_qubits)
        self.classifier_head = nn.Linear(n_qubits, 1)  # used ONLY for pretraining the encoder;
                                                         # discarded once quantum_model.py takes over

    def forward(self, x, return_embedding: bool = False):
        feats = self.features(x).flatten(1)         # (batch, 512)
        embedding = torch.tanh(self.bottleneck(feats)) * (3.14159265 / 2)  # scale to angle range
        if return_embedding:
            return embedding
        return torch.sigmoid(self.classifier_head(embedding))


def build_dataloaders(data_dir: str, batch_size: int = 32):
    """
    data_dir must contain train/ and test/ subfolders, each with one subfolder per class
    (torchvision.datasets.ImageFolder convention).
    """
    train_ds = datasets.ImageFolder(f"{data_dir}/train", transform=IMAGE_TRANSFORM)
    test_ds = datasets.ImageFolder(f"{data_dir}/test", transform=IMAGE_TRANSFORM)
    logger.info(f"Classes found: {train_ds.classes} | train={len(train_ds)} | test={len(test_ds)}")
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        train_ds.classes,
    )


def train_cnn_bottleneck(data_dir: str, n_qubits: int = 8, epochs: int = 10, lr: float = 1e-4):
    """
    Pretrains the CNN encoder end-to-end (with its own temporary classifier head) so the
    bottleneck embeddings are already discriminative BEFORE being handed to the quantum layer.
    After this, `model.forward(x, return_embedding=True)` gives you the 8-dim vector to feed
    into `quantum_model.HybridQuantumClassifier`.
    """
    train_loader, test_loader, classes = build_dataloaders(data_dir)
    model = CNNBottleneckEncoder(n_qubits=n_qubits)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = nn.BCELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            # NOTE: for multi-class datasets (glioma/meningioma/pituitary/notumor), binarize
            # here as (labels != notumor_class_idx) for a tumor-present/absent target, or swap
            # BCELoss for CrossEntropyLoss with a multi-unit classifier_head for full multi-class.
            optimizer.zero_grad()
            preds = model(images)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"[CNN encoder] epoch {epoch+1}/{epochs} - avg loss: {total_loss/len(train_loader):.4f}")

    return model, classes
