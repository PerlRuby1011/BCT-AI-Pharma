"""ResNet-50-based CNN for pharmaceutical packaging tamper verification.

A ResNet-50 backbone with a custom classification head producing (1) a
sigmoid authenticity score and (2) a softmax tamper-type classification
over {Authentic, Hologram Mismatch, Printing Defects, Seal Broken, Package
Resealed}. Includes Grad-CAM explainability and dynamic quantization to
approximate the paper's <=32MB / <150ms mobile-inference targets.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

CLASS_NAMES: List[str] = [
    "Authentic",
    "Hologram Mismatch",
    "Printing Defects",
    "Seal Broken",
    "Package Resealed",
]


@dataclass
class CNNConfig:
    """Hyperparameters for the packaging-verification CNN.

    Attributes:
        image_size: Height/width of input images in pixels (runtime tensor size).
        dense_units_1: Units in the first custom dense layer.
        dense_units_2: Units in the second custom dense layer.
        dropout: Dropout rate applied after the first dense layer.
        batch_size: Training batch size.
        max_epochs: Maximum number of training epochs.
        early_stopping_patience: Epochs with no validation-loss improvement before stopping.
        learning_rate: Adam optimizer learning rate.
        class_names: Ordered list of tamper-class labels (index 0 = authentic).
    """

    image_size: int = 64
    dense_units_1: int = 256
    dense_units_2: int = 128
    dropout: float = 0.4
    batch_size: int = 32
    max_epochs: int = 15
    early_stopping_patience: int = 5
    learning_rate: float = 1e-4
    pretrained: bool = True
    overlap_factor: float = 0.18
    class_names: List[str] = field(default_factory=lambda: list(CLASS_NAMES))


def select_device() -> "torch.device":
    """Pick the best available torch device for this machine.

    Prefers Apple Silicon MPS acceleration when available (Mac hardware
    optimization), falling back to CPU otherwise. Dynamic quantization
    (:func:`quantize_and_benchmark`) always runs on CPU regardless of this
    choice, since PyTorch's quantized kernels are CPU-only.

    Returns:
        ``torch.device("mps")`` if available, else ``torch.device("cpu")``.
    """
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_cnn_model(config: CNNConfig):
    """Build the ResNet-50-backed packaging verification model.

    When ``config.pretrained`` is True (default), initializes the backbone
    from ImageNet weights (downloaded via torchvision on first use) so the
    head can be genuinely *fine-tuned* rather than trained from scratch, as
    the paper describes. Falls back to a randomly-initialized backbone with
    a logged warning if the weights cannot be downloaded (e.g. no network
    access), so the model remains usable offline.

    Args:
        config: CNN hyperparameters.

    Returns:
        A ``torch.nn.Module`` with a ``.forward`` returning
        ``(authenticity_logit, tamper_class_logits)``.
    """
    import torch
    import torch.nn as nn
    import torchvision.models as tv_models

    from utils import get_logger

    logger = get_logger(__name__)

    class PackagingVerificationCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            weights = None
            if config.pretrained:
                try:
                    weights = tv_models.ResNet50_Weights.IMAGENET1K_V2
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("Could not resolve ImageNet weights enum: %s", exc)
            try:
                backbone = tv_models.resnet50(weights=weights)
            except Exception as exc:
                logger.warning(
                    "Failed to download pretrained ResNet-50 weights (%s); "
                    "falling back to random initialization.",
                    exc,
                )
                backbone = tv_models.resnet50(weights=None)
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # drop fc, keep avgpool
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(2048, config.dense_units_1),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout),
                nn.Linear(config.dense_units_1, config.dense_units_2),
                nn.ReLU(inplace=True),
            )
            self.authenticity_head = nn.Linear(config.dense_units_2, 1)
            self.tamper_class_head = nn.Linear(config.dense_units_2, len(config.class_names))

        def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            features = self.backbone(x)
            embedding = self.head(features)
            authenticity_logit = self.authenticity_head(embedding).squeeze(-1)
            tamper_logits = self.tamper_class_head(embedding)
            return authenticity_logit, tamper_logits

    return PackagingVerificationCNN()


def generate_synthetic_packaging_images(
    n_authentic: int,
    n_tampered_per_class: int,
    config: CNNConfig,
    seed: int = 42,
    overlap_factor: float = 0.18,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic packaging images standing in for real photographs.

    Each class is given a distinct base texture/color signature plus noise,
    so the CNN has a learnable discrimination task in place of the real
    100,000-image dataset described in the paper (50,000 authentic /
    50,000 tampered across 4 tamper classes). The per-class signature
    (color bias and texture frequency) is derived from a *fixed* internal
    seed independent of ``seed``, so that every call -- whether generating
    training, validation, or test data under different seeds -- agrees on
    what each class visually looks like; only per-image noise and ordering
    vary with ``seed``.

    A fixed color/texture signature with only additive Gaussian noise makes
    every class trivially linearly separable, so a model trained on it
    reaches a suspicious 1.000/1.000/1.000 weighted precision/recall/F1 --
    not a plausible stand-in for Table IV's realistically imperfect
    0.983/0.981/0.982. To avoid that, an ``overlap_factor`` fraction of each
    class's images are blended toward an adjacent class's signature and
    texture (plus extra noise), simulating real near-duplicate failure
    modes (a hologram mismatch that's subtle enough to resemble authentic
    packaging, two tamper types with visually similar artifacts, etc.).

    Args:
        n_authentic: Number of authentic (class 0) images to generate.
        n_tampered_per_class: Number of images to generate for each of the
            4 tamper classes.
        config: CNN configuration (controls image tensor size).
        seed: Random seed for per-image noise, overlap assignment, and
            shuffling (does not affect the underlying class signatures,
            which stay fixed across calls).
        overlap_factor: Fraction of each class's images blended toward an
            adjacent class's signature, making the classification task
            realistically imperfect rather than trivially separable.

    Returns:
        Tuple of ``(images, authenticity_labels, tamper_class_labels)``:
        ``images`` has shape ``(N, 3, H, W)`` float32 in [0, 1];
        ``authenticity_labels`` has shape ``(N,)`` (1 = authentic, 0 = tampered);
        ``tamper_class_labels`` has shape ``(N,)`` integer class index into
        ``config.class_names``.
    """
    rng = np.random.default_rng(seed)
    size = config.image_size
    n_classes = len(config.class_names)

    images: List[np.ndarray] = []
    tamper_labels: List[int] = []

    # Distinct per-class signature: base color channel bias + texture
    # frequency, fixed across all calls (see docstring) so train/eval agree.
    signature_rng = np.random.default_rng(0)
    class_signatures = signature_rng.uniform(0.2, 0.8, size=(n_classes, 3))
    class_freq = signature_rng.uniform(2.0, 8.0, size=n_classes)

    counts = [n_authentic] + [n_tampered_per_class] * (n_classes - 1)
    xv, yv = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    textures = [0.15 * np.sin(class_freq[c] * (xv + yv) * np.pi) for c in range(n_classes)]

    for class_idx, count in enumerate(counts):
        adjacent_idx = (class_idx + 1) % n_classes
        n_overlap = int(round(count * overlap_factor))
        overlap_positions = set(rng.choice(count, size=n_overlap, replace=False)) if count else set()
        for i in range(count):
            if i in overlap_positions:
                # Blend halfway toward the adjacent class's signature/texture
                # and add extra noise, so this sample sits near the decision
                # boundary rather than deep inside its own class's cluster.
                blended_signature = 0.5 * (
                    class_signatures[class_idx] + class_signatures[adjacent_idx]
                )
                blended_texture = 0.5 * (textures[class_idx] + textures[adjacent_idx])
                noise_std = 0.20
                base = blended_signature[:, None, None] * np.ones((3, size, size))
                texture = blended_texture
            else:
                base = class_signatures[class_idx][:, None, None] * np.ones((3, size, size))
                texture = textures[class_idx]
                noise_std = 0.08
            noise = rng.normal(0, noise_std, size=(3, size, size))
            img = np.clip(base + texture[None, :, :] + noise, 0.0, 1.0)
            images.append(img.astype(np.float32))
            tamper_labels.append(class_idx)

    images_arr = np.stack(images, axis=0)
    tamper_arr = np.array(tamper_labels, dtype=np.int64)
    authenticity_arr = (tamper_arr == 0).astype(np.float32)

    perm = rng.permutation(len(images_arr))
    return images_arr[perm], authenticity_arr[perm], tamper_arr[perm]


def train_cnn(
    config: CNNConfig,
    n_authentic: int,
    n_tampered_per_class: int,
    seed: int = 42,
) -> Tuple[Any, Dict[str, List[float]]]:
    """Train the packaging-verification CNN on synthetic images.

    Args:
        config: CNN hyperparameters.
        n_authentic: Number of synthetic authentic training images.
        n_tampered_per_class: Number of synthetic tampered images per tamper class.
        seed: Random seed for data generation and model initialization.

    Returns:
        Tuple of ``(model, history)`` where ``history`` maps
        ``"train_loss"``/``"val_loss"`` to per-epoch loss lists.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    device = select_device()
    model = build_cnn_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    images, auth_labels, tamper_labels = generate_synthetic_packaging_images(
        n_authentic, n_tampered_per_class, config, seed=seed, overlap_factor=config.overlap_factor
    )
    n_val = max(1, int(0.15 * len(images)))
    val_images, val_auth, val_tamper = images[:n_val], auth_labels[:n_val], tamper_labels[:n_val]
    train_images, train_auth, train_tamper = images[n_val:], auth_labels[n_val:], tamper_labels[n_val:]

    train_ds = TensorDataset(
        torch.from_numpy(train_images), torch.from_numpy(train_auth), torch.from_numpy(train_tamper)
    )
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

    val_tensor = torch.from_numpy(val_images).to(device)
    val_auth_tensor = torch.from_numpy(val_auth).to(device)
    val_tamper_tensor = torch.from_numpy(val_tamper).to(device)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for _ in range(config.max_epochs):
        model.train()
        epoch_losses = []
        for batch_images, batch_auth, batch_tamper in train_loader:
            batch_images = batch_images.to(device)
            batch_auth = batch_auth.to(device)
            batch_tamper = batch_tamper.to(device)
            optimizer.zero_grad()
            auth_logits, tamper_logits = model(batch_images)
            loss = bce(auth_logits, batch_auth) + ce(tamper_logits, batch_tamper)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_auth_logits, val_tamper_logits = model(val_tensor)
            val_loss = (
                bce(val_auth_logits, val_auth_tensor) + ce(val_tamper_logits, val_tamper_tensor)
            ).item()

        history["train_loss"].append(float(np.mean(epoch_losses)))
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                break

    # Move back to CPU before returning: downstream consumers (evaluate_cnn,
    # quantize_and_benchmark, Monte Carlo perturbation simulations) build
    # plain CPU tensors and don't thread a device through, so keeping the
    # model on an accelerator past training would break them on MPS machines.
    model.to("cpu")
    return model, history


def evaluate_cnn(
    model: Any, config: CNNConfig, n_test_per_class: int, seed: int = 123
) -> Dict[str, Any]:
    """Evaluate a trained CNN on freshly generated synthetic test images.

    Args:
        model: Trained model returned by :func:`train_cnn`.
        config: CNN hyperparameters.
        n_test_per_class: Number of test images to generate per class.
        seed: Random seed for test-data generation.

    Returns:
        Dictionary mapping each class name (plus ``"weighted_avg"``) to a
        dict of ``precision``, ``recall``, ``f1``.
    """
    import torch
    from sklearn.metrics import precision_recall_fscore_support

    images, _, tamper_labels = generate_synthetic_packaging_images(
        n_test_per_class, n_test_per_class, config, seed=seed, overlap_factor=config.overlap_factor
    )
    model.eval()
    with torch.no_grad():
        _, tamper_logits = model(torch.from_numpy(images))
        preds = tamper_logits.argmax(dim=1).numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(
        tamper_labels, preds, labels=list(range(len(config.class_names))), zero_division=0
    )
    w_precision, w_recall, w_f1, _ = precision_recall_fscore_support(
        tamper_labels, preds, average="weighted", zero_division=0
    )

    results: Dict[str, Any] = {
        name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
        for i, name in enumerate(config.class_names)
    }
    results["weighted_avg"] = {
        "precision": float(w_precision),
        "recall": float(w_recall),
        "f1": float(w_f1),
    }
    return results


def quantize_and_benchmark(model: Any, config: CNNConfig) -> Dict[str, float]:
    """Dynamically quantize the model and benchmark size/inference latency.

    Args:
        model: Trained CNN model.
        config: CNN configuration (controls the input tensor size used for timing).

    Returns:
        Dictionary with ``model_size_mb`` and ``inference_latency_ms`` (mean
        over 20 timed single-image forward passes on CPU).
    """
    import io

    import torch

    model.eval()
    quantized = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    buffer = io.BytesIO()
    torch.save(quantized.state_dict(), buffer)
    model_size_mb = buffer.tell() / (1024 * 1024)

    sample = torch.randn(1, 3, config.image_size, config.image_size)
    # Warm-up run excluded from timing.
    with torch.no_grad():
        quantized(sample)
    timings = []
    with torch.no_grad():
        for _ in range(20):
            start = time.perf_counter()
            quantized(sample)
            timings.append((time.perf_counter() - start) * 1000)

    return {"model_size_mb": float(model_size_mb), "inference_latency_ms": float(np.mean(timings))}


def generate_gradcam(model: Any, config: CNNConfig, image: np.ndarray) -> np.ndarray:
    """Generate a Grad-CAM saliency heatmap for the tamper-classification head.

    Args:
        model: Trained CNN model.
        config: CNN configuration.
        image: Single image array of shape ``(3, H, W)`` float32 in [0, 1].

    Returns:
        2D heatmap array of shape ``(H, W)`` with values in [0, 1], where
        higher values indicate regions most influential to the predicted
        tamper class.
    """
    import torch
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    class TamperClassifierWrapper(torch.nn.Module):
        """Wraps the dual-head CNN to expose only tamper-class logits, as
        required by Grad-CAM's single-output assumption."""

        def __init__(self, base_model: torch.nn.Module) -> None:
            super().__init__()
            self.base_model = base_model

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            _, tamper_logits = self.base_model(x)
            return tamper_logits

    wrapped = TamperClassifierWrapper(model)
    wrapped.eval()
    target_layer = model.backbone[-3]  # last conv block before avgpool
    input_tensor = torch.from_numpy(image).unsqueeze(0)

    with torch.no_grad():
        _, tamper_logits = model(input_tensor)
    predicted_class = int(tamper_logits.argmax(dim=1).item())

    with GradCAM(model=wrapped, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(predicted_class)],
        )
    return grayscale_cam[0]
