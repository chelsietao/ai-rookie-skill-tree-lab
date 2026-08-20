"""Deep-learning experiments with early stopping and scheduler traces."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rookie_lab.deep.models import Autoencoder, MLPClassifier, SequenceClassifier, initialize


def _classification_train(
    model: nn.Module,
    train: TensorDataset,
    valid: TensorDataset,
    epochs: int,
    seed: int,
    optimizer_name: str = "adam",
    scheduler_name: str = "cosine",
    class_weighted: bool = False,
) -> dict[str, object]:
    torch.manual_seed(seed)
    loader = DataLoader(
        train, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed)
    )
    optimizers = {
        "sgd": lambda: torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9),
        "adam": lambda: torch.optim.Adam(model.parameters(), lr=0.01),
        "rmsprop": lambda: torch.optim.RMSprop(model.parameters(), lr=0.01),
    }
    if optimizer_name not in optimizers:
        raise ValueError(f"unknown optimizer: {optimizer_name}")
    optimizer = optimizers[optimizer_name]()
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    elif scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    elif scheduler_name == "warmup":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.2, end_factor=1.0, total_iters=max(1, min(3, epochs))
        )
    else:
        raise ValueError(f"unknown scheduler: {scheduler_name}")
    weights = None
    if class_weighted:
        counts = torch.bincount(train.tensors[1], minlength=2).float()
        weights = counts.sum() / (2 * counts.clamp_min(1))
    criterion = nn.CrossEntropyLoss(weight=weights)
    best_loss, best_state, patience = float("inf"), deepcopy(model.state_dict()), 0
    learning_rates: list[float] = []
    train_losses: list[float] = []
    valid_losses: list[float] = []
    for _ in range(epochs):
        model.train()
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss = float(criterion(model(train.tensors[0]), train.tensors[1]))
            valid_loss = float(criterion(model(valid.tensors[0]), valid.tensors[1]))
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        learning_rates.append(float(optimizer.param_groups[0]["lr"]))
        scheduler.step()
        if valid_loss < best_loss - 1e-5:
            best_loss, best_state, patience = valid_loss, deepcopy(model.state_dict()), 0
        else:
            patience += 1
            if patience >= 3:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probability = model(valid.tensors[0]).softmax(1)[:, 1].numpy()
    return {
        "accuracy": float(accuracy_score(valid.tensors[1].numpy(), probability >= 0.5)),
        "roc_auc": float(roc_auc_score(valid.tensors[1].numpy(), probability)),
        "best_validation_loss": best_loss,
        "epochs_ran": len(learning_rates),
        "learning_rates": learning_rates,
        "train_loss": train_losses,
        "validation_loss": valid_losses,
        "overfitting_gap": valid_losses[-1] - train_losses[-1],
        "optimizer": optimizer_name,
        "scheduler": scheduler_name,
        "class_weighted": class_weighted,
    }


def _tabular(data_dir: Path, profile: str, seed: int) -> dict[str, object]:
    frame = pd.read_csv(data_dir / "customers.csv.gz")
    columns = ["age", "tenure_days", "income", "visits_30d", "support_tickets", "loyalty_score"]
    features = SimpleImputer(strategy="median").fit_transform(frame[columns])
    features = StandardScaler().fit_transform(features).astype("float32")
    target = frame["churned"].to_numpy("int64")
    x_train, x_valid, y_train, y_valid = train_test_split(
        features, target, test_size=0.25, stratify=target, random_state=seed
    )
    train = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    valid = TensorDataset(torch.from_numpy(x_valid), torch.from_numpy(y_valid))
    epochs = 3 if profile == "ci" else 8
    activation_results: dict[str, object] = {}
    for name in ("relu", "leaky_relu", "gelu", "sigmoid", "tanh"):
        model = MLPClassifier(features.shape[1], activation_name=name)
        initialize(model, "kaiming" if name in {"relu", "leaky_relu"} else "xavier")
        activation_results[name] = _classification_train(model, train, valid, epochs, seed)
    normalization_results: dict[str, object] = {}
    for name in ("none", "batch", "layer", "group"):
        model = MLPClassifier(features.shape[1], activation_name="relu", norm=name)
        initialize(model, "kaiming")
        normalization_results[name] = _classification_train(model, train, valid, epochs, seed)
    optimizer_results: dict[str, object] = {}
    for name in ("sgd", "adam", "rmsprop"):
        model = MLPClassifier(features.shape[1], activation_name="relu", norm="layer")
        initialize(model, "kaiming")
        optimizer_results[name] = _classification_train(
            model, train, valid, epochs, seed, optimizer_name=name
        )
    scheduler_results: dict[str, object] = {}
    for name in ("warmup", "step", "cosine"):
        model = MLPClassifier(features.shape[1], activation_name="relu", norm="layer")
        initialize(model, "kaiming")
        scheduler_results[name] = _classification_train(
            model, train, valid, epochs, seed, scheduler_name=name
        )
    weighted_model = MLPClassifier(features.shape[1], activation_name="relu", norm="layer")
    initialize(weighted_model, "kaiming")
    weighted_result = _classification_train(
        weighted_model, train, valid, epochs, seed, class_weighted=True
    )
    return {
        "activations": activation_results,
        "normalization": normalization_results,
        "optimizers": optimizer_results,
        "schedulers": scheduler_results,
        "class_weighting": weighted_result,
    }


def _sequences(data_dir: Path, profile: str, seed: int) -> dict[str, object]:
    payload = np.load(data_dir / "sequences.npz")
    x, y = payload["x"], payload["y"]
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=seed
    )
    train = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    valid = TensorDataset(torch.from_numpy(x_valid), torch.from_numpy(y_valid))
    epochs = 3 if profile == "ci" else 8
    return {
        cell: _classification_train(SequenceClassifier(cell), train, valid, epochs, seed)
        for cell in ("rnn", "lstm", "gru")
    }


def _autoencoder(data_dir: Path, profile: str, seed: int) -> dict[str, object]:
    frame = pd.read_csv(data_dir / "transactions.csv.gz")
    category = pd.get_dummies(frame["category"], dtype=float)
    values = np.column_stack((frame[["amount", "day"]].to_numpy(), category.to_numpy()))
    values = StandardScaler().fit_transform(values).astype("float32")
    labels = frame["is_anomaly"].to_numpy()
    train_values = torch.from_numpy(values[labels == 0])
    model = Autoencoder(values.shape[1])
    initialize(model, "xavier")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    epochs = 3 if profile == "ci" else 8
    loader = DataLoader(TensorDataset(train_values), batch_size=128, shuffle=True)
    losses: list[float] = []
    for _ in range(epochs):
        epoch_losses: list[float] = []
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_losses)))
    with torch.no_grad():
        reconstruction = model(torch.from_numpy(values)).numpy()
    error = np.square(values - reconstruction).mean(1)
    threshold = float(np.quantile(error[labels == 0], 0.99))
    prediction = error >= threshold
    return {
        "anomaly_roc_auc": float(roc_auc_score(labels, error)),
        "latent_dimensions": 2,
        "threshold_p99_normal": threshold,
        "precision_at_threshold": float(
            ((prediction == 1) & (labels == 1)).sum() / max(1, prediction.sum())
        ),
        "recall_at_threshold": float(
            ((prediction == 1) & (labels == 1)).sum() / max(1, (labels == 1).sum())
        ),
        "training_loss": losses,
    }


def run(data_dir: Path, profile: str, seed: int = 20260820) -> dict[str, object]:
    torch.set_num_threads(2)
    return {
        "framework": "PyTorch",
        "tabular_mlp_ablations": _tabular(data_dir, profile, seed),
        "sequence_models": _sequences(data_dir, profile, seed),
        "autoencoder": _autoencoder(data_dir, profile, seed),
        "training_tricks": [
            "Xavier/Kaiming initialization",
            "ReLU/LeakyReLU/GELU/Sigmoid/Tanh",
            "BatchNorm/LayerNorm/GroupNorm",
            "dropout",
            "early stopping",
            "warmup/step/cosine learning-rate scheduling",
            "SGD/Adam/RMSProp",
            "class weighting",
        ],
    }
