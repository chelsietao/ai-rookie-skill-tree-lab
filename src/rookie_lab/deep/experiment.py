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
) -> dict[str, object]:
    torch.manual_seed(seed)
    loader = DataLoader(
        train, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss()
    best_loss, best_state, patience = float("inf"), deepcopy(model.state_dict()), 0
    learning_rates: list[float] = []
    for _ in range(epochs):
        model.train()
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = float(criterion(model(valid.tensors[0]), valid.tensors[1]))
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
    for name in ("none", "batch", "layer"):
        model = MLPClassifier(features.shape[1], activation_name="relu", norm=name)
        initialize(model, "kaiming")
        normalization_results[name] = _classification_train(model, train, valid, epochs, seed)
    return {"activations": activation_results, "normalization": normalization_results}


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
    for _ in range(epochs):
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        reconstruction = model(torch.from_numpy(values)).numpy()
    error = np.square(values - reconstruction).mean(1)
    return {"anomaly_roc_auc": float(roc_auc_score(labels, error)), "latent_dimensions": 2}


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
            "BatchNorm/LayerNorm",
            "dropout",
            "early stopping",
            "cosine learning-rate scheduling",
        ],
    }
