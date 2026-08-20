"""Customer-product node classification and link prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from rookie_lab.graph.gcn import GCN, normalized_adjacency


def _build(
    data_dir: Path, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, list[tuple[int, int]]]:
    customers = pd.read_csv(data_dir / "customers.csv.gz")
    graph = pd.read_csv(data_dir / "graph_edges.csv.gz")
    customer_ids = {value: index for index, value in enumerate(customers["customer_id"])}
    product_values = sorted(graph["product_id"].unique())
    product_ids = {value: len(customers) + index for index, value in enumerate(product_values)}
    pairs = [(customer_ids[row.customer_id], product_ids[row.product_id]) for row in graph.itertuples()]
    edges = torch.tensor(pairs, dtype=torch.long).T
    nodes = len(customers) + len(product_values)
    adjacency = normalized_adjacency(edges, nodes)

    columns = ["age", "tenure_days", "income", "visits_30d", "support_tickets", "loyalty_score"]
    customer_features = SimpleImputer(strategy="median").fit_transform(customers[columns])
    customer_features = StandardScaler().fit_transform(customer_features)
    features = np.zeros((nodes, 8), dtype="float32")
    features[: len(customers), :6] = customer_features
    rng = np.random.default_rng(seed)
    features[len(customers) :, 6:] = rng.normal(0, 1, (len(product_values), 2))
    labels = torch.from_numpy(customers["churned"].to_numpy("int64", copy=True))
    return torch.from_numpy(features), labels, adjacency, len(customers), pairs


def run(data_dir: Path, profile: str, seed: int = 20260820) -> dict[str, object]:
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    features, labels, adjacency, customer_count, pairs = _build(data_dir, seed)
    indices = np.arange(customer_count)
    train_idx, valid_idx = train_test_split(indices, test_size=0.3, stratify=labels, random_state=seed)
    train_tensor, valid_tensor = torch.from_numpy(train_idx), torch.from_numpy(valid_idx)
    customer_degree = np.bincount([left for left, _ in pairs], minlength=customer_count)
    baseline_features = np.column_stack([features[:customer_count].numpy(), np.log1p(customer_degree)])
    baseline = LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed).fit(
        baseline_features[train_idx], labels[train_tensor].numpy()
    )
    baseline_probability = baseline.predict_proba(baseline_features[valid_idx])[:, 1]
    model = GCN(features.shape[1])
    positive = int(labels[train_tensor].sum())
    negative = len(train_tensor) - positive
    weights = torch.tensor([1.0, negative / max(1, positive)])
    criterion = nn.CrossEntropyLoss(weight=weights)
    rng = np.random.default_rng(seed)
    shuffled = [pairs[index] for index in rng.permutation(len(pairs))]
    split = int(len(shuffled) * 0.8)
    train_positive = shuffled[: min(split, 5000)]
    valid_positive = shuffled[split : split + min(1000, len(shuffled) - split)]
    positive_set = set(pairs)
    product_start = customer_count

    def sample_negatives(count: int) -> list[tuple[int, int]]:
        values: list[tuple[int, int]] = []
        while len(values) < count:
            pair = (
                int(rng.integers(0, customer_count)),
                int(rng.integers(product_start, len(features))),
            )
            if pair not in positive_set:
                values.append(pair)
        return values

    train_negative = sample_negatives(len(train_positive))
    valid_negative = sample_negatives(len(valid_positive))
    train_links = torch.tensor(train_positive + train_negative, dtype=torch.long)
    train_link_labels = torch.cat([torch.ones(len(train_positive)), torch.zeros(len(train_negative))])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=1e-4)
    losses: list[float] = []
    for _ in range(15 if profile == "ci" else 40):
        optimizer.zero_grad()
        logits, embeddings = model(features, adjacency)
        node_loss = criterion(logits[train_tensor], labels[train_tensor])
        link_logits = (embeddings[train_links[:, 0]] * embeddings[train_links[:, 1]]).sum(1)
        link_loss = nn.functional.binary_cross_entropy_with_logits(link_logits, train_link_labels)
        loss = node_loss + 0.35 * link_loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        logits, embeddings = model(features, adjacency)
        probability = logits[valid_tensor].softmax(1)[:, 1].numpy()

    link_pairs = valid_positive + valid_negative
    link_scores = [float((embeddings[left] * embeddings[right]).sum()) for left, right in link_pairs]
    link_labels = [1] * len(valid_positive) + [0] * len(valid_negative)
    node_degree = np.bincount(np.asarray(pairs).reshape(-1), minlength=len(features))
    popularity_scores = [
        float(np.log1p(node_degree[left]) + np.log1p(node_degree[right])) for left, right in link_pairs
    ]
    return {
        "node_classification": {
            "accuracy": float(accuracy_score(labels[valid_tensor].numpy(), probability >= 0.5)),
            "roc_auc": float(roc_auc_score(labels[valid_tensor].numpy(), probability)),
        },
        "traditional_baseline": {
            "model": "class-weighted logistic regression on tabular features plus node degree",
            "node_roc_auc": float(roc_auc_score(labels[valid_tensor].numpy(), baseline_probability)),
        },
        "link_prediction": {
            "gcn_roc_auc": float(roc_auc_score(link_labels, link_scores)),
            "degree_popularity_baseline_roc_auc": float(roc_auc_score(link_labels, popularity_scores)),
        },
        "training_loss": {"first": losses[0], "last": losses[-1], "curve": losses},
        "graph_statistics": {
            "customer_nodes": customer_count,
            "product_nodes": len(features) - customer_count,
            "mean_customer_degree": float(customer_degree.mean()),
            "density": float(len(pairs) / (customer_count * (len(features) - customer_count))),
        },
        "nodes": len(features),
        "edges": len(pairs),
        "implementation": "manual PyTorch sparse GCN; no PyG or DGL",
        "scope": "transductive graph with joint node and link-reconstruction losses",
    }
