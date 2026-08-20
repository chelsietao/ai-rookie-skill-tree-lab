"""A two-layer GCN using only PyTorch sparse matrix operations."""

from __future__ import annotations

import torch
from torch import nn


class GraphConvolution(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.linear = nn.Linear(inputs, outputs, bias=False)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.sparse.mm(adjacency, features))


class GCN(nn.Module):
    def __init__(self, inputs: int, hidden: int = 24, classes: int = 2):
        super().__init__()
        self.first = GraphConvolution(inputs, hidden)
        self.second = GraphConvolution(hidden, hidden)
        self.classifier = nn.Linear(hidden, classes)

    def forward(
        self, features: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = torch.relu(self.first(features, adjacency))
        embeddings = self.second(embeddings, adjacency)
        return self.classifier(torch.relu(embeddings)), embeddings


def normalized_adjacency(edges: torch.Tensor, nodes: int) -> torch.Tensor:
    reverse = edges.flip(0)
    identity = torch.arange(nodes)
    indices = torch.cat([edges, reverse, torch.stack([identity, identity])], dim=1)
    values = torch.ones(indices.shape[1])
    adjacency = torch.sparse_coo_tensor(
        indices, values, (nodes, nodes), check_invariants=False
    ).coalesce()
    degree = torch.zeros(nodes).scatter_add_(0, adjacency.indices()[0], adjacency.values())
    row, column = adjacency.indices()
    normalized = (
        adjacency.values() * degree[row].clamp_min(1).pow(-0.5) * degree[column].clamp_min(1).pow(-0.5)
    )
    return torch.sparse_coo_tensor(
        adjacency.indices(), normalized, adjacency.shape, check_invariants=False
    ).coalesce()
