import torch

from rookie_lab.graph.gcn import GCN, normalized_adjacency


def test_sparse_gcn_shapes() -> None:
    edges = torch.tensor([[0, 1], [2, 2]])
    adjacency = normalized_adjacency(edges, 3)
    logits, embeddings = GCN(4, hidden=5)(torch.ones(3, 4), adjacency)
    assert logits.shape == (3, 2)
    assert embeddings.shape == (3, 5)
    assert adjacency.is_sparse


def test_normalized_adjacency_is_symmetric_and_has_self_loops() -> None:
    edges = torch.tensor([[0, 1], [1, 2]])
    dense = normalized_adjacency(edges, 3).to_dense()
    assert torch.allclose(dense, dense.T)
    assert torch.all(torch.diag(dense) > 0)
