"""Run clustering and dimension-reduction comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NUMERIC = ["age", "tenure_days", "income", "visits_30d", "support_tickets", "loyalty_score"]


def _score(features: np.ndarray, labels: np.ndarray) -> float | None:
    valid = labels >= 0
    unique = np.unique(labels[valid])
    if valid.sum() < 3 or len(unique) < 2:
        return None
    return float(silhouette_score(features[valid], labels[valid]))


def run(data_dir: Path, profile: str, seed: int = 20260820) -> dict[str, object]:
    frame = pd.read_csv(data_dir / "customers.csv.gz")
    sample = frame.sample(min(len(frame), 500 if profile == "ci" else 1500), random_state=seed)
    pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    features = pipeline.fit_transform(sample[NUMERIC])
    clusterers = {
        "kmeans": KMeans(n_clusters=4, n_init=10, random_state=seed),
        "dbscan": DBSCAN(eps=1.15, min_samples=8),
        "agglomerative": AgglomerativeClustering(n_clusters=4),
    }
    clustering: dict[str, object] = {}
    labels_by_name: dict[str, np.ndarray] = {}
    for name, model in clusterers.items():
        labels = model.fit_predict(features)
        labels_by_name[name] = labels
        clustering[name] = {
            "clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
            "noise_rate": float(np.mean(labels == -1)),
            "silhouette": _score(features, labels),
            "cluster_sizes": {str(label): int((labels == label).sum()) for label in sorted(set(labels))},
        }

    repeated_labels = [
        KMeans(n_clusters=4, n_init=10, random_state=seed + offset).fit_predict(features)
        for offset in range(5)
    ]
    pairwise_ari = [
        adjusted_rand_score(repeated_labels[left], repeated_labels[right])
        for left in range(len(repeated_labels))
        for right in range(left + 1, len(repeated_labels))
    ]

    pca = PCA(n_components=2, random_state=seed).fit_transform(features)
    tsne = TSNE(n_components=2, perplexity=min(30, len(features) // 5), random_state=seed).fit_transform(
        features
    )
    embedding = umap.UMAP(n_components=2, n_neighbors=15, random_state=seed).fit_transform(features)
    figure_dir = Path("artifacts/figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for axis, values, title in zip(axes, (pca, tsne, embedding), ("PCA", "t-SNE", "UMAP"), strict=True):
        sns.scatterplot(
            x=values[:, 0], y=values[:, 1], hue=labels_by_name["kmeans"], ax=axis, legend=False, s=14
        )
        axis.set_title(title)
    fig.tight_layout()
    fig.savefig(figure_dir / f"dimension-reduction-{profile}.png", dpi=150)
    plt.close(fig)
    return {
        "rows": len(sample),
        "clustering": clustering,
        "stability": {
            "method": "five K-Means restarts with pairwise adjusted Rand index",
            "mean_ari": float(np.mean(pairwise_ari)),
            "min_ari": float(np.min(pairwise_ari)),
            "pairwise_ari": [float(value) for value in pairwise_ari],
        },
        "dimension_reduction": {
            "pca_explained_variance": float(
                PCA(n_components=2).fit(features).explained_variance_ratio_.sum()
            ),
            "methods": ["PCA", "t-SNE", "UMAP"],
        },
    }
