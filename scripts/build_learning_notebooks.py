"""Build the seven substantial, executable learning notebooks from reviewable source cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from textwrap import dedent
from typing import Any


def md(source: str) -> dict[str, Any]:
    lines = dedent(source).strip().splitlines(True)
    cell_id = hashlib.sha1(("markdown:" + "".join(lines)).encode()).hexdigest()[:12]
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": lines}


def code(source: str, tags: list[str] | None = None) -> dict[str, Any]:
    lines = dedent(source).strip().splitlines(True)
    cell_id = hashlib.sha1(("code:" + "".join(lines)).encode()).hexdigest()[:12]
    return {
        "cell_type": "code",
        "id": cell_id,
        "execution_count": None,
        "metadata": {"tags": tags or []},
        "outputs": [],
        "source": lines,
    }


SETUP = code(
    """
    # Local/VS Code and Colab bootstrap. No secret or API key is used.
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch

    IN_COLAB = "google.colab" in sys.modules
    if IN_COLAB:
        repo = Path("/content/ai-rookie-skill-tree-lab")
        if not repo.exists():
            subprocess.run(
                ["git", "clone", "https://github.com/chelsietao/ai-rookie-skill-tree-lab.git", str(repo)],
                check=True,
            )
        os.chdir(repo)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[full,dev]"], check=True)
    elif Path.cwd().name == "notebooks":
        os.chdir(Path.cwd().parent)

    ROOT = Path.cwd().resolve()
    DATA_DIR = ROOT / "data/generated/ci"
    METRICS_DIR = ROOT / "artifacts/metrics/ci"
    RUN_TRAINING = False  # Set True when you want to regenerate the chapter's metrics.
    assert (ROOT / "pyproject.toml").exists(), "Open the repository root or notebooks directory."
    print({"environment": "Colab" if IN_COLAB else "local/VS Code", "root": str(ROOT)})
    """,
    ["parameters"],
)


def ensure_data() -> dict[str, Any]:
    return code(
        """
        if not (DATA_DIR / "manifest.json").exists():
            subprocess.run(
                [sys.executable, "scripts/prepare_data.py", "--profile", "ci", "--output", str(DATA_DIR)],
                check=True,
            )
        subprocess.run([sys.executable, "scripts/validate_data.py", "--data-dir", str(DATA_DIR)], check=True)
        manifest = json.loads((DATA_DIR / "manifest.json").read_text())
        print(manifest["dataset_version"], manifest["rows"])
        """
    )


def run_or_load(script: str, metric: str, config: str | None = None) -> dict[str, Any]:
    command = [
        "sys.executable",
        f'"scripts/{script}"',
        '"--data-dir"',
        "str(DATA_DIR)",
        '"--profile"',
        '"ci"',
        '"--output"',
        f'str(METRICS_DIR / "{metric}.json")',
    ]
    command_text = ", ".join(command)
    config_line = f'print("config", Path("{config}").read_text())' if config else ""
    return code(
        f"""
        metric_path = METRICS_DIR / "{metric}.json"
        if RUN_TRAINING or not metric_path.exists():
            metric_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([{command_text}], check=True)
        result = json.loads(metric_path.read_text())
        {config_line}
        print("loaded", metric_path, "dataset", result.get("_run", {{}}).get("dataset_version"))
        """
    )


def book(title: str, purpose: str, cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cells": [
            md(
                f"""
                # {title}

                {purpose}

                This is a learning and interview notebook, not a thin launcher. It combines mechanism,
                executable evidence, diagnostics, limitations, and experiments. Run from top to bottom;
                keep `RUN_TRAINING=False` to inspect committed/CI results, or turn it on to reproduce them.
                """
            ),
            SETUP,
            *cells,
        ],
        "metadata": {
            "accelerator": "CPU",
            "colab": {"name": title, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


BOOKS: dict[str, dict[str, Any]] = {
    "01_data_foundations.ipynb": book(
        "01 · Data Foundations and Contracts",
        "Build one deterministic multimodal retail world and prove its identities, foreign keys, missingness, imbalance, shapes, checksums, and privacy boundary.",
        [
            md("""## Learning objectives

- Distinguish a data generator from a random table dump.
- Read schema, primary-key, foreign-key, label, and modality contracts.
- Measure missingness and imbalance before modeling.
- Explain why fixed seeds and file checksums are related but different reproducibility controls.
- Identify leakage risks before writing a model."""),
            md("""## World design

The portfolio profile contains 20,000 customers, 100,000 purchases, 50,000 browse events,
10,000 temporal sequences, 5,000 support messages, 5,000 images/bounding boxes, and a
customer–product bipartite graph. CI uses the same mechanisms at smaller scale.

The labels are deliberately learnable but noisy. That lets us detect broken training while still
retaining non-perfect metrics and failure cases."""),
            ensure_data(),
            code("""
            import pandas as pd
            import yaml

            schema = yaml.safe_load(Path("data/schemas/retail_world.yaml").read_text())
            display(pd.DataFrame(schema["tables"]["customers"]["columns"].items(), columns=["column", "contract"]))
            schema["modalities"]
            """),
            md("""## Relational tables

Primary keys answer “which record is this?” Foreign keys answer “which entity does it belong to?”
Validation must test both uniqueness and referential integrity; checking only `not null` is not enough."""),
            code("""
            customers = pd.read_csv(DATA_DIR / "customers.csv.gz")
            transactions = pd.read_csv(DATA_DIR / "transactions.csv.gz")
            views = pd.read_csv(DATA_DIR / "product_views.csv.gz")
            texts = pd.read_csv(DATA_DIR / "support_texts.csv.gz")
            display(customers.head())
            display(views.head())
            print({
                "unique_customer_pk": customers.customer_id.is_unique,
                "unique_transaction_pk": transactions.transaction_id.is_unique,
                "view_fk_valid": views.customer_id.isin(customers.customer_id).all(),
            })
            """),
            md("""## Missing data

`income` and `loyalty_score` have injected missingness. Median imputation is robust and simple,
but it shrinks variance and can hide informative missingness. In production, ask whether data are
MCAR, MAR, or MNAR and consider a missing-indicator feature."""),
            code("""
            import matplotlib.pyplot as plt
            import seaborn as sns

            missing = customers.isna().mean().sort_values(ascending=False)
            display(missing[missing > 0].rename("missing_rate").to_frame())
            ax = missing[missing > 0].plot.bar(title="Injected missingness")
            ax.set_ylabel("fraction missing")
            plt.show()
            """),
            code("""
            from sklearn.impute import SimpleImputer

            before = customers[["income", "loyalty_score"]].describe().T
            imputed = SimpleImputer(strategy="median").fit_transform(customers[["income", "loyalty_score"]])
            after = pd.DataFrame(imputed, columns=["income", "loyalty_score"]).describe().T
            display(before.add_prefix("before_" ).join(after.add_prefix("after_")))
            """),
            md("""## Imbalanced targets

Churn and anomalous transactions are rare by design. A classifier that predicts zero for every
record can have high accuracy. Always report positive rate, PR-AUC, recall, and a confusion matrix."""),
            code("""
            rates = pd.Series({
                "churn": customers.churned.mean(),
                "transaction_anomaly": transactions.is_anomaly.mean(),
                "view_conversion": views.converted.mean(),
                "positive_sentiment": texts.sentiment.mean(),
            })
            display(rates.rename("positive_rate").to_frame())
            rates.plot.bar(title="Label prevalence across tasks")
            plt.axhline(0.5, linestyle="--", color="grey")
            plt.show()
            """),
            md("""## Browse-event mechanism

Conversion probability depends on dwell time and page depth through a logistic mechanism. This is
a known ground truth for debugging feature engineering; it is not a claim about real shoppers."""),
            code("""
            summary = views.groupby(pd.qcut(views.dwell_seconds, 5, duplicates="drop"), observed=True).agg(
                conversion_rate=("converted", "mean"), mean_depth=("page_depth", "mean"), rows=("view_id", "size")
            )
            display(summary)
            sns.lineplot(data=views.sample(min(1000, len(views)), random_state=20260820), x="dwell_seconds", y="converted")
            plt.xscale("log")
            plt.show()
            """),
            md("""## NumPy modalities

Tabular CSV files are not enough for RNN, vision, detection, and graph practice. NPZ preserves typed
arrays; JSON stores variable-length bounding-box records; the edge CSV remains auditable."""),
            code("""
            import numpy as np

            sequences = np.load(DATA_DIR / "sequences.npz")
            images = np.load(DATA_DIR / "product_images.npz")
            boxes = json.loads((DATA_DIR / "detection_labels.json").read_text())
            edges = pd.read_csv(DATA_DIR / "graph_edges.csv.gz")
            print({"sequence_x": sequences["x"].shape, "sequence_y": sequences["y"].shape})
            print({"image_x": images["x"].shape, "image_y": images["y"].shape})
            print({"boxes": len(boxes), "unique_edges": len(edges)})
            """),
            code("""
            fig, axes = plt.subplots(2, 5, figsize=(12, 5))
            for axis, image, label in zip(axes.flat, images["x"][:10], images["y"][:10], strict=True):
                axis.imshow(image.transpose(1, 2, 0))
                axis.set_title(f"class={label}")
                axis.axis("off")
            plt.tight_layout()
            """),
            md("""## Determinism and integrity

A seed reproduces pseudo-random choices. A SHA-256 checksum proves that a concrete file has not
changed. The manifest versions the generator and records both controls."""),
            code("""
            checksum_frame = pd.DataFrame(
                [{"file": name, "sha256_prefix": digest[:16]} for name, digest in manifest["sha256"].items()]
            )
            display(checksum_frame)
            assert len(manifest["sha256"]) == 8
            assert manifest["contains_real_pii"] is False
            """),
            md("""## Leakage checklist

- Split entities before fitting imputers/scalers.
- Put preprocessing inside cross-validation Pipelines.
- Do not aggregate future `day` values into past features.
- Do not let the same customer leak across entity-level evaluation when measuring new-customer performance.
- Keep label-generation columns out of model features unless the task explicitly permits them."""),
            code("""
            print(Path("sql/sqlite/schema.sql").read_text())
            print("MySQL and MSSQL versions use the same customer/view relationship and indexed access path.")
            """),
            md("""## Exercises and interview checkpoint

1. Double missingness and add a missing indicator; compare CV ROC-AUC.
2. Make conversion rarer and compare accuracy with PR-AUC.
3. Design an out-of-time split using `day`.
4. Explain why synthetic IDs are not automatically anonymous if generated from real source rows.
5. State exactly what the checksums prove—and what they do not prove."""),
        ],
    ),
    "02_classical_ml.ipynb": book(
        "02 · Classical Supervised Learning",
        "Benchmark classification and regression families under one fold-safe preprocessing contract, then analyze imbalance, calibration, errors, latency, and model-selection tradeoffs.",
        [
            md("""## Learning objectives

- Compare linear, neighbor, margin, tree, ensemble, boosting, and MLP estimators fairly.
- Explain Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier score, and ECE.
- Interpret calibration separately from discrimination.
- Use Pipeline and cross-validation without preprocessing leakage.
- Select a model using more than the largest score."""),
            ensure_data(),
            run_or_load("run_classical.py", "classical", "configs/models/classical.yaml"),
            code("""
            import pandas as pd

            classification = pd.DataFrame(result["classification"]).T
            regression = pd.DataFrame(result["regression"]).T
            display(classification.sort_values("roc_auc", ascending=False))
            display(regression.sort_values("r2", ascending=False))
            """),
            md("""## Metric map

For binary predictions, precision is $TP/(TP+FP)$ and recall is $TP/(TP+FN)$. F1 is their harmonic
mean. ROC-AUC evaluates ranking across thresholds; PR-AUC focuses on the positive class. Brier score
is mean squared probability error. ECE bins probabilities and compares confidence with observed rate."""),
            code("""
            label_rate = pd.read_csv(DATA_DIR / "customers.csv.gz").churned.mean()
            baseline = {"all_negative_accuracy": 1 - label_rate, "random_pr_auc": label_rate}
            baseline
            """),
            md("""## Accuracy trap

Several models can exceed 95% accuracy by predicting almost no churn. Use the confusion matrix and
recall to see whether the model actually identifies positives."""),
            code("""
            view = classification[["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]].sort_values("roc_auc")
            view.plot.barh(figsize=(10, 7), title="Classification metrics: same split, different conclusions")
            plt.xlim(0, 1)
            plt.show()
            """),
            code("""
            best = result["best_classifier"]
            matrix = classification.loc[best, "confusion_matrix"]
            print("best by ROC-AUC", best, "confusion matrix", matrix)
            from IPython.display import Image, display
            display(Image(filename=result["diagnostic_figure"]))
            """),
            md("""## Calibration is not ranking

A model may rank churners well but output probabilities that are systematically too high or low.
Lower Brier/ECE is better; compare them only after confirming the score really is probability-like.
The SVC sigmoid margin here is explicitly an educational approximation, not fitted calibration."""),
            code("""
            calibration = classification[["roc_auc", "brier", "ece_10_bin"]].sort_values("brier")
            display(calibration)
            calibration.plot.scatter(x="roc_auc", y="brier", s=70, title="Discrimination vs calibration")
            for name, row in calibration.iterrows():
                plt.annotate(name, (row.roc_auc, row.brier), fontsize=8)
            plt.show()
            """),
            md("""## Fold-safe preprocessing

Numeric features use median imputation and scaling; categorical features use most-frequent
imputation and one-hot encoding. Because these transformations live inside Pipeline, every CV fold
fits its own statistics using training rows only."""),
            code("""
            from rookie_lab.classical.preprocessing import preprocessor

            print(preprocessor())
            print("CV folds", result["cross_validation"]["folds"])
            print("CV mean", result["cross_validation"]["logistic_roc_auc_mean"])
            """),
            md("""## Imbalanced-learn evidence

Random oversampling changes the training distribution to 50% positive. It must happen only inside
training folds in a real tuning workflow. Oversampling before the split duplicates information into
validation and produces optimistic results."""),
            code("""
            pd.Series(result["class_imbalance"]).plot.bar(title="Positive rate before/after resampling")
            plt.ylim(0, 0.55)
            plt.show()
            """),
            md("""## Regression task

MAE is in currency units and robust to large residuals; RMSE penalizes large misses more strongly;
$R^2$ compares residual variance with a mean-target baseline and can be negative."""),
            code("""
            display(regression[["mae", "rmse", "r2", "fit_seconds", "inference_ms_per_1000"]].sort_values("r2", ascending=False))
            """),
            md("""## Runtime and deployment tradeoff

Training speed, batch latency, calibration, interpretability, and dependency size matter. The best
ROC-AUC model is not automatically the correct product model."""),
            code("""
            efficiency = classification.explode("confusion_matrix") if False else classification
            efficiency.plot.scatter(x="inference_ms_per_1000", y="roc_auc", s=80, logx=True)
            for name, row in efficiency.iterrows():
                plt.annotate(name, (row.inference_ms_per_1000, row.roc_auc), fontsize=8)
            plt.title("ROC-AUC versus inference time")
            plt.show()
            """),
            md("""## Error-slice exercise

Load `artifacts/models/best_classical.joblib`, score the held-out set, and compare recall by region,
channel, and tenure quartile. A global metric can hide a failing subgroup. This repository does not
claim fairness from synthetic demographic slices."""),
            code("""
            print("Leaderboard")
            display(pd.DataFrame(result["classification_leaderboard"]))
            print("Local XGBoost backend:", result["xgboost_backend"])
            """),
            md("""## Exercises and interview checkpoint

1. Tune the decision threshold for recall ≥ 0.70 and report precision.
2. Wrap SVC in `CalibratedClassifierCV` and compare Brier/ECE.
3. Explain why Lasso and Ridge are almost tied on the synthetic spend target.
4. Replace random CV with an out-of-time split.
5. Choose one model for batch scoring and defend the choice without saying “highest AUC.”"""),
        ],
    ),
    "03_unsupervised.ipynb": book(
        "03 · Unsupervised Learning and AutoML",
        "Compare clustering assumptions, test stability, inspect PCA/t-SNE/UMAP, and preserve complete Grid/Random/Optuna search histories without preprocessing leakage.",
        [
            md("""## Learning objectives

- Explain the geometry assumed by K-Means, DBSCAN, and agglomerative clustering.
- Treat silhouette as one diagnostic rather than ground truth.
- Distinguish PCA's linear variance objective from t-SNE/UMAP neighborhood visualizations.
- Measure cluster stability across restarts.
- Compare exhaustive, random, and Bayesian/TPE search with a finite budget."""),
            ensure_data(),
            run_or_load("run_unsupervised.py", "unsupervised"),
            code("""
            import pandas as pd
            clustering = pd.DataFrame(result["unsupervised"]["clustering"]).T
            display(clustering)
            display(pd.DataFrame(result["automl"]["leaderboard"]))
            """),
            md("""## Why scaling comes first

Distance-based methods are dominated by large-scale columns. Median imputation and StandardScaler
are fitted before clustering. In supervised CV they must be inside the Pipeline; unsupervised
exploration still needs a clearly documented fit population."""),
            code("""
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler

            customers = pd.read_csv(DATA_DIR / "customers.csv.gz")
            columns = ["age", "tenure_days", "income", "visits_30d", "support_tickets", "loyalty_score"]
            raw = SimpleImputer(strategy="median").fit_transform(customers[columns])
            scaled = StandardScaler().fit_transform(raw)
            display(pd.DataFrame({"raw_std": raw.std(0), "scaled_std": scaled.std(0)}, index=columns))
            """),
            md("""## K-Means objective

K-Means alternates assignment and centroid updates to minimize within-cluster squared distance.
It favors convex, similarly scaled clusters and requires $k$ in advance."""),
            code("""
            import numpy as np

            def one_kmeans_step(values, centers):
                distances = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(2)
                labels = distances.argmin(1)
                new_centers = np.vstack([values[labels == index].mean(0) for index in range(len(centers))])
                return labels, new_centers

            demo = scaled[:100, :2]
            labels, centers = one_kmeans_step(demo, demo[[0, 25, 50, 75]])
            print("one assignment/update step", np.bincount(labels), centers.shape)
            """),
            md("""## DBSCAN and hierarchy

DBSCAN can mark sparse points as noise and discover non-convex shapes, but `eps` is scale-sensitive.
Agglomerative clustering builds a merge hierarchy; this experiment cuts it at four clusters."""),
            code("""
            for name, row in result["unsupervised"]["clustering"].items():
                print(name, "silhouette", row["silhouette"], "noise", row["noise_rate"], "sizes", row["cluster_sizes"])
            """),
            md("""## Stability

A visually pleasing clustering can be unstable. Five K-Means restarts are compared with pairwise
Adjusted Rand Index (ARI), which is permutation-invariant and chance-adjusted."""),
            code("""
            stability = result["unsupervised"]["stability"]
            display(pd.Series(stability["pairwise_ari"]).describe())
            pd.Series(stability["pairwise_ari"]).plot.hist(bins=8, title="Pairwise ARI across restarts")
            plt.show()
            """),
            md("""## PCA from linear algebra

After centering, PCA directions are right singular vectors of the data matrix. The first components
maximize explained variance but need not preserve nonlinear neighborhoods."""),
            code("""
            centered = scaled - scaled.mean(0)
            _, singular_values, vectors = np.linalg.svd(centered, full_matrices=False)
            explained = singular_values**2 / (len(centered) - 1)
            ratio = explained / explained.sum()
            print("manual first-two explained variance", ratio[:2].sum())
            print("recorded PCA result", result["unsupervised"]["dimension_reduction"]["pca_explained_variance"])
            """),
            code("""
            from IPython.display import Image, display
            display(Image(filename="artifacts/figures/dimension-reduction-ci.png"))
            """),
            md("""## Do not over-interpret t-SNE/UMAP

Axes have no direct business meaning, global distances can be distorted, and different seeds or
hyperparameters may change the picture. Use embeddings to generate hypotheses, not labels."""),
            md("""## AutoML search strategies

Grid is exhaustive over a small declared set. Random search spends budget on sampled combinations.
Optuna TPE uses previous trials to bias future proposals. All three still require a valid CV design."""),
            code("""
            histories = []
            for search in ("grid_search", "random_search", "optuna_tpe"):
                for row in result["automl"][search]["history"]:
                    histories.append({"search": search, "score": row.get("mean_cv_roc_auc", row.get("value")), "params": str(row["params"])})
            history = pd.DataFrame(histories)
            display(history)
            history.assign(trial=history.groupby("search").cumcount()).pivot(index="trial", columns="search", values="score").plot(marker="o")
            plt.ylabel("CV ROC-AUC")
            plt.show()
            """),
            code("""
            print("Leakage control:", result["automl"]["leakage_control"])
            print("Winner:", result["automl"]["leaderboard"][0])
            """),
            md("""## Exercises and interview checkpoint

1. Sweep DBSCAN `eps` and plot clusters/noise/silhouette together.
2. Repeat stability with bootstrap samples rather than seeds only.
3. Compare PCA before versus after scaling.
4. Explain why choosing the best CV score still has selection bias.
5. Add a nested-CV outer loop and report the difference."""),
        ],
    ),
    "04_deep_learning.ipynb": book(
        "04 · Deep Learning Fundamentals",
        "Use PyTorch and TensorFlow/Keras to study gradients, MLP/autoencoder/recurrent models, activation and normalization choices, optimizers, schedulers, class weighting, early stopping, and overfitting.",
        [
            md("""## Learning objectives

- Trace forward pass, loss, backpropagation, optimizer step, and scheduler step.
- Compare five activations, two initialization families, four normalization choices, and three optimizers.
- Read training/validation curves instead of reporting only final accuracy.
- Compare RNN/LSTM/GRU under controlled data and epochs.
- Use reconstruction error for anomaly ranking and threshold decisions."""),
            ensure_data(),
            md("""## Autograd in one cell

For $y=(wx+b)^2$, PyTorch builds a dynamic graph and accumulates gradients after `backward()`.
Optimizers then mutate parameters; `zero_grad()` prevents unintended accumulation."""),
            code("""
            import torch

            x = torch.tensor([2.0])
            weight = torch.tensor([3.0], requires_grad=True)
            bias = torch.tensor([1.0], requires_grad=True)
            loss = (weight * x + bias).pow(2).mean()
            loss.backward()
            print({"loss": loss.item(), "d_loss_d_weight": weight.grad.item(), "d_loss_d_bias": bias.grad.item()})
            """),
            run_or_load("run_deep.py", "deep", "configs/models/deep.yaml"),
            code("""
            import pandas as pd

            ablations = result["pytorch"]["tabular_mlp_ablations"]
            activation_table = pd.DataFrame(ablations["activations"]).T
            normalization_table = pd.DataFrame(ablations["normalization"]).T
            optimizer_table = pd.DataFrame(ablations["optimizers"]).T
            display(activation_table[["roc_auc", "best_validation_loss", "overfitting_gap"]])
            display(normalization_table[["roc_auc", "best_validation_loss", "overfitting_gap"]])
            display(optimizer_table[["roc_auc", "best_validation_loss", "overfitting_gap"]])
            """),
            md("""## Activations and initialization

ReLU is cheap but can die; LeakyReLU retains a negative slope; GELU gates smoothly; sigmoid/tanh
saturate. Kaiming initialization matches rectifiers; Xavier targets stable variance for symmetric
activations. Short synthetic runs may rank them noisily—inspect multiple seeds before generalizing."""),
            code("""
            import numpy as np
            import matplotlib.pyplot as plt

            values = torch.linspace(-5, 5, 300)
            functions = {
                "relu": torch.relu(values), "leaky_relu": torch.nn.functional.leaky_relu(values),
                "gelu": torch.nn.functional.gelu(values), "sigmoid": torch.sigmoid(values), "tanh": torch.tanh(values),
            }
            for name, output in functions.items():
                plt.plot(values, output, label=name)
            plt.legend(); plt.title("Activation functions"); plt.show()
            """),
            md("""## Normalization

BatchNorm uses batch statistics, LayerNorm normalizes within each example, and GroupNorm divides
channels/features into groups. Small batches can make BatchNorm noisy; tabular GroupNorm here is an
educational parity exercise rather than a default recommendation."""),
            code("""
            normalization_table["roc_auc"].plot.bar(title="Normalization ablation")
            plt.ylim(0, 1)
            plt.show()
            """),
            md("""## Optimizers

SGD with momentum has simple dynamics, Adam adapts first/second moments, and RMSProp scales by a
moving squared-gradient average. A fair comparison should tune learning rate per optimizer; this
controlled quick run intentionally holds most other choices fixed."""),
            code("""
            optimizer_table[["roc_auc", "best_validation_loss"]].plot.bar(subplots=True, figsize=(9, 6), title=["ROC-AUC", "Validation loss"])
            plt.tight_layout(); plt.show()
            """),
            md("""## Learning-rate schedules

Warmup gradually increases the learning rate, StepLR drops it discretely, and cosine annealing
decays smoothly. The recorded trace proves scheduler behavior rather than only naming it."""),
            code("""
            schedulers = ablations["schedulers"]
            for name, values in schedulers.items():
                plt.plot(values["learning_rates"], marker="o", label=name)
            plt.xlabel("epoch"); plt.ylabel("learning rate"); plt.legend(); plt.show()
            """),
            md("""## Early stopping and overfitting

The best validation checkpoint is restored. `overfitting_gap = final validation loss - final train
loss`; a large positive gap is a warning, not a proof. Curves and multiple seeds are stronger evidence."""),
            code("""
            chosen = ablations["class_weighting"]
            plt.plot(chosen["train_loss"], marker="o", label="train")
            plt.plot(chosen["validation_loss"], marker="o", label="validation")
            plt.title(f"Class-weighted MLP; gap={chosen['overfitting_gap']:.3f}")
            plt.legend(); plt.show()
            """),
            md("""## Class weighting

Inverse-frequency weights increase loss contribution from rare churners. They can improve ranking or
recall while reducing threshold accuracy and making optimization less stable. Tune the decision
threshold after training rather than assuming 0.5 is correct."""),
            code("""
            print("class-weighted", {key: chosen[key] for key in ("accuracy", "roc_auc", "class_weighted", "overfitting_gap")})
            """),
            md("""## Sequence models

RNN repeatedly transforms hidden state; LSTM adds input/forget/output gates and a cell state; GRU
uses update/reset gates with fewer parameters. The synthetic target depends on recent amount trend."""),
            code("""
            pytorch_sequence = pd.DataFrame(result["pytorch"]["sequence_models"]).T
            keras_sequence = pd.DataFrame(result["tensorflow"]["sequence_models"]).T
            display(pytorch_sequence[["accuracy", "roc_auc", "overfitting_gap"]])
            display(keras_sequence[["roc_auc", "epochs_ran", "parameters"]])
            """),
            md("""## Framework parity

PyTorch and Keras use the same generated sequences but not identical initialization, architecture,
batch order, or optimizer implementation. Compare workflow and trends; do not treat score differences
as a framework benchmark."""),
            code("""
            print("Keras tabular MLP", result["tensorflow"]["tabular_mlp"])
            """),
            md("""## Autoencoder anomaly detection

The autoencoder learns normal transaction reconstruction. Reconstruction error ranks anomalies;
the 99th percentile of normal error creates an explicit threshold with precision/recall tradeoff."""),
            code("""
            anomaly = result["pytorch"]["autoencoder"]
            display(pd.Series(anomaly).drop("training_loss"))
            plt.plot(anomaly["training_loss"], marker="o"); plt.title("Autoencoder epoch loss"); plt.show()
            """),
            md("""## TensorBoard

Run `tensorboard --logdir artifacts/tensorboard` locally. TensorBoard is useful for synchronized
curves and comparisons, but event files do not replace versioned configs and JSON summaries."""),
            md("""## Exercises and interview checkpoint

1. Repeat every ablation across five seeds and add confidence intervals.
2. Tune optimizer-specific learning rates.
3. Plot threshold precision/recall for autoencoder error.
4. Create an intentionally overfit MLP and diagnose it from curves.
5. Explain why the quick-profile Keras scores are not evidence that one framework is superior."""),
        ],
    ),
    "05_computer_vision.ipynb": book(
        "05 · Computer Vision Architecture Zoo and Detection",
        "Study convolution, pooling, augmentation, seven educational architecture families, a transfer-learning mechanics proxy, one-object detection, IoU, precision/recall, mAP, efficiency, and visual errors.",
        [
            md("""## Learning objectives

- Implement and visualize convolution/pooling behavior.
- Locate the defining block in VGG, ResNet, MobileNet, EfficientNet, SqueezeNet, and ViT.
- Compare accuracy, parameters, runtime, and training curves.
- Compute IoU and distinguish mAP@0.50 from mAP@0.50:0.95.
- Inspect detection errors instead of relying on a single score."""),
            ensure_data(),
            code("""
            import numpy as np
            import matplotlib.pyplot as plt

            bundle = np.load(DATA_DIR / "product_images.npz")
            images, labels = bundle["x"], bundle["y"]
            fig, axes = plt.subplots(2, 6, figsize=(12, 4))
            for axis, image, label in zip(axes.flat, images[:12], labels[:12], strict=True):
                axis.imshow(image.transpose(1, 2, 0)); axis.set_title(f"class {label}"); axis.axis("off")
            plt.tight_layout()
            """),
            md("""## Convolution

A convolution kernel shares weights over spatial locations. Padding controls output size; stride
controls downsampling; channels let filters combine color and learned feature maps."""),
            code("""
            sample = torch.tensor(images[0:1].astype("float32") / 255)
            edge_kernel = torch.tensor([[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]])
            kernel = torch.zeros(1, 3, 3, 3)
            kernel[0, 0] = edge_kernel
            edges = torch.nn.functional.conv2d(sample, kernel, padding=1)
            fig, axes = plt.subplots(1, 2, figsize=(7, 3))
            axes[0].imshow(images[0].transpose(1, 2, 0)); axes[0].set_title("input")
            axes[1].imshow(edges[0, 0], cmap="coolwarm"); axes[1].set_title("edge response")
            plt.show()
            """),
            md("""## Pooling and augmentation

Max pooling preserves strong local responses; average pooling summarizes a region. Horizontal flips
and Gaussian noise add invariances here. An augmentation is valid only if it preserves the label."""),
            code("""
            max_pool = torch.nn.functional.max_pool2d(sample, 2)
            avg_pool = torch.nn.functional.avg_pool2d(sample, 2)
            print(sample.shape, max_pool.shape, avg_pool.shape)
            """),
            run_or_load("run_vision.py", "vision", "configs/models/vision.yaml"),
            code("""
            import pandas as pd
            zoo = pd.DataFrame(result["classification"]).T
            display(zoo[["accuracy", "parameters", "fit_seconds", "batch32_inference_ms"]].sort_values("accuracy", ascending=False))
            """),
            md("""## Architecture map

- **VGG:** repeated 3×3 convolution stacks.
- **ResNet:** identity skip connections stabilize deeper optimization.
- **MobileNet:** depthwise spatial convolution plus pointwise channel mixing.
- **EfficientNet-style:** depthwise blocks, smooth activation, squeeze/excitation.
- **SqueezeNet:** fire modules squeeze then expand channels.
- **ViT:** patch embedding, position encoding, multi-head self-attention.

These are intentionally tiny teaching variants, not canonical ImageNet models."""),
            code("""
            from rookie_lab.vision.models import model_zoo
            for name, model in model_zoo().items():
                print("\\n", name, "parameters", sum(p.numel() for p in model.parameters()))
                print(model)
            """),
            md("""## Quick-profile failures matter

Short training can leave MobileNet/EfficientNet underfit even when simpler models reach perfect
synthetic accuracy. This is evidence about optimization budget, not architecture quality."""),
            code("""
            zoo["accuracy"].sort_values().plot.barh(title="Quick-profile architecture accuracy")
            plt.xlim(0, 1.05); plt.show()
            for name, row in result["classification"].items():
                plt.plot(row["training_loss"], marker="o", label=name)
            plt.yscale("log"); plt.legend(ncol=2); plt.title("Training loss"); plt.show()
            """),
            code("""
            display(pd.DataFrame(result["pooling_ablation"]).T[["accuracy", "fit_seconds", "training_loss"]])
            """),
            md("""## Transfer-learning mechanics proxy

The experiment pretrains a small CNN for two epochs, freezes its convolutional representation,
resets the classifier, and adapts the head for one epoch. Because source and target are the same
synthetic domain, this demonstrates mechanics—not external knowledge transfer."""),
            code("""
            transfer = result["transfer_learning_proxy"]
            display(pd.DataFrame({
                "frozen_transfer": transfer["frozen_feature_transfer"],
                "scratch": transfer["one_epoch_from_scratch"],
            }).T[["accuracy", "fit_seconds", "training_loss"]])
            print(transfer["limitation"])
            """),
            md("""## Intersection over Union

$IoU = area(B_p \\cap B_t) / area(B_p \\cup B_t)$. A correct class with a poorly localized box can
still fail detection. Higher IoU thresholds demand more precise localization."""),
            code("""
            from rookie_lab.vision.experiment import _iou
            prediction = torch.tensor([[0.1, 0.1, 0.8, 0.8]])
            target = torch.tensor([[0.2, 0.2, 0.9, 0.9]])
            print("example IoU", _iou(prediction, target).item())
            """),
            md("""## Detection metrics

This dataset has exactly one prediction and one target per image. AP is averaged across classes and
IoU thresholds under that simplified protocol. It is not COCO multi-object mAP with NMS/crowds."""),
            code("""
            detection = result["object_detection"]
            display(pd.Series({key: value for key, value in detection.items() if not isinstance(value, dict) and key != "error_figure"}))
            pd.Series(detection["ap_by_iou_threshold"]).plot(marker="o", title="AP versus IoU threshold")
            plt.ylabel("mean AP"); plt.show()
            """),
            code("""
            from IPython.display import Image, display
            display(Image(filename=detection["error_figure"]))
            """),
            md("""## Efficiency frontier

Parameter count is a storage proxy; measured batch latency depends on hardware, threads, and runtime.
Compare both with accuracy rather than assuming smaller always means faster."""),
            code("""
            zoo.plot.scatter(x="parameters", y="batch32_inference_ms", s=100, logx=True, logy=True)
            for name, row in zoo.iterrows():
                plt.annotate(name, (row.parameters, row.batch32_inference_ms))
            plt.title("Model size versus measured latency"); plt.show()
            """),
            md("""## Exercises and interview checkpoint

1. Run five seeds and plot accuracy confidence intervals.
2. Add brightness augmentation and justify label preservation.
3. Replace the one-object protocol with multiple objects and NMS.
4. Explain why mAP@0.50 is much higher than mAP@0.50:0.95.
5. Identify one defining source-code block for every architecture."""),
        ],
    ),
    "06_nlp_transformers.ipynb": book(
        "06 · NLP, Recurrent Models, BERT, and GPT",
        "Move from TF-IDF and local tokenization through LSTM/GRU baselines, self-attention, masked-language pretraining, classification fine-tuning, causal generation, perplexity, and honest synthetic limitations.",
        [
            md("""## Learning objectives

- Establish TF-IDF baselines before neural models.
- Inspect deterministic local tokenization and padding masks.
- Compare LSTM/GRU, bidirectional BERT-style attention, and causal GPT-style attention.
- Distinguish pretraining objective from downstream fine-tuning.
- Interpret perplexity and generated samples without claiming general language ability."""),
            ensure_data(),
            code("""
            import pandas as pd
            texts = pd.read_csv(DATA_DIR / "support_texts.csv.gz")
            display(texts.sample(10, random_state=20260820))
            display(pd.crosstab(texts.intent, texts.sentiment, normalize="index"))
            """),
            md("""## TF-IDF baseline

TF-IDF increases weight for terms frequent in a document but uncommon in the corpus. Logistic
regression then learns a linear boundary. On templated synthetic text this can be extremely strong;
that does not imply semantic understanding."""),
            run_or_load("run_nlp.py", "nlp"),
            code("""
            print("TF-IDF", result["tfidf"])
            """),
            md("""## Local tokenizer

The vocabulary is built only from generated text. `[CLS]` starts a sequence, `[SEP]` ends it,
`[PAD]` fills fixed length, `[UNK]` handles unseen words, and `[MASK]` supports MLM."""),
            code("""
            from rookie_lab.nlp.tokenizer import LocalTokenizer
            tokenizer = LocalTokenizer(texts.text.tolist())
            ids, mask = tokenizer.batch(texts.text.head(3).tolist())
            print("vocabulary", len(tokenizer.tokens))
            display(pd.DataFrame(ids.numpy()))
            display(pd.DataFrame(mask.numpy()))
            print(tokenizer.decode(ids[0].tolist()))
            """),
            md("""## LSTM and GRU text baselines

Word embeddings feed recurrent cells. LSTM uses a cell state and three gates; GRU is smaller with
update/reset gates. The last hidden state drives sentiment classification."""),
            code("""
            recurrent = pd.DataFrame(result["recurrent_baselines"]).T
            display(recurrent[["accuracy", "roc_auc", "parameters", "training_loss"]])
            """),
            md("""## Scaled dot-product self-attention

$Attention(Q,K,V)=softmax(QK^T/\\sqrt{d_k})V$. Multi-head attention learns several projections;
padding and causal masks determine which tokens are visible."""),
            code("""
            import torch

            torch.manual_seed(7)
            tokens = torch.randn(1, 4, 8)
            scores = tokens @ tokens.transpose(-1, -2) / np.sqrt(tokens.shape[-1])
            weights = scores.softmax(-1)
            attended = weights @ tokens
            print("weights sum", weights.sum(-1))
            display(pd.DataFrame(weights[0].numpy()))
            """),
            md("""## BERT-style encoder

BERT sees left and right context. Here one encoder block with four heads is first optimized on masked
tokens, then copied into a sentiment classifier. The model is random-initialized and tiny."""),
            code("""
            bert = result["bert"]
            print(bert["architecture"])
            print(bert["attention"])
            display(pd.DataFrame({"scratch": bert["from_scratch"], "pretrained": bert["pretrained_then_finetuned"]}).T)
            """),
            md("""## Pretraining versus scratch

One CI MLM update can have zero meaningful loss change and does not guarantee downstream benefit.
The experiment intentionally records that limitation. Portfolio mode uses more updates but remains
far from foundation-model pretraining."""),
            code("""
            plt.plot(bert["masked_language_pretraining_loss"], marker="o")
            plt.title("Masked-language pretraining loss"); plt.xlabel("update"); plt.show()
            """),
            md("""## GPT-style decoder and causal mask

GPT predicts the next token and cannot attend to future positions. The upper-triangular causal mask
sets future attention logits to negative infinity before softmax."""),
            code("""
            length = 8
            causal_mask = torch.triu(torch.ones(length, length), diagonal=1).bool()
            display(pd.DataFrame(causal_mask.numpy()).style.background_gradient(cmap="Reds"))
            """),
            md("""## Perplexity and generation

Perplexity is $exp(cross\\ entropy)$. Lower is better on the same tokenization/data, but it does not
measure factuality, helpfulness, or safety. Always inspect outputs."""),
            code("""
            gpt = result["gpt"]
            print("perplexity", gpt["perplexity"])
            display(pd.Series(gpt["greedy_generations"], name="generation").to_frame())
            plt.plot(gpt["training_loss"]); plt.title("Causal-LM batch loss"); plt.show()
            """),
            md("""## NLU versus NLG

Intent/sentiment are NLU-style classification tasks. Causal generation is NLG. A model can score
perfectly on synthetic intent labels while producing repetitive text; those are different abilities."""),
            code("""
            summary = pd.DataFrame([
                {"model": "TF-IDF", "sentiment_roc_auc": result["tfidf"]["sentiment_roc_auc"]},
                *({"model": name.upper(), "sentiment_roc_auc": values["roc_auc"]} for name, values in result["recurrent_baselines"].items()),
                {"model": "Tiny BERT scratch", "sentiment_roc_auc": bert["from_scratch"]["roc_auc"]},
                {"model": "Tiny BERT pretrained", "sentiment_roc_auc": bert["pretrained_then_finetuned"]["roc_auc"]},
            ])
            display(summary.sort_values("sentiment_roc_auc", ascending=False))
            """),
            md("""## Exercises and interview checkpoint

1. Add a validation split to GPT and report validation perplexity.
2. Increase MLM steps and test whether transfer consistently helps across seeds.
3. Show one attention matrix for a sentence and state why it is not a causal explanation.
4. Add character n-gram TF-IDF and compare robustness to misspellings.
5. Explain why perfect synthetic classification is not evidence of general NLP capability."""),
        ],
    ),
    "07_gnn.ipynb": book(
        "07 · Graph Neural Networks",
        "Represent a customer–product bipartite graph, normalize sparse adjacency, implement message passing, train a manual PyTorch GCN, compare node/link tasks with traditional baselines, and retain negative results.",
        [
            md("""## Learning objectives

- Map external IDs to contiguous node indices.
- Build undirected sparse adjacency with self-loops and symmetric normalization.
- Explain graph convolution as neighbor aggregation plus a learned linear map.
- Separate node classification from link prediction evaluation.
- Compare GNN scores against degree/tabular baselines before claiming graph value."""),
            ensure_data(),
            code("""
            import pandas as pd
            edges = pd.read_csv(DATA_DIR / "graph_edges.csv.gz")
            customers = pd.read_csv(DATA_DIR / "customers.csv.gz")
            display(edges.head())
            print({"customers": customers.customer_id.nunique(), "products": edges.product_id.nunique(), "edges": len(edges)})
            """),
            md("""## Bipartite representation

Edges connect customer nodes only to product nodes. There are no customer–customer or product–product
edges in the raw graph. Transactions and browse events both contribute unique relationships."""),
            code("""
            customer_degree = edges.groupby("customer_id").size()
            product_degree = edges.groupby("product_id").size()
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
            customer_degree.hist(ax=axes[0], bins=30); axes[0].set_title("Customer degree")
            product_degree.hist(ax=axes[1], bins=30); axes[1].set_title("Product degree")
            plt.tight_layout(); plt.show()
            """),
            md("""## Normalized adjacency

The GCN uses $\\hat A=D^{-1/2}(A+I)D^{-1/2}$. Self-loops retain a node's representation; symmetric
degree normalization prevents high-degree neighbors from dominating every update."""),
            code("""
            import torch
            from rookie_lab.graph.gcn import normalized_adjacency

            tiny_edges = torch.tensor([[0, 1], [2, 2]])
            tiny_adjacency = normalized_adjacency(tiny_edges, 3)
            display(pd.DataFrame(tiny_adjacency.to_dense().numpy()).round(3))
            assert torch.allclose(tiny_adjacency.to_dense(), tiny_adjacency.to_dense().T)
            """),
            md("""## Message passing by hand

One graph-convolution layer computes normalized neighbor aggregation and then a linear projection.
Sparse multiplication avoids materializing the full adjacency for larger graphs."""),
            code("""
            features = torch.eye(3)
            aggregated = torch.sparse.mm(tiny_adjacency, features)
            display(pd.DataFrame(aggregated.numpy()).round(3))
            """),
            run_or_load("run_graph.py", "graph"),
            code("""
            print(result["graph_statistics"])
            print(result["implementation"])
            """),
            md("""## Node classification

Customer features occupy the first six dimensions; product nodes receive synthetic features. A
two-layer GCN predicts customer churn. The comparison baseline uses the same tabular customer
features plus node degree in class-weighted logistic regression."""),
            code("""
            node_comparison = pd.Series({
                "GCN": result["node_classification"]["roc_auc"],
                "Logistic + degree": result["traditional_baseline"]["node_roc_auc"],
            })
            display(node_comparison.rename("node_roc_auc").to_frame())
            node_comparison.plot.bar(title="Does message passing beat the baseline?"); plt.ylim(0, 1); plt.show()
            """),
            md("""## Link prediction and negative sampling

Positive links are observed customer–product pairs. Negatives are sampled from absent pairs. The
GCN scores an edge with the dot product of endpoint embeddings. Sampling policy changes task
difficulty and must be documented."""),
            code("""
            display(pd.Series(result["link_prediction"], name="roc_auc").to_frame())
            """),
            md("""## Honest negative result

In the quick run, the GCN link score can be below the degree-popularity baseline. That means the
current product features, graph generator, objective, or training budget does not justify claiming
superior graph link prediction. Keeping this result is stronger research evidence than hiding it."""),
            code("""
            loss = result["training_loss"]["curve"]
            plt.plot(loss); plt.title("Joint node + link reconstruction loss"); plt.xlabel("epoch"); plt.show()
            print("loss change", loss[-1] - loss[0])
            """),
            md("""## Transductive limitation

All nodes are present while training, even though only training customer labels are used. This is
transductive learning. A true new-customer/product deployment needs inductive features and an
evaluation split that withholds nodes, not only labels or edges."""),
            code("""
            print({"scope": result["scope"], "nodes": result["nodes"], "edges": result["edges"]})
            """),
            md("""## Exercises and interview checkpoint

1. Use hard negatives with popular products rather than uniform negatives.
2. Hold out entire customers and explain the inductive changes required.
3. Add product category features and test link ROC-AUC.
4. Compare one versus two graph-convolution layers for oversmoothing.
5. State why a falling joint loss does not guarantee better node and link metrics."""),
        ],
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("notebooks"))
    parser.add_argument(
        "--check",
        type=Path,
        help="Verify that committed notebooks exactly match this reviewable source.",
    )
    args = parser.parse_args()
    if args.check:
        mismatches = []
        for filename, payload in BOOKS.items():
            expected = json.dumps(payload, indent=1, ensure_ascii=False) + "\n"
            path = args.check / filename
            if not path.exists() or path.read_text() != expected:
                mismatches.append(filename)
        if mismatches:
            raise SystemExit(f"notebooks are stale or missing: {', '.join(mismatches)}")
        print(f"verified {len(BOOKS)} generated notebooks in {args.check}")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, payload in BOOKS.items():
        (args.output / filename).write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {len(BOOKS)} deep-learning-path notebooks to {args.output}")


if __name__ == "__main__":
    main()
