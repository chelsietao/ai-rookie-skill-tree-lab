"""Generate a coherent, fictional retail world for every lab."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class DataConfig:
    profile: str
    seed: int
    customers: int
    transactions: int
    views: int
    sequences: int
    texts: int
    images: int
    products: int
    sequence_length: int
    image_size: int

    @classmethod
    def from_yaml(cls, path: Path) -> DataConfig:
        return cls(**yaml.safe_load(path.read_text()))


REGIONS = np.array(["north", "central", "south", "east"])
CHANNELS = np.array(["organic", "search", "referral", "partner"])
CATEGORIES = np.array(["food", "home", "beauty", "sports", "books"])
POSITIVE = ("helpful service", "fast delivery", "great quality", "easy checkout")
NEGATIVE = ("late delivery", "damaged product", "refund needed", "poor support")
INTENTS = ("delivery", "refund", "product", "account")


def _write_csv_gzip(frame: pd.DataFrame, path: Path) -> None:
    """Write byte-reproducible gzip by removing the current-time header."""
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write an np.load-compatible archive with fixed ZIP timestamps and ordering."""
    with zipfile.ZipFile(path, "w") as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.save(buffer, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compresslevel=9)


def _customers(config: DataConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = config.customers
    age = rng.integers(18, 75, n)
    tenure = rng.integers(1, 1800, n)
    visits = rng.poisson(7, n)
    support = rng.poisson(0.8, n)
    income = rng.lognormal(10.5, 0.45, n)
    loyalty = np.clip(0.25 + 0.03 * visits - 0.05 * support + rng.normal(0, 0.18, n), 0, 1)
    churn_logit = -2.7 + 0.65 * support - 0.12 * visits - 0.001 * tenure + 1.2 * (loyalty < 0.25)
    churn_p = 1 / (1 + np.exp(-churn_logit))
    churn = rng.binomial(1, churn_p)
    annual_spend = np.maximum(
        20,
        80 + 18 * visits + 0.004 * income + 0.06 * tenure - 45 * support + rng.normal(0, 120, n),
    )
    frame = pd.DataFrame(
        {
            "customer_id": [f"cus_{i:06d}" for i in range(n)],
            "age": age,
            "tenure_days": tenure,
            "region": rng.choice(REGIONS, n, p=[0.42, 0.22, 0.29, 0.07]),
            "channel": rng.choice(CHANNELS, n),
            "income": income.round(2),
            "visits_30d": visits,
            "support_tickets": support,
            "loyalty_score": loyalty.round(5),
            "churned": churn,
            "annual_spend": annual_spend.round(2),
        }
    )
    frame.loc[rng.random(n) < 0.12, "loyalty_score"] = np.nan
    frame.loc[rng.random(n) < 0.05, "income"] = np.nan
    return frame


def _transactions(config: DataConfig, customers: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = config.transactions
    customer_idx = rng.integers(0, len(customers), n)
    category = rng.choice(CATEGORIES, n)
    base = np.array([18, 55, 35, 70, 25])
    category_map = {name: value for name, value in zip(CATEGORIES, base, strict=True)}
    amount = np.array([category_map[value] for value in category]) * rng.lognormal(0, 0.55, n)
    anomaly = rng.random(n) < 0.018
    amount[anomaly] *= rng.uniform(4, 9, anomaly.sum())
    return pd.DataFrame(
        {
            "transaction_id": [f"txn_{i:08d}" for i in range(n)],
            "customer_id": customers.iloc[customer_idx]["customer_id"].to_numpy(),
            "product_id": [f"prd_{value:05d}" for value in rng.integers(0, config.products, n)],
            "day": rng.integers(0, 365, n),
            "category": category,
            "amount": amount.round(2),
            "is_anomaly": anomaly.astype(int),
        }
    )


def _views(config: DataConfig, customers: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Create browse events with a learnable but imperfect conversion mechanism."""
    n = config.views
    customer_idx = rng.integers(0, len(customers), n)
    product_idx = rng.integers(0, config.products, n)
    dwell = np.clip(rng.lognormal(3.2, 0.75, n), 2, 900)
    page_depth = rng.integers(1, 9, n)
    conversion_logit = -4.0 + 0.012 * dwell + 0.18 * page_depth
    converted = rng.binomial(1, 1 / (1 + np.exp(-conversion_logit)))
    return pd.DataFrame(
        {
            "view_id": [f"view_{i:08d}" for i in range(n)],
            "customer_id": customers.iloc[customer_idx]["customer_id"].to_numpy(),
            "product_id": [f"prd_{value:05d}" for value in product_idx],
            "day": rng.integers(0, 365, n),
            "dwell_seconds": dwell.round(2),
            "page_depth": page_depth,
            "converted": converted,
        }
    )


def _sequences(config: DataConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n, length = config.sequences, config.sequence_length
    latent = rng.normal(0, 1, (n, 1, 1))
    trend = np.linspace(-0.8, 0.8, length)[None, :, None]
    amounts = np.maximum(0, 30 + 12 * latent + 8 * trend + rng.normal(0, 8, (n, length, 1)))
    activity = rng.poisson(np.clip(2.5 + latent + trend, 0.2, 8), (n, length, 1))
    category = rng.integers(0, len(CATEGORIES), (n, length, 1)) / len(CATEGORIES)
    sequence = np.concatenate([amounts / 100, activity / 10, category], axis=2).astype("float32")
    target = ((amounts[:, -3:, 0].mean(1) + 12 * latent[:, 0, 0]) > 45).astype("int64")
    return sequence, target


def _texts(config: DataConfig, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(config.texts):
        sentiment = int(rng.random() > 0.45)
        intent = INTENTS[index % len(INTENTS)]
        phrase = rng.choice(POSITIVE if sentiment else NEGATIVE)
        text = f"{phrase}; question about {intent}; order {index % 97}"
        rows.append(
            {"text_id": f"txt_{index:06d}", "text": text, "sentiment": sentiment, "intent": intent}
        )
    return pd.DataFrame(rows)


def _images(
    config: DataConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    size, n = config.image_size, config.images
    images = np.zeros((n, 3, size, size), dtype="uint8")
    labels = np.zeros(n, dtype="int64")
    boxes: list[dict[str, object]] = []
    colors = ((235, 70, 70), (70, 210, 90), (70, 110, 235))
    for index in range(n):
        label = index % 3
        margin = int(rng.integers(3, 9))
        x1, y1 = margin, int(rng.integers(3, 9))
        x2, y2 = size - margin, size - int(rng.integers(3, 9))
        image = Image.new("RGB", (size, size), (20, 20, 25))
        draw = ImageDraw.Draw(image)
        if label == 0:
            draw.rectangle((x1, y1, x2, y2), fill=colors[label])
        elif label == 1:
            draw.ellipse((x1, y1, x2, y2), fill=colors[label])
        else:
            draw.polygon(((size // 2, y1), (x2, y2), (x1, y2)), fill=colors[label])
        array = np.asarray(image).transpose(2, 0, 1)
        noise = rng.normal(0, 8, array.shape)
        images[index] = np.clip(array + noise, 0, 255).astype("uint8")
        labels[index] = label
        boxes.append({"image_id": index, "label": int(label), "bbox_xyxy": [x1, y1, x2, y2]})
    return images, labels, boxes


def generate(config: DataConfig, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    customers = _customers(config, rng)
    transactions = _transactions(config, customers, rng)
    views = _views(config, customers, rng)
    sequences, sequence_targets = _sequences(config, rng)
    texts = _texts(config, rng)
    images, image_labels, boxes = _images(config, rng)
    graph = (
        pd.concat(
            [transactions[["customer_id", "product_id"]], views[["customer_id", "product_id"]]],
            ignore_index=True,
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )

    _write_csv_gzip(customers, output / "customers.csv.gz")
    _write_csv_gzip(transactions, output / "transactions.csv.gz")
    _write_csv_gzip(views, output / "product_views.csv.gz")
    _write_csv_gzip(texts, output / "support_texts.csv.gz")
    _write_csv_gzip(graph, output / "graph_edges.csv.gz")
    _write_npz(output / "sequences.npz", x=sequences, y=sequence_targets)
    _write_npz(output / "product_images.npz", x=images, y=image_labels)
    (output / "detection_labels.json").write_text(json.dumps(boxes, indent=2))

    data_files = [
        "customers.csv.gz",
        "transactions.csv.gz",
        "product_views.csv.gz",
        "support_texts.csv.gz",
        "graph_edges.csv.gz",
        "sequences.npz",
        "product_images.npz",
        "detection_labels.json",
    ]
    checksums = {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in data_files}
    manifest: dict[str, object] = {
        "generator": "rookie-lab-synthetic-v2",
        "dataset_version": f"{config.profile}-{config.seed}-v2",
        "fictional": True,
        "contains_real_pii": False,
        "config": asdict(config),
        "rows": {
            "customers": len(customers),
            "transactions": len(transactions),
            "views": len(views),
            "sequences": len(sequences),
            "texts": len(texts),
            "images": len(images),
            "graph_edges": len(graph),
        },
        "sha256": checksums,
        "label_rates": {
            "churn": float(customers["churned"].mean()),
            "transaction_anomaly": float(transactions["is_anomaly"].mean()),
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
