"""TensorFlow/Keras parity experiment using the same local sequence data."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def run(data_dir: Path, profile: str, seed: int = 20260820) -> dict[str, object]:
    import tensorflow as tf
    from tensorflow import keras

    tf.keras.utils.set_random_seed(seed)
    payload = np.load(data_dir / "sequences.npz")
    x, y = payload["x"], payload["y"]
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=seed
    )
    epochs = 2 if profile == "ci" else 5
    results: dict[str, object] = {}
    for name, layer in {
        "simple_rnn": keras.layers.SimpleRNN(20),
        "lstm": keras.layers.LSTM(20),
        "gru": keras.layers.GRU(20),
    }.items():
        model = keras.Sequential(
            [keras.Input(shape=x.shape[1:]), layer, keras.layers.Dense(1, activation="sigmoid")]
        )
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        callbacks = [keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)]
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_valid, y_valid),
            epochs=epochs,
            batch_size=64,
            callbacks=callbacks,
            verbose=0,
        )
        probability = model.predict(x_valid, verbose=0).reshape(-1)
        results[name] = {
            "roc_auc": float(roc_auc_score(y_valid, probability)),
            "epochs_ran": len(history.history["loss"]),
            "train_loss": [float(value) for value in history.history["loss"]],
            "validation_loss": [float(value) for value in history.history["val_loss"]],
            "parameters": int(model.count_params()),
        }

    frame = pd.read_csv(data_dir / "customers.csv.gz")
    columns = ["age", "tenure_days", "income", "visits_30d", "support_tickets", "loyalty_score"]
    features = SimpleImputer(strategy="median").fit_transform(frame[columns])
    features = StandardScaler().fit_transform(features).astype("float32")
    target = frame["churned"].to_numpy("float32")
    x_tab_train, x_tab_valid, y_tab_train, y_tab_valid = train_test_split(
        features, target, test_size=0.25, stratify=target, random_state=seed
    )
    mlp = keras.Sequential(
        [
            keras.Input(shape=(features.shape[1],)),
            keras.layers.Dense(32, activation="relu", kernel_initializer="he_uniform"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.15),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    mlp.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    mlp_history = mlp.fit(
        x_tab_train,
        y_tab_train,
        validation_data=(x_tab_valid, y_tab_valid),
        epochs=epochs,
        batch_size=64,
        callbacks=[keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)],
        verbose=0,
    )
    mlp_probability = mlp.predict(x_tab_valid, verbose=0).reshape(-1)
    tabular = {
        "roc_auc": float(roc_auc_score(y_tab_valid, mlp_probability)),
        "epochs_ran": len(mlp_history.history["loss"]),
        "train_loss": [float(value) for value in mlp_history.history["loss"]],
        "validation_loss": [float(value) for value in mlp_history.history["val_loss"]],
        "parameters": int(mlp.count_params()),
    }
    return {
        "framework": "TensorFlow/Keras",
        "version": tf.__version__,
        "tabular_mlp": tabular,
        "sequence_models": results,
    }
