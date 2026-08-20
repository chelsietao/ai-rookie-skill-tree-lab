"""TensorFlow/Keras parity experiment using the same local sequence data."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

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
        }
    return {"framework": "TensorFlow/Keras", "version": tf.__version__, "sequence_models": results}
