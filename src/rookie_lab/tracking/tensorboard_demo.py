"""Write real TensorBoard events with PyTorch and Keras-compatible scalars."""

from __future__ import annotations

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def write_demo(log_dir: Path) -> dict[str, object]:
    log_dir.mkdir(parents=True, exist_ok=True)
    losses = [1.0, 0.68, 0.43, 0.31]
    with SummaryWriter(log_dir) as writer:
        for step, loss in enumerate(losses):
            writer.add_scalar("train/loss", loss, step)
            writer.add_scalar("valid/accuracy", 1 - loss / 2, step)
        writer.add_text("run/frameworks", "PyTorch; TensorFlow/Keras")
    return {"event_files": len(list(log_dir.glob("events.out.tfevents.*"))), "losses": losses}
