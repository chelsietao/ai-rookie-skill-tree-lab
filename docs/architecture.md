# Architecture and experiment contract

All labs consume artifacts produced by `rookie_lab.data.generate`; no lab privately changes the
ground truth. Every supervised evaluation fits preprocessing on training folds only. Full and CI
profiles share code but differ in rows, epochs, estimators, and search trials.

Each experiment returns JSON-serializable metrics. The orchestration script writes portfolio
metrics to `artifacts/metrics/` and CI metrics to an ignored subdirectory. TensorBoard event files,
checkpoints, the SQLite registry, and generated data remain local and reproducible.

The implementation deliberately uses tiny architectures. Names such as ResNet, MobileNet,
EfficientNet, SqueezeNet, ViT, BERT, and GPT refer to educational variants that preserve the defining
block—not canonical full-size pretrained models.

