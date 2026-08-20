# Architecture and experiment contract

All labs consume artifacts produced by `rookie_lab.data.generate`; no lab privately changes the
ground truth. Every supervised evaluation fits preprocessing on training folds only. Full and CI
profiles share code but differ in rows, epochs, estimators, and search trials.

Each experiment returns JSON-serializable metrics enriched with lab, profile, seed, data version,
configuration path, elapsed time, process RSS, and environment. The orchestration script writes portfolio
metrics to `artifacts/metrics/` and CI metrics to an ignored subdirectory. TensorBoard event files,
checkpoints, the SQLite registry, and generated data remain local and reproducible.

The implementation deliberately uses tiny architectures. Names such as ResNet, MobileNet,
EfficientNet, SqueezeNet, ViT, BERT, and GPT refer to educational variants that preserve the defining
block—not canonical full-size pretrained models.

The repository uses three separations: generated data is reconstructable and ignored; metrics and
diagnostic figures are committed; binary models, TensorBoard events, and the SQLite registry remain
local. Thirteen notebooks teach and inspect the package, while reusable behavior stays in `src` and
is exercised through 30 unit, integration, and safety tests.
