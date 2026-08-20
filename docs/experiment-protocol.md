# Experiment protocol

## Profiles and split policy

`ci` verifies behavior quickly. `portfolio` produces the committed research evidence. Both use seed
`20260820`. Supervised tabular experiments use fixed stratified holdout data and preprocessing inside
each pipeline. Cross-validation refits the full pipeline per fold. Sequence, text, image, and graph
experiments use fixed seeded splits documented in source.

## Selection and metrics

- Classification: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier score, 10-bin ECE,
  confusion matrix, fit time, and inference time.
- Regression: MAE, RMSE, R², fit time, and inference time.
- Clustering: silhouette, noise rate, cluster sizes, and restart stability by pairwise ARI.
- Detection: class accuracy, IoU, precision/recall at .50, AP@.50, and mean AP at .50:.95.
- Language modeling: training loss and perplexity under a fixed local tokenizer.
- Graph: node and link ROC-AUC, full loss curve, and non-GNN baselines.

Model selection uses validation/CV ranking and never the test target. The default classification
threshold is intentionally exposed as a limitation: ranking, calibration, and operational recall can
disagree strongly on the imbalanced churn task.

## Evidence contract

Every lab must have executable source, a declared configuration/profile, automated tests, a
machine-readable metric artifact, at least one diagnostic or registered artifact, explanatory
documentation, and explicit limitations. The generated reproducibility report records data version,
seed, wall time, peak-process RSS snapshot, and runtime environment for each full-profile lab.

## Interpretation boundary

The data is synthetic and deliberately learnable. Perfect image or text scores validate the code
path, not real-world generalization. Negative graph comparisons are preserved. No result supports a
financial, credit, employment, insurance, fraud, medical, or customer-impact claim.
