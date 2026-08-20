# Research report

## Question

Can the complete AI-rookie skill template be demonstrated on a normal CPU without API keys,
private data, or misleading claims of large-scale training?

## Method

A seeded synthetic retail world connects tabular, sequential, visual, textual, and graph tasks.
Models are compared with fixed splits and task-appropriate metrics. CI uses smaller rows and epochs;
the committed results come from the portfolio profile.

## Findings

- Class imbalance makes accuracy misleading: several classifiers achieve high accuracy while F1 is
  near zero. CatBoost's ROC-AUC is 0.8035; balanced Logistic Regression is 0.8011.
- Lasso, Ridge, and ordinary least squares are nearly tied because the spend target is mostly linear.
- RNN/LSTM/GRU succeed because the sequence generator contains strong recency and trend signals.
- The autoencoder detects injected amount anomalies with ROC-AUC 0.9578.
- Vision and text classification reach perfect scores because labels are deliberately simple. They
  verify implementation and training, not external generalization.
- Tiny GPT learns the synthetic support grammar to perplexity 2.7528.
- The detector reaches mean IoU 0.8248 and mAP@.50:.95 0.7988 under a deliberately simplified
  one-object protocol; this is not COCO-scale detection evidence.
- The GCN node task reaches ROC-AUC 0.7001 but loses to logistic regression plus degree at 0.8004.
  GCN link prediction reaches 0.5553 and loses to degree popularity at 0.5913. These retained negative
  results indicate that the current node features, graph generator, and transductive objective do not
  justify claiming graph-model superiority.

## Reproducibility

The generator, split seeds, profiles, source code, metrics, tests, and environment constraints are
versioned. Generated records and model binaries are excluded because they can be reconstructed. The
portfolio v2 manifest covers 20k customers, 100k transactions, 50k views, 10k sequences, 5k texts,
5k images, and 148,923 graph edges. Thirty automated tests, notebook-source checks, Ruff, a live
MySQL CI service, and an optional scheduled full-lab workflow guard the evidence.
