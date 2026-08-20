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
- Tiny GPT learns the synthetic support grammar to perplexity 2.7445.
- The GCN node task learns churn structure (ROC-AUC 0.7339); link prediction remains modest at
  0.5612, showing that the random product features and weak preference structure limit the task.

## Reproducibility

The generator, split seeds, profiles, source code, metrics, tests, and environment constraints are
versioned. Generated records and model binaries are excluded because they can be reconstructed.

