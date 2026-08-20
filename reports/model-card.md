# Model card

The repository contains multiple small educational models rather than one production model.

- Intended use: learning, interviews, reproducible local experiments.
- Out-of-scope: credit, fraud, employment, medical, insurance, investment, or customer decisions.
- Data: deterministic fictional retail data.
- Metrics: held-out synthetic performance plus cross-validation, calibration, timing, stability,
  detection AP by IoU threshold, language-model perplexity, and graph baselines where applicable.
- Risks: synthetic shortcuts, misleading accuracy under imbalance, architecture-name overclaiming,
  and treating tutorial performance as real-world validation.
- Mitigations: explicit scope labels, PR-AUC/F1/calibration, baseline comparisons, limitations, data
  card, secret scan, notebook execution, automated tests, and evidence matrix.

The saved classical pipeline is a demonstration artifact, not a deployable decision model. Tiny
BERT/GPT and named vision variants are trained from scratch on simple local synthetic data. The GCN
is transductive and underperforms traditional baselines in the portfolio run.
