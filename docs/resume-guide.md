# Resume and portfolio guide

## Suggested project entry

**AI Rookie Skill Tree Lab — reproducible multimodal ML benchmark**

- Built a deterministic synthetic retail data system spanning 20,000 customers, 100,000
  transactions, 50,000 browsing events, 10,000 sequences, 5,000 texts, 5,000 images, bounding
  boxes, and a customer-product graph; added data contracts, checksums, PII checks, and quality tests.
- Benchmarked 19 classical classification/regression estimators with fold-safe preprocessing,
  cross-validation, imbalance handling, calibration, latency, ROC/PR, confusion-matrix, and
  regression diagnostics.
- Implemented CPU-scale PyTorch and TensorFlow/Keras MLP, autoencoder, RNN/LSTM/GRU experiments;
  compared activations, initialization, normalization, optimizers, schedulers, class weighting,
  early stopping, and overfitting curves.
- Compared seven educational vision architectures, a one-object detector, local Tiny BERT/GPT,
  and a manual sparse GCN against task-appropriate baselines; retained limitations and negative
  results instead of reporting only favorable metrics.
- Packaged the work with CLI/FastAPI, TensorBoard, SQLite model/artifact registry, MySQL CI,
  MSSQL dialect checks, GitHub/GitLab workflows, tests, versioned configs, and notebook tutorials.

Use measured results from `artifacts/metrics/` when adding numbers. Do not copy a metric from an old
run after changing data or code.

## Repository tour for an interviewer

1. `README.md`: question, architecture, reproducible commands, and honest result summary.
2. `data/schemas/` and `src/rookie_lab/data/`: data design and validation.
3. `notebooks/`: narrated experiments and error analysis.
4. `src/rookie_lab/`: tested implementation.
5. `artifacts/metrics/`: machine-readable evidence with run metadata.
6. `docs/official-skill-matrix.md`: official skill → code → result/test traceability.
7. `reports/limitations.md`: what the repository does not prove.

## Claims to avoid

- “Production-ready” — the API has no production authentication or monitoring.
- “State of the art” — models and labels are intentionally small and synthetic.
- “Large-scale training” — portfolio data is larger than CI data but training remains CPU-scale.
- “Real transfer learning” — the vision transfer experiment is a same-domain mechanics proxy.
- “Financial AI” — this is a fictional retail benchmark and not financial advice or underwriting.
