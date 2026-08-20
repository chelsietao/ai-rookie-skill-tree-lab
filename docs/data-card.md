# Synthetic retail data card

All records are fictional and generated from documented probability distributions with seed `20260820`.
No record is copied from a person, merchant, employer, bank, or customer system.

## Composition

| Modality | Portfolio rows | Purpose |
|---|---:|---|
| Customers | 20,000 | churn classification and spend regression |
| Transactions | 100,000 | temporal aggregation, anomaly detection, graph edges |
| Product views | 50,000 | event modeling, conversion analysis, graph edges |
| Sequences | 10,000 | RNN/LSTM/GRU sequence classification |
| Support texts | 5,000 | TF-IDF, LSTM/GRU, BERT and GPT |
| Product images | 5,000 | seven classifiers and one-object detection |
| Graph edges | 148,923 | customer–product node/link experiments |

The machine-readable contract is `data/schemas/retail_world.yaml`. The manifest records generator
version, seed, row counts, and SHA-256 checksums for all eight generated data files. A five-table,
20-row-per-table preview is committed under `data/samples`; full generated files are ignored because
they are deterministic and reconstructable.

## Designed properties

- Customer income and loyalty include missing values for imputation experiments.
- Churn prevalence is about 4%, forcing imbalance-aware metrics and resampling.
- Transaction anomalies have known injected labels; event time enables leakage exercises.
- View dwell time and page depth influence a fictional conversion label.
- Images contain simple generated shapes and one bounding box per image.
- Text uses a deliberately templated support grammar.
- Graph edges combine transaction and view interactions.

## Validation and known risks

Validation checks unique IDs, binary labels, foreign keys, array alignment, nonempty text, designed
missingness, fictional/PII flags, and complete checksums. These checks prove internal consistency,
not realism. The simple generators cause very high image and text scores; demographic columns can
encode undesirable proxy relationships; synthetic performance must never be treated as evidence for
real people. No raw name, address, email, phone, account, credential, or production identifier exists.

This data is educational. It must not be used to make financial, employment, health, credit, or
commercial decisions about real people.
