# Limitations

- All data is synthetic; metrics do not estimate real retail performance.
- Tiny named architectures preserve defining blocks but are not canonical pretrained checkpoints.
- The object detector assumes one centered synthetic object per image.
- BERT pre-training uses a small vocabulary and corpus; GPT learns templated language.
- GCN adjacency is transductive and link prediction is only modestly above random.
- XGBoost requires `libomp` on macOS. This machine reports a clearly labelled sklearn fallback;
  Linux full CI exercises the native package.
- MSSQL is validated through schema/dialect compilation, not a live licensed server. MySQL has a live
  ephemeral CI service; SQLite is exercised locally.
- No result supports decisions about real people.

