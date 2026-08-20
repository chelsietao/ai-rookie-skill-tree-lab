# Limitations

- All data is synthetic; metrics do not estimate real retail performance.
- Tiny named architectures preserve defining blocks but are not canonical pretrained checkpoints.
- The object detector assumes exactly one synthetic object and reports an educational approximation
  of AP across IoU thresholds, not an official COCO evaluator with multiple classes/crowds.
- BERT pre-training uses a small vocabulary and corpus; GPT learns templated language.
- GCN adjacency is transductive. In the portfolio run both GCN node and link scores are below their
  traditional baselines; falling joint loss does not establish task superiority.
- XGBoost requires `libomp` on macOS. This machine reports a clearly labelled sklearn fallback;
  Linux full CI exercises the native package.
- MSSQL is validated through schema/dialect compilation, not a live licensed server. MySQL has a live
  ephemeral CI service; SQLite is exercised locally.
- No result supports decisions about real people.
- Wall time, memory, and floating-point details can vary across hardware even with fixed seeds.
- The local FastAPI demo lacks authentication, rate limiting, drift monitoring, feature lineage,
  rollout controls, and a production registry; it binds to localhost by default.
