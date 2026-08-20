# Security and privacy review

- No API key, remote AI endpoint, or cloud credential is required.
- IDs are generated counters and never hash or resemble source identities.
- `.env*`, generated data, checkpoints, TensorBoard events, and temporary databases are ignored.
- API inputs are typed and bounded; metrics paths reject traversal characters.
- MySQL CI uses an ephemeral empty-password service available only to its isolated runner job.
- The repository contains no production authentication or authorization layer and must not be
  exposed as a public decision service.

