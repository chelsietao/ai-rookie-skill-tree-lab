# Completion standard audit

An official skill is **Verified** only when it belongs to an evidence route with executable code,
tests, declared configuration/profile, measured result or behavior check, artifact/figure where
applicable, technical explanation, and an explicit limitation. A README mention alone is not enough.

| Evidence class | Acceptance check |
|---|---|
| Executable code | Imported by a script, CLI, API, test, or notebook that has been run |
| Test | Deterministic assertion covers behavior or a data/SQL/API contract |
| Configuration | YAML is versioned and named in run metadata |
| Result | JSON contains seed, dataset version, runtime, memory, environment, and task metrics |
| Figure | Generated from a named profile; simple synthetic labels are disclosed |
| Technical explanation | Defines mechanism, interpretation, limitation, and next experiment |
| Failure evidence | Negative or modest results remain in the report and are not filtered out |

CI is the fast acceptance profile. The portfolio profile is the resume-quality reproduction run.
