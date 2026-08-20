# Interview guide

Use this repository as evidence, not as a keyword list. A strong answer follows:

> problem → data contract → baseline → experiment control → metric → error analysis → limitation → next step

## Data and validation

**Why synthetic data?** It makes the entire multimodal system public, deterministic, free of PII,
and cheap to reproduce. The limitation is construct validity: strong scores show implementation
correctness on known mechanisms, not external business performance.

**How is leakage prevented?** Imputation, scaling, and encoding are inside scikit-learn Pipelines
that are refit in each cross-validation fold. Test data never fits preprocessing statistics.

**Why use ROC-AUC and PR-AUC together?** Churn is rare. ROC-AUC measures ranking across thresholds,
while PR-AUC is more sensitive to false positives under imbalance. Accuracy alone can reward an
all-negative model.

## Model families

**Linear versus tree models:** Linear/logistic models offer coefficients, stable baselines, and
fast inference. Trees capture interactions and nonlinear thresholds but can overfit and require
calibration checks. This repo compares both under one split and preprocessing contract.

**RNN, LSTM, GRU:** The plain RNN is simple but vulnerable to vanishing gradients. LSTM adds input,
forget, and output gates; GRU combines gates and is cheaper. The experiment keeps data and training
controls constant so only the recurrent cell changes.

**CNN families:** VGG stacks convolutions; ResNet adds identity skips; MobileNet uses depthwise
separable convolution; EfficientNet-style blocks add smooth activations and channel attention;
SqueezeNet uses fire modules; ViT tokenizes image patches and applies self-attention.

**BERT versus GPT:** BERT is a bidirectional encoder trained here with masked-token prediction,
then used for classification. GPT is a causal decoder trained to predict the next token. Both are
tiny random-initialized models, so the evidence is about mechanics rather than language quality.

**GCN:** A graph convolution aggregates normalized neighbor representations. Compare its node ROC-
AUC with the degree-plus-tabular logistic baseline before claiming that graph structure helps.

## Experiment and systems questions

**What makes a run reproducible?** Fixed seed, versioned generator/config, dataset checksum,
pipeline-contained preprocessing, JSON metrics, environment/runtime metadata, and a one-command
quick/full runner.

**What would change in production?** Replace synthetic data, add temporal/out-of-time validation,
feature lineage, model monitoring, authentication/authorization, drift checks, a managed registry,
and load/security tests. Do not use this educational API for decisions.

## Practice prompts

1. Find a high-accuracy/low-F1 classifier and explain the threshold behavior.
2. Compare Brier score and ECE for two churn models.
3. Explain why t-SNE coordinates are not cluster probabilities.
4. Diagnose an overfitting gap from training and validation loss curves.
5. Explain why the detector's one-object mAP is simpler than COCO mAP.
6. Defend or reject the claim that pre-training helped Tiny BERT in this run.
7. Compare GCN link prediction with the degree-popularity baseline.
8. Design an out-of-time split using the `day` fields.
