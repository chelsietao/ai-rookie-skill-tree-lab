# Official AI rookie skill coverage

Source snapshot: SinoPac Turing Plan skill tree, inspected 2026-08-20.
Ranks 1 and 2 are the 114 active nodes in the official AI rookie template.
Every row points to executable evidence; hierarchy and duplicate-case nodes are retained.

| Official path | Rank | Evidence | Status |
|---|---:|---|---|
| `/AI` | 1 | `README.md; reports/research-report.md` | Implemented |
| `/AI/model structure` | 1 | `README.md; reports/research-report.md` | Implemented |
| `/AI/model structure/CNN` | 2 | `src/rookie_lab/vision/` | Implemented |
| `/AI/model structure/CNN/EfficientNet` | 1 | `src/rookie_lab/vision/` | Implemented |
| `/AI/model structure/CNN/MobileNet` | 1 | `src/rookie_lab/vision/` | Implemented |
| `/AI/model structure/CNN/ResNet` | 1 | `src/rookie_lab/vision/` | Implemented |
| `/AI/model structure/CNN/VGG` | 1 | `src/rookie_lab/vision/` | Implemented |
| `/AI/model structure/CNN/object detection` | 1 | `src/rookie_lab/vision/` | Implemented |
| `/AI/model structure/GNN` | 1 | `src/rookie_lab/graph/` | Implemented |
| `/AI/model structure/KNN` | 2 | `src/rookie_lab/classical/` | Implemented |
| `/AI/model structure/MLP` | 2 | `src/rookie_lab/classical/` | Implemented |
| `/AI/model structure/autoencoder` | 1 | `src/rookie_lab/deep/` | Implemented |
| `/AI/model structure/linear model` | 1 | `README.md; reports/research-report.md` | Implemented |
| `/AI/model structure/linear model/lasso` | 2 | `src/rookie_lab/classical/` | Implemented |
| `/AI/model structure/linear model/linear regression` | 2 | `src/rookie_lab/classical/` | Implemented |
| `/AI/model structure/linear model/logistic regression` | 2 | `src/rookie_lab/classical/` | Implemented |
| `/AI/model structure/linear model/ridge` | 2 | `src/rookie_lab/classical/` | Implemented |
| `/AI/model structure/RNN` | 1 | `src/rookie_lab/deep/` | Implemented |
| `/AI/model structure/RNN/GRU` | 2 | `src/rookie_lab/deep/` | Implemented |
| `/AI/model structure/RNN/LSTM` | 2 | `src/rookie_lab/deep/` | Implemented |
| `/AI/model structure/SVM` | 2 | `src/rookie_lab/classical/` | Implemented |
| `/AI/model structure/SVM/SVC` | 2 | `src/rookie_lab/classical/` | Implemented |
| `/AI/model structure/SVM/SVR` | 2 | `src/rookie_lab/classical/` | Implemented |
| `/AI/model structure/tree-based` | 2 | `src/rookie_lab/classical/` | Implemented |
| `/AI/model structure/tree-based/CatBoost` | 2 | `src/rookie_lab/classical/` | Implemented |
| `/AI/model structure/tree-based/decision tree` | 2 | `src/rookie_lab/classical/` | Implemented |
| `/AI/model structure/tree-based/Extra-Trees` | 2 | `src/rookie_lab/classical/` | Implemented |
| `/AI/model structure/tree-based/random forests` | 2 | `src/rookie_lab/classical/` | Implemented |
| `/AI/model structure/tree-based/XGBoost` | 2 | `src/rookie_lab/classical/` | Implemented |
| `/AI/model structure/transformer` | 1 | `src/rookie_lab/nlp/` | Implemented |
| `/AI/model structure/transformer/BERT` | 1 | `src/rookie_lab/nlp/` | Implemented |
| `/AI/model structure/transformer/GPT` | 1 | `src/rookie_lab/nlp/` | Implemented |
| `/AI/model structure/transformer/ViT` | 1 | `src/rookie_lab/vision/models.py` | Implemented |
| `/AI/application` | 1 | `README.md; reports/research-report.md` | Implemented |
| `/AI/application/computer vision` | 1 | `README.md; reports/research-report.md` | Implemented |
| `/AI/application/NLP` | 1 | `src/rookie_lab/nlp/` | Implemented |
| `/AI/learning strategy` | 1 | `README.md; reports/research-report.md` | Implemented |
| `/AI/learning strategy/unsupervised learning` | 1 | `src/rookie_lab/unsupervised/` | Implemented |
| `/AI/learning strategy/unsupervised learning/clustering` | 2 | `src/rookie_lab/unsupervised/` | Implemented |
| `/AI/learning strategy/unsupervised learning/dimension reduction` | 2 | `src/rookie_lab/unsupervised/` | Implemented |
| `/AI/learning strategy/pre-training` | 2 | `src/rookie_lab/nlp/` | Implemented |
| `/AI/learning strategy/supervised learning` | 2 | `src/rookie_lab/classical/` | Implemented |
| `/AI/learning strategy/ensemble learning` | 1 | `src/rookie_lab/classical/` | Implemented |
| `/AI/training trick` | 1 | `README.md; reports/research-report.md` | Implemented |
| `/AI/training trick/data preprocess` | 2 | `src/rookie_lab/data/; src/rookie_lab/classical/preprocessing.py` | Implemented |
| `/AI/training trick/data preprocess/data augmentation` | 1 | `src/rookie_lab/data/; src/rookie_lab/classical/preprocessing.py` | Implemented |
| `/AI/training trick/data preprocess/missing value imputation` | 2 | `src/rookie_lab/data/; src/rookie_lab/classical/preprocessing.py` | Implemented |
| `/AI/training trick/parameter tuning` | 1 | `src/rookie_lab/unsupervised/` | Implemented |
| `/AI/training trick/pooling` | 2 | `src/rookie_lab/vision/experiment.py` | Implemented |
| `/AI/training trick/initialization` | 1 | `src/rookie_lab/deep/` | Implemented |
| `/AI/training trick/activation` | 2 | `src/rookie_lab/deep/` | Implemented |
| `/AI/training trick/attention` | 2 | `src/rookie_lab/deep/` | Implemented |
| `/AI/training trick/learning rate scheduling` | 2 | `src/rookie_lab/deep/` | Implemented |
| `/AI/training trick/normalization layer` | 2 | `src/rookie_lab/deep/` | Implemented |
| `/AI/training trick/validation` | 2 | `README.md; reports/research-report.md` | Implemented |
| `/AI/training trick/validation/early stopping` | 2 | `src/rookie_lab/deep/` | Implemented |
| `/AI/training trick/validation/cross-validation` | 2 | `src/rookie_lab/classical/` | Implemented |
| `/AI/automl` | 1 | `src/rookie_lab/unsupervised/` | Implemented |
| `/AI/editor` | 1 | `src/rookie_lab/software/; docs/tooling.md` | Implemented |
| `/AI/editor/colab` | 2 | `src/rookie_lab/software/; docs/tooling.md` | Implemented |
| `/AI/editor/jupyter notebook` | 2 | `src/rookie_lab/software/; docs/tooling.md` | Implemented |
| `/AI/framework` | 1 | `README.md; reports/research-report.md` | Implemented |
| `/AI/framework/Keras` | 2 | `src/rookie_lab/deep/` | Implemented |
| `/AI/framework/PyTorch` | 2 | `src/rookie_lab/deep/` | Implemented |
| `/AI/framework/scikit-learn` | 2 | `README.md; reports/research-report.md` | Implemented |
| `/AI/framework/TensorFlow` | 2 | `src/rookie_lab/deep/` | Implemented |
| `/AI/tracking` | 1 | `src/rookie_lab/tracking/` | Implemented |
| `/AI/tracking/TensorBoard` | 1 | `src/rookie_lab/tracking/` | Implemented |
| `/AI/model structure/CNN/SqueezeNet` | 1 | `src/rookie_lab/vision/` | Implemented |
| `/AI/model structure/knn` | 2 | `src/rookie_lab/classical/` | Implemented |
| `/AI/model structure/mlp` | 2 | `src/rookie_lab/classical/` | Implemented |
| `/AI/training trick/data preprocess/normalization` | 2 | `src/rookie_lab/data/; src/rookie_lab/classical/preprocessing.py` | Implemented |
| `/AI/training trick/data preprocess/numpy` | 2 | `src/rookie_lab/data/; src/rookie_lab/classical/preprocessing.py` | Implemented |
| `/AI/training trick/data preprocess/pandas` | 2 | `src/rookie_lab/data/; src/rookie_lab/classical/preprocessing.py` | Implemented |
| `/AI/training trick/data preprocess/imbalanced-learn` | 1 | `src/rookie_lab/data/; src/rookie_lab/classical/preprocessing.py` | Implemented |
| `/AI/training trick/data preprocess/sklearn.preprocessing` | 2 | `src/rookie_lab/data/; src/rookie_lab/classical/preprocessing.py` | Implemented |
| `/AI/training trick/early stopping` | 2 | `src/rookie_lab/deep/` | Implemented |
| `/AI/framework/pytorch` | 2 | `src/rookie_lab/deep/` | Implemented |
| `/AI/framework/sklearn` | 2 | `README.md; reports/research-report.md` | Implemented |
| `/AI/framework/tensorflow` | 2 | `src/rookie_lab/deep/` | Implemented |
| `/AI/visualization` | 1 | `src/rookie_lab/unsupervised/experiment.py` | Implemented |
| `/AI/visualization/matplotlib` | 2 | `src/rookie_lab/unsupervised/experiment.py` | Implemented |
| `/AI/visualization/seaborn` | 2 | `src/rookie_lab/unsupervised/experiment.py` | Implemented |
| `/AI/model structure/CNN/convolution` | 2 | `src/rookie_lab/vision/` | Implemented |
| `/software engineering` | 1 | `README.md; reports/research-report.md` | Implemented |
| `/software engineering/backend` | 1 | `README.md; reports/research-report.md` | Implemented |
| `/software engineering/backend/logging` | 1 | `src/rookie_lab/software/; docs/tooling.md` | Implemented |
| `/software engineering/database` | 1 | `src/rookie_lab/database/; sql/` | Implemented |
| `/software engineering/database/RDBMS` | 1 | `src/rookie_lab/database/; sql/` | Implemented |
| `/software engineering/database/RDBMS/MSSQL` | 1 | `src/rookie_lab/database/; sql/` | Implemented |
| `/software engineering/database/RDBMS/MySQL` | 1 | `src/rookie_lab/database/; sql/` | Implemented |
| `/software engineering/VCS` | 1 | `README.md; reports/research-report.md` | Implemented |
| `/software engineering/VCS/git` | 2 | `src/rookie_lab/software/; docs/tooling.md` | Implemented |
| `/software engineering/VCS/registry` | 2 | `README.md; reports/research-report.md` | Implemented |
| `/software engineering/VCS/registry/GitHub` | 2 | `src/rookie_lab/software/; docs/tooling.md` | Implemented |
| `/software engineering/VCS/registry/GitLab` | 2 | `src/rookie_lab/software/; docs/tooling.md` | Implemented |
| `/software engineering/visualization` | 1 | `src/rookie_lab/unsupervised/experiment.py` | Implemented |
| `/software engineering/visualization/matplotlib` | 2 | `src/rookie_lab/unsupervised/experiment.py` | Implemented |
| `/software engineering/visualization/seaborn` | 2 | `src/rookie_lab/unsupervised/experiment.py` | Implemented |
| `/software engineering/data analysis` | 2 | `README.md; reports/research-report.md` | Implemented |
| `/software engineering/data analysis/numpy` | 2 | `src/rookie_lab/data/; src/rookie_lab/classical/preprocessing.py` | Implemented |
| `/software engineering/data analysis/pandas` | 2 | `src/rookie_lab/data/; src/rookie_lab/classical/preprocessing.py` | Implemented |
| `/software engineering/editor` | 1 | `src/rookie_lab/software/; docs/tooling.md` | Implemented |
| `/software engineering/editor/vi` | 2 | `docs/tooling.md` | Implemented |
| `/software engineering/editor/VS Code` | 2 | `src/rookie_lab/software/; docs/tooling.md` | Implemented |
| `/software engineering/language` | 1 | `README.md; reports/research-report.md` | Implemented |
| `/software engineering/language/Python` | 2 | `src/rookie_lab/software/; docs/tooling.md` | Implemented |
| `/software engineering/file format` | 1 | `README.md; reports/research-report.md` | Implemented |
| `/software engineering/file format/JSON` | 2 | `src/rookie_lab/software/; docs/tooling.md` | Implemented |
| `/software engineering/file format/YAML` | 2 | `src/rookie_lab/software/; docs/tooling.md` | Implemented |
| `/software engineering/backend/logging/standard output` | 2 | `src/rookie_lab/software/; docs/tooling.md` | Implemented |
| `/software engineering/serialization` | 1 | `src/rookie_lab/software/; docs/tooling.md` | Implemented |
| `/software engineering/serialization/JSON` | 2 | `src/rookie_lab/software/; docs/tooling.md` | Implemented |
| `/software engineering/serialization/YAML` | 2 | `src/rookie_lab/software/; docs/tooling.md` | Implemented |
