# Machine Learning From Scratch

A self-contained, from-first-principles implementation of the important ML
algorithms, one per problem type from the standard ML problem-type reference:

| Requirement | Problem type | Implemented |
|---|---|---|
| Predict a numerical value | Regression | `LinearRegression`, `GradientBoostingRegressor` |
| Predict a category | Classification | `LogisticRegression`, `LinearSVM`, `KernelSVM`, `GradientBoostingClassifier` |
| Order items by relevance | Ranking | `LambdaMART` |
| Find natural groups | Clustering | `KMeans`, `DBSCAN` |
| Reduce feature count | Dimensionality reduction | `PCA`, `Autoencoder` |
| Find unusual observations | Anomaly detection | `IsolationForest`, `LocalOutlierFactor` |
| Predict future values | Forecasting | `ARIMA`, `BoostingForecaster` |
| Generate new content | Generative modelling | `DiffusionModel` (DDPM), `AutoregressiveCharModel` |
| Learn actions through rewards | Sequential decision-making | `QLearningAgent`, `PPOAgent` |

Everything is implemented with **NumPy only** (no scikit-learn, no PyTorch). Each
algorithm carries comments explaining the math so you can read, understand, and
re-implement it yourself.

## Layout

```
ml-from-scratch/
├── src/
│   ├── utils/           # metrics, preprocessing, synthetic data
│   ├── regression/
│   ├── classification/
│   ├── ranking/
│   ├── clustering/
│   ├── dimensionality_reduction/
│   ├── anomaly_detection/
│   ├── forecasting/
│   ├── generative/
│   └── sequential_decision/
├── examples/            # runnable demos, one per problem type
├── tests/               # smoke tests
└── docs/                # learning path & notes
```

## Setup

```bash
cd ml-from-scratch
pip install numpy pytest
```

## Run an example

```bash
PYTHONPATH=src python examples/regression_demo.py
```

## Run tests

```bash
PYTHONPATH=src pytest -q
```

## Suggested learning order

1. `regression/linear_regression.py` — closed form + gradient descent.
2. `classification/logistic_regression.py` — sigmoid, cross-entropy, softmax.
3. `classification/svm.py` — hinge loss, subgradients, SMO dual.
4. `clustering/kmeans.py`, `dbscan.py`.
5. `dimensionality_reduction/pca.py`, `autoencoder.py`.
6. `regression/boosting.py` — gradient boosting; then `ranking/lambdamart.py`.
7. `anomaly_detection/` — isolation forest, LOF.
8. `forecasting/` — ARIMA, boosting forecaster.
9. `generative/` — diffusion, autoregressive.
10. `sequential_decision/` — Q-learning, PPO.

## Notes

- Implementations favor clarity over speed; they are educational.
- Random seeds are exposed for reproducibility.
- Each module is independent; import via `PYTHONPATH=src`.
- See `README_MATH.md` for first-principles math of every algorithm.
- Roughgarden algorithms/complexity/game-theory live in the separate `../roughgarden/` folder.
- Practical applied mathematics (Howison) lives in the separate `../practical_applied_math/` folder.
