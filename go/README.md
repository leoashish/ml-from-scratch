# Machine Learning From Scratch (Go)

Go port of the Python project: same algorithms, implemented with only the
standard library. Each package mirrors a problem type.

## Layout

```
ml-from-scratch-go/
├── mat/          # tiny matrix/vector library
├── regression/   # linear regression, gradient boosting
├── classification/ # logistic regression, SVM
├── ranking/      # LambdaMART
├── clustering/   # KMeans, DBSCAN
├── dimred/       # PCA
├── anomaly/      # Isolation Forest, LOF
├── forecasting/  # ARIMA, boosting forecaster
├── generative/   # diffusion, autoregressive
├── rl/           # Q-learning, PPO
└── examples/
```

## Run

```bash
cd ml-from-scratch/go
go run ./examples/
```

## Build

```bash
go build ./...
```

See the Python project (`../ml-from-scratch`) for the reference implementation
and detailed notes.
