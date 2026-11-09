# Algorithms From First Principles

This document explains every algorithm in this repository from first principles:
the intuition, the model, the objective/loss, and the update rule. Read with the
code in `src/`.

---

## 1. Linear Regression

**Problem**: predict a number `y` from features `x`.

**Model**: `ŷ = wᵀx + b`.

**Objective**: minimize Mean Squared Error
```
L(w,b) = (1/n) Σ (y_i - (wᵀx_i + b))²
```

**Closed form**: set gradient to zero:
```
w = (XᵀX)⁻¹ Xᵀy   (with a bias column of ones in X)
```
Ridge adds L2: `w = (XᵀX + λI)⁻¹ Xᵀy`.

**Gradient descent**: `w -= lr * (2/n) Xᵀ(Xw - y)`.

**Code**: `src/regression/linear_regression.py`.

---

## 2. Logistic Regression

**Problem**: predict a probability / class.

**Model**: `p = σ(wᵀx + b)`, `σ(z) = 1/(1+e⁻ᶻ)`.

**Objective**: binary cross-entropy
```
L = - (1/n) Σ [ y_i log p_i + (1-y_i) log(1-p_i) ]
```

**Gradient**: `∇w L = (1/n) Xᵀ(p - y)`. Update `w -= lr * ∇`.

**Multiclass**: softmax `p_k = e^{z_k}/Σ_j e^{z_j}`, cross-entropy
`L = -(1/n) Σ_i Σ_k y_{ik} log p_{ik}`. Gradient `Xᵀ(P - Y)/n`.

**Code**: `src/classification/logistic_regression.py`.

---

## 3. Support Vector Machine (SVM)

**Problem**: find the maximum-margin separating hyperplane.

**Model**: `f(x) = wᵀx + b`, predict `sign(f)`.

**Primal objective**:
```
min  ½||w||² + C Σ ξ_i
s.t. y_i (wᵀx_i + b) ≥ 1 - ξ_i,  ξ_i ≥ 0
```
Hinge loss form: `L = ½||w||² + C Σ max(0, 1 - y_i f_i)`.

**Subgradient update** on violating points (`y_i f_i < 1`):
```
w -= lr*(w/n - C y_i x_i),   b += lr*C*y_i
```

**Dual / Kernel**: maximize
```
Σ α_i - ½ Σ_i Σ_j α_i α_j y_i y_j K(x_i, x_j)
s.t. 0 ≤ α_i ≤ C,  Σ α_i y_i = 0
```
Solved by SMO (coordinate ascent, two alphas at a time). Prediction
`f(x) = Σ α_i y_i K(x_i, x) + b`. Kernels: linear, RBF `exp(-γ||x-x'||²)`, poly.

**Code**: `src/classification/svm.py`.

---

## 4. K-Means

**Problem**: partition points into K clusters minimizing within-cluster SSE.

**Objective**: `Σ_k Σ_{x∈C_k} ||x - μ_k||²`.

**Algorithm (Lloyd)**:
1. Initialize centroids (k-means++).
2. Assign each point to nearest centroid.
3. Recompute centroids as cluster means.
4. Repeat until convergence.

**Code**: `src/clustering/kmeans.py`.

---

## 5. DBSCAN

**Problem**: density-based clustering; finds arbitrary shapes and noise.

**Definitions**:
- Core point: has ≥ `min_samples` points within radius `eps`.
- Border point: within eps of a core point but not core.
- Noise: neither.

**Algorithm**: for each unvisited point, if core, start a new cluster and
expand by adding all density-reachable points (BFS over eps-neighbors).

**Code**: `src/clustering/dbscan.py`.

---

## 6. PCA

**Problem**: reduce dimensions while preserving variance.

**Method**: center data `Xc = X - mean`. Compute covariance
`C = (1/(n-1)) Xcᵀ Xc`. Eigendecompose `C = V Λ Vᵀ`. Top-k eigenvectors
form `W`. Projection `Z = Xc W`. Explained variance ratio = eigenvalue / sum.

**Code**: `src/dimensionality_reduction/pca.py`.

---

## 7. Autoencoder

**Problem**: learn a compressed representation.

**Model**: encoder `z = f_θ(x)`, decoder `x̂ = g_φ(z)`. Loss reconstruction
MSE `||x - x̂||²`. Train both networks with backpropagation (chain rule
through each layer: `∂L/∂W = upstream_grad * input`).

**Code**: `src/dimensionality_reduction/autoencoder.py`.

---

## 8. Gradient Boosting

**Problem**: build a strong model from weak trees.

**Additive model**: `F_m(x) = F_{m-1}(x) + η h_m(x)`.

**Idea**: each new tree `h_m` fits the negative gradient of the loss w.r.t.
`F_{m-1}`. For MSE, negative gradient = residual `y - F_{m-1}(x)`.

**Classification**: gradient of log-loss gives `y - p` where `p = σ(F)`.

**Code**: `src/regression/boosting.py`.

---

## 9. LambdaMART (Ranking)

**Problem**: order items by relevance (optimize NDCG).

**Approach**: gradient boosting of trees where gradients are "lambdas".

For a query with items, for each pair (i,j) where rel_i > rel_j:
```
λ_ij = |ΔNDCG_ij| * 1/(1 + exp(s_i - s_j))
λ_i += λ_ij,  λ_j -= λ_ij
```
`ΔNDCG_ij` is the change in NDCG if i and j were swapped. Trees fit lambdas;
leaf values use Newton step `Σλ / Σμ` where `μ = λ(1-λ-ish)` (second derivative).

**Code**: `src/ranking/lambdamart.py`.

---

## 10. Isolation Forest

**Problem**: find anomalies.

**Idea**: anomalies are easier to isolate (fewer random splits). Build random
trees (random feature + random threshold). Anomaly score:
```
s(x) = 2^{-E[h(x)] / c(n)}
```
where `E[h(x)]` is expected path length and `c(n)` normalizes. Higher score =>
more anomalous.

**Code**: `src/anomaly_detection/isolation_forest.py`.

---

## 11. LOF (Local Outlier Factor)

**Problem**: find local outliers.

**Method**: for each point, compute k-distance and local reachability density
```
lrd(p) = k / Σ_{o∈N_k(p)} reach-dist(p,o)
```
`LOF(p) = (1/k) Σ_{o∈N_k(p)} lrd(o)/lrd(p)`. Values >> 1 indicate outliers.

**Code**: `src/anomaly_detection/lof.py`.

---

## 12. ARIMA

**Problem**: forecast a time series.

**Model**: ARIMA(p,d,q):
- `d`: difference the series d times to make it stationary.
- AR(p): `y_t = c + Σ_{i=1}^p φ_i y_{t-i} + ε_t`.
- MA(q): `y_t = c + ε_t + Σ_{j=1}^q θ_j ε_{t-j}`.

**Fitting**: Yule–Walker equations for AR coefficients (Toeplitz solve); MA
via residual regression. Forecast iteratively; invert differencing by cumsum.

**Code**: `src/forecasting/arima.py`.

---

## 13. Boosting Forecaster

**Problem**: forecast via ML.

**Method**: create supervised dataset with lag features
`X_t = [y_{t-1}, ..., y_{t-L}]`, `y_t = y_t`. Fit gradient boosting
regressor. Forecast recursively: predict next, append to history, repeat.

**Code**: `src/forecasting/boosting_forecaster.py`.

---

## 14. Diffusion (DDPM)

**Problem**: generate new data.

**Forward process**: `x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε`, `ε ~ N(0,I)`.

**Reverse**: learn `ε_θ(x_t, t)` with a neural net (MSE loss). Sample by
starting from noise and iteratively denoising:
```
x_{t-1} = (1/√α_t)(x_t - (1-α_t)/√(1-ᾱ_t) ε_θ(x_t,t)) + σ_t z
```

**Code**: `src/generative/diffusion.py`.

---

## 15. Autoregressive Model

**Problem**: generate sequences.

**Model**: `P(x_t | x_{<t})`. Factorize `P(x) = Π_t P(x_t | x_{<t})`.

**Training**: MLP maps context (one-hot) to next-token logits; cross-entropy
loss. **Generation**: sample token, append to context, repeat.

**Code**: `src/generative/autoregressive.py`.

---

## 16. Q-Learning

**Problem**: learn a policy from rewards.

**Update**: `Q(s,a) ← Q(s,a) + α [ r + γ max_{a'} Q(s',a') - Q(s,a) ]`.

Epsilon-greedy action selection. Tabular over (state, action).

**Code**: `src/sequential_decision/q_learning.py`.

---

## 17. PPO

**Problem**: policy gradient for continuous/large action spaces.

**Objective (clipped)**:
```
L^CLIP = E[ min(r_t(θ) Â_t, clip(r_t(θ), 1-ε, 1+ε) Â_t) ]
```
where `r_t = π_θ(a_t|s_t)/π_{θ_old}(a_t|s_t)` and `Â_t` is GAE advantage.
Plus value loss and entropy bonus. Update with gradient ascent.

**GAE**: `Â_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`, `δ_t = r_t + γ V(s_{t+1}) - V(s_t)`.

**Code**: `src/sequential_decision/ppo.py`.

---

## Further reading
- `docs/learning_path.md` for suggested order.
- `README.md` for project layout and how to run.
