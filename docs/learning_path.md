# Step-by-step learning path

This document walks through each algorithm from first principles: the problem,
the model, the loss/objective, and the update rule. Read alongside the code.

## 1. Linear Regression (regression/linear_regression.py)

Model: `y = w^T x + b`. Loss: MSE `L = (1/n) sum (y_i - (w^T x_i+b))^2`.

Closed form: `w = (X^T X)^-1 X^T y` (with bias column). Ridge adds `lambda I`.

Gradient descent: `w -= lr * (2/n) X^T (Xw - y)`.

## 2. Logistic Regression (classification/logistic_regression.py)

Model: `p = sigma(w^T x + b)`, `sigma(z)=1/(1+e^-z)`. Loss: binary cross-entropy
`L = -[y log p + (1-y) log(1-p)]`. Gradient: `X^T (p - y)/n`.

Multiclass: softmax + cross-entropy; gradient `X^T (P - Y)/n`.

## 3. SVM (classification/svm.py)

Linear: minimize `0.5||w||^2 + C sum hinge(w;x_i,y_i)` where hinge=max(0,1-y_i f_i).
Subgradient update on violating points.

Kernel: dual `max sum a_i - 0.5 sum_i,j a_i a_j y_i y_j K(x_i,x_j)` s.t. `0<=a_i<=C`, `sum a_i y_i=0`.
Solved with SMO (coordinate ascent).

## 4. K-Means (clustering/kmeans.py)

Minimize within-cluster SSE. Lloyd: assign each point to nearest centroid, then
recompute centroids as cluster means. Repeat.

## 5. DBSCAN (clustering/dbscan.py)

Density: a point is core if it has >= min_samples within eps. Clusters are
connected components of core points (via eps-neighborhood); border points
attached to a cluster; otherwise noise.

## 6. PCA (dimensionality_reduction/pca.py)

Center data, compute covariance, eigendecompose, take top-k eigenvectors.
Projection: `Z = (X - mean) W`.

## 7. Autoencoder (dimensionality_reduction/autoencoder.py)

Neural net encoder `z = f(x)`, decoder `x_hat = g(z)`. Loss MSE. Train with
backpropagation (chain rule through layers).

## 8. Gradient Boosting (regression/boosting.py)

Additive model `F_m = F_{m-1} + lr * h_m` where `h_m` fits negative gradient
of loss (residuals for MSE). Trees are weak learners.

## 9. LambdaMART (ranking/lambdamart.py)

Ranking via gradient boosting where gradients are "lambdas": for each pair
(i,j) with differing relevance, lambda_ij = |ΔNDCG| * sigmoid-based weight.
Trees fit lambdas; leaf values use Newton step (mu).

## 10. Isolation Forest (anomaly_detection/isolation_forest.py)

Anomalies isolate in fewer random splits. Score = 2^(-E[path length]/c(n)).
Shorter path => more anomalous.

## 11. LOF (anomaly_detection/lof.py)

Local density via k-nearest neighbors. LOF_i = average ratio of neighbor
density to point density. >>1 => outlier.

## 12. ARIMA (forecasting/arima.py)

Difference d times for stationarity. Fit AR(p) (Yule-Walker) and MA(q)
(residual regression). Forecast iteratively, invert differencing.

## 13. Boosting Forecaster (forecasting/boosting_forecaster.py)

Turn time series into supervised (lag features -> next value), fit gradient
boosting regressor, forecast recursively.

## 14. Diffusion (generative/diffusion.py)

Forward: gradually add Gaussian noise over T steps. Reverse: learn to predict
noise with a neural net; sample by denoising from pure noise.

## 15. Autoregressive (generative/autoregressive.py)

Model `P(x_t | x_{<t})`. Train MLP on (context -> next token). Generate by
sampling one token at a time and appending.

## 16. Q-Learning (sequential_decision/q_learning.py)

Tabular: `Q(s,a) += lr * (r + gamma max_a' Q(s',a') - Q(s,a))`. Epsilon-greedy.

## 17. PPO (sequential_decision/ppo.py)

Actor-critic with clipped surrogate objective:
`L = E[min(r_t A_t, clip(r_t,1-eps,1+eps) A_t)]` plus value loss and entropy.
GAE for advantages.
