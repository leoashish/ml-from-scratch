"""Smoke tests for all algorithms."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.utils.data import make_regression, make_classification, make_blobs, make_time_series, make_ranking
from src.utils.preprocessing import train_test_split
from src.utils.metrics import accuracy, r2_score


def test_linear_regression():
    X, y = make_regression(n_samples=200, noise=1.0, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)
    from src.regression.linear_regression import LinearRegression
    m = LinearRegression().fit(Xtr, ytr)
    assert r2_score(yte, m.predict(Xte)) > 0.7


def test_logistic_regression():
    X, y = make_classification(n_samples=200, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)
    from src.classification.logistic_regression import LogisticRegression
    m = LogisticRegression(lr=0.1, n_iters=300).fit(Xtr, ytr)
    assert accuracy(yte, m.predict(Xte)) > 0.85


def test_svm():
    X, y = make_classification(n_samples=200, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)
    from src.classification.svm import LinearSVM, KernelSVM
    for M in (LinearSVM(n_iters=200), KernelSVM(max_passes=3)):
        M.fit(Xtr, ytr)
        assert accuracy(yte, M.predict(Xte)) > 0.85


def test_boosting_classifier():
    X, y = make_classification(n_samples=200, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)
    from src.regression.boosting import GradientBoostingClassifier
    m = GradientBoostingClassifier(n_estimators=50).fit(Xtr, ytr)
    assert accuracy(yte, m.predict(Xte)) > 0.85


def test_kmeans():
    X, y = make_blobs(n_samples=200, centers=3, random_state=0)
    from src.clustering.kmeans import KMeans
    m = KMeans(n_clusters=3, random_state=0).fit(X)
    assert len(np.unique(m.labels_)) == 3


def test_dbscan():
    X, y = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=0)
    from src.clustering.dbscan import DBSCAN
    m = DBSCAN(eps=1.0, min_samples=5).fit(X)
    assert len(np.unique(m.labels_)) >= 2


def test_pca():
    X, y = make_blobs(n_samples=200, centers=3, n_features=5, random_state=0)
    from src.dimensionality_reduction.pca import PCA
    m = PCA(n_components=2).fit(X)
    Z = m.transform(X)
    assert Z.shape == (200, 2)


def test_autoencoder():
    X, y = make_blobs(n_samples=100, centers=2, n_features=6, random_state=0)
    from src.dimensionality_reduction.autoencoder import Autoencoder
    m = Autoencoder(input_dim=6, hidden_dims=(4,), latent_dim=2, n_iters=20, random_state=0).fit(X)
    assert m.encode(X).shape == (100, 2)


def test_isolation_forest():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.standard_normal((100, 2)), rng.uniform(5, 6, (10, 2))])
    from src.anomaly_detection.isolation_forest import IsolationForest
    m = IsolationForest(contamination=0.1, random_state=0).fit(X)
    assert (m.predict(X) == -1).sum() >= 5


def test_lof():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.standard_normal((100, 2)), rng.uniform(5, 6, (10, 2))])
    from src.anomaly_detection.lof import LocalOutlierFactor
    m = LocalOutlierFactor(contamination=0.1).fit(X)
    assert (m.fit_predict(X) == -1).sum() >= 5


def test_arima():
    y = make_time_series(n=80, random_state=0)
    from src.forecasting.arima import ARIMA
    m = ARIMA(p=2, d=1, q=1).fit(y)
    f = m.forecast(5)
    assert f.shape == (5,)
    assert np.all(np.isfinite(f))


def test_boosting_forecaster():
    y = make_time_series(n=80, random_state=0)
    from src.forecasting.boosting_forecaster import BoostingForecaster
    m = BoostingForecaster(n_lags=6, n_estimators=20).fit(y)
    f = m.forecast(5)
    assert f.shape == (5,)


def test_lambdamart():
    X, y, qid = make_ranking(n_queries=10, n_items_per_query=8, random_state=0)
    from src.ranking.lambdamart import LambdaMART
    m = LambdaMART(n_estimators=10, max_depth=2, random_state=0).fit(X, y, qid)
    p = m.predict(X)
    assert p.shape == (len(y),)


def test_diffusion():
    from src.utils.data import make_moons
    X, y = make_moons(n_samples=200, random_state=0)
    from src.generative.diffusion import DiffusionModel
    m = DiffusionModel(data_dim=2, T=10, n_iters=50, random_state=0).fit(X)
    S = m.sample(20, seed=1)
    assert S.shape == (20, 2)


def test_autoregressive():
    from src.generative.autoregressive import AutoregressiveCharModel
    m = AutoregressiveCharModel(context=3, hidden=8, n_iters=20, random_state=0).fit("abc " * 20)
    out = m.generate("ab", length=10, seed=1)
    assert len(out) == 12


def test_q_learning():
    from src.sequential_decision.q_learning import GridWorld, QLearningAgent
    env = GridWorld(size=4)
    a = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions, random_state=0)
    r = a.train(env, episodes=100)
    assert len(r) == 100


def test_ppo():
    from src.sequential_decision.ppo import CartPole, PPOAgent
    env = CartPole()
    a = PPOAgent(random_state=0)
    rh = a.train(env, total_timesteps=512, rollout_len=128)
    assert len(rh) >= 1
