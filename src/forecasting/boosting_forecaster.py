"""Forecasting via gradient boosting on lag features."""

import numpy as np
from ..regression.boosting import GradientBoostingRegressor


class BoostingForecaster:
    def __init__(self, n_lags=12, n_estimators=100, learning_rate=0.1,
                 max_depth=3, random_state=None):
        self.n_lags = n_lags
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.model_ = None
        self.history_ = None

    def _make_features(self, series):
        series = np.asarray(series, float)
        n = len(series)
        X = []
        y = []
        for t in range(self.n_lags, n):
            X.append(series[t - self.n_lags:t][::-1])
            y.append(series[t])
        return np.array(X), np.array(y)

    def fit(self, series):
        series = np.asarray(series, float)
        self.history_ = series.copy()
        X, y = self._make_features(series)
        self.model_ = GradientBoostingRegressor(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            max_depth=self.max_depth, random_state=self.random_state)
        self.model_.fit(X, y)
        return self

    def forecast(self, steps=10):
        history = list(self.history_)
        out = []
        for _ in range(steps):
            feat = np.array(history[-self.n_lags:][::-1])
            pred = self.model_.predict(feat.reshape(1, -1))[0]
            out.append(pred)
            history.append(pred)
        return np.array(out)
