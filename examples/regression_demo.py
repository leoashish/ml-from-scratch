"""Demo: regression algorithms."""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.data import make_regression
from src.utils.preprocessing import train_test_split
from src.utils.metrics import r2_score, rmse
from src.regression.linear_regression import LinearRegression
from src.regression.boosting import GradientBoostingRegressor

X, y = make_regression(n_samples=500, n_features=5, noise=1.0, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)

lr = LinearRegression(method="normal")
lr.fit(Xtr, ytr)
print("LinearRegression  R2:", round(r2_score(yte, lr.predict(Xte)), 4),
      "RMSE:", round(rmse(yte, lr.predict(Xte)), 4))

gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
gb.fit(Xtr, ytr)
print("GradientBoosting  R2:", round(r2_score(yte, gb.predict(Xte)), 4),
      "RMSE:", round(rmse(yte, gb.predict(Xte)), 4))
print("true coef (approx):", np.round(lr.coef_, 3))
