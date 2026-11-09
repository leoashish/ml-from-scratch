"""Demo: forecasting."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.utils.data import make_time_series
from src.forecasting.arima import ARIMA
from src.forecasting.boosting_forecaster import BoostingForecaster

y = make_time_series(n=150, trend=0.05, seasonality=12, noise=1.0, random_state=0)

ar = ARIMA(p=3, d=1, q=2)
ar.fit(y)
print("ARIMA forecast:", np.round(ar.forecast(5), 3))

bf = BoostingForecaster(n_lags=12, n_estimators=100, random_state=0)
bf.fit(y)
print("Boosting forecast:", np.round(bf.forecast(5), 3))
print("last observed:", np.round(y[-3:], 3))
