"""Demo: anomaly detection."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.anomaly_detection.isolation_forest import IsolationForest
from src.anomaly_detection.lof import LocalOutlierFactor

rng = np.random.default_rng(0)
X = np.vstack([rng.standard_normal((200, 2)), rng.uniform(5, 6, (20, 2))])

for name, model in [("IsolationForest", IsolationForest(contamination=0.1, random_state=0)),
                    ("LOF", LocalOutlierFactor(contamination=0.1))]:
    pred = model.fit_predict(X)
    print(f"{name:16s} anomalies detected: {(pred == -1).sum()}")
