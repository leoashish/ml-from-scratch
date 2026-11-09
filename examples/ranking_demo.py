"""Demo: ranking with LambdaMART."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.utils.data import make_ranking
from src.utils.metrics import ndcg_at_k
from src.ranking.lambdamart import LambdaMART

X, y, qid = make_ranking(n_queries=30, n_items_per_query=10, random_state=0)
lm = LambdaMART(n_estimators=30, max_depth=3, random_state=0)
lm.fit(X, y, qid)
pred = lm.predict(X)
ndcgs = [ndcg_at_k(y[qid == q], pred[qid == q], k=5) for q in np.unique(qid)]
print("LambdaMART mean NDCG@5:", round(float(np.mean(ndcgs)), 4))
