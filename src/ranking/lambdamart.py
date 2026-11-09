"""LambdaMART: gradient boosted decision trees for ranking (NDCG optimization).

Implements the LambdaMART algorithm: at each boosting round, for every query
we compute per-document lambda-gradients (pairwise, weighted by change in NDCG)
and fit a regression tree; leaf values use a Newton (mu) step.
"""

import numpy as np


class _RankTree:
    def __init__(self, max_depth=3, min_samples_leaf=5):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree_ = None

    def fit(self, X, lam, mu):
        X = np.asarray(X, float)
        lam = np.asarray(lam, float).ravel()
        mu = np.asarray(mu, float).ravel()
        self.tree_ = self._build(X, lam, mu, depth=0)
        return self

    def _build(self, X, lam, mu, depth):
        node = {"value": float(np.sum(lam) / (np.sum(mu) + 1e-6))}
        if depth >= self.max_depth or len(lam) < 2 * self.min_samples_leaf:
            return node
        best = self._best_split(X, lam, mu)
        if best is None:
            return node
        feat, thr = best
        left_mask = X[:, feat] <= thr
        right_mask = ~left_mask
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            return node
        node["feature"] = int(feat)
        node["threshold"] = float(thr)
        node["left"] = self._build(X[left_mask], lam[left_mask], mu[left_mask], depth + 1)
        node["right"] = self._build(X[right_mask], lam[right_mask], mu[right_mask], depth + 1)
        return node

    def _best_split(self, X, lam, mu):
        n, d = X.shape
        # split criterion: maximize reduction in weighted variance of lambdas
        total_lam = lam.sum()
        total_mu = mu.sum() + 1e-6
        # use variance reduction weighted by mu
        best_gain = 0.0
        best = None
        for feat in range(d):
            vals = np.unique(X[:, feat])
            if len(vals) <= 1:
                continue
            thresholds = (vals[1:] + vals[:-1]) / 2
            for thr in thresholds:
                lm = X[:, feat] <= thr
                rm = ~lm
                if lm.sum() < self.min_samples_leaf or rm.sum() < self.min_samples_leaf:
                    continue
                # gain = weighted variance reduction
                def wvar(mask):
                    w = mu[mask]
                    if w.sum() <= 0:
                        return 0.0
                    mean = np.sum(w * lam[mask]) / np.sum(w)
                    return np.sum(w * (lam[mask] - mean) ** 2)
                parent = wvar(np.ones(n, dtype=bool))
                gain = parent - (wvar(lm) + wvar(rm))
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best = (feat, thr)
        return best

    def predict(self, X):
        X = np.asarray(X, float)
        out = np.zeros(len(X))
        for i in range(len(X)):
            out[i] = self._predict_one(X[i], self.tree_)
        return out

    def _predict_one(self, x, node):
        if "feature" not in node:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])


class LambdaMART:
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3,
                 min_samples_leaf=5, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees_ = []
        self.init_ = 0.0

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def fit(self, X, y, qid):
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        qid = np.asarray(qid).ravel()
        n = len(y)
        F = np.zeros(n)
        rng = np.random.default_rng(self.random_state)
        queries = np.unique(qid)

        for m in range(self.n_estimators):
            lambdas = np.zeros(n)
            mus = np.zeros(n)
            lam = lambdas
            mu = mus
            # compute per-query lambdas
            for q in queries:
                mask = qid == q
                idx = np.where(mask)[0]
                scores = F[idx]
                rels = y[idx]
                # sort by score desc
                order = np.argsort(scores)[::-1]
                # ideal dcg
                # compute lambda for each pair
                # map position
                # NDCG normalization
                # compute gain per position for ideal
                sorted_rels_ideal = np.sort(rels)[::-1]
                # discounts
                positions = np.arange(1, len(idx) + 1)
                discounts = 1.0 / np.log2(positions + 1)
                # ideal dcg
                idcg = np.sum((2 ** sorted_rels_ideal - 1) * discounts)
                if idcg == 0:
                    idcg = 1.0
                # for each pair i (higher should be above j)
                for a in range(len(idx)):
                    for b in range(len(idx)):
                        if rels[order[a]] <= rels[order[b]]:
                            continue
                        # a has higher relevance than b; want score[a] > score[b]
                        # current: positions pa, pb in 'order'
                        ia = order[a]
                        ib = order[b]
                        # positions (0-based) in sorted order
                        pa = a
                        pb = b
                        if pa < pb:
                            # already correct order, maybe still gradient small; lambdaMART only swaps? 
                            # We still add gradient for pairs out of order in general formulation uses all pairs weighted by delta ndcg.
                            pass
                        # delta NDCG if swap positions pa and pb
                        # gains
                        gain_a = (2 ** rels[ia] - 1)
                        gain_b = (2 ** rels[ib] - 1)
                        # dcg contribution change approx
                        disc_pa = discounts[pa]
                        disc_pb = discounts[pb]
                        delta_ndcg = (gain_a * (disc_pb - disc_pa) + gain_b * (disc_pa - disc_pb)) / idcg
                        # abs
                        delta = abs(delta_ndcg)
                        if delta == 0:
                            continue
                        # pairwise lambda
                        s_diff = scores[ia] - scores[ib]
                        # rho = 1/(1+exp(s_diff))
                        rho = 1.0 / (1.0 + np.exp(s_diff))
                        # gradient contribution
                        lam[ia] += delta * rho
                        lam[ib] -= delta * rho
                        # mu (second deriv) ~ rho*(1-rho)*delta
                        mus[ia] += delta * rho * (1 - rho)
                        mus[ib] += delta * rho * (1 - rho)
            # ensure mus positive
            mus = np.clip(mus, 1e-6, None)
            # fit tree
            tree = _RankTree(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
            tree.fit(X, lambdas, mus)
            update = tree.predict(X)
            F += self.learning_rate * update
            self.trees_.append(tree)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        F = np.zeros(len(X))
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
        return F
