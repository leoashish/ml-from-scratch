"""ARIMA(p,d,q) from first principles.

Approach:
- Difference the series d times (store last values for inversion).
- Fit AR(p) coefficients via Yule-Walker (stable).
- Fit MA(q) coefficients by regressing residuals on lagged residuals.
- Forecast iteratively and invert differencing.
"""

import numpy as np


class ARIMA:
    def __init__(self, p=1, d=1, q=1):
        self.p = p
        self.d = d
        self.q = q
        self.ar_coef_ = None
        self.ma_coef_ = None
        self.intercept_ = 0.0
        self.residuals_ = None
        self.diff_series_ = None
        self.history_ = None
        self._last_vals = None  # last values of each differencing level

    def _difference(self, series, d):
        s = np.asarray(series, float).copy()
        last_vals = [s[-1]] if False else []
        # store last value before each diff
        levels = [s.copy()]
        for _ in range(d):
            last_vals.append(s[0])  # not used
            # store last d values? We'll store last value of current series
            last_vals_level = s[-1]
            last_vals.append(last_vals_level)
            s = np.diff(s)
            levels.append(s.copy())
        return s, last_vals, levels

    def fit(self, series):
        series = np.asarray(series, float).ravel()
        self.history_ = series.copy()
        s, last_vals, levels = self._difference(series, self.d)
        self.diff_series_ = s
        self._levels = levels
        p, q = self.p, self.q
        n = len(s)
        # center
        mean_s = float(np.mean(s))
        sc = s - mean_s

        # Yule-Walker for AR
        ar = np.zeros(p)
        if p > 0:
            # autocorrelations
            r = np.array([np.correlate(sc, sc, mode="full")[len(sc) - 1 - k] / n for k in range(p + 1)])
            # solve Toeplitz
            from numpy.linalg import solve
            R = np.zeros((p, p))
            for i in range(p):
                for j in range(p):
                    R[i, j] = r[abs(i - j)]
            rhs = r[1:]
            try:
                ar = solve(R + 1e-6 * np.eye(p), rhs)
            except np.linalg.LinAlgError:
                ar = np.zeros(p)
        # residuals from AR
        resid = np.zeros(n)
        for t in range(n):
            val = mean_s
            for i in range(1, p + 1):
                if t - i >= 0:
                    val += ar[i - 1] * sc[t - i]
            resid[t] = s[t] - val
        # MA: regress resid[t] on resid[t-1..t-q]
        ma = np.zeros(q)
        if q > 0:
            rows = []
            targets = []
            for t in range(q, n):
                row = [resid[t - j] for j in range(1, q + 1)]
                rows.append(row)
                targets.append(resid[t])
            A = np.array(rows)
            b = np.array(targets)
            if len(A) > 0:
                coef, *_ = np.linalg.lstsq(A, b, rcond=None)
                ma = coef
        # recompute residuals with MA included (a few iterations)
        for _ in range(3):
            resid = np.zeros(n)
            for t in range(n):
                val = mean_s
                for i in range(1, p + 1):
                    if t - i >= 0:
                        val += ar[i - 1] * (s[t - i] - mean_s)
                for j in range(1, q + 1):
                    if t - j >= 0:
                        val += ma[j - 1] * resid[t - j]
                resid[t] = s[t] - val
        self.ar_coef_ = ar
        self.ma_coef_ = ma
        self.intercept_ = mean_s
        self.residuals_ = resid
        return self

    def forecast(self, steps=10):
        s_hist = list(self.diff_series_.copy())
        resid_hist = list(self.residuals_.copy())
        mean_s = self.intercept_
        p, q = self.p, self.q
        ar = self.ar_coef_
        ma = self.ma_coef_
        out_diff = []
        for _ in range(steps):
            t = len(s_hist)
            val = mean_s
            for i in range(1, p + 1):
                if t - i >= 0 and (t - i) < len(s_hist):
                    val += ar[i - 1] * (s_hist[-i] - mean_s)
            for j in range(1, q + 1):
                if len(resid_hist) - j >= 0:
                    val += ma[j - 1] * resid_hist[-j]
            out_diff.append(val)
            s_hist.append(val)
            resid_hist.append(0.0)
        out_diff = np.array(out_diff)
        # invert differencing
        # levels: levels[0]=original, levels[1]=1st diff, ...
        # For d=1: y = last_original + cumsum(diff)
        if self.d == 0:
            return out_diff
        if self.d == 1:
            cur = self.history_[-1]
            inv = []
            for df in out_diff:
                cur = cur + df
                inv.append(cur)
            return np.array(inv)
        # d>1: invert level by level
        # last value of series at each level
        # levels[k] is k-th differenced series; to invert, start from last value of levels[d-1]? 
        # General: invert from highest diff down to original using last values of each level
        inv = out_diff.copy()
        # level from d-1 down to 0
        for level in range(self.d - 1, -1, -1):
            # last value of the series at this level before forecast
            if level == 0:
                last = self.history_[-1]
            else:
                last = self._levels[level][-1] if hasattr(self, "_levels") else 0.0
            cur = last
            new = []
            for df in inv:
                cur = cur + df
                new.append(cur)
            inv = np.array(new)
        return inv
