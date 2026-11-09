package forecasting

import (
	"mlfromscratch/mat"
	"mlfromscratch/regression"
)

type BoostingForecaster struct {
	NLags        int
	NEstimators  int
	LearningRate float64
	MaxDepth     int
	Seed         int64
	Model        *regression.GradientBoostingRegressor
	History      mat.Vec
}

func NewBoostingForecaster(nLags, nEst, maxDepth int, lr float64, seed int64) *BoostingForecaster {
	if nLags == 0 {
		nLags = 12
	}
	if nEst == 0 {
		nEst = 100
	}
	if maxDepth == 0 {
		maxDepth = 3
	}
	if lr == 0 {
		lr = 0.1
	}
	return &BoostingForecaster{NLags: nLags, NEstimators: nEst, LearningRate: lr, MaxDepth: maxDepth, Seed: seed}
}

func (f *BoostingForecaster) makeFeatures(series mat.Vec) (mat.Matrix, mat.Vec) {
	n := len(series)
	if n <= f.NLags {
		return nil, nil
	}
	rows := n - f.NLags
	X := mat.NewMatrix(rows, f.NLags)
	y := make(mat.Vec, rows)
	for t := f.NLags; t < n; t++ {
		for lag := 0; lag < f.NLags; lag++ {
			X[t-f.NLags][lag] = series[t-1-lag]
		}
		y[t-f.NLags] = series[t]
	}
	return X, y
}

func (f *BoostingForecaster) Fit(series mat.Vec) {
	f.History = series.Clone()
	X, y := f.makeFeatures(series)
	f.Model = regression.NewGradientBoostingRegressor(f.NEstimators, f.LearningRate, f.MaxDepth, 1, f.Seed)
	f.Model.Fit(X, y)
}

func (f *BoostingForecaster) Forecast(steps int) mat.Vec {
	hist := f.History.Clone()
	out := make(mat.Vec, steps)
	for s := 0; s < steps; s++ {
		feat := mat.NewMatrix(1, f.NLags)
		for lag := 0; lag < f.NLags; lag++ {
			feat[0][lag] = hist[len(hist)-1-lag]
		}
		pred := f.Model.Predict(feat)
		out[s] = pred[0]
		hist = append(hist, pred[0])
	}
	return out
}
