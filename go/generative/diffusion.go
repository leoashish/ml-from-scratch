package generative

import (
	"math"
	"math/rand"

	"mlfromscratch/mat"
)

type DiffusionModel struct {
	DataDim  int
	Hidden   int
	T        int
	BetaStart float64
	BetaEnd   float64
	LR       float64
	NIters   int
	BatchSize int
	Seed     int64

	Betas     []float64
	Alphas    []float64
	AlphaBar  []float64

	// MLP weights: input (dataDim+1) -> hidden -> hidden -> dataDim
	W1 mat.Matrix
	B1 mat.Vec
	W2 mat.Matrix
	B2 mat.Vec
	W3 mat.Matrix
	B3 mat.Vec
}

func NewDiffusion(dataDim, hidden, T int, betaStart, betaEnd, lr float64, nIters, batchSize int, seed int64) *DiffusionModel {
	if hidden == 0 {
		hidden = 64
	}
	if T == 0 {
		T = 50
	}
	if betaStart == 0 {
		betaStart = 1e-4
	}
	if betaEnd == 0 {
		betaEnd = 0.02
	}
	if lr == 0 {
		lr = 1e-3
	}
	if nIters == 0 {
		nIters = 2000
	}
	if batchSize == 0 {
		batchSize = 64
	}
	m := &DiffusionModel{DataDim: dataDim, Hidden: hidden, T: T, BetaStart: betaStart, BetaEnd: betaEnd, LR: lr, NIters: nIters, BatchSize: batchSize, Seed: seed}
	m.Betas = make([]float64, T)
	m.Alphas = make([]float64, T)
	m.AlphaBar = make([]float64, T)
	for t := 0; t < T; t++ {
		m.Betas[t] = betaStart + (betaEnd-betaStart)*float64(t)/float64(T-1)
		m.Alphas[t] = 1 - m.Betas[t]
	}
	ab := 1.0
	for t := 0; t < T; t++ {
		ab *= m.Alphas[t]
		m.AlphaBar[t] = ab
	}
	rng := rand.New(rand.NewSource(seed))
	inDim := dataDim + 1
	m.W1 = randnMat(inDim, hidden, rng)
	m.B1 = make(mat.Vec, hidden)
	m.W2 = randnMat(hidden, hidden, rng)
	m.B2 = make(mat.Vec, hidden)
	m.W3 = randnMat(hidden, dataDim, rng)
	m.B3 = make(mat.Vec, dataDim)
	for i := range m.W1 {
		for j := range m.W1[i] {
			m.W1[i][j] *= 0.01
		}
	}
	for i := range m.W2 {
		for j := range m.W2[i] {
			m.W2[i][j] *= 0.01
		}
	}
	for i := range m.W3 {
		for j := range m.W3[i] {
			m.W3[i][j] *= 0.01
		}
	}
	return m
}

func randnMat(r, c int, rng *rand.Rand) mat.Matrix {
	m := mat.NewMatrix(r, c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			// box-muller
			u1 := rng.Float64()
			u2 := rng.Float64()
			m[i][j] = math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
		}
	}
	return m
}

func (m *DiffusionModel) forward(x mat.Matrix, t []int) mat.Matrix {
	// x: (batch, dataDim), t: (batch,)
	batch := x.Rows()
	inDim := m.DataDim + 1
	inp := mat.NewMatrix(batch, inDim)
	for i := 0; i < batch; i++ {
		for j := 0; j < m.DataDim; j++ {
			inp[i][j] = x[i][j]
		}
		inp[i][m.DataDim] = float64(t[i]) / float64(m.T)
	}
	// layer1
	h1 := mat.NewMatrix(batch, m.Hidden)
	for i := 0; i < batch; i++ {
		for j := 0; j < m.Hidden; j++ {
			s := m.B1[j]
			for k := 0; k < inDim; k++ {
				s += inp[i][k] * m.W1[k][j]
			}
			if s > 0 {
				h1[i][j] = s
			}
		}
	}
	h2 := mat.NewMatrix(batch, m.Hidden)
	for i := 0; i < batch; i++ {
		for j := 0; j < m.Hidden; j++ {
			s := m.B2[j]
			for k := 0; k < m.Hidden; k++ {
				s += h1[i][k] * m.W2[k][j]
			}
			if s > 0 {
				h2[i][j] = s
			}
		}
	}
	out := mat.NewMatrix(batch, m.DataDim)
	for i := 0; i < batch; i++ {
		for j := 0; j < m.DataDim; j++ {
			s := m.B3[j]
			for k := 0; k < m.Hidden; k++ {
				s += h2[i][k] * m.W3[k][j]
			}
			out[i][j] = s
		}
	}
	return out
}

func (m *DiffusionModel) Fit(X mat.Matrix) {
	n := X.Rows()
	rng := rand.New(rand.NewSource(m.Seed + 7))
	for it := 0; it < m.NIters; it++ {
		// batch
		batch := m.BatchSize
		if batch > n {
			batch = n
		}
		x0 := mat.NewMatrix(batch, m.DataDim)
		tt := make([]int, batch)
		eps := mat.NewMatrix(batch, m.DataDim)
		for i := 0; i < batch; i++ {
			idx := rng.Intn(n)
			for j := 0; j < m.DataDim; j++ {
				x0[i][j] = X[idx][j]
				eps[i][j] = randn(rng)
			}
			tt[i] = 1 + rng.Intn(m.T-1)
		}
		// x_t
		xt := mat.NewMatrix(batch, m.DataDim)
		for i := 0; i < batch; i++ {
			t := tt[i]
			ab := m.AlphaBar[t-1]
			sa := math.Sqrt(ab)
			sb := math.Sqrt(1 - ab)
			for j := 0; j < m.DataDim; j++ {
				xt[i][j] = sa*x0[i][j] + sb*eps[i][j]
			}
		}
		pred := m.forward(xt, tt)
		// gradient of MSE(pred, eps): d = 2*(pred-eps)/batch
		// backprop manually
		// recompute intermediates
		inDim := m.DataDim + 1
		inp := mat.NewMatrix(batch, inDim)
		for i := 0; i < batch; i++ {
			for j := 0; j < m.DataDim; j++ {
				inp[i][j] = xt[i][j]
			}
			inp[i][m.DataDim] = float64(tt[i]) / float64(m.T)
		}
		h1pre := mat.NewMatrix(batch, m.Hidden)
		h1 := mat.NewMatrix(batch, m.Hidden)
		for i := 0; i < batch; i++ {
			for j := 0; j < m.Hidden; j++ {
				s := m.B1[j]
				for k := 0; k < inDim; k++ {
					s += inp[i][k] * m.W1[k][j]
				}
				h1pre[i][j] = s
				if s > 0 {
					h1[i][j] = s
				}
			}
		}
		h2pre := mat.NewMatrix(batch, m.Hidden)
		h2 := mat.NewMatrix(batch, m.Hidden)
		for i := 0; i < batch; i++ {
			for j := 0; j < m.Hidden; j++ {
				s := m.B2[j]
				for k := 0; k < m.Hidden; k++ {
					s += h1[i][k] * m.W2[k][j]
				}
				h2pre[i][j] = s
				if s > 0 {
					h2[i][j] = s
				}
			}
		}
		// diff matrix (batch, dataDim)
		diff := mat.NewMatrix(batch, m.DataDim)
		for i := 0; i < batch; i++ {
			for j := 0; j < m.DataDim; j++ {
				diff[i][j] = (pred[i][j] - eps[i][j]) * 2 / float64(batch)
			}
		}
		// grad W3, B3
		dW3 := mat.NewMatrix(m.Hidden, m.DataDim)
		db3 := make(mat.Vec, m.DataDim)
		for i := 0; i < batch; i++ {
			for k := 0; k < m.Hidden; k++ {
				for j := 0; j < m.DataDim; j++ {
					dW3[k][j] += h2[i][k] * diff[i][j]
				}
			}
			for j := 0; j < m.DataDim; j++ {
				db3[j] += diff[i][j]
			}
		}
		// dh2
		dh2 := mat.NewMatrix(batch, m.Hidden)
		for i := 0; i < batch; i++ {
			for k := 0; k < m.Hidden; k++ {
				s := 0.0
				for j := 0; j < m.DataDim; j++ {
					s += diff[i][j] * m.W3[k][j]
				}
				if h2pre[i][k] > 0 {
					dh2[i][k] = s
				}
			}
		}
		dW2 := mat.NewMatrix(m.Hidden, m.Hidden)
		db2 := make(mat.Vec, m.Hidden)
		for i := 0; i < batch; i++ {
			for a := 0; a < m.Hidden; a++ {
				for b := 0; b < m.Hidden; b++ {
					dW2[a][b] += h1[i][a] * dh2[i][b]
				}
				db2[a] += 0 // placeholder
			}
		}
		for i := 0; i < batch; i++ {
			for b := 0; b < m.Hidden; b++ {
				db2[b] += dh2[i][b]
			}
		}
		// dh1
		dh1 := mat.NewMatrix(batch, m.Hidden)
		for i := 0; i < batch; i++ {
			for a := 0; a < m.Hidden; a++ {
				s := 0.0
				for b := 0; b < m.Hidden; b++ {
					s += dh2[i][b] * m.W2[a][b]
				}
				if h1pre[i][a] > 0 {
					dh1[i][a] = s
				}
			}
		}
		dW1 := mat.NewMatrix(inDim, m.Hidden)
		db1 := make(mat.Vec, m.Hidden)
		for i := 0; i < batch; i++ {
			for a := 0; a < inDim; a++ {
				for b := 0; b < m.Hidden; b++ {
					dW1[a][b] += inp[i][a] * dh1[i][b]
				}
			}
			for b := 0; b < m.Hidden; b++ {
				db1[b] += dh1[i][b]
			}
		}
		// update
		for i := 0; i < inDim; i++ {
			for j := 0; j < m.Hidden; j++ {
				m.W1[i][j] -= m.LR * dW1[i][j]
			}
		}
		for j := 0; j < m.Hidden; j++ {
			m.B1[j] -= m.LR * db1[j]
		}
		for i := 0; i < m.Hidden; i++ {
			for j := 0; j < m.Hidden; j++ {
				m.W2[i][j] -= m.LR * dW2[i][j]
			}
		}
		for j := 0; j < m.Hidden; j++ {
			m.B2[j] -= m.LR * db2[j]
		}
		for i := 0; i < m.Hidden; i++ {
			for j := 0; j < m.DataDim; j++ {
				m.W3[i][j] -= m.LR * dW3[i][j]
			}
		}
		for j := 0; j < m.DataDim; j++ {
			m.B3[j] -= m.LR * db3[j]
		}
	}
}

func randn(rng *rand.Rand) float64 {
	u1 := rng.Float64()
	u2 := rng.Float64()
	return math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
}

func (m *DiffusionModel) Sample(n int, seed int64) mat.Matrix {
	rng := rand.New(rand.NewSource(seed))
	x := mat.NewMatrix(n, m.DataDim)
	for i := 0; i < n; i++ {
		for j := 0; j < m.DataDim; j++ {
			x[i][j] = randn(rng)
		}
	}
	for t := m.T; t >= 1; t-- {
		z := mat.NewMatrix(n, m.DataDim)
		if t > 1 {
			for i := 0; i < n; i++ {
				for j := 0; j < m.DataDim; j++ {
					z[i][j] = randn(rng)
				}
			}
		}
		ab := m.AlphaBar[t-1]
		abPrev := 1.0
		if t > 1 {
			abPrev = m.AlphaBar[t-2]
		}
		betaT := m.Betas[t-1]
		alphaT := m.Alphas[t-1]
		tt := make([]int, n)
		for i := 0; i < n; i++ {
			tt[i] = t
		}
		epsPred := m.forward(x, tt)
		// mean
		coef := (1 - alphaT) / math.Sqrt(1-ab)
		for i := 0; i < n; i++ {
			for j := 0; j < m.DataDim; j++ {
				mean := (x[i][j] - coef*epsPred[i][j]) / math.Sqrt(alphaT)
				variance := betaT * (1 - abPrev) / (1 - ab)
				x[i][j] = mean + math.Sqrt(variance)*z[i][j]
			}
		}
	}
	return x
}
