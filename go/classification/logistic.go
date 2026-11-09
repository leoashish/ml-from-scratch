package classification

import (
	"math"

	"mlfromscratch/mat"
)

type LogisticRegression struct {
	LR           float64
	NIters       int
	L2           float64
	FitIntercept bool
	Multi        string // "softmax" (binary via sigmoid)

	W        mat.Vec   // binary weights including bias
	Coef     mat.Vec
	Intercept float64
	Classes  []float64
	LossHist []float64
	// multiclass
	Wm       mat.Matrix // (d+1) x K
	IsBinary bool
}

func NewLogisticRegression(lr float64, nIters int, l2 float64, fitIntercept bool) *LogisticRegression {
	if lr == 0 {
		lr = 0.1
	}
	if nIters == 0 {
		nIters = 1000
	}
	return &LogisticRegression{LR: lr, NIters: nIters, L2: l2, FitIntercept: fitIntercept, Multi: "softmax", IsBinary: true}
}

func sigmoid(x float64) float64 {
	if x < -30 {
		return 0
	}
	if x > 30 {
		return 1
	}
	return 1.0 / (1.0 + math.Exp(-x))
}

func (m *LogisticRegression) addBias(X mat.Matrix) mat.Matrix {
	if !m.FitIntercept {
		return X
	}
	r := mat.NewMatrix(X.Rows(), X.Cols()+1)
	for i := 0; i < X.Rows(); i++ {
		r[i][0] = 1
		for j := 0; j < X.Cols(); j++ {
			r[i][j+1] = X[i][j]
		}
	}
	return r
}

func (m *LogisticRegression) Fit(X mat.Matrix, y mat.Vec) {
	// detect classes
	classSet := map[float64]bool{}
	for _, v := range y {
		classSet[v] = true
	}
	m.Classes = make([]float64, 0, len(classSet))
	for c := range classSet {
		m.Classes = append(m.Classes, c)
	}
	// sort
	for i := 1; i < len(m.Classes); i++ {
		for j := i; j > 0 && m.Classes[j-1] > m.Classes[j]; j-- {
			m.Classes[j-1], m.Classes[j] = m.Classes[j], m.Classes[j-1]
		}
	}
	K := len(m.Classes)
	Xb := m.addBias(X)
	n, d := Xb.Rows(), Xb.Cols()

	if K == 2 {
		m.IsBinary = true
		// map y to 0/1 where class[1]=1
		yb := make(mat.Vec, n)
		for i := 0; i < n; i++ {
			if y[i] == m.Classes[1] {
				yb[i] = 1
			}
		}
		w := mat.NewVec(d)
		for it := 0; it < m.NIters; it++ {
			// z = Xb w
			z := mat.MatVec(Xb, w)
			p := make(mat.Vec, n)
			for i := 0; i < n; i++ {
				p[i] = sigmoid(z[i])
			}
			grad := mat.NewVec(d)
			for i := 0; i < n; i++ {
				for j := 0; j < d; j++ {
					grad[j] += Xb[i][j] * (p[i] - yb[i])
				}
			}
			for j := 0; j < d; j++ {
				grad[j] /= float64(n)
				if m.L2 > 0 {
					reg := m.L2 / float64(n) * w[j]
					if m.FitIntercept && j == 0 {
						reg = 0
					}
					grad[j] += reg
				}
				w[j] -= m.LR * grad[j]
			}
			// loss
			eps := 1e-12
			loss := 0.0
			for i := 0; i < n; i++ {
				loss += -(yb[i]*math.Log(p[i]+eps) + (1-yb[i])*math.Log(1-p[i]+eps))
			}
			m.LossHist = append(m.LossHist, loss/float64(n))
		}
		m.W = w
		if m.FitIntercept {
			m.Intercept = w[0]
			m.Coef = make(mat.Vec, d-1)
			copy(m.Coef, w[1:])
		} else {
			m.Intercept = 0
			m.Coef = w.Clone()
		}
		return
	}

	// multiclass softmax
	m.IsBinary = false
	Y := mat.NewMatrix(n, K)
	for i := 0; i < n; i++ {
		for k := 0; k < K; k++ {
			if y[i] == m.Classes[k] {
				Y[i][k] = 1
			}
		}
	}
	W := mat.NewMatrix(d, K)
	for it := 0; it < m.NIters; it++ {
		// Z = Xb @ W  (n x K)
		Z := mat.Mul(Xb, W)
		// stable softmax
		P := mat.NewMatrix(n, K)
		for i := 0; i < n; i++ {
			mx := Z[i][0]
			for k := 1; k < K; k++ {
				if Z[i][k] > mx {
					mx = Z[i][k]
				}
			}
			s := 0.0
			for k := 0; k < K; k++ {
				P[i][k] = math.Exp(Z[i][k] - mx)
				s += P[i][k]
			}
			for k := 0; k < K; k++ {
				P[i][k] /= s
			}
		}
		// grad = Xb^T (P-Y)/n
		grad := mat.NewMatrix(d, K)
		for i := 0; i < n; i++ {
			for j := 0; j < d; j++ {
				for k := 0; k < K; k++ {
					grad[j][k] += Xb[i][j] * (P[i][k] - Y[i][k])
				}
			}
		}
		for j := 0; j < d; j++ {
			for k := 0; k < K; k++ {
				grad[j][k] /= float64(n)
				if m.L2 > 0 {
					reg := m.L2 / float64(n) * W[j][k]
					if m.FitIntercept && j == 0 {
						reg = 0
					}
					grad[j][k] += reg
				}
				W[j][k] -= m.LR * grad[j][k]
			}
		}
	}
	m.Wm = W
	if m.FitIntercept {
		m.Intercept = 0 // store as vector? We'll keep in Wm row 0
		m.Coef = nil
	}
}

func (m *LogisticRegression) PredictProba(X mat.Matrix) mat.Matrix {
	Xb := m.addBias(X)
	if m.IsBinary {
		z := mat.MatVec(Xb, m.W)
		out := mat.NewMatrix(X.Rows(), 2)
		for i := 0; i < X.Rows(); i++ {
			p := sigmoid(z[i])
			out[i][0] = 1 - p
			out[i][1] = p
		}
		return out
	}
	Z := mat.Mul(Xb, m.Wm)
	P := mat.NewMatrix(X.Rows(), len(m.Classes))
	for i := 0; i < X.Rows(); i++ {
		mx := Z[i][0]
		for k := 1; k < len(m.Classes); k++ {
			if Z[i][k] > mx {
				mx = Z[i][k]
			}
		}
		s := 0.0
		for k := 0; k < len(m.Classes); k++ {
			P[i][k] = math.Exp(Z[i][k] - mx)
			s += P[i][k]
		}
		for k := 0; k < len(m.Classes); k++ {
			P[i][k] /= s
		}
	}
	return P
}

func (m *LogisticRegression) Predict(X mat.Matrix) mat.Vec {
	proba := m.PredictProba(X)
	out := make(mat.Vec, X.Rows())
	for i := 0; i < X.Rows(); i++ {
		// argmax
		best := 0
		for k := 1; k < len(m.Classes); k++ {
			if proba[i][k] > proba[i][best] {
				best = k
			}
		}
		out[i] = m.Classes[best]
	}
	return out
}
