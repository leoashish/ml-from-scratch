package regression

import (
	"mlfromscratch/mat"
)

// LinearRegression supports closed-form (normal) and gradient descent.
type LinearRegression struct {
	Method      string // "normal" or "gd"
	LR          float64
	NIters      int
	L2          float64
	FitIntercept bool

	Coef      mat.Vec
	Intercept float64
	W         mat.Vec // includes bias if FitIntercept
	LossHist  []float64
}

func NewLinearRegression(method string, lr float64, nIters int, l2 float64, fitIntercept bool) *LinearRegression {
	if method == "" {
		method = "normal"
	}
	if lr == 0 {
		lr = 0.01
	}
	if nIters == 0 {
		nIters = 1000
	}
	return &LinearRegression{Method: method, LR: lr, NIters: nIters, L2: l2, FitIntercept: fitIntercept}
}

func (m *LinearRegression) addBias(X mat.Matrix) mat.Matrix {
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

func (m *LinearRegression) Fit(X mat.Matrix, y mat.Vec) {
	Xb := m.addBias(X)
	n, d := Xb.Rows(), Xb.Cols()
	if m.Method == "normal" {
		// A = Xb^T Xb + l2 I (no reg on bias)
		XtX := mat.Mul(mat.Transpose(Xb), Xb)
		A := XtX.Clone()
		for i := 0; i < d; i++ {
			reg := m.L2
			if m.FitIntercept && i == 0 {
				reg = 0
			}
			A[i][i] += reg
		}
		Xty := mat.MatVec(mat.Transpose(Xb), y)
		w, _ := mat.Solve(A, Xty)
		m.W = w
	} else {
		m.W = mat.NewVec(d)
		for it := 0; it < m.NIters; it++ {
			pred := mat.MatVec(Xb, m.W)
			// grad = 2/n Xb^T(pred-y)
			res := mat.VecSub(pred, y)
			grad := mat.MatVec(mat.Transpose(Xb), res)
			for i := range grad {
				grad[i] *= 2.0 / float64(n)
			}
			if m.L2 > 0 {
				for i := range m.W {
					reg := m.L2 * 2 / float64(n) * m.W[i]
					if m.FitIntercept && i == 0 {
						reg = 0
					}
					grad[i] += reg
				}
			}
			for i := range m.W {
				m.W[i] -= m.LR * grad[i]
			}
			// loss
			pred2 := mat.MatVec(Xb, m.W)
			s := 0.0
			for i := range pred2 {
				dd := pred2[i] - y[i]
				s += dd * dd
			}
			m.LossHist = append(m.LossHist, s/float64(n))
		}
	}
	if m.FitIntercept {
		m.Intercept = m.W[0]
		m.Coef = make(mat.Vec, d-1)
		copy(m.Coef, m.W[1:])
	} else {
		m.Intercept = 0
		m.Coef = m.W.Clone()
	}
}

func (m *LinearRegression) Predict(X mat.Matrix) mat.Vec {
	out := make(mat.Vec, X.Rows())
	for i := 0; i < X.Rows(); i++ {
		s := m.Intercept
		for j := 0; j < X.Cols(); j++ {
			s += m.Coef[j] * X[i][j]
		}
		out[i] = s
	}
	return out
}
