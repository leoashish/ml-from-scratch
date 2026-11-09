package classification

import (
	"math"
	"math/rand"

	"mlfromscratch/mat"
)

// LinearSVM via hinge loss subgradient.
type LinearSVM struct {
	C            float64
	LR           float64
	NIters       int
	FitIntercept bool
	W            mat.Vec
	B            float64
	Classes      []float64
}

func NewLinearSVM(C, lr float64, nIters int, fitIntercept bool) *LinearSVM {
	if C == 0 {
		C = 1
	}
	if lr == 0 {
		lr = 0.001
	}
	if nIters == 0 {
		nIters = 1000
	}
	return &LinearSVM{C: C, LR: lr, NIters: nIters, FitIntercept: fitIntercept}
}

func (m *LinearSVM) Fit(X mat.Matrix, y mat.Vec) {
	// classes
	m.Classes = []float64{0, 1}
	// detect
	set := map[float64]bool{}
	for _, v := range y {
		set[v] = true
	}
	m.Classes = m.Classes[:0]
	for c := range set {
		m.Classes = append(m.Classes, c)
	}
	for i := 1; i < len(m.Classes); i++ {
		for j := i; j > 0 && m.Classes[j-1] > m.Classes[j]; j-- {
			m.Classes[j-1], m.Classes[j] = m.Classes[j], m.Classes[j-1]
		}
	}
	n, d := X.Rows(), X.Cols()
	yb := make(mat.Vec, n)
	for i := 0; i < n; i++ {
		if y[i] == m.Classes[1] {
			yb[i] = 1
		} else {
			yb[i] = -1
		}
	}
	w := mat.NewVec(d)
	b := 0.0
	for it := 0; it < m.NIters; it++ {
		// vectorized subgradient
		margin := make(mat.Vec, n)
		for i := 0; i < n; i++ {
			s := b
			for j := 0; j < d; j++ {
				s += w[j] * X[i][j]
			}
			margin[i] = yb[i] * s
		}
		// gradient
		dw := make(mat.Vec, d)
		db := 0.0
		for j := 0; j < d; j++ {
			dw[j] = w[j]
		}
		for i := 0; i < n; i++ {
			if margin[i] < 1 {
				for j := 0; j < d; j++ {
					dw[j] -= m.C * yb[i] * X[i][j]
				}
				db -= m.C * yb[i]
			}
		}
		for j := 0; j < d; j++ {
			dw[j] /= float64(n)
		}
		db /= float64(n)
		for j := 0; j < d; j++ {
			w[j] -= m.LR * dw[j]
		}
		b -= m.LR * db
	}
	m.W = w
	m.B = b
}

func (m *LinearSVM) Predict(X mat.Matrix) mat.Vec {
	out := make(mat.Vec, X.Rows())
	for i := 0; i < X.Rows(); i++ {
		s := m.B
		for j := 0; j < X.Cols(); j++ {
			s += m.W[j] * X[i][j]
		}
		if s >= 0 {
			out[i] = m.Classes[1]
		} else {
			out[i] = m.Classes[0]
		}
	}
	return out
}

// KernelSVM with simplified SMO (RBF/linear/poly)
type KernelSVM struct {
	C       float64
	Kernel  string
	Gamma   float64
	Degree  int
	Coef0   float64
	Tol     float64
	MaxPass int
	Seed    int64

	Alphas  mat.Vec
	B       float64
	X       mat.Matrix
	Y       mat.Vec
	Classes []float64
	GammaSet bool
}

func NewKernelSVM(C float64, kernel string, gamma float64, degree int, maxPass int, seed int64) *KernelSVM {
	if C == 0 {
		C = 1
	}
	if kernel == "" {
		kernel = "rbf"
	}
	if degree == 0 {
		degree = 3
	}
	if maxPass == 0 {
		maxPass = 5
	}
	if gamma == 0 {
		gamma = -1 // set later
	}
	return &KernelSVM{C: C, Kernel: kernel, Gamma: gamma, Degree: degree, Tol: 1e-3, MaxPass: maxPass, Seed: seed}
}

func (m *KernelSVM) kernelFunc(x1, x2 mat.Vec) float64 {
	switch m.Kernel {
	case "linear":
		return mat.Dot(x1, x2)
	case "rbf":
		if m.Gamma < 0 {
			m.Gamma = 1.0 / float64(len(x1))
		}
		sq := 0.0
		for j := range x1 {
			d := x1[j] - x2[j]
			sq += d * d
		}
		return math.Exp(-m.Gamma * sq)
	case "poly":
		return math.Pow(mat.Dot(x1, x2)+m.Coef0, float64(m.Degree))
	}
	return mat.Dot(x1, x2)
}

func (m *KernelSVM) KMatrix(X mat.Matrix) mat.Matrix {
	n := X.Rows()
	K := mat.NewMatrix(n, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			K[i][j] = m.kernelFunc(X[i], X[j])
		}
	}
	return K
}

func (m *KernelSVM) Fit(X mat.Matrix, y mat.Vec) {
	set := map[float64]bool{}
	for _, v := range y {
		set[v] = true
	}
	m.Classes = m.Classes[:0]
	for c := range set {
		m.Classes = append(m.Classes, c)
	}
	for i := 1; i < len(m.Classes); i++ {
		for j := i; j > 0 && m.Classes[j-1] > m.Classes[j]; j-- {
			m.Classes[j-1], m.Classes[j] = m.Classes[j], m.Classes[j-1]
		}
	}
	n := X.Rows()
	if m.Gamma < 0 && m.Kernel == "rbf" {
		m.Gamma = 1.0 / float64(X.Cols())
	}
	yb := make(mat.Vec, n)
	for i := 0; i < n; i++ {
		if y[i] == m.Classes[1] {
			yb[i] = 1
		} else {
			yb[i] = -1
		}
	}
	m.X = X.Clone()
	m.Y = yb
	K := m.KMatrix(X)
	alphas := mat.NewVec(n)
	b := 0.0
	rng := rand.New(rand.NewSource(m.Seed))
	passes := 0
	for passes < m.MaxPass {
		numChanged := 0
		for i := 0; i < n; i++ {
			// Ei
			Ei := 0.0
			for j := 0; j < n; j++ {
				Ei += alphas[j] * yb[j] * K[j][i]
			}
			Ei += b - yb[i]
			if (yb[i]*Ei < -m.Tol && alphas[i] < m.C) || (yb[i]*Ei > m.Tol && alphas[i] > 0) {
				j := i
				for j == i {
					j = rng.Intn(n)
				}
				Ej := 0.0
				for l := 0; l < n; l++ {
					Ej += alphas[l] * yb[l] * K[l][j]
				}
				Ej += b - yb[j]
				aiOld, ajOld := alphas[i], alphas[j]
				var L, H float64
				if yb[i] != yb[j] {
					L = math.Max(0, alphas[j]-alphas[i])
					H = math.Min(m.C, m.C+alphas[j]-alphas[i])
				} else {
					L = math.Max(0, alphas[i]+alphas[j]-m.C)
					H = math.Min(m.C, alphas[i]+alphas[j])
				}
				if L >= H {
					continue
				}
				eta := 2*K[i][j] - K[i][i] - K[j][j]
				if eta >= 0 {
					continue
				}
				alphas[j] -= yb[j] * (Ei - Ej) / eta
				if alphas[j] > H {
					alphas[j] = H
				}
				if alphas[j] < L {
					alphas[j] = L
				}
				if math.Abs(alphas[j]-ajOld) < 1e-5 {
					continue
				}
				alphas[i] += yb[i] * yb[j] * (ajOld - alphas[j])
				b1 := b - Ei - yb[i]*(alphas[i]-aiOld)*K[i][i] - yb[j]*(alphas[j]-ajOld)*K[i][j]
				b2 := b - Ej - yb[i]*(alphas[i]-aiOld)*K[i][j] - yb[j]*(alphas[j]-ajOld)*K[j][j]
				if alphas[i] > 0 && alphas[i] < m.C {
					b = b1
				} else if alphas[j] > 0 && alphas[j] < m.C {
					b = b2
				} else {
					b = (b1 + b2) / 2
				}
				numChanged++
			}
		}
		if numChanged == 0 {
			passes++
		} else {
			passes = 0
		}
	}
	m.Alphas = alphas
	m.B = b
}

func (m *KernelSVM) Decision(X mat.Matrix) mat.Vec {
	out := make(mat.Vec, X.Rows())
	for i := 0; i < X.Rows(); i++ {
		s := m.B
		for j := 0; j < m.X.Rows(); j++ {
			s += m.Alphas[j] * m.Y[j] * m.kernelFunc(X[i], m.X[j])
		}
		out[i] = s
	}
	return out
}

func (m *KernelSVM) Predict(X mat.Matrix) mat.Vec {
	df := m.Decision(X)
	out := make(mat.Vec, X.Rows())
	for i := range df {
		if df[i] >= 0 {
			out[i] = m.Classes[1]
		} else {
			out[i] = m.Classes[0]
		}
	}
	return out
}
