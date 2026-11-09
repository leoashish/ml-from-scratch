package dimred

import (
	"math"

	"mlfromscratch/mat"
)

type PCA struct {
	NComponents int
	Components  mat.Matrix // k x d
	Mean        mat.Vec
	Explained   mat.Vec
}

func NewPCA(n int) *PCA {
	return &PCA{NComponents: n}
}

func (p *PCA) Fit(X mat.Matrix) {
	n, d := X.Rows(), X.Cols()
	if p.NComponents == 0 {
		p.NComponents = d
	}
	// mean
	mean := mat.ColMeans(X)
	p.Mean = mean
	// center
	Xc := mat.NewMatrix(n, d)
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			Xc[i][j] = X[i][j] - mean[j]
		}
	}
	// covariance (d x d)
	Cov := mat.NewMatrix(d, d)
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			s := 0.0
			for r := 0; r < n; r++ {
				s += Xc[r][i] * Xc[r][j]
			}
			Cov[i][j] = s / float64(n-1)
		}
	}
	// eigendecomposition (symmetric) via Jacobi
	eigVals, eigVecs := jacobiEigen(Cov)
	// sort descending
	order := make([]int, d)
	for i := range order {
		order[i] = i
	}
	for i := 1; i < d; i++ {
		for j := i; j > 0 && eigVals[order[j-1]] < eigVals[order[j]]; j-- {
			order[j-1], order[j] = order[j], order[j-1]
		}
	}
	k := p.NComponents
	if k > d {
		k = d
	}
	p.Components = mat.NewMatrix(k, d)
	p.Explained = make(mat.Vec, k)
	total := 0.0
	for _, v := range eigVals {
		total += v
	}
	for i := 0; i < k; i++ {
		idx := order[i]
		p.Explained[i] = eigVals[idx] / total
		for j := 0; j < d; j++ {
			p.Components[i][j] = eigVecs[j][idx]
		}
	}
}

func (p *PCA) Transform(X mat.Matrix) mat.Matrix {
	n := X.Rows()
	d := X.Cols()
	k := p.Components.Rows()
	out := mat.NewMatrix(n, k)
	for i := 0; i < n; i++ {
		for c := 0; c < k; c++ {
			s := 0.0
			for j := 0; j < d; j++ {
				s += (X[i][j] - p.Mean[j]) * p.Components[c][j]
			}
			out[i][c] = s
		}
	}
	return out
}

func (p *PCA) FitTransform(X mat.Matrix) mat.Matrix {
	p.Fit(X)
	return p.Transform(X)
}

func (p *PCA) InverseTransform(Z mat.Matrix) mat.Matrix {
	n := Z.Rows()
	d := len(p.Mean)
	out := mat.NewMatrix(n, d)
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			s := p.Mean[j]
			for c := 0; c < Z.Cols(); c++ {
				s += Z[i][c] * p.Components[c][j]
			}
			out[i][j] = s
		}
	}
	return out
}

// Jacobi eigenvalue algorithm for symmetric matrices
func jacobiEigen(A mat.Matrix) (mat.Vec, mat.Matrix) {
	n := A.Rows()
	// copy
	D := A.Clone()
	V := mat.Identity(n)
	for sweep := 0; sweep < 50; sweep++ {
		off := 0.0
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				off += math.Abs(D[i][j])
			}
		}
		if off < 1e-10 {
			break
		}
		for p := 0; p < n; p++ {
			for q := p + 1; q < n; q++ {
				if math.Abs(D[p][q]) < 1e-12 {
					continue
				}
				theta := (D[q][q] - D[p][p]) / (2 * D[p][q])
				var t float64
				if theta >= 0 {
					t = 1.0 / (theta + math.Sqrt(theta*theta+1))
				} else {
					t = -1.0 / (-theta + math.Sqrt(theta*theta+1))
				}
				c := 1.0 / math.Sqrt(t*t+1)
				s := t * c
				// rotate
				for k := 0; k < n; k++ {
					Dpk := D[p][k]
					Dqk := D[q][k]
					D[p][k] = c*Dpk - s*Dqk
					D[q][k] = s*Dpk + c*Dqk
				}
				for k := 0; k < n; k++ {
					Dkp := D[k][p]
					Dkq := D[k][q]
					D[k][p] = c*Dkp - s*Dkq
					D[k][q] = s*Dkp + c*Dkq
				}
				for k := 0; k < n; k++ {
					Vkp := V[k][p]
					Vkq := V[k][q]
					V[k][p] = c*Vkp - s*Vkq
					V[k][q] = s*Vkp + c*Vkq
				}
			}
		}
	}
	eigVals := make(mat.Vec, n)
	for i := 0; i < n; i++ {
		eigVals[i] = D[i][i]
	}
	return eigVals, V
}
