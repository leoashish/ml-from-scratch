// Package mat provides a tiny matrix/vector library used by the ML algorithms.
package mat

import (
	"fmt"
	"math"
	"math/rand"
)

// Matrix is a row-major 2D slice.
type Matrix [][]float64

// Vec is a 1D slice.
type Vec []float64

func NewMatrix(rows, cols int) Matrix {
	m := make(Matrix, rows)
	for i := range m {
		m[i] = make([]float64, cols)
	}
	return m
}

func NewVec(n int) Vec { return make(Vec, n) }

func (m Matrix) Rows() int {
	if len(m) == 0 {
		return 0
	}
	return len(m)
}

func (m Matrix) Cols() int {
	if len(m) == 0 {
		return 0
	}
	return len(m[0])
}

func (m Matrix) Clone() Matrix {
	r := NewMatrix(m.Rows(), m.Cols())
	for i := range m {
		copy(r[i], m[i])
	}
	return r
}

func (v Vec) Clone() Vec {
	r := make(Vec, len(v))
	copy(r, v)
	return r
}

func Add(a, b Matrix) Matrix {
	r := NewMatrix(a.Rows(), a.Cols())
	for i := range a {
		for j := range a[i] {
			r[i][j] = a[i][j] + b[i][j]
		}
	}
	return r
}

func Sub(a, b Matrix) Matrix {
	r := NewMatrix(a.Rows(), a.Cols())
	for i := range a {
		for j := range a[i] {
			r[i][j] = a[i][j] - b[i][j]
		}
	}
	return r
}

func Scale(m Matrix, s float64) Matrix {
	r := m.Clone()
	for i := range r {
		for j := range r[i] {
			r[i][j] *= s
		}
	}
	return r
}

func Mul(a, b Matrix) Matrix {
	// a: (m x k), b: (k x n) -> (m x n)
	m, k := a.Rows(), a.Cols()
	n := b.Cols()
	r := NewMatrix(m, n)
	for i := 0; i < m; i++ {
		for l := 0; l < k; l++ {
			if a[i][l] == 0 {
				continue
			}
			ail := a[i][l]
			for j := 0; j < n; j++ {
				r[i][j] += ail * b[l][j]
			}
		}
	}
	return r
}

func Transpose(m Matrix) Matrix {
	r := NewMatrix(m.Cols(), m.Rows())
	for i := range m {
		for j := range m[i] {
			r[j][i] = m[i][j]
		}
	}
	return r
}

// Vec ops
func VecAdd(a, b Vec) Vec {
	r := make(Vec, len(a))
	for i := range a {
		r[i] = a[i] + b[i]
	}
	return r
}

func VecSub(a, b Vec) Vec {
	r := make(Vec, len(a))
	for i := range a {
		r[i] = a[i] - b[i]
	}
	return r
}

func VecScale(v Vec, s float64) Vec {
	r := v.Clone()
	for i := range r {
		r[i] *= s
	}
	return r
}

func Dot(a, b Vec) float64 {
	s := 0.0
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func MatVec(m Matrix, v Vec) Vec {
	r := make(Vec, m.Rows())
	for i := range m {
		s := 0.0
		for j := range m[i] {
			s += m[i][j] * v[j]
		}
		r[i] = s
	}
	return r
}

func Outer(a, b Vec) Matrix {
	r := NewMatrix(len(a), len(b))
	for i := range a {
		for j := range b {
			r[i][j] = a[i] * b[j]
		}
	}
	return r
}

// Solve linear system A x = b via Gaussian elimination with partial pivot.
func Solve(A Matrix, b Vec) (Vec, error) {
	n := len(A)
	// augment
	M := NewMatrix(n, n+1)
	for i := 0; i < n; i++ {
		copy(M[i], A[i])
		M[i][n] = b[i]
	}
	// elimination
	for col := 0; col < n; col++ {
		// pivot
		pivot := col
		best := math.Abs(M[col][col])
		for r := col + 1; r < n; r++ {
			if math.Abs(M[r][col]) > best {
				best = math.Abs(M[r][col])
				pivot = r
			}
		}
		if best < 1e-12 {
			return nil, fmt.Errorf("singular matrix")
		}
		if pivot != col {
			M[col], M[pivot] = M[pivot], M[col]
		}
		// normalize
		pv := M[col][col]
		for j := col; j <= n; j++ {
			M[col][j] /= pv
		}
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			factor := M[r][col]
			if factor != 0 {
				for j := col; j <= n; j++ {
					M[r][j] -= factor * M[col][j]
				}
			}
		}
	}
	x := make(Vec, n)
	for i := 0; i < n; i++ {
		x[i] = M[i][n]
	}
	return x, nil
}

// Inverse via solve on identity.
func Inverse(m Matrix) (Matrix, error) {
	n := m.Rows()
	inv := NewMatrix(n, n)
	for i := 0; i < n; i++ {
		b := make(Vec, n)
		b[i] = 1
		col, err := Solve(m, b)
		if err != nil {
			return nil, err
		}
		for r := 0; r < n; r++ {
			inv[r][i] = col[r]
		}
	}
	return inv, nil
}

func Identity(n int) Matrix {
	m := NewMatrix(n, n)
	for i := 0; i < n; i++ {
		m[i][i] = 1
	}
	return m
}

func (m Matrix) String() string {
	return fmt.Sprintf("%v", [][]float64(m))
}

// RandnMat fills matrix with standard normal using provided rng.
func RandnMat(rows, cols int, rng *rand.Rand) Matrix {
	m := NewMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			u1 := rng.Float64()
			u2 := rng.Float64()
			m[i][j] = math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
		}
	}
	return m
}

// Utility: column means
func ColMeans(m Matrix) Vec {
	r := make(Vec, m.Cols())
	for j := 0; j < m.Cols(); j++ {
		for i := 0; i < m.Rows(); i++ {
			r[j] += m[i][j]
		}
		r[j] /= float64(m.Rows())
	}
	return r
}

func ColStd(m Matrix, mean Vec) Vec {
	r := make(Vec, m.Cols())
	for j := 0; j < m.Cols(); j++ {
		s := 0.0
		for i := 0; i < m.Rows(); i++ {
			d := m[i][j] - mean[j]
			s += d * d
		}
		r[j] = math.Sqrt(s / float64(m.Rows()))
		if r[j] == 0 {
			r[j] = 1
		}
	}
	return r
}
