package anomaly

import (
	"math"

	"mlfromscratch/mat"
)

type LOF struct {
	NNeighbors    int
	Contamination float64
	Threshold     float64
	X             mat.Matrix
	KnnDist       []float64
	KnnIdx        [][]int
	LRD           []float64
	LOF           []float64
}

func NewLOF(k int, contamination float64) *LOF {
	if k == 0 {
		k = 20
	}
	if contamination == 0 {
		contamination = 0.1
	}
	return &LOF{NNeighbors: k, Contamination: contamination}
}

func (m *LOF) Fit(X mat.Matrix) {
	n := X.Rows()
	k := m.NNeighbors
	if k >= n {
		k = n - 1
	}
	m.X = X.Clone()
	// distance matrix
	D := make([][]float64, n)
	for i := 0; i < n; i++ {
		D[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			sq := 0.0
			for c := 0; c < X.Cols(); c++ {
				dd := X[i][c] - X[j][c]
				sq += dd * dd
			}
			D[i][j] = math.Sqrt(sq)
		}
	}
	m.KnnDist = make([]float64, n)
	m.KnnIdx = make([][]int, n)
	for i := 0; i < n; i++ {
		// sort indices by distance excluding i
		order := make([]int, 0, n-1)
		for j := 0; j < n; j++ {
			if j != i {
				order = append(order, j)
			}
		}
		// insertion sort
		for a := 1; a < len(order); a++ {
			for b := a; b > 0 && D[i][order[b-1]] > D[i][order[b]]; b-- {
				order[b-1], order[b] = order[b], order[b-1]
			}
		}
		m.KnnIdx[i] = order[:k]
		m.KnnDist[i] = D[i][order[k-1]]
	}
	m.LRD = make([]float64, n)
	for i := 0; i < n; i++ {
		rdSum := 0.0
		for _, j := range m.KnnIdx[i] {
			rd := m.KnnDist[j]
			if D[i][j] > rd {
				rd = D[i][j]
			}
			rdSum += rd
		}
		if rdSum > 0 {
			m.LRD[i] = float64(k) / rdSum
		}
	}
	m.LOF = make([]float64, n)
	for i := 0; i < n; i++ {
		if m.LRD[i] == 0 {
			m.LOF[i] = 1
			continue
		}
		s := 0.0
		for _, j := range m.KnnIdx[i] {
			if m.LRD[i] > 0 {
				s += m.LRD[j] / m.LRD[i]
			}
		}
		m.LOF[i] = s / float64(k)
	}
	// threshold
	sorted := append([]float64(nil), m.LOF...)
	for i := 1; i < len(sorted); i++ {
		for j := i; j > 0 && sorted[j-1] > sorted[j]; j-- {
			sorted[j-1], sorted[j] = sorted[j], sorted[j-1]
		}
	}
	q := 1.0 - m.Contamination
	idx := int(q * float64(len(sorted)))
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	m.Threshold = sorted[idx]
}

func (m *LOF) FitPredict(X mat.Matrix) []int {
	m.Fit(X)
	out := make([]int, X.Rows())
	for i := range m.LOF {
		if m.LOF[i] >= m.Threshold {
			out[i] = -1
		} else {
			out[i] = 1
		}
	}
	return out
}
