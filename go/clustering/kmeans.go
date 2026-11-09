package clustering

import (
	"math"
	"math/rand"

	"mlfromscratch/mat"
)

type KMeans struct {
	NClusters int
	MaxIters  int
	Tol       float64
	NInit     int
	Seed      int64
	Centers   mat.Matrix
	Labels    []int
	Inertia   float64
}

func NewKMeans(k, maxIters, nInit int, seed int64) *KMeans {
	if k == 0 {
		k = 3
	}
	if maxIters == 0 {
		maxIters = 300
	}
	if nInit == 0 {
		nInit = 10
	}
	return &KMeans{NClusters: k, MaxIters: maxIters, Tol: 1e-4, NInit: nInit, Seed: seed}
}

func (m *KMeans) Fit(X mat.Matrix) {
	n, d := X.Rows(), X.Cols()
	rng := rand.New(rand.NewSource(m.Seed))
	bestInertia := math.Inf(1)
	var bestCenters mat.Matrix
	var bestLabels []int
	for init := 0; init < m.NInit; init++ {
		// kmeans++ init
		centers := mat.NewMatrix(m.NClusters, d)
		centers[0] = X[rng.Intn(n)]
		for c := 1; c < m.NClusters; c++ {
			// distances
			D := make([]float64, n)
			for i := 0; i < n; i++ {
				mn := math.Inf(1)
				for cc := 0; cc < c; cc++ {
					sq := 0.0
					for j := 0; j < d; j++ {
						dd := X[i][j] - centers[cc][j]
						sq += dd * dd
					}
					if sq < mn {
						mn = sq
					}
				}
				D[i] = mn
			}
			// probs
			sum := 0.0
			for i := 0; i < n; i++ {
				sum += D[i]
			}
			r := rng.Float64() * sum
			acc := 0.0
			chosen := n - 1
			for i := 0; i < n; i++ {
				acc += D[i]
				if acc >= r {
					chosen = i
					break
				}
			}
			for j := 0; j < d; j++ {
				centers[c][j] = X[chosen][j]
			}
		}
		// iterate
		labels := make([]int, n)
		for it := 0; it < m.MaxIters; it++ {
			// assign
			for i := 0; i < n; i++ {
				best := 0
				bestD := math.Inf(1)
				for c := 0; c < m.NClusters; c++ {
					sq := 0.0
					for j := 0; j < d; j++ {
						dd := X[i][j] - centers[c][j]
						sq += dd * dd
					}
					if sq < bestD {
						bestD = sq
						best = c
					}
				}
				labels[i] = best
			}
			// update
			newCenters := mat.NewMatrix(m.NClusters, d)
			counts := make([]int, m.NClusters)
			for i := 0; i < n; i++ {
				c := labels[i]
				counts[c]++
				for j := 0; j < d; j++ {
					newCenters[c][j] += X[i][j]
				}
			}
			for c := 0; c < m.NClusters; c++ {
				if counts[c] > 0 {
					for j := 0; j < d; j++ {
						newCenters[c][j] /= float64(counts[c])
					}
				} else {
					for j := 0; j < d; j++ {
						newCenters[c][j] = centers[c][j]
					}
				}
			}
			// shift
			shift := 0.0
			for c := 0; c < m.NClusters; c++ {
				for j := 0; j < d; j++ {
					dd := newCenters[c][j] - centers[c][j]
					shift += dd * dd
				}
			}
			centers = newCenters
			if math.Sqrt(shift) < m.Tol {
				break
			}
		}
		// final assign
		for i := 0; i < n; i++ {
			best := 0
			bestD := math.Inf(1)
			for c := 0; c < m.NClusters; c++ {
				sq := 0.0
				for j := 0; j < d; j++ {
					dd := X[i][j] - centers[c][j]
					sq += dd * dd
				}
				if sq < bestD {
					bestD = sq
					best = c
				}
			}
			labels[i] = best
		}
		// inertia
		inertia := 0.0
		for i := 0; i < n; i++ {
			c := labels[i]
			for j := 0; j < d; j++ {
				dd := X[i][j] - centers[c][j]
				inertia += dd * dd
			}
		}
		if inertia < bestInertia {
			bestInertia = inertia
			bestCenters = centers.Clone()
			bestLabels = append([]int(nil), labels...)
		}
	}
	m.Centers = bestCenters
	m.Labels = bestLabels
	m.Inertia = bestInertia
}

func (m *KMeans) Predict(X mat.Matrix) []int {
	n, d := X.Rows(), X.Cols()
	out := make([]int, n)
	for i := 0; i < n; i++ {
		best := 0
		bestD := math.Inf(1)
		for c := 0; c < m.NClusters; c++ {
			sq := 0.0
			for j := 0; j < d; j++ {
				dd := X[i][j] - m.Centers[c][j]
				sq += dd * dd
			}
			if sq < bestD {
				bestD = sq
				best = c
			}
		}
		out[i] = best
	}
	return out
}
