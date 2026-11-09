package clustering

import (
	"math"

	"mlfromscratch/mat"
)

type DBSCAN struct {
	Eps        float64
	MinSamples int
	Labels     []int
}

func NewDBSCAN(eps float64, minSamples int) *DBSCAN {
	if eps == 0 {
		eps = 0.5
	}
	if minSamples == 0 {
		minSamples = 5
	}
	return &DBSCAN{Eps: eps, MinSamples: minSamples}
}

func (m *DBSCAN) Fit(X mat.Matrix) {
	n := X.Rows()
	// distance matrix
	D := make([][]float64, n)
	for i := 0; i < n; i++ {
		D[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			sq := 0.0
			for k := 0; k < X.Cols(); k++ {
				dd := X[i][k] - X[j][k]
				sq += dd * dd
			}
			D[i][j] = math.Sqrt(sq)
		}
	}
	labels := make([]int, n)
	for i := range labels {
		labels[i] = -2 // unvisited
	}
	clusterID := 0
	for i := 0; i < n; i++ {
		if labels[i] != -2 {
			continue
		}
		// neighbors
		nbrs := []int{}
		for j := 0; j < n; j++ {
			if D[i][j] <= m.Eps {
				nbrs = append(nbrs, j)
			}
		}
		if len(nbrs) < m.MinSamples {
			labels[i] = -1
			continue
		}
		labels[i] = clusterID
		seeds := []int{}
		for _, nb := range nbrs {
			if nb != i {
				seeds = append(seeds, nb)
			}
		}
		for len(seeds) > 0 {
			j := seeds[len(seeds)-1]
			seeds = seeds[:len(seeds)-1]
			if labels[j] == -1 {
				labels[j] = clusterID
			}
			if labels[j] != -2 {
				continue
			}
			labels[j] = clusterID
			nbrsJ := []int{}
			for k := 0; k < n; k++ {
				if D[j][k] <= m.Eps {
					nbrsJ = append(nbrsJ, k)
				}
			}
			if len(nbrsJ) >= m.MinSamples {
				for _, nb := range nbrsJ {
					if labels[nb] == -2 || labels[nb] == -1 {
						seeds = append(seeds, nb)
					}
				}
			}
		}
		clusterID++
	}
	m.Labels = labels
}
