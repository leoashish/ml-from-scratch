package anomaly

import (
	"math"
	"math/rand"

	"mlfromscratch/mat"
)

type isoNode struct {
	isLeaf    bool
	size      int
	depth     int
	feature   int
	threshold float64
	left      *isoNode
	right     *isoNode
}

type isoTree struct {
	root     *isoNode
	maxDepth int
}

func (t *isoTree) build(X mat.Matrix, depth int) *isoNode {
	n, d := X.Rows(), X.Cols()
	node := &isoNode{size: n, depth: depth}
	if depth >= t.maxDepth || n <= 1 {
		node.isLeaf = true
		return node
	}
	feat := rand.Intn(d)
	col := make([]float64, n)
	for i := 0; i < n; i++ {
		col[i] = X[i][feat]
	}
	mn, mx := col[0], col[0]
	for _, v := range col {
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}
	if mn == mx {
		node.isLeaf = true
		return node
	}
	thr := mn + rand.Float64()*(mx-mn)
	var leftX, rightX mat.Matrix
	for i := 0; i < n; i++ {
		if X[i][feat] < thr {
			leftX = append(leftX, X[i])
		} else {
			rightX = append(rightX, X[i])
		}
	}
	if len(leftX) == 0 || len(rightX) == 0 {
		node.isLeaf = true
		return node
	}
	node.isLeaf = false
	node.feature = feat
	node.threshold = thr
	node.left = t.build(leftX, depth+1)
	node.right = t.build(rightX, depth+1)
	return node
}

func (t *isoTree) pathLength(x mat.Vec, node *isoNode, depth int) float64 {
	if node == nil {
		node = t.root
	}
	if node.isLeaf {
		size := node.size
		if size <= 1 {
			return float64(depth)
		}
		// c(size)
		h := math.Log(float64(size-1)) + 0.5772156649
		c := 2*h - 2*float64(size-1)/float64(size)
		return float64(depth) + c
	}
	if x[node.feature] < node.threshold {
		return t.pathLength(x, node.left, depth+1)
	}
	return t.pathLength(x, node.right, depth+1)
}

type IsolationForest struct {
	NEstimators   int
	MaxSamples    int
	Contamination float64
	Seed          int64
	Trees         []*isoTree
	Threshold     float64
}

func NewIsolationForest(nEst, maxSamples int, contamination float64, seed int64) *IsolationForest {
	if nEst == 0 {
		nEst = 100
	}
	if maxSamples == 0 {
		maxSamples = 256
	}
	if contamination == 0 {
		contamination = 0.1
	}
	return &IsolationForest{NEstimators: nEst, MaxSamples: maxSamples, Contamination: contamination, Seed: seed}
}

func cFactor(n int) float64 {
	if n <= 1 {
		return 0
	}
	if n == 2 {
		return 1
	}
	h := math.Log(float64(n-1)) + 0.5772156649
	return 2*h - 2*float64(n-1)/float64(n)
}

func (f *IsolationForest) Fit(X mat.Matrix) {
	n := X.Rows()
	ms := f.MaxSamples
	if ms > n {
		ms = n
	}
	rand.Seed(f.Seed)
	f.Trees = nil
	for t := 0; t < f.NEstimators; t++ {
		idx := rand.Perm(n)[:ms]
		sub := mat.NewMatrix(ms, X.Cols())
		for i, id := range idx {
			for j := 0; j < X.Cols(); j++ {
				sub[i][j] = X[id][j]
			}
		}
		tree := &isoTree{maxDepth: int(math.Ceil(math.Log2(float64(ms))))}
		tree.root = tree.build(sub, 0)
		f.Trees = append(f.Trees, tree)
	}
	// threshold on training scores
	scores := f.DecisionFunction(X)
	// percentile
	sorted := append([]float64(nil), scores...)
	for i := 1; i < len(sorted); i++ {
		for j := i; j > 0 && sorted[j-1] > sorted[j]; j-- {
			sorted[j-1], sorted[j] = sorted[j], sorted[j-1]
		}
	}
	// threshold at (1-contamination) quantile
	q := 1.0 - f.Contamination
	idx := int(q * float64(len(sorted)))
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	f.Threshold = sorted[idx]
}

func (f *IsolationForest) DecisionFunction(X mat.Matrix) []float64 {
	ms := f.MaxSamples
	if ms > X.Rows() {
		ms = X.Rows()
	}
	cms := cFactor(ms)
	out := make([]float64, X.Rows())
	for i := 0; i < X.Rows(); i++ {
		pl := 0.0
		for _, tree := range f.Trees {
			pl += tree.pathLength(X[i], nil, 0)
		}
		pl /= float64(len(f.Trees))
		out[i] = math.Pow(2, -pl/cms)
	}
	return out
}

func (f *IsolationForest) Predict(X mat.Matrix) []int {
	scores := f.DecisionFunction(X)
	out := make([]int, X.Rows())
	for i := range scores {
		if scores[i] >= f.Threshold {
			out[i] = -1
		} else {
			out[i] = 1
		}
	}
	return out
}
