package regression

import (
	"math"
	"math/rand"

	"mlfromscratch/mat"
)

// ---- Decision Tree Regressor ----

type TreeNode struct {
	IsLeaf    bool
	Value     float64
	Feature   int
	Threshold float64
	Left      *TreeNode
	Right     *TreeNode
}

type DecisionTreeRegressor struct {
	MaxDepth         int
	MinSamplesLeaf   int
	Root             *TreeNode
}

func NewDecisionTreeRegressor(maxDepth, minSamplesLeaf int) *DecisionTreeRegressor {
	if maxDepth == 0 {
		maxDepth = 3
	}
	if minSamplesLeaf == 0 {
		minSamplesLeaf = 1
	}
	return &DecisionTreeRegressor{MaxDepth: maxDepth, MinSamplesLeaf: minSamplesLeaf}
}

func (t *DecisionTreeRegressor) Fit(X mat.Matrix, y mat.Vec) {
	t.Root = t.build(X, y, 0)
}

func (t *DecisionTreeRegressor) build(X mat.Matrix, y mat.Vec, depth int) *TreeNode {
	node := &TreeNode{Value: mean(y)}
	if depth >= t.MaxDepth || len(y) < 2*t.MinSamplesLeaf || variance(y) == 0 {
		node.IsLeaf = true
		return node
	}
	bestFeat, bestThr, bestGain := -1, 0.0, 0.0
	parentVar := variance(y) * float64(len(y))
	n, d := X.Rows(), X.Cols()
	for feat := 0; feat < d; feat++ {
		// candidate thresholds
		vals := make([]float64, n)
		for i := 0; i < n; i++ {
			vals[i] = X[i][feat]
		}
		// unique sorted
		sortedVals := append([]float64(nil), vals...)
		// simple: try midpoints between consecutive sorted unique
		// collect unique
		uniq := uniqueSorted(sortedVals)
		for k := 1; k < len(uniq); k++ {
			thr := (uniq[k-1] + uniq[k]) / 2
			// split
			var leftY, rightY []float64
			var leftIdx, rightIdx []int
			for i := 0; i < n; i++ {
				if X[i][feat] <= thr {
					leftY = append(leftY, y[i])
					leftIdx = append(leftIdx, i)
				} else {
					rightY = append(rightY, y[i])
					rightIdx = append(rightIdx, i)
				}
			}
			if len(leftY) < t.MinSamplesLeaf || len(rightY) < t.MinSamplesLeaf {
				continue
			}
			gain := parentVar - (varianceSlice(leftY)*float64(len(leftY)) + varianceSlice(rightY)*float64(len(rightY)))
			if gain > bestGain+1e-12 {
				bestGain = gain
				bestFeat = feat
				bestThr = thr
			}
		}
	}
	if bestFeat == -1 {
		node.IsLeaf = true
		return node
	}
	// split data
	var leftX, rightX mat.Matrix
	var leftY, rightY mat.Vec
	for i := 0; i < n; i++ {
		if X[i][bestFeat] <= bestThr {
			leftX = append(leftX, X[i])
			leftY = append(leftY, y[i])
		} else {
			rightX = append(rightX, X[i])
			rightY = append(rightY, y[i])
		}
	}
	node.IsLeaf = false
	node.Feature = bestFeat
	node.Threshold = bestThr
	node.Left = t.build(leftX, leftY, depth+1)
	node.Right = t.build(rightX, rightY, depth+1)
	return node
}

func (t *DecisionTreeRegressor) Predict(X mat.Matrix) mat.Vec {
	out := make(mat.Vec, X.Rows())
	for i := 0; i < X.Rows(); i++ {
		out[i] = t.predictOne(X[i])
	}
	return out
}

func (t *DecisionTreeRegressor) predictOne(x mat.Vec) float64 {
	cur := t.Root
	for !cur.IsLeaf {
		if x[cur.Feature] <= cur.Threshold {
			cur = cur.Left
		} else {
			cur = cur.Right
		}
	}
	return cur.Value
}

// ---- Gradient Boosting Regressor ----

type GradientBoostingRegressor struct {
	NEstimators    int
	LearningRate   float64
	MaxDepth       int
	MinSamplesLeaf int
	RandomState    int64
	Trees          []*DecisionTreeRegressor
	Init           float64
}

func NewGradientBoostingRegressor(nEst int, lr float64, maxDepth, minLeaf int, seed int64) *GradientBoostingRegressor {
	if nEst == 0 {
		nEst = 100
	}
	if lr == 0 {
		lr = 0.1
	}
	if maxDepth == 0 {
		maxDepth = 3
	}
	if minLeaf == 0 {
		minLeaf = 1
	}
	return &GradientBoostingRegressor{NEstimators: nEst, LearningRate: lr, MaxDepth: maxDepth, MinSamplesLeaf: minLeaf, RandomState: seed}
}

func (g *GradientBoostingRegressor) Fit(X mat.Matrix, y mat.Vec) {
	n := len(y)
	g.Init = mean(y)
	F := make(mat.Vec, n)
	for i := range F {
		F[i] = g.Init
	}
	rng := rand.New(rand.NewSource(g.RandomState))
	for m := 0; m < g.NEstimators; m++ {
		resid := make(mat.Vec, n)
		for i := 0; i < n; i++ {
			resid[i] = y[i] - F[i]
		}
		tree := NewDecisionTreeRegressor(g.MaxDepth, g.MinSamplesLeaf)
		tree.Fit(X, resid)
		upd := tree.Predict(X)
		for i := 0; i < n; i++ {
			F[i] += g.LearningRate * upd[i]
		}
		g.Trees = append(g.Trees, tree)
		_ = rng
	}
}

func (g *GradientBoostingRegressor) Predict(X mat.Matrix) mat.Vec {
	out := make(mat.Vec, X.Rows())
	for i := range out {
		out[i] = g.Init
	}
	for _, tree := range g.Trees {
		upd := tree.Predict(X)
		for i := range out {
			out[i] += g.LearningRate * upd[i]
		}
	}
	return out
}

// ---- helpers ----

func mean(v mat.Vec) float64 {
	if len(v) == 0 {
		return 0
	}
	s := 0.0
	for _, x := range v {
		s += x
	}
	return s / float64(len(v))
}

func variance(v mat.Vec) float64 {
	if len(v) == 0 {
		return 0
	}
	m := mean(v)
	s := 0.0
	for _, x := range v {
		d := x - m
		s += d * d
	}
	return s / float64(len(v))
}

func varianceSlice(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	m := 0.0
	for _, x := range v {
		m += x
	}
	m /= float64(len(v))
	s := 0.0
	for _, x := range v {
		d := x - m
		s += d * d
	}
	return s / float64(len(v))
}

func uniqueSorted(v []float64) []float64 {
	cp := append([]float64(nil), v...)
	// sort
	for i := 1; i < len(cp); i++ {
		for j := i; j > 0 && cp[j-1] > cp[j]; j-- {
			cp[j-1], cp[j] = cp[j], cp[j-1]
		}
	}
	out := []float64{}
	for i, x := range cp {
		if i == 0 || x != cp[i-1] {
			out = append(out, x)
		}
	}
	return out
}

// GradientBoostingClassifier (binary, log-loss)
type GradientBoostingClassifier struct {
	NEstimators  int
	LearningRate float64
	MaxDepth     int
	Trees        []*DecisionTreeRegressor
	Init         float64
	Classes      []float64
}

func NewGradientBoostingClassifier(nEst int, lr float64, maxDepth int, seed int64) *GradientBoostingClassifier {
	if nEst == 0 {
		nEst = 100
	}
	if lr == 0 {
		lr = 0.1
	}
	if maxDepth == 0 {
		maxDepth = 3
	}
	return &GradientBoostingClassifier{NEstimators: nEst, LearningRate: lr, MaxDepth: maxDepth}
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

func (g *GradientBoostingClassifier) Fit(X mat.Matrix, y mat.Vec) {
	// classes
	g.Classes = []float64{0, 1}
	// map y to 0/1 already assumed 0/1
	n := len(y)
	p := 0.0
	for _, v := range y {
		p += v
	}
	p /= float64(n)
	if p <= 0 {
		p = 1e-6
	}
	if p >= 1 {
		p = 1 - 1e-6
	}
	g.Init = math.Log(p / (1 - p))
	F := make(mat.Vec, n)
	for i := range F {
		F[i] = g.Init
	}
	for m := 0; m < g.NEstimators; m++ {
		resid := make(mat.Vec, n)
		for i := 0; i < n; i++ {
			resid[i] = y[i] - sigmoid(F[i])
		}
		tree := NewDecisionTreeRegressor(g.MaxDepth, 1)
		tree.Fit(X, resid)
		upd := tree.Predict(X)
		for i := 0; i < n; i++ {
			F[i] += g.LearningRate * upd[i]
		}
		g.Trees = append(g.Trees, tree)
	}
}

func (g *GradientBoostingClassifier) PredictProba(X mat.Matrix) mat.Vec {
	out := make(mat.Vec, X.Rows())
	F := make(mat.Vec, X.Rows())
	for i := range F {
		F[i] = g.Init
	}
	for _, tree := range g.Trees {
		upd := tree.Predict(X)
		for i := range F {
			F[i] += g.LearningRate * upd[i]
		}
	}
	for i := range out {
		out[i] = sigmoid(F[i])
	}
	return out
}

func (g *GradientBoostingClassifier) Predict(X mat.Matrix) mat.Vec {
	proba := g.PredictProba(X)
	out := make(mat.Vec, X.Rows())
	for i := range proba {
		if proba[i] >= 0.5 {
			out[i] = 1
		} else {
			out[i] = 0
		}
	}
	return out
}
