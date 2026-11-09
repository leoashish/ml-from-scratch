package ranking

import (
	"math"

	"mlfromscratch/mat"
	"mlfromscratch/regression"
)

// Reuse regression tree via exported? We'll implement own simple tree for lambdas.
type rankNode struct {
	isLeaf    bool
	value     float64
	feature   int
	threshold float64
	left      *rankNode
	right     *rankNode
}

type rankTree struct {
	maxDepth int
	minLeaf  int
	root     *rankNode
}

func (t *rankTree) build(X mat.Matrix, lam, mu mat.Vec, depth int) *rankNode {
	node := &rankNode{value: weightedMean(lam, mu)}
	if depth >= t.maxDepth || len(lam) < 2*t.minLeaf {
		node.isLeaf = true
		return node
	}
	bestFeat, bestThr := -1, 0.0
	bestGain := 0.0
	n, d := X.Rows(), X.Cols()
	for feat := 0; feat < d; feat++ {
		uniq := uniqueSortedCol(X, feat)
		for k := 1; k < len(uniq); k++ {
			thr := (uniq[k-1] + uniq[k]) / 2
			var lm, rm []int
			for i := 0; i < n; i++ {
				if X[i][feat] <= thr {
					lm = append(lm, i)
				} else {
					rm = append(rm, i)
				}
			}
			if len(lm) < t.minLeaf || len(rm) < t.minLeaf {
				continue
			}
			// gain = weighted variance reduction
			parent := wvar(lam, mu, allIdx(n))
			gain := parent - (wvarIdx(lam, mu, lm) + wvarIdx(lam, mu, rm))
			if gain > bestGain+1e-12 {
				bestGain = gain
				bestFeat = feat
				bestThr = thr
			}
		}
	}
	if bestFeat == -1 {
		node.isLeaf = true
		return node
	}
	var leftX, rightX mat.Matrix
	leftLam, leftMu, rightLam, rightMu := mat.Vec{}, mat.Vec{}, mat.Vec{}, mat.Vec{}
	for i := 0; i < n; i++ {
		if X[i][bestFeat] <= bestThr {
			leftX = append(leftX, X[i])
			leftLam = append(leftLam, lam[i])
			leftMu = append(leftMu, mu[i])
		} else {
			rightX = append(rightX, X[i])
			rightLam = append(rightLam, lam[i])
			rightMu = append(rightMu, mu[i])
		}
	}
	node.isLeaf = false
	node.feature = bestFeat
	node.threshold = bestThr
	node.left = t.build(leftX, leftLam, leftMu, depth+1)
	node.right = t.build(rightX, rightLam, rightMu, depth+1)
	return node
}

func (t *rankTree) predictOne(x mat.Vec) float64 {
	cur := t.root
	for !cur.isLeaf {
		if x[cur.feature] <= cur.threshold {
			cur = cur.left
		} else {
			cur = cur.right
		}
	}
	return cur.value
}

func (t *rankTree) Predict(X mat.Matrix) mat.Vec {
	out := make(mat.Vec, X.Rows())
	for i := 0; i < X.Rows(); i++ {
		out[i] = t.predictOne(X[i])
	}
	return out
}

type LambdaMART struct {
	NEstimators   int
	LearningRate  float64
	MaxDepth      int
	MinSamplesLeaf int
	Seed          int64
	Trees         []*rankTree
}

func NewLambdaMART(nEst int, lr float64, maxDepth, minLeaf int, seed int64) *LambdaMART {
	if nEst == 0 {
		nEst = 50
	}
	if lr == 0 {
		lr = 0.1
	}
	if maxDepth == 0 {
		maxDepth = 3
	}
	if minLeaf == 0 {
		minLeaf = 5
	}
	return &LambdaMART{NEstimators: nEst, LearningRate: lr, MaxDepth: maxDepth, MinSamplesLeaf: minLeaf, Seed: seed}
}

func (l *LambdaMART) Fit(X mat.Matrix, y mat.Vec, qid []int) {
	n := X.Rows()
	F := make(mat.Vec, n)
	// queries
	qset := map[int]bool{}
	for _, q := range qid {
		qset[q] = true
	}
	queries := make([]int, 0, len(qset))
	for q := range qset {
		queries = append(queries, q)
	}
	for m := 0; m < l.NEstimators; m++ {
		lambdas := make(mat.Vec, n)
		mus := make(mat.Vec, n)
		for _, q := range queries {
			// indices for this query
			idx := []int{}
			for i := 0; i < n; i++ {
				if qid[i] == q {
					idx = append(idx, i)
				}
			}
			mm := len(idx)
			scores := make(mat.Vec, mm)
			rels := make(mat.Vec, mm)
			for a := 0; a < mm; a++ {
				scores[a] = F[idx[a]]
				rels[a] = y[idx[a]]
			}
			// ideal dcg
			sortedRels := make(mat.Vec, mm)
			copy(sortedRels, rels)
			// sort desc
			for a := 1; a < mm; a++ {
				for b := a; b > 0 && sortedRels[b-1] < sortedRels[b]; b-- {
					sortedRels[b-1], sortedRels[b] = sortedRels[b], sortedRels[b-1]
				}
			}
			discounts := make(mat.Vec, mm)
			for a := 0; a < mm; a++ {
				discounts[a] = 1.0 / math.Log2(float64(a+2))
			}
			idcg := 0.0
			for a := 0; a < mm; a++ {
				idcg += (math.Pow(2, sortedRels[a]) - 1) * discounts[a]
			}
			if idcg == 0 {
				idcg = 1
			}
			// pairs
			for a := 0; a < mm; a++ {
				for b := 0; b < mm; b++ {
					if rels[a] <= rels[b] {
						continue
					}
					ia := idx[a]
					ib := idx[b]
					// positions in score order
					// find ranks
					order := make([]int, mm)
					for x := 0; x < mm; x++ {
						order[x] = x
					}
					// sort indices by score desc
					for x := 1; x < mm; x++ {
						for y2 := x; y2 > 0 && scores[order[y2-1]] < scores[order[y2]]; y2-- {
							order[y2-1], order[y2] = order[y2], order[y2-1]
						}
					}
					pa, pb := 0, 0
					for x := 0; x < mm; x++ {
						if order[x] == a {
							pa = x
						}
						if order[x] == b {
							pb = x
						}
					}
					gainA := math.Pow(2, rels[a]) - 1
					gainB := math.Pow(2, rels[b]) - 1
					delta := math.Abs(gainA*(discounts[pb]-discounts[pa])+gainB*(discounts[pa]-discounts[pb])) / idcg
					if delta == 0 {
						continue
					}
					sDiff := scores[a] - scores[b]
					rho := 1.0 / (1.0 + math.Exp(sDiff))
					lambdas[ia] += delta * rho
					lambdas[ib] -= delta * rho
					mus[ia] += delta * rho * (1 - rho)
					mus[ib] += delta * rho * (1 - rho)
				}
			}
		}
		for i := range mus {
			if mus[i] < 1e-6 {
				mus[i] = 1e-6
			}
		}
		tree := &rankTree{maxDepth: l.MaxDepth, minLeaf: l.MinSamplesLeaf}
		tree.root = tree.build(X, lambdas, mus, 0)
		upd := tree.Predict(X)
		for i := 0; i < n; i++ {
			F[i] += l.LearningRate * upd[i]
		}
		l.Trees = append(l.Trees, tree)
	}
}

func (l *LambdaMART) Predict(X mat.Matrix) mat.Vec {
	out := make(mat.Vec, X.Rows())
	for _, tree := range l.Trees {
		upd := tree.Predict(X)
		for i := range out {
			out[i] += l.LearningRate * upd[i]
		}
	}
	return out
}

// helpers
func weightedMean(lam, mu mat.Vec) float64 {
	num, den := 0.0, 0.0
	for i := range lam {
		num += lam[i]
		den += mu[i]
	}
	if den == 0 {
		return 0
	}
	return num / den
}

func allIdx(n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = i
	}
	return out
}

func wvar(lam, mu mat.Vec, idx []int) float64 {
	// weighted variance of lam over idx with weights mu
	wsum := 0.0
	for _, i := range idx {
		wsum += mu[i]
	}
	if wsum == 0 {
		return 0
	}
	mean := 0.0
	for _, i := range idx {
		mean += mu[i] * lam[i]
	}
	mean /= wsum
	s := 0.0
	for _, i := range idx {
		d := lam[i] - mean
		s += mu[i] * d * d
	}
	return s
}

func wvarIdx(lam, mu mat.Vec, idx []int) float64 {
	return wvar(lam, mu, idx)
}

func uniqueSortedCol(X mat.Matrix, feat int) []float64 {
	n := X.Rows()
	vals := make([]float64, n)
	for i := 0; i < n; i++ {
		vals[i] = X[i][feat]
	}
	// sort
	for i := 1; i < n; i++ {
		for j := i; j > 0 && vals[j-1] > vals[j]; j-- {
			vals[j-1], vals[j] = vals[j], vals[j-1]
		}
	}
	out := []float64{}
	for i, v := range vals {
		if i == 0 || v != vals[i-1] {
			out = append(out, v)
		}
	}
	return out
}

// avoid unused import
var _ = regression.NewDecisionTreeRegressor
var _ = mat.NewMatrix
