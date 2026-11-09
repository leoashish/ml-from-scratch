package generative

import (
	"math"
	"math/rand"

	"mlfromscratch/mat"
)

type AutoRegChar struct {
	Context  int
	Hidden   int
	LR       float64
	NIters   int
	BatchSize int
	Seed     int64

	Chars     []rune
	CharToIdx map[rune]int
	IdxToChar map[int]rune
	Vocab     int

	W1 mat.Matrix
	B1 mat.Vec
	W2 mat.Matrix
	B2 mat.Vec
}

func NewAutoRegChar(context, hidden int, lr float64, nIters, batchSize int, seed int64) *AutoRegChar {
	if context == 0 {
		context = 4
	}
	if hidden == 0 {
		hidden = 32
	}
	if lr == 0 {
		lr = 0.1
	}
	if nIters == 0 {
		nIters = 500
	}
	if batchSize == 0 {
		batchSize = 64
	}
	return &AutoRegChar{Context: context, Hidden: hidden, LR: lr, NIters: nIters, BatchSize: batchSize, Seed: seed}
}

func (m *AutoRegChar) Fit(text string) {
	runes := []rune(text)
	set := map[rune]bool{}
	for _, c := range runes {
		set[c] = true
	}
	m.Chars = make([]rune, 0, len(set))
	for c := range set {
		m.Chars = append(m.Chars, c)
	}
	// sort
	for i := 1; i < len(m.Chars); i++ {
		for j := i; j > 0 && m.Chars[j-1] > m.Chars[j]; j-- {
			m.Chars[j-1], m.Chars[j] = m.Chars[j], m.Chars[j-1]
		}
	}
	m.Vocab = len(m.Chars)
	m.CharToIdx = map[rune]int{}
	m.IdxToChar = map[int]rune{}
	for i, c := range m.Chars {
		m.CharToIdx[c] = i
		m.IdxToChar[i] = c
	}
	rng := rand.New(rand.NewSource(m.Seed))
	inDim := m.Context * m.Vocab
	m.W1 = randnMat(inDim, m.Hidden, rng)
	for i := range m.W1 {
		for j := range m.W1[i] {
			m.W1[i][j] *= 0.01
		}
	}
	m.B1 = make(mat.Vec, m.Hidden)
	m.W2 = randnMat(m.Hidden, m.Vocab, rng)
	for i := range m.W2 {
		for j := range m.W2[i] {
			m.W2[i][j] *= 0.01
		}
	}
	m.B2 = make(mat.Vec, m.Vocab)

	// make training data
	n := len(runes) - m.Context
	if n <= 0 {
		return
	}
	X := mat.NewMatrix(n, inDim)
	y := make([]int, n)
	for t := m.Context; t < len(runes); t++ {
		i := t - m.Context
		for lag := 0; lag < m.Context; lag++ {
			c := runes[t-m.Context+lag]
			idx := m.CharToIdx[c]
			X[i][lag*m.Vocab+idx] = 1
		}
		y[i] = m.CharToIdx[runes[t]]
	}
	for it := 0; it < m.NIters; it++ {
		bs := m.BatchSize
		if bs > n {
			bs = n
		}
		// random batch
		for b := 0; b < bs; b++ {
			ri := rng.Intn(n)
			// single sample gradient (SGD)
			// forward
			h := make(mat.Vec, m.Hidden)
			hpre := make(mat.Vec, m.Hidden)
			for j := 0; j < m.Hidden; j++ {
				s := m.B1[j]
				for k := 0; k < inDim; k++ {
					s += X[ri][k] * m.W1[k][j]
				}
				hpre[j] = s
				h[j] = math.Tanh(s)
			}
			logits := make(mat.Vec, m.Vocab)
			for j := 0; j < m.Vocab; j++ {
				s := m.B2[j]
				for k := 0; k < m.Hidden; k++ {
					s += h[k] * m.W2[k][j]
				}
				logits[j] = s
			}
			// softmax
			mx := logits[0]
			for j := 1; j < m.Vocab; j++ {
				if logits[j] > mx {
					mx = logits[j]
				}
			}
			probs := make(mat.Vec, m.Vocab)
			s := 0.0
			for j := 0; j < m.Vocab; j++ {
				probs[j] = math.Exp(logits[j] - mx)
				s += probs[j]
			}
			for j := 0; j < m.Vocab; j++ {
				probs[j] /= s
			}
			// grad
			// dlogits = probs - onehot(y)
			dlogits := probs.Clone()
			dlogits[y[ri]] -= 1
			// update W2,B2
			for k := 0; k < m.Hidden; k++ {
				for j := 0; j < m.Vocab; j++ {
					m.W2[k][j] -= m.LR * h[k] * dlogits[j]
				}
			}
			for j := 0; j < m.Vocab; j++ {
				m.B2[j] -= m.LR * dlogits[j]
			}
			// dh
			dh := make(mat.Vec, m.Hidden)
			for k := 0; k < m.Hidden; k++ {
				s := 0.0
				for j := 0; j < m.Vocab; j++ {
					s += dlogits[j] * m.W2[k][j]
				}
				dh[k] = s * (1 - h[k]*h[k])
			}
			for k := 0; k < inDim; k++ {
				for j := 0; j < m.Hidden; j++ {
					m.W1[k][j] -= m.LR * X[ri][k] * dh[j]
				}
			}
			for j := 0; j < m.Hidden; j++ {
				m.B1[j] -= m.LR * dh[j]
			}
		}
	}
}

func (m *AutoRegChar) Generate(seedText string, length int, temperature float64, seed int64) string {
	rng := rand.New(rand.NewSource(seed))
	if temperature <= 0 {
		temperature = 1
	}
	runes := []rune(seedText)
	ctx := make([]int, m.Context)
	for i := 0; i < m.Context; i++ {
		c := rune(0)
		if i >= m.Context-len(runes) {
			c = runes[i-(m.Context-len(runes))]
		}
		// if seed shorter, pad with 0
		if i < m.Context-len(runes) {
			ctx[i] = 0
			continue
		}
		ctx[i] = m.CharToIdx[c]
	}
	if len(runes) >= m.Context {
		for i := 0; i < m.Context; i++ {
			ctx[i] = m.CharToIdx[runes[len(runes)-m.Context+i]]
		}
	}
	out := []rune(seedText)
	for s := 0; s < length; s++ {
		vec := make(mat.Vec, m.Context*m.Vocab)
		for lag := 0; lag < m.Context; lag++ {
			vec[lag*m.Vocab+ctx[lag]] = 1
		}
		h := make(mat.Vec, m.Hidden)
		for j := 0; j < m.Hidden; j++ {
			sv := m.B1[j]
			for k := 0; k < m.Context*m.Vocab; k++ {
				sv += vec[k] * m.W1[k][j]
			}
			h[j] = math.Tanh(sv)
		}
		logits := make(mat.Vec, m.Vocab)
		for j := 0; j < m.Vocab; j++ {
			sv := m.B2[j]
			for k := 0; k < m.Hidden; k++ {
				sv += h[k] * m.W2[k][j]
			}
			logits[j] = sv / temperature
		}
		mx := logits[0]
		for j := 1; j < m.Vocab; j++ {
			if logits[j] > mx {
				mx = logits[j]
			}
		}
		probs := make(mat.Vec, m.Vocab)
		sv := 0.0
		for j := 0; j < m.Vocab; j++ {
			probs[j] = math.Exp(logits[j] - mx)
			sv += probs[j]
		}
		for j := 0; j < m.Vocab; j++ {
			probs[j] /= sv
		}
		// sample
		r := rng.Float64()
		acc := 0.0
		chosen := m.Vocab - 1
		for j := 0; j < m.Vocab; j++ {
			acc += probs[j]
			if acc >= r {
				chosen = j
				break
			}
		}
		out = append(out, m.IdxToChar[chosen])
		ctx = append(ctx[1:], chosen)
	}
	return string(out)
}
