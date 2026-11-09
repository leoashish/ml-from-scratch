package forecasting

import (
	"math"

	"mlfromscratch/mat"
)

type ARIMA struct {
	P, D, Q int
	AR      mat.Vec
	MA      mat.Vec
	Intercept float64
	Resid   mat.Vec
	Diff    mat.Vec
	History mat.Vec
	Levels  []mat.Vec
}

func NewARIMA(p, d, q int) *ARIMA {
	return &ARIMA{P: p, D: d, Q: q}
}

func (m *ARIMA) difference(series mat.Vec, d int) (mat.Vec, []mat.Vec) {
	s := series.Clone()
	levels := []mat.Vec{s.Clone()}
	for i := 0; i < d; i++ {
		ns := make(mat.Vec, len(s)-1)
		for j := 1; j < len(s); j++ {
			ns[j-1] = s[j] - s[j-1]
		}
		s = ns
		levels = append(levels, s.Clone())
	}
	return s, levels
}

func (m *ARIMA) Fit(series mat.Vec) {
	m.History = series.Clone()
	s, levels := m.difference(series, m.D)
	m.Diff = s
	m.Levels = levels
	n := len(s)
	p, q := m.P, m.Q
	// center
	meanS := 0.0
	for _, v := range s {
		meanS += v
	}
	meanS /= float64(n)
	sc := make(mat.Vec, n)
	for i := range s {
		sc[i] = s[i] - meanS
	}
	// Yule-Walker AR
	ar := make(mat.Vec, p)
	if p > 0 {
		// autocorr
		r := make(mat.Vec, p+1)
		for k := 0; k <= p; k++ {
			sv := 0.0
			for t := 0; t+k < n; t++ {
				sv += sc[t] * sc[t+k]
			}
			r[k] = sv / float64(n)
		}
		// solve Toeplitz R a = r[1:]
		R := mat.NewMatrix(p, p)
		for i := 0; i < p; i++ {
			for j := 0; j < p; j++ {
				R[i][j] = r[int(math.Abs(float64(i-j)))]
			}
		}
		rhs := make(mat.Vec, p)
		for i := 0; i < p; i++ {
			rhs[i] = r[i+1]
		}
		sol, err := mat.Solve(R, rhs)
		if err == nil {
			ar = sol
		}
	}
	// residuals from AR
	resid := make(mat.Vec, n)
	for t := 0; t < n; t++ {
		val := meanS
		for i := 1; i <= p; i++ {
			if t-i >= 0 {
				val += ar[i-1] * (s[t-i] - meanS)
			}
		}
		resid[t] = s[t] - val
	}
	// MA via regression resid[t] on resid[t-1..]
	ma := make(mat.Vec, q)
	if q > 0 {
		rows := [][]float64{}
		targets := []float64{}
		for t := q; t < n; t++ {
			row := make([]float64, q)
			for j := 1; j <= q; j++ {
				row[j-1] = resid[t-j]
			}
			rows = append(rows, row)
			targets = append(targets, resid[t])
		}
		// least squares via normal equations
		// A^T A beta = A^T b
		k := q
		AtA := mat.NewMatrix(k, k)
		Atb := make(mat.Vec, k)
		for r := 0; r < len(rows); r++ {
			for a := 0; a < k; a++ {
				Atb[a] += rows[r][a] * targets[r]
				for b := 0; b < k; b++ {
					AtA[a][b] += rows[r][a] * rows[r][b]
				}
			}
		}
		sol, err := mat.Solve(AtA, Atb)
		if err == nil {
			ma = sol
		}
	}
	// refine residuals with MA
	resid = make(mat.Vec, n)
	for t := 0; t < n; t++ {
		val := meanS
		for i := 1; i <= p; i++ {
			if t-i >= 0 {
				val += ar[i-1] * (s[t-i] - meanS)
			}
		}
		for j := 1; j <= q; j++ {
			if t-j >= 0 {
				val += ma[j-1] * resid[t-j]
			}
		}
		resid[t] = s[t] - val
	}
	m.AR = ar
	m.MA = ma
	m.Intercept = meanS
	m.Resid = resid
}

func (m *ARIMA) Forecast(steps int) mat.Vec {
	sHist := m.Diff.Clone()
	rHist := m.Resid.Clone()
	p, q := m.P, m.Q
	out := make(mat.Vec, steps)
	for step := 0; step < steps; step++ {
		val := m.Intercept
		for i := 1; i <= p; i++ {
			if len(sHist)-i >= 0 {
				val += m.AR[i-1] * (sHist[len(sHist)-i] - m.Intercept)
			}
		}
		for j := 1; j <= q; j++ {
			if len(rHist)-j >= 0 {
				val += m.MA[j-1] * rHist[len(rHist)-j]
			}
		}
		out[step] = val
		sHist = append(sHist, val)
		rHist = append(rHist, 0)
	}
	// invert differencing
	if m.D == 0 {
		return out
	}
	if m.D == 1 {
		inv := make(mat.Vec, steps)
		cur := m.History[len(m.History)-1]
		for i := 0; i < steps; i++ {
			cur += out[i]
			inv[i] = cur
		}
		return inv
	}
	// d>1
	inv := out.Clone()
	for level := m.D - 1; level >= 0; level-- {
		var last float64
		if level == 0 {
			last = m.History[len(m.History)-1]
		} else {
			last = m.Levels[level][len(m.Levels[level])-1]
		}
		cur := last
		new := make(mat.Vec, steps)
		for i := 0; i < steps; i++ {
			cur += inv[i]
			new[i] = cur
		}
		inv = new
	}
	return inv
}
