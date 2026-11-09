package rl

import (
	"math"
	"math/rand"

	"mlfromscratch/mat"
)

// CartPole environment
type CartPole struct {
	State         mat.Vec
	Steps         int
	Gravity       float64
	MassCart      float64
	MassPole      float64
	TotalMass     float64
	Length        float64
	PoleMassLen   float64
	ForceMag      float64
	Tau           float64
	ThetaThresh   float64
	XThresh       float64
}

func NewCartPole() *CartPole {
	cp := &CartPole{Gravity: 9.8, MassCart: 1.0, MassPole: 0.1, Length: 0.5, ForceMag: 10.0, Tau: 0.02, ThetaThresh: 12 * 2 * math.Pi / 360, XThresh: 2.4}
	cp.TotalMass = cp.MassCart + cp.MassPole
	cp.PoleMassLen = cp.MassPole * cp.Length
	return cp
}

func (c *CartPole) Reset() mat.Vec {
	c.State = make(mat.Vec, 4)
	for i := range c.State {
		c.State[i] = (rand.Float64()*2 - 1) * 0.05
	}
	c.Steps = 0
	return c.State.Clone()
}

func (c *CartPole) Step(action int) (mat.Vec, float64, bool) {
	x, xdot, theta, thetadot := c.State[0], c.State[1], c.State[2], c.State[3]
	force := c.ForceMag
	if action == 0 {
		force = -c.ForceMag
	}
	costheta := math.Cos(theta)
	sintheta := math.Sin(theta)
	temp := (force + c.PoleMassLen*thetadot*thetadot*sintheta) / c.TotalMass
	thetaacc := (c.Gravity*sintheta - costheta*temp) / (c.Length * (4.0/3.0 - c.MassPole*costheta*costheta/c.TotalMass))
	xacc := temp - c.PoleMassLen*thetaacc*costheta/c.TotalMass
	x = x + c.Tau*xdot
	xdot = xdot + c.Tau*xacc
	theta = theta + c.Tau*thetadot
	thetadot = thetadot + c.Tau*thetaacc
	c.State = mat.Vec{x, xdot, theta, thetadot}
	c.Steps++
	done := math.Abs(x) > c.XThresh || math.Abs(theta) > c.ThetaThresh || c.Steps >= 500
	reward := 1.0
	if math.Abs(x) > c.XThresh || math.Abs(theta) > c.ThetaThresh {
		reward = 0.0
	}
	return c.State.Clone(), reward, done
}

// MLP for actor/critic
type mlp struct {
	W1 mat.Matrix
	B1 mat.Vec
	W2 mat.Matrix
	B2 mat.Vec
}

func newMLP(in, hidden, out int, seed int64) *mlp {
	rng := rand.New(rand.NewSource(seed))
	m := &mlp{}
	m.W1 = mat.RandnMat(in, hidden, rng)
	for i := range m.W1 {
		for j := range m.W1[i] {
			m.W1[i][j] *= math.Sqrt(2.0 / float64(in))
		}
	}
	m.B1 = make(mat.Vec, hidden)
	m.W2 = mat.RandnMat(hidden, out, rng)
	for i := range m.W2 {
		for j := range m.W2[i] {
			m.W2[i][j] *= math.Sqrt(2.0 / float64(hidden))
		}
	}
	m.B2 = make(mat.Vec, out)
	return m
}

func (m *mlp) forward(X mat.Matrix) (mat.Matrix, mat.Matrix, mat.Matrix) {
	// returns output, hidden, pre-activation
	batch := X.Rows()
	in := X.Cols()
	hidden := len(m.B1)
	outDim := len(m.B2)
	hpre := mat.NewMatrix(batch, hidden)
	h := mat.NewMatrix(batch, hidden)
	for i := 0; i < batch; i++ {
		for j := 0; j < hidden; j++ {
			s := m.B1[j]
			for k := 0; k < in; k++ {
				s += X[i][k] * m.W1[k][j]
			}
			hpre[i][j] = s
			if s > 0 {
				h[i][j] = s
			}
		}
	}
	out := mat.NewMatrix(batch, outDim)
	for i := 0; i < batch; i++ {
		for j := 0; j < outDim; j++ {
			s := m.B2[j]
			for k := 0; k < hidden; k++ {
				s += h[i][k] * m.W2[k][j]
			}
			out[i][j] = s
		}
	}
	return out, h, hpre
}

type PPOAgent struct {
	StateDim  int
	ActionDim int
	Hidden    int
	LR        float64
	Gamma     float64
	Lam       float64
	ClipEps   float64
	Epochs    int
	BatchSize int
	Seed      int64

	Actor  *mlp
	Critic *mlp
	rng    *rand.Rand
}

func NewPPO(stateDim, actionDim, hidden int, lr, gamma, lam, clip float64, epochs, batchSize int, seed int64) *PPOAgent {
	if hidden == 0 {
		hidden = 64
	}
	if lr == 0 {
		lr = 3e-4
	}
	if gamma == 0 {
		gamma = 0.99
	}
	if lam == 0 {
		lam = 0.95
	}
	if clip == 0 {
		clip = 0.2
	}
	if epochs == 0 {
		epochs = 4
	}
	if batchSize == 0 {
		batchSize = 64
	}
	return &PPOAgent{StateDim: stateDim, ActionDim: actionDim, Hidden: hidden, LR: lr, Gamma: gamma, Lam: lam, ClipEps: clip, Epochs: epochs, BatchSize: batchSize, Seed: seed, Actor: newMLP(stateDim, hidden, actionDim, seed), Critic: newMLP(stateDim, hidden, 1, seed+1), rng: rand.New(rand.NewSource(seed))}
}

func softmax(v mat.Vec) mat.Vec {
	out := make(mat.Vec, len(v))
	mx := v[0]
	for i := 1; i < len(v); i++ {
		if v[i] > mx {
			mx = v[i]
		}
	}
	s := 0.0
	for i := range v {
		out[i] = math.Exp(v[i] - mx)
		s += out[i]
	}
	for i := range out {
		out[i] /= s
	}
	return out
}

func (a *PPOAgent) Act(state mat.Vec) (int, float64, float64) {
	X := mat.NewMatrix(1, a.StateDim)
	for j := 0; j < a.StateDim; j++ {
		X[0][j] = state[j]
	}
	logits, _, _ := a.Actor.forward(X)
	lv := make(mat.Vec, a.ActionDim)
	for j := 0; j < a.ActionDim; j++ {
		lv[j] = logits[0][j]
	}
	probs := softmax(lv)
	action := 0
	r := a.rng.Float64()
	acc := 0.0
	for j := 0; j < a.ActionDim; j++ {
		acc += probs[j]
		if acc >= r {
			action = j
			break
		}
	}
	logp := math.Log(probs[action] + 1e-8)
	vals, _, _ := a.Critic.forward(X)
	value := vals[0][0]
	return action, logp, value
}

func (a *PPOAgent) Train(env *CartPole, totalSteps, rolloutLen int) []float64 {
	rewards := []float64{}
	state := env.Reset()
	for len(rewards)*rolloutLen < totalSteps {
		states := make([]mat.Vec, 0, rolloutLen)
		actions := make([]int, 0, rolloutLen)
		logps := make([]float64, 0, rolloutLen)
		rewardsR := make([]float64, 0, rolloutLen)
		dones := make([]float64, 0, rolloutLen)
		values := make([]float64, 0, rolloutLen)
		for step := 0; step < rolloutLen; step++ {
			act, logp, val := a.Act(state)
			ns, r, done := env.Step(act)
			states = append(states, state.Clone())
			actions = append(actions, act)
			logps = append(logps, logp)
			rewardsR = append(rewardsR, r)
			dones = append(dones, boolToFloat(done))
			values = append(values, val)
			state = ns
			if done {
				state = env.Reset()
			}
		}
		// next value
		nextVal := 0.0
		if dones[rolloutLen-1] == 0 {
			X := mat.NewMatrix(1, a.StateDim)
			for j := 0; j < a.StateDim; j++ {
				X[0][j] = state[j]
			}
			v, _, _ := a.Critic.forward(X)
			nextVal = v[0][0]
		}
		// GAE
		adv := make([]float64, rolloutLen)
		gae := 0.0
		for t := rolloutLen - 1; t >= 0; t-- {
			nv := nextVal
			if t < rolloutLen-1 {
				nv = values[t+1]
			}
			delta := rewardsR[t] + a.Gamma*nv*(1-dones[t]) - values[t]
			gae = delta + a.Gamma*a.Lam*(1-dones[t])*gae
			adv[t] = gae
		}
		returns := make([]float64, rolloutLen)
		for t := 0; t < rolloutLen; t++ {
			returns[t] = adv[t] + values[t]
		}
		// normalize adv
		meanA, varA := 0.0, 0.0
		for _, v := range adv {
			meanA += v
		}
		meanA /= float64(rolloutLen)
		for _, v := range adv {
			d := v - meanA
			varA += d * d
		}
		varA /= float64(rolloutLen)
		stdA := math.Sqrt(varA + 1e-8)
		for t := range adv {
			adv[t] = (adv[t] - meanA) / stdA
		}
		// update for epochs
		for epoch := 0; epoch < a.Epochs; epoch++ {
			perm := a.rng.Perm(rolloutLen)
			for start := 0; start < rolloutLen; start += a.BatchSize {
				end := start + a.BatchSize
				if end > rolloutLen {
					end = rolloutLen
				}
				bs := end - start
				// build batch
				Xb := mat.NewMatrix(bs, a.StateDim)
				ab := make([]int, bs)
				oldlp := make([]float64, bs)
				retb := make([]float64, bs)
				advb := make([]float64, bs)
				for i := 0; i < bs; i++ {
					idx := perm[start+i]
					for j := 0; j < a.StateDim; j++ {
						Xb[i][j] = states[idx][j]
					}
					ab[i] = actions[idx]
					oldlp[i] = logps[idx]
					retb[i] = returns[idx]
					advb[i] = adv[idx]
				}
				// forward
				logits, h, hpre := a.Actor.forward(Xb)
				vals, vh, _ := a.Critic.forward(Xb)
				// log probs of taken actions
				logp := make([]float64, bs)
				for i := 0; i < bs; i++ {
					// softmax
					mx := logits[i][0]
					for j := 1; j < a.ActionDim; j++ {
						if logits[i][j] > mx {
							mx = logits[i][j]
						}
					}
					s := 0.0
					for j := 0; j < a.ActionDim; j++ {
						s += math.Exp(logits[i][j] - mx)
					}
					p := math.Exp(logits[i][ab[i]]-mx) / s
					logp[i] = math.Log(p + 1e-8)
				}
				// ratio
				ratio := make([]float64, bs)
				for i := 0; i < bs; i++ {
					ratio[i] = math.Exp(logp[i] - oldlp[i])
				}
				// clipped mask
				// gradient of actor loss (simplified): use analytic gradient
				// dlogits
				dlogits := mat.NewMatrix(bs, a.ActionDim)
				for i := 0; i < bs; i++ {
					clipped := ratio[i] > 1+a.ClipEps || ratio[i] < 1-a.ClipEps
					coef := 0.0
					if !clipped {
						coef = advb[i] * ratio[i]
					}
					// d logp/d logits
					// softmax probs
					mx := logits[i][0]
					for j := 1; j < a.ActionDim; j++ {
						if logits[i][j] > mx {
							mx = logits[i][j]
						}
					}
					s := 0.0
					probs := make([]float64, a.ActionDim)
					for j := 0; j < a.ActionDim; j++ {
						probs[j] = math.Exp(logits[i][j] - mx)
						s += probs[j]
					}
					for j := 0; j < a.ActionDim; j++ {
						probs[j] /= s
					}
					for j := 0; j < a.ActionDim; j++ {
						dlp := 0.0
						if j == ab[i] {
							dlp = 1 - probs[j]
						} else {
							dlp = -probs[j]
						}
						dlogits[i][j] += -coef * dlp
					}
				}
				// scale
				for i := 0; i < bs; i++ {
					for j := 0; j < a.ActionDim; j++ {
						dlogits[i][j] /= float64(bs)
					}
				}
				// backprop actor
				// h relu (hpre)
				// grad W2
				dW2 := mat.NewMatrix(a.Hidden, a.ActionDim)
				db2 := make(mat.Vec, a.ActionDim)
				for i := 0; i < bs; i++ {
					for k := 0; k < a.Hidden; k++ {
						for j := 0; j < a.ActionDim; j++ {
							dW2[k][j] += h[i][k] * dlogits[i][j]
						}
					}
					for j := 0; j < a.ActionDim; j++ {
						db2[j] += dlogits[i][j]
					}
				}
				// dh
				dh := mat.NewMatrix(bs, a.Hidden)
				for i := 0; i < bs; i++ {
					for k := 0; k < a.Hidden; k++ {
						s := 0.0
						for j := 0; j < a.ActionDim; j++ {
							s += dlogits[i][j] * a.Actor.W2[k][j]
						}
						if hpre[i][k] > 0 {
							dh[i][k] = s
						}
					}
				}
				dW1 := mat.NewMatrix(a.StateDim, a.Hidden)
				db1 := make(mat.Vec, a.Hidden)
				for i := 0; i < bs; i++ {
					for k := 0; k < a.StateDim; k++ {
						for j := 0; j < a.Hidden; j++ {
							dW1[k][j] += Xb[i][k] * dh[i][j]
						}
					}
					for j := 0; j < a.Hidden; j++ {
						db1[j] += dh[i][j]
					}
				}
				// update actor
				for i := 0; i < a.StateDim; i++ {
					for j := 0; j < a.Hidden; j++ {
						a.Actor.W1[i][j] -= a.LR * dW1[i][j]
					}
				}
				for j := 0; j < a.Hidden; j++ {
					a.Actor.B1[j] -= a.LR * db1[j]
				}
				for i := 0; i < a.Hidden; i++ {
					for j := 0; j < a.ActionDim; j++ {
						a.Actor.W2[i][j] -= a.LR * dW2[i][j]
					}
				}
				for j := 0; j < a.ActionDim; j++ {
					a.Actor.B2[j] -= a.LR * db2[j]
				}

				// critic gradient
				// value loss MSE: d/dval = 2*(val-ret)/bs
				dv := make(mat.Vec, bs)
				for i := 0; i < bs; i++ {
					v := vals[i][0]
					dv[i] = 2 * (v - retb[i]) / float64(bs)
				}
				// backprop critic
				dW2c := mat.NewMatrix(a.Hidden, 1)
				db2c := make(mat.Vec, 1)
				for i := 0; i < bs; i++ {
					for k := 0; k < a.Hidden; k++ {
						dW2c[k][0] += vh[i][k] * dv[i]
					}
					db2c[0] += dv[i]
				}
				dhc := mat.NewMatrix(bs, a.Hidden)
				for i := 0; i < bs; i++ {
					for k := 0; k < a.Hidden; k++ {
						s := dv[i] * a.Critic.W2[k][0]
						// need hpre critic; recompute
						_ = s
					}
				}
				// recompute critic hpre
				hpreC := mat.NewMatrix(bs, a.Hidden)
				for i := 0; i < bs; i++ {
					for k := 0; k < a.Hidden; k++ {
						s := a.Critic.B1[k]
						for j := 0; j < a.StateDim; j++ {
							s += Xb[i][j] * a.Critic.W1[j][k]
						}
						hpreC[i][k] = s
					}
				}
				for i := 0; i < bs; i++ {
					for k := 0; k < a.Hidden; k++ {
						s := dv[i] * a.Critic.W2[k][0]
						if hpreC[i][k] > 0 {
							dhc[i][k] = s
						}
					}
				}
				dW1c := mat.NewMatrix(a.StateDim, a.Hidden)
				db1c := make(mat.Vec, a.Hidden)
				for i := 0; i < bs; i++ {
					for k := 0; k < a.StateDim; k++ {
						for j := 0; j < a.Hidden; j++ {
							dW1c[k][j] += Xb[i][k] * dhc[i][j]
						}
					}
					for j := 0; j < a.Hidden; j++ {
						db1c[j] += dhc[i][j]
					}
				}
				// update critic
				for i := 0; i < a.StateDim; i++ {
					for j := 0; j < a.Hidden; j++ {
						a.Critic.W1[i][j] -= a.LR * dW1c[i][j]
					}
				}
				for j := 0; j < a.Hidden; j++ {
					a.Critic.B1[j] -= a.LR * db1c[j]
				}
				for i := 0; i < a.Hidden; i++ {
					a.Critic.W2[i][0] -= a.LR * dW2c[i][0]
				}
				a.Critic.B2[0] -= a.LR * db2c[0]
			}
		}
		// episode reward sum
		sumR := 0.0
		for _, r := range rewardsR {
			sumR += r
		}
		rewards = append(rewards, sumR)
	}
	return rewards
}

func boolToFloat(b bool) float64 {
	if b {
		return 1
	}
	return 0
}
