package rl

import (
	"math/rand"

	"mlfromscratch/mat"
)

type GridWorld struct {
	Size  int
	Goal  [2]int
	Pit   [2]int
	Pos   [2]int
}

func NewGridWorld(size int, goal, pit [2]int) *GridWorld {
	return &GridWorld{Size: size, Goal: goal, Pit: pit}
}

func (g *GridWorld) Reset() int {
	g.Pos = [2]int{0, 0}
	return g.state()
}

func (g *GridWorld) state() int {
	return g.Pos[0]*g.Size + g.Pos[1]
}

func (g *GridWorld) Step(action int) (int, float64, bool) {
	dirs := [][2]int{{0, 1}, {0, -1}, {1, 0}, {-1, 0}}
	dr, dc := dirs[action][0], dirs[action][1]
	nr, nc := g.Pos[0]+dr, g.Pos[1]+dc
	if nr >= 0 && nr < g.Size && nc >= 0 && nc < g.Size {
		g.Pos = [2]int{nr, nc}
	}
	reward := -0.1
	done := false
	if g.Pos == g.Goal {
		reward = 10
		done = true
	} else if g.Pos == g.Pit {
		reward = -10
		done = true
	}
	return g.state(), reward, done
}

type QLearningAgent struct {
	NStates      int
	NActions     int
	LR           float64
	Gamma        float64
	Epsilon      float64
	EpsilonDecay float64
	EpsilonMin   float64
	Seed         int64
	Q            mat.Matrix
	rng          *rand.Rand
}

func NewQLearning(nStates, nActions int, lr, gamma, eps, decay, minEps float64, seed int64) *QLearningAgent {
	return &QLearningAgent{NStates: nStates, NActions: nActions, LR: lr, Gamma: gamma, Epsilon: eps, EpsilonDecay: decay, EpsilonMin: minEps, Seed: seed, Q: mat.NewMatrix(nStates, nActions), rng: rand.New(rand.NewSource(seed))}
}

func (a *QLearningAgent) Act(state int) int {
	if a.rng.Float64() < a.Epsilon {
		return a.rng.Intn(a.NActions)
	}
	best := 0
	bestV := a.Q[state][0]
	for ac := 1; ac < a.NActions; ac++ {
		if a.Q[state][ac] > bestV {
			bestV = a.Q[state][ac]
			best = ac
		}
	}
	return best
}

func (a *QLearningAgent) Update(s, act int, r float64, sNext int, done bool) {
	target := r
	if !done {
		mx := a.Q[sNext][0]
		for ac := 1; ac < a.NActions; ac++ {
			if a.Q[sNext][ac] > mx {
				mx = a.Q[sNext][ac]
			}
		}
		target += a.Gamma * mx
	}
	a.Q[s][act] += a.LR * (target - a.Q[s][act])
}

func (a *QLearningAgent) Train(env *GridWorld, episodes, maxSteps int) []float64 {
	rewards := make([]float64, episodes)
	for ep := 0; ep < episodes; ep++ {
		s := env.Reset()
		total := 0.0
		for step := 0; step < maxSteps; step++ {
			act := a.Act(s)
			sNext, r, done := env.Step(act)
			a.Update(s, act, r, sNext, done)
			s = sNext
			total += r
			if done {
				break
			}
		}
		a.Epsilon = mathMax(a.EpsilonMin, a.Epsilon*a.EpsilonDecay)
		rewards[ep] = total
	}
	return rewards
}

func mathMax(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
