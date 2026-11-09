"""Q-Learning from first principles.

Tabular Q-learning with epsilon-greedy on a simple GridWorld.
"""

import numpy as np


class GridWorld:
    def __init__(self, size=5, goal=(4, 4), pit=(2, 2)):
        self.size = size
        self.goal = goal
        self.pit = pit
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # R,L,D,U
        self.n_actions = 4
        self.n_states = size * size
        self.reset()

    def reset(self):
        self.pos = (0, 0)
        return self._state()

    def _state(self):
        return self.pos[0] * self.size + self.pos[1]

    def step(self, action):
        dr, dc = self.actions[action]
        r, c = self.pos
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.size and 0 <= nc < self.size:
            self.pos = (nr, nc)
        # reward
        if self.pos == self.goal:
            reward = 10.0
            done = True
        elif self.pos == self.pit:
            reward = -10.0
            done = True
        else:
            reward = -0.1
            done = False
        return self._state(), reward, done

    def render_q(self, Q):
        # not necessary
        pass


class QLearningAgent:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 random_state=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.Q = np.zeros((n_states, n_actions))

    def act(self, state):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        # greedy with tie-break
        q = self.Q[state]
        maxq = q.max()
        actions = np.where(q == maxq)[0]
        return int(self.rng.choice(actions))

    def update(self, s, a, r, s_next, done):
        target = r
        if not done:
            target += self.gamma * np.max(self.Q[s_next])
        self.Q[s, a] += self.lr * (target - self.Q[s, a])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env, episodes=500, max_steps=100):
        rewards = []
        for ep in range(episodes):
            s = env.reset()
            total = 0.0
            for _ in range(max_steps):
                a = self.act(s)
                s_next, r, done = env.step(a)
                self.update(s, a, r, s_next, done)
                s = s_next
                total += r
                if done:
                    break
            self.decay_epsilon()
            rewards.append(total)
        return rewards
