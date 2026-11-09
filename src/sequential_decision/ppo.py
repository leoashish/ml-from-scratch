"""Proximal Policy Optimization (PPO) from first principles.

Actor-critic with clipped surrogate objective. Includes a small CartPole
environment implemented from physics so the agent can train without gym.
"""

import numpy as np


class CartPole:
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02
        self.theta_threshold = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4
        self.state = None
        self.steps = 0

    def reset(self):
        self.state = np.random.uniform(-0.05, 0.05, size=4)
        self.steps = 0
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1
        done = (abs(x) > self.x_threshold or abs(theta) > self.theta_threshold
                or self.steps >= 500)
        reward = 1.0 if not (abs(x) > self.x_threshold or abs(theta) > self.theta_threshold) else 0.0
        return self.state.copy(), reward, done


def _relu(x):
    return np.maximum(0, x)


class _MLP:
    def __init__(self, in_dim, hidden, out_dim, random_state=None):
        rng = np.random.default_rng(random_state)
        self.W1 = rng.standard_normal((in_dim, hidden)) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((hidden, out_dim)) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(out_dim)

    def forward(self, X):
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        z1 = X @ self.W1 + self.b1
        h = _relu(z1)
        z2 = h @ self.W2 + self.b2
        return z2, h, z1, X

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]


class PPOAgent:
    def __init__(self, state_dim=4, action_dim=2, hidden=64, lr=3e-4,
                 gamma=0.99, lam=0.95, clip_eps=0.2, epochs=4,
                 batch_size=64, random_state=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.rng = np.random.default_rng(random_state)
        self.actor = _MLP(state_dim, hidden, action_dim, random_state=random_state)
        self.critic = _MLP(state_dim, hidden, 1, random_state=random_state + 1 if random_state is not None else 1)
        self.lr = lr

    def _softmax(self, logits):
        logits = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=-1, keepdims=True)

    def act(self, state):
        logits, *_ = self.actor.forward(state)
        logits = logits.ravel()
        probs = self._softmax(logits)
        action = int(self.rng.choice(self.action_dim, p=probs))
        logp = np.log(probs[action] + 1e-8)
        value, *_ = self.critic.forward(state)
        value = float(value.ravel()[0])
        return action, logp, value

    def _gae(self, rewards, values, dones, next_value):
        advantages = np.zeros(len(rewards))
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * (next_value if t == len(rewards) - 1 else values[t + 1]) * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def update(self, states, actions, old_logps, returns, advantages):
        states = np.asarray(states, float)
        actions = np.asarray(actions, int)
        old_logps = np.asarray(old_logps, float)
        returns = np.asarray(returns, float)
        advantages = np.asarray(advantages, float)
        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        n = len(states)
        for epoch in range(self.epochs):
            idx = self.rng.permutation(n)
            for start in range(0, n, self.batch_size):
                mb = idx[start:start + self.batch_size]
                s = states[mb]
                a = actions[mb]
                olp = old_logps[mb]
                ret = returns[mb]
                adv = advantages[mb]
                # forward
                logits, h, z1, X = self.actor.forward(s)
                values, vh, vz, vX = self.critic.forward(s)
                values = values.ravel()
                probs = self._softmax(logits)
                # log prob of taken actions
                logp = np.log(probs[np.arange(len(a)), a] + 1e-8)
                # ratio
                ratio = np.exp(logp - olp)
                # clipped surrogate
                surr1 = ratio * adv
                surr2 = np.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                actor_loss = -np.mean(np.minimum(surr1, surr2))
                # value loss
                value_loss = np.mean((values - ret) ** 2)
                # entropy
                entropy = -np.mean(np.sum(probs * np.log(probs + 1e-8), axis=1))
                # gradients (manual, combined)
                # We'll do finite-difference-ish? Better: analytic backprop for actor and critic separately.
                # Actor gradient: dL/dlogits
                # d actor_loss wrt logits:
                # ratio = exp(logp-olp); d ratio/d logits = ratio * d logp/d logits
                # d logp_i / d logits_{i,a} = 1 - probs_{i,a}; for other actions = -probs
                dlogits = np.zeros_like(logits)
                # gradient of -min(surr1,surr2) averaged
                # clipped mask
                clipped = (ratio > 1 + self.clip_eps) | (ratio < 1 - self.clip_eps)
                # for non-clipped use ratio*adv; for clipped use 0 (surrogate flat)
                # derivative of surr1 wrt logits = adv * ratio * dlogp/dlogits
                # derivative of surr2 (clipped) = 0 when clipped
                coef = np.where(clipped, 0.0, adv * ratio)  # shape (mb,)
                # d logp/d logits
                for i in range(len(a)):
                    dlogits[i, a[i]] += -coef[i] * (1 - probs[i, a[i]])
                    for j in range(self.action_dim):
                        if j != a[i]:
                            dlogits[i, j] += -coef[i] * (-probs[i, j])
                dlogits /= len(a)
                # entropy gradient: d/dlogits of -entropy? We maximize entropy, so loss -= entropy_coeff*entropy
                # entropy = -sum p log p; d entropy/d logits = ... ; we add small entropy bonus
                entropy_coef = 0.01
                # gradient of -entropy_coef*entropy wrt logits = -entropy_coef * dentropy/dlogits
                # dentropy/dlogits_k = -p_k (log p_k + 1) + p_k * sum... standard: dH/dz_k = -p_k (log p_k - H?) 
                # Simpler: use autograd-like finite diff? We'll approximate entropy gradient numerically small.
                # Skip entropy gradient for simplicity (minor).
                # Backprop actor
                # h relu
                dz2 = dlogits
                dW2 = h.T @ dz2
                db2 = dz2.sum(axis=0)
                dh = dz2 @ self.actor.W2.T
                dh = dh * (z1 > 0)
                dW1 = X.T @ dh
                db1 = dh.sum(axis=0)
                # update actor
                self.actor.W1 -= self.lr * dW1
                self.actor.b1 -= self.lr * db1
                self.actor.W2 -= self.lr * dW2
                self.actor.b2 -= self.lr * db2
                # Critic gradient
                dv = (values - ret) * 2 / len(ret)  # d MSE
                # backprop critic
                dz2c = dv.reshape(-1, 1)
                dW2c = vh.T @ dz2c
                db2c = dz2c.sum(axis=0)
                dhc = dz2c @ self.critic.W2.T
                dhc = dhc * (vz > 0)
                dW1c = vX.T @ dhc
                db1c = dhc.sum(axis=0)
                self.critic.W1 -= self.lr * dW1c
                self.critic.b1 -= self.lr * db1c
                self.critic.W2 -= self.lr * dW2c
                self.critic.b2 -= self.lr * db2c
        return actor_loss + value_loss

    def train(self, env, total_timesteps=20000, rollout_len=512):
        rewards_history = []
        state = env.reset()
        while len(rewards_history) * rollout_len < total_timesteps:
            states = []
            actions = []
            logps = []
            rewards = []
            dones = []
            values = []
            for _ in range(rollout_len):
                a, logp, v = self.act(state)
                ns, r, done = env.step(a)
                states.append(state.copy())
                actions.append(a)
                logps.append(logp)
                rewards.append(r)
                dones.append(float(done))
                values.append(v)
                state = ns
                if done:
                    state = env.reset()
            next_value = 0.0
            if not dones[-1]:
                nv, *_ = self.critic.forward(state)
                next_value = float(nv.ravel()[0])
            adv, ret = self._gae(np.array(rewards), np.array(values), np.array(dones), next_value)
            self.update(states, actions, logps, ret, adv)
            rewards_history.append(float(np.sum(rewards)))
        return rewards_history
