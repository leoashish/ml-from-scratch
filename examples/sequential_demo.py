"""Demo: sequential decision-making."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.sequential_decision.q_learning import GridWorld, QLearningAgent
from src.sequential_decision.ppo import CartPole, PPOAgent

# Q-learning
env = GridWorld(size=5)
agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions, random_state=0)
rewards = agent.train(env, episodes=300)
print("Q-learning last-10 avg reward:", round(float(__import__('numpy').mean(rewards[-10:])), 3))

# PPO
env2 = CartPole()
ppo = PPOAgent(random_state=0)
rh = ppo.train(env2, total_timesteps=2048, rollout_len=256)
print("PPO rollout rewards:", [round(r, 1) for r in rh])
