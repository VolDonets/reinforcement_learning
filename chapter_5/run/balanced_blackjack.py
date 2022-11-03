import random
import gym

from chapter_5.monte_carlo.policy import q_greedy_policy
from chapter_5.monte_carlo.q_monte_carlo import train_q_monte_carlo
from chapter_5.utils.plot_q import plot_q
import numpy as np


env = gym.make('Blackjack-v0')

seed = 0
random.seed(seed)
np.random.seed(seed)
env.seed(seed)

Q = train_q_monte_carlo(env, 1000_000, gamma=.5)
plot_q(Q)

episodes = 10000
wins = 0
losses = 0
draws = 0

for e in range(1, episodes+1):
    reward = q_greedy_policy(env, Q)
    if reward > 0:
        wins += 1
    elif reward < 0:
        losses += 1
    else:
        draws += 1

print(f'Wins: {wins} | Losses: {losses} | Draws: {draws} | Total: {episodes}')
