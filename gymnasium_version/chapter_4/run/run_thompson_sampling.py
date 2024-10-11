from gymnasium_version.chapter_4.policy.thompson_sampling import thompson_sampling_policy
import random
from gymnasium_version.chapter_4.gym.multiarmed_bandit_env import get_bandit_env_5
import matplotlib.pyplot as plt
import numpy as np


def run_thompson_sampling(balance_, env_, visualize=False):
    state = env_.reset()
    rewards_ = []

    for i in range(balance_):
        if i % 50 == 0:
            action = thompson_sampling_policy(state, visualize, plot_title = f'Iteration: {i}')
        else:
            action = thompson_sampling_policy(state, False, plot_title = f'Iteration: {i}')

        state, reward, done, debug = env_.step(action)
        rewards_.append(reward)

    env_.close()
    return env_, rewards_


if __name__ == '__main__':
    seed = 3
    random.seed(seed)

    balance = 1_000
    env = get_bandit_env_5()
    env, rewards = run_thompson_sampling(balance, env, visualize=True)
    env.render()

    cum_rewards = np.cumsum(rewards)
    plt.plot(cum_rewards)
    plt.title('Thompson Sampling Policy')
    plt.xlabel('Trials')
    plt.ylabel('Reward')
    plt.show()
