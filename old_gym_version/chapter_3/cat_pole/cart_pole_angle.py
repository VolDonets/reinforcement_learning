from time import sleep
import gym
import random


# init env
env = gym.make('CartPole-v1')

# making the script reproducible
seed = 0
random.seed(seed)
env.seed(seed)

print(f'Action Space: {env.action_space}\n')
print(f'Observation Space: {env.observation_space}')

# run 10 episodes in a row
episodes = 10
for i in range(episodes):
    state = env.reset()
    reward_sum = 0

    while True:
        env.render()
        action = 1 if state[2] > 0 else 0
        state, reward, done, debug = env.step(action)
        reward_sum += reward
        sleep(.01)
        if done:
            print(f'Episode {i} reward: {reward_sum}')
            sleep(1)
            break

env.close()
