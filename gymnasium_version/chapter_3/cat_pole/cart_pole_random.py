from time import sleep
import gymnasium as gym
import random


# init env
env = gym.make('CartPole-v1', render_mode="human")

# making the script reproducible
seed = 0
random.seed(seed)
# env.seed(seed)

print(f'Action Space: {env.action_space}\n')
print(f'Observation Space: {env.observation_space}')

# run 10 episodes in a row
episodes = 10
# init_state = env.reset()
for i in range(episodes):
    init_state = env.reset()
    reward_sum = 0

    while True:
        env.render()
        random_action = random.randint(0, 1)
        state, reward, done, trunc, info = env.step(random_action)
        reward_sum += reward
        sleep(0.1)
        if done:
            print(f'Episode {i} reward: {reward_sum}')
            sleep(0.1)
            break

env.close()
