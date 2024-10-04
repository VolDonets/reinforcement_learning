from time import sleep
import gymnasium as gym


env = gym.make('MountainCar-v0', render_mode="human")

episodes = 10

for i in range(episodes):
    init_state = env.reset()
    reward_sum = 0
    while True:
        env.render()
        random_action = env.action_space.sample()
        state, reward, done, trunc, info = env.step(random_action)
        reward_sum += reward
        sleep(0.1)
        if done:
            print(f'Episode_{i}, reward: {reward_sum}')
            sleep(1)
            break

env.close()
