from time import sleep
import gym


old_pos = 0.0

env = gym.make('MountainCar-v0')

print(f'Action Space: {env.action_space}')
print(f'Observation Space: {env.observation_space}')

episodes = 3

for i in range(episodes):
    state = env.reset()
    reward_sum = 0
    while True:
        env.render()
        action = 1
        if old_pos > state[0]:
            action = 0
        elif old_pos < state[0]:
            action = 2
        old_pos = state[0]
        state, reward, done, debug = env.step(action)
        # print(state)
        reward_sum += reward
        sleep(0.1)
        if done:
            print(f'Episode_{i}, reward: {reward_sum}')
            sleep(1)
            break

env.close()
