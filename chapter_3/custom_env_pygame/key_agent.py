import random
from time import sleep
from catch_coins_env import CatchCoinsEnv

env = CatchCoinsEnv()

init_state = env.reset()
# -1: left
#  0: stay
# +1: right
action_space = [-1, 0, 1]

for _ in range(1000):
    env.render('ascii')
    print("push step: ")
    action = int(input())
    action -= 2
    state, reward, done, debug = env.step(action)
    sleep(.2)

env.close()
