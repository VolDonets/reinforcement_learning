import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def select_action(state_, model_net):
    with torch.no_grad():
        # t.max(1) will return the largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        return model_net(state_).max(1).indices.view(1, 1)


env = gym.make("CartPole-v1")
state, info = env.reset()
n_observations = len(state)
n_actions = env.action_space.n

model = DQN(n_observations, n_actions)
model.load_state_dict(torch.load('./torch_model/policy_model_weights.pth'))
# model.load_state_dict(torch.load('./torch_model/target_model_weights.pth'))

model.eval()  # Set to evaluation mode

env = gym.make('CartPole-v1', render_mode="human")

# making the script reproducible
seed = 0
random.seed(seed)
# env.seed(seed)

print(f'Action Space: {env.action_space}\n')
print(f'Observation Space: {env.observation_space}')

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# run 10 episodes in a row
episodes = 10
for i in range(episodes):
    state = env.reset()[0]
    reward_sum = 0

    while True:
        env.render()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action = select_action(state, model)
        action = action.item()
        state, reward, done, trunc, info = env.step(action)
        reward_sum += reward
        if done:
            print(f'Episode {i} reward: {reward_sum}')
            break

env.close()
