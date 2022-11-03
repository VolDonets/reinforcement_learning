import sys
from math import ceil
from random import random, randint
from time import sleep
import gym
import collections
import os


class CatchCoinsEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    # metadata = {"render.modes": ["ascii"]}

    def __init__(self, display_width=10, display_height=10, density=.8):
        self.display_width = display_width
        self.display_height = display_height
        self.density = density
        self.display = collections.deque(maxlen=display_height)
        self.last_action = None
        self.last_reward = None
        self.total_score = 0
        self.v_position = 0
        self.game_scr = None

    def step(self, action):
        self.last_action = action
        self.v_position = min(max(self.v_position + action, 0), self.display_width - 1)
        reward = self.display[0][self.v_position]
        self.last_reward = reward
        self.total_score += reward
        self.display.append(self.line_generator())
        state = self.display, self.v_position
        done = False
        info = {}
        return state, reward, done, info

    def reset(self):
        for _ in range(self.display_height):
            self.display.append(self.line_generator())
        self.v_position = ceil(self.display_width / 2)
        state = self.display, self.v_position
        return state

    def line_generator(self):
        line = [0] * self.display_width
        if random() > (1 - self.density):
            r = random()
            if r < .6:
                v = 1
            elif r < .9:
                v = 2
            else:
                v = 3

            line[randint(0, self.display_width - 1)] = v
        return line

    def render(self, mode="human"):
        if mode == "human":
            self._render_human()
        elif mode == "ascii":
            self._render_ascii()
        else:
            raise Exception('Not Implemented')

    def _render_ascii(self):
        # the next line works in bash console only
        os.system('clear')
        outfile = sys.stdout
        area = []
        for i in range(self.display_height):
            line = self.display[self.display_height - 1 - i]
            row = []

            for j in range(len(line)):
                p = line[j]
                if p > 0:
                    row.append(str(p))
                    if i > 0 and area[-1][j] == ' ':
                        area[-1][j] = '|'
                    if i > 1 and area[-2][j] == ' ':
                        area[-2][j] = '.'
                else:
                    row.append(' ')

            area.append(row)

        pos_line = (['_'] * self.display_width)
        pos_line[self.v_position] = str(self.last_reward) if self.last_reward else 'V'
        area.append(pos_line)
        outfile.write(f'\nTotal score: {self.total_score}\n')
        outfile.write('\n'.join(' '.join(line) for line in area) + '\n')

    def _render_human(self):
        from catch_coins_screen import CatchCoinsScreen
        if not self.game_scr:
            self.game_scr = CatchCoinsScreen(
                h=self.display_height,
                w=self.display_width
            )

        if self.last_reward:
            self.game_scr.plus()
            sleep(.1)

        self.game_scr.update(
            self.display,
            self.v_position,
            self.total_score
        )
