import numpy as np
from pettingzoo.classic import texas_holdem_v4
import copy

########
########
## randomly chooses a valid action
class LooseCannonAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.regret_sum = np.zeros(num_actions)
        self.strategy_sum = np.zeros(num_actions)
        self.strategy = np.ones(num_actions) / num_actions

    def step(self, mask):
        valid_actions = np.where(mask == 1)[0]
        if len(valid_actions) > 0:
            return np.random.choice(valid_actions)
        else:
            return 0  # If no valid actions, return a default action (e.g., 0)