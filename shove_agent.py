import numpy as np
from typing import Callable, Dict, List, NewType, cast
from pettingzoo.classic import texas_holdem_v4
import copy

import agent_refactor

########
########
class ShoveAgent1:
    ''' Agent that always shoves all-in
    '''
    
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def choose_action(self, mask):
        if mask[4] == 1:
            return mask[4]
        
        valid_actions = np.where(mask == 1)[0]
        if len(valid_actions) > 2:
            return np.random.choice(valid_actions)
        else:
            return 0  # If no valid actions, return a default action (e.g., 0)
        
########
########
class ShoveAgent2:
    ''' Agent that always shoves all-in (implemented w/ Ethan stuff)
    '''
    info_sets: Dict[str, InfoSet]
    def __init__(self, *, create_new_history: Callable[[], History], epochs: int, n_players: int = 2):
        self.n_players = n_players
        self.epochs = epochs
        self.create_new_history = create_new_history
        self.info_sets = {}
        self.tracker = InfoSetTracker()

    def choose_action(self, mask):
        if mask[4] == 1:
            return mask[4]
        
        valid_actions = np.where(mask == 1)[0]
        if len(valid_actions) > 2:
            return np.random.choice(valid_actions)
        else:
            return 0  # If no valid actions, return a default action (e.g., 0)