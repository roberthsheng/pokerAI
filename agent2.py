import numpy as np
from pettingzoo.classic import texas_holdem_v4
import copy
import treys

import logging

logging.basicConfig(level=logging.INFO)
logging.info('This is a debug message')

## logging observation space to be human-readable
obs_dict = {0:'A', 1:'2', 2:'3', 3:'4', 4:'5', 6:'7', 7:'8', 8:'9', 9:'10', 10:'J', 11:'Q', 12:'K'}
def translate_obs(obs_space):
    log = ''
    obs = np.where(obs_space == 1)[0]
    for o in obs:
        if o < 52:
            if o <= 12:
                log += 'S'
            elif o <= 25:
                log += 'H'
            elif o <= 38:
                log += 'D'
            elif o <= 51:
                log += 'C'
            log += obs_dict[o % 13] + '\n'
    logging.info(log)


########
## IN PROGRESS
########
## only makes value bets
class ValueAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.vals = {0:'A', 1:'2', 2:'3', 3:'4', 4:'5', 5:'6', 6:'7', 7:'8', 8:'9', 9:'10', 10:'J', 11:'Q', 12:'K'}
        self.suits = {0:'s', 1:'h', 2:'d', 3:'c'}
        self.evaluator = Evaluator()
        self.board = []
        self.hand = []
    
    def set_hand(self, hand):
        self.hand = hand

    def set_board(self, observation):
        obs_space = observation['observation']
        card_idxs = np.where(obs_space == 1)[0]

        board = []
        for c_idx in card_idxs:
            if c_idx not in self.hand:
                board.append(self.get_card(c_idx))

    def get_card(self, card_index):
        suit = self.vals[card_index / 13]            
        val = self.vals[card_index % 13]
        return suit+val

    def choose_action(self, mask):
        rank = self.evaluator.evaluate(self.board, self.hand)
        percentage = 1.0 - self.evaluator.get_five_card_rank_percentage(rank)

        pot = 0 # get value of pot
        valid_actions = np.where(mask == 1)[0]

        if mask[0] == 1 and len(valid_actions) == 1:
            return 0

        if mask[1] == 1:
            # CHECK IF FIRST MOVER
            first_mover = 

            if not first_mover:
                # FIND COST TO CALL
                call_price = 
                ev = percentage * (pot + call_price) - call_price
                if ev < 0:
                    valid_actions.pop(0)
        if mask[2] == 1:
            ev = percentage * (pot + pot/2) - pot/2
            if ev < 0:
                valid_actions.pop(0)
        if mask[3] == 1:
            ev = percentage * (pot + pot) - pot
            if ev < 0:
                valid_actions.pop(0)
        if mask[4] == 1:
            # GET CHIP STACK
            stack = 

            ev = percentage * (pot + stack) - stack
            if ev < 0:
                valid_actions.pop(0)
        
        # randomly choose among +EV actions
        if len(valid_actions) > 0:
            return np.random.choice(valid_actions)
        else:
            return 0  # If no valid actions, return a default action (e.g., 0)

########
########
## always shoves all in
class ShoveAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.regret_sum = np.zeros(num_actions)
        self.strategy_sum = np.zeros(num_actions)
        self.strategy = np.ones(num_actions) / num_actions

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
## randomly chooses a valid action
class LooseCannonAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.regret_sum = np.zeros(num_actions)
        self.strategy_sum = np.zeros(num_actions)
        self.strategy = np.ones(num_actions) / num_actions

    def choose_action(self, mask):
        valid_actions = np.where(mask == 1)[0]
        if len(valid_actions) > 0:
            return np.random.choice(valid_actions)
        else:
            return 0  # If no valid actions, return a default action (e.g., 0)