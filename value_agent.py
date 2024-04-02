import numpy as np
from typing import Callable, Dict, List, NewType, cast
from pettingzoo.classic import texas_holdem_v4
import copy
import treys

import agent_refactor

'''
TODO: implement new info_set methods:
- hand() ==> gets player's hand
- board() ==> gets current board
- first_mover() ==> checks if player is the first-mover in current round of betting
- last_bet() ==> gets value of last-placed bet if exists, otherwise 0
- stack() ==> gets chip stack
'''     

#################
## IN PROGRESS ##
#################
class ValueAgent2:
    '''Agent that randomly selects form +EV actions (impl'd w/ Ethan's structure)'''
    vals = {0:'A', 1:'2', 2:'3', 3:'4', 4:'5', 5:'6', 6:'7', 7:'8', 8:'9', 9:'10', 10:'J', 11:'Q', 12:'K'}
    suits = {0:'s', 1:'h', 2:'d', 3:'c'}
    info_sets: Dict[str, InfoSet]
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.evaluator = Evaluator()

        self.hand = []
        self.board = []

    def _get_info_set(self, h: History):
        '''Get info set of current player for a given history.'''
        info_set_key = h.info_set()
        if info_set_key not in self.info_sets:
            self.info_sets[info_set_key] = h.new_info_set()
        return self.info_sets[info_set_key]

    def set_info(self, h: History): # should be history 0
        info_set = self._get_info_set(h)
        self.hand = info_set.hand()
        self.board = info_set.board()

    def get_card(self, card_index):
        suit = self.vals[card_index / 13]            
        val = self.vals[card_index % 13]
        return suit+val

    def choose_action(self, mask, h: History):
        info_set = self._get_info_set(h)

        rank = self.evaluator.evaluate(self.board, self.hand)
        percentage = 1.0 - self.evaluator.get_five_card_rank_percentage(rank)

        pot = info_set.pot() # get value of pot
        valid_actions = np.where(mask == 1)[0]

        if mask[0] == 1 and len(valid_actions) == 1:
            return 0

        if mask[1] == 1:
            # CHECK IF FIRST MOVER
            first_mover = info_set.first_mover()

            if not first_mover:
                # FIND COST TO CALL
                call_price = info_set.last_bet()
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
            stack = info_set.stack()

            ev = percentage * (pot + stack) - stack
            if ev < 0:
                valid_actions.pop(0)
        
        # randomly choose among +EV actions
        if len(valid_actions) > 0:
            return np.random.choice(valid_actions)
        else:
            return 0  # If no valid actions, return a default action (e.g., 0)

'''OLD V'''
# #################
# ## IN PROGRESS ##
# #################
# ## only makes value bets
# class ValueAgent:
#     def __init__(self, num_actions):
#         self.num_actions = num_actions
#         self.vals = {0:'A', 1:'2', 2:'3', 3:'4', 4:'5', 5:'6', 6:'7', 7:'8', 8:'9', 9:'10', 10:'J', 11:'Q', 12:'K'}
#         self.suits = {0:'s', 1:'h', 2:'d', 3:'c'}
#         self.evaluator = Evaluator()
#         self.board = []
#         self.hand = []
    
#     def set_hand(self, hand):
#         self.hand = hand

#     def set_board(self, observation):
#         obs_space = observation['observation']
#         card_idxs = np.where(obs_space == 1)[0]

#         board = []
#         for c_idx in card_idxs:
#             if c_idx not in self.hand:
#                 board.append(self.get_card(c_idx))

#     def get_card(self, card_index):
#         suit = self.vals[card_index / 13]            
#         val = self.vals[card_index % 13]
#         return suit+val

#     def choose_action(self, mask):
#         rank = self.evaluator.evaluate(self.board, self.hand)
#         percentage = 1.0 - self.evaluator.get_five_card_rank_percentage(rank)

#         pot = 0 # get value of pot
#         valid_actions = np.where(mask == 1)[0]

#         if mask[0] == 1 and len(valid_actions) == 1:
#             return 0

#         if mask[1] == 1:
#             # CHECK IF FIRST MOVER
#             first_mover = 

#             if not first_mover:
#                 # FIND COST TO CALL
#                 call_price = 
#                 ev = percentage * (pot + call_price) - call_price
#                 if ev < 0:
#                     valid_actions.pop(0)
#         if mask[2] == 1:
#             ev = percentage * (pot + pot/2) - pot/2
#             if ev < 0:
#                 valid_actions.pop(0)
#         if mask[3] == 1:
#             ev = percentage * (pot + pot) - pot
#             if ev < 0:
#                 valid_actions.pop(0)
#         if mask[4] == 1:
#             # GET CHIP STACK
#             stack = 

#             ev = percentage * (pot + stack) - stack
#             if ev < 0:
#                 valid_actions.pop(0)
        
#         # randomly choose among +EV actions
#         if len(valid_actions) > 0:
#             return np.random.choice(valid_actions)
#         else:
#             return 0  # If no valid actions, return a default action (e.g., 0)

##################################################
# import logging
# logging.basicConfig(level=logging.INFO)
# logging.info('This is a debug message')

## logging observation space to be human-readable
# obs_dict = {0:'A', 1:'2', 2:'3', 3:'4', 4:'5', 6:'7', 7:'8', 8:'9', 9:'10', 10:'J', 11:'Q', 12:'K'}
# def translate_obs(obs_space):
#     log = ''
#     obs = np.where(obs_space == 1)[0]
#     for o in obs:
#         if o < 52:
#             if o <= 12:
#                 log += 'S'
#             elif o <= 25:
#                 log += 'H'
#             elif o <= 38:
#                 log += 'D'
#             elif o <= 51:
#                 log += 'C'
#             log += obs_dict[o % 13] + '\n'
#     logging.info(log)