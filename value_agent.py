import numpy as np
from typing import Callable, Dict, List, NewType, cast
from pettingzoo.classic import texas_holdem_v4
import copy
import treys # pip install

from equity_agent import calculate_equity

'''
TODO: implement new info_set methods:
- first_mover() ==> checks if player is the first-mover in current round of betting
- last_bet() ==> gets value of last-placed bet if exists, otherwise 0
- stack() ==> gets chip stack
'''     

#################
## IN PROGRESS ##
#################
class ValueAgent:
    '''Agent that randomly selects form +EV actions'''
    def __init__(self, env):
        self.env = env

    def get_state(self, observation):
        # need to get hand/community card data which seems inaccessible via vanilla observations
        unwrapped_env = self.env.unwrapped
        raw_data = unwrapped_env.env.game.get_state(unwrapped_env.env.get_player_id())
        hole_cards = raw_data['hand'] 
        community_cards = raw_data['public_cards'] 
        pot = raw_data['pot']
        current_bet = max(raw_data['all_chips'])
        amount_to_play = current_bet - raw_data['my_chips']
        state = {
            'raw_obs': observation,
            'hole_cards': hole_cards,
            'community_cards': community_cards, 
            'pot': pot,
            'amount_to_play': amount_to_play
        }
        return state 

    def choose_action(self, observation):
        state = self.get_state(observation)
        print(state)

        mask = observation['action_mask']
        valid_actions = np.where(mask == 1)[0]

        ## remove valid actions based on whether they are +EV
        # fold
        if len(valid_actions) == 1 and mask[0] == 1:
            return 0

        # check/call
        if mask[1] == 1:
            # call
            if state['amount_to_play'] > 0:
                hand_equity = calculate_equity(state['hole_cards'], state['community_cards'])
                pot_odds = state['amount_to_play'] / (state['pot'] + state['amount_to_play']) 
                if hand_equity <= pot_odds:
                    valid_actions.pop(0)
        
        # raise half pot
        if mask[2] == 1:
            hand_equity = calculate_equity(state['hole_cards'], state['community_cards'])
            pot_odds = 0
            if state['amount_to_play'] == 0:
                pot_odds = (state['pot']/2) / (state['pot'] + state['pot']/2)
            else:
                pay = state['amount_to_play'] + (state['amount_to_play'] + state['pot'])/2 # price to call + current pot halved
                win = state['pot'] + pay
                pot_odds = pay / win
                
            if hand_equity <= pot_odds:
                valid_actions.pop(0)
        
        # raise pot
        if mask[3] == 1:
            hand_equity = calculate_equity(state['hole_cards'], state['community_cards'])
            pot_odds = 0
            if state['amount_to_play'] == 0:
                pot_odds = (state['pot']) / (state['pot']*2)
            else:
                pay = state['amount_to_play'] + (state['amount_to_play'] + state['pot']) # price to call + current pot
                win = state['pot'] + pay
                pot_odds = pay / win
                
            if hand_equity <= pot_odds:
                valid_actions.pop(0)
        
        # all in
        if mask[4] == 1:
            hand_equity = calculate_equity(state['hole_cards'], state['community_cards'])
            pot_odds = 0
            if state['amount_to_play'] == 0 or state['amount_to_play'] >= state['stack']:
                pot_odds = (state['stack']) / (state['pot'] + state['stack'])
            else:
                pay = state['stack'] # price to call + current pot
                win = state['pot'] + pay
                pot_odds = pay / state['stack']
            if hand_equity <= pot_odds:
                valid_actions.pop(0)
        
        # randomly choose among +EV actions
        if len(valid_actions) > 0:
            return np.random.choice(valid_actions)
        else:
            return 0  # If no valid actions, return a default action (e.g., 0)

# #################
# ## IN PROGRESS ##
# #################
# class ValueAgent2:
#     '''Agent that randomly selects form +EV actions (impl'd w/ Ethan's structure)'''
#     vals = {0:'A', 1:'2', 2:'3', 3:'4', 4:'5', 5:'6', 6:'7', 7:'8', 8:'9', 9:'10', 10:'J', 11:'Q', 12:'K'}
#     suits = {0:'s', 1:'h', 2:'d', 3:'c'}
#     info_sets: Dict[str, InfoSet]
#     def __init__(self, num_actions):
#         self.num_actions = num_actions
#         self.evaluator = Evaluator()

#         self.hand = []
#         self.board = []

#     def _get_info_set(self, h: History):
#         '''Get info set of current player for a given history.'''
#         info_set_key = h.info_set()
#         if info_set_key not in self.info_sets:
#             self.info_sets[info_set_key] = h.new_info_set()
#         return self.info_sets[info_set_key]

#     def set_info(self, h: History): # should be history 0
#         info_set = self._get_info_set(h)
#         self.hand = info_set.hand()
#         self.board = info_set.board()

#     def get_card(self, card_index):
#         suit = self.vals[card_index / 13]            
#         val = self.vals[card_index % 13]
#         return suit+val

#     def choose_action(self, mask, h: History):
#         info_set = self._get_info_set(h)

#         rank = self.evaluator.evaluate(self.board, self.hand)
#         percentage = 1.0 - self.evaluator.get_five_card_rank_percentage(rank)

#         pot = info_set.pot() # get value of pot
#         valid_actions = np.where(mask == 1)[0]

#         if mask[0] == 1 and len(valid_actions) == 1:
#             return 0

#         if mask[1] == 1:
#             # CHECK IF FIRST MOVER
#             first_mover = info_set.first_mover()

#             if not first_mover:
#                 # FIND COST TO CALL
#                 call_price = info_set.last_bet()
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
#             stack = info_set.stack()

#             ev = percentage * (pot + stack) - stack
#             if ev < 0:
#                 valid_actions.pop(0)
        
#         # randomly choose among +EV actions
#         if len(valid_actions) > 0:
#             return np.random.choice(valid_actions)
#         else:
#             return 0  # If no valid actions, return a default action (e.g., 0)