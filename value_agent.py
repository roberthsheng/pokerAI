import numpy as np
from typing import Callable, Dict, List, NewType, cast
from pettingzoo.classic import texas_holdem_v4
import copy

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
    def __init__(self, env, agent_idx=1):
        self.env = env
        self.agent_idx = agent_idx

    def get_state(self, observation):
        # need to get hand/community card data which seems inaccessible via vanilla observations
        unwrapped_env = self.env.unwrapped
        raw_data = unwrapped_env.env.game.get_state(unwrapped_env.env.get_player_id())
        hole_cards = raw_data['hand'] 
        community_cards = raw_data['public_cards'] 
        pot = raw_data['pot']
        current_bet = max(raw_data['all_chips'])
        amount_to_play = current_bet - raw_data['my_chips']
        stack = raw_data['stakes']
        state = {
            'raw_obs': observation,
            'hole_cards': hole_cards,
            'community_cards': community_cards, 
            'pot': pot,
            'amount_to_play': amount_to_play,
            'stack': stack
        }
        return state 

    def step(self, observation):
        state = self.get_state(observation)
        print(state)

        mask = observation['action_mask']
        valid_actions = np.where(mask == 1)[0]

        stack = state['stack'][self.agent_idx]

        ## remove valid actions based on whether they are +EV
        # fold
        if len(valid_actions) == 1 and mask[0] == 1:
            return 0

        options = []

        # check/call
        if mask[1] == 1:
            # check/call
            hand_equity = calculate_equity(state['hole_cards'], state['community_cards'])
            pot_odds = min(state['amount_to_play'], stack) / (state['pot'] + min(state['amount_to_play'], stack))

            if hand_equity >= pot_odds:
                options.append(1)

        # raise half pot
        if mask[2] == 1:
            hand_equity = calculate_equity(state['hole_cards'], state['community_cards'])
            pay = state['amount_to_play'] + (state['amount_to_play'] + state['pot'])/2 # price to call + current pot halved
            win = state['pot'] + pay + (state['amount_to_play'] + state['pot'])/2 # to win pot + paid + opponent calling half pot
            pot_odds = pay / win

            if hand_equity >= pot_odds:
                options.append(2)
        
        # raise pot
        if mask[3] == 1:
            hand_equity = calculate_equity(state['hole_cards'], state['community_cards'])
            pay = state['amount_to_play'] + (state['amount_to_play'] + state['pot']) # price to call + current pot
            win = state['pot'] + pay + (state['amount_to_play'] + state['pot']) # to win pot + paid + opponent calling pot
            pot_odds = pay / win

            if hand_equity >= pot_odds:
                options.append(3)
        
        # all in
        if mask[4] == 1 and stack > state['amount_to_play'] + state['pot']:
            hand_equity = calculate_equity(state['hole_cards'], state['community_cards'])
            pot_odds = stack / (state['pot'] + stack + (stack - state['amount_to_play'])) # to win pot + paid + opponent calling stack

            if hand_equity < pot_odds:
                options.append(4)
        
        print(options)
        # randomly choose among +EV actions
        if len(options) > 0:
            return np.random.choice(options)
        else:
            return 0  # If no valid actions, return a default action (e.g., 0)