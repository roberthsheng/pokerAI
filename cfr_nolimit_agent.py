import numpy as np
import collections

import os
import pickle
from equity_calculator import calculate_equity
from rlcard.utils.utils import *

class CFRAgent():
    ''' Implement CFR (chance sampling) algorithm
    '''

    def __init__(self, env, model_path='./cfr_model'):
        ''' Initilize Agent

        Args:
            env (Env): Env class
        '''
        self.use_raw = False
        self.env = env
        self.model_path = model_path

        # A policy is a dict state_str -> action probabilities
        self.policy = collections.defaultdict(list)
        self.average_policy = collections.defaultdict(np.array)

        # Regret is a dict state_str -> action regrets
        self.regrets = collections.defaultdict(np.array)

        self.iteration = 0

    def train(self):
        ''' Do one iteration of CFR
        '''
        self.iteration += 1
        # Firstly, traverse tree to compute counterfactual regret for each player
        # The regrets are recorded in traversal
        for player_id in range(self.env.num_players):
            self.env.reset()
            self.go_to_random_state(player_id) # generate a trajectory, then roll back to a random state
            probs = np.ones(self.env.num_players)
            self.traverse_tree(probs, player_id) # find the regret for the action we did at that state-- execute all possible actions, then follow policy exactly and get rewards

        # Update policy
        self.update_policy()

    def go_to_random_state(self, player_id):
            trajectory_len = self.generate_trajectory()
            if (trajectory_len == 0) or (trajectory_len == 1 and player_id == self.env.get_player_id()):
            # may need fix, if there's an empty trajectory-- e.g your opponent played first and folded immediately, then just do CFR on the first step 
                self.env.reset()
                return

            if self.env.get_player_id() != player_id: # I made the last move
                numbers = np.arange(1, trajectory_len+1)
                odd_numbers = numbers[numbers % 2 == 1]
                random_number = np.random.choice(odd_numbers)
                
            else: # need even number of step backs
                numbers = np.arange(1, trajectory_len+1)
                even_numbers = numbers[numbers % 2 == 0]
                random_number = np.random.choice(even_numbers)
       
            for i in range(random_number): # step back to the random state we chose
                self.env.step_back()
            
    def generate_trajectory(self):
        '''
            Bring the environment to a terminal state by playing according to policy
        '''
        counter = 0 
        while True: 
            if (self.env.is_over()):
                break
            player_id = self.env.get_player_id()
            state_for_act = self.env.get_state(player_id)
            action, info = self.eval_step(state_for_act)
            # print(f"Player {player_id} played {action}")
    
            # Keep traversing the child state
            self.env.step(action)
            counter += 1

        return counter 

    def traverse_tree(self, probs, player_id):
        ''' Traverse the game tree, update the regrets

        Args:
            probs: The reach probability of the current node
            player_id: The player to update the value

        Returns:
            state_utilities (list): The expected utilities for all the players
        '''
        if self.env.is_over():
            return self.env.get_payoffs()

        current_player = self.env.get_player_id()

        action_utilities = {} # want to compute how much better this current action may be
        state_utility = np.zeros(self.env.num_players)
        obs, legal_actions = self.get_state(current_player)
        action_probs = self.action_probs(obs, legal_actions, self.policy)

        for action in legal_actions: # try all possible actions "counterfactually"
            action_prob = action_probs[action]
            new_probs = probs.copy()
            new_probs[current_player] *= action_prob

            # Keep traversing the child state
            self.env.step(action)
            # print(action)
            utility = self.traverse_lazy(player_id) # this just computes utility if we follow policy exactly from here on out 
            self.env.step_back()

            state_utility += action_prob * utility
            action_utilities[action] = utility

        if not current_player == player_id:
            return state_utility

        # If it is current player, we record the policy and compute regret
        player_prob = probs[current_player]
        counterfactual_prob = (np.prod(probs[:current_player]) *
                                np.prod(probs[current_player + 1:]))
        player_state_utility = state_utility[current_player]

        if obs not in self.regrets:
            self.regrets[obs] = np.zeros(self.env.num_actions)
        if obs not in self.average_policy:
            self.average_policy[obs] = np.zeros(self.env.num_actions)
        for action in legal_actions:
            action_prob = action_probs[action]
            regret = counterfactual_prob * (action_utilities[action][current_player]
                    - player_state_utility)
            self.regrets[obs][action] += regret
            self.average_policy[obs][action] += self.iteration * player_prob * action_prob
        return state_utility

    def traverse_lazy(self, player_id):
        ''' Traverse the game tree, get the outcome from following policy exactly at every step 

        Args:
            player_id: The player to get the payoff for 

        Returns:
            state_utilities (nparray): zeros for all other players, payoffs for player_id 
        '''
        if self.env.is_over():
            return self.env.get_payoffs()

        current_player = self.env.get_player_id()

        action_utilities = {} # want to compute how much better this current action may be
        state_utility = np.zeros(self.env.num_players)
        obs, legal_actions = self.get_state(current_player)

        state_for_act = self.env.get_state(player_id)
        action, info = self.eval_step(state_for_act)

        # Keep traversing the child state
        self.env.step(action)
        utility = self.traverse_lazy(player_id) 
        self.env.step_back()

        state_utility +=  utility
        action_utilities[action] = utility

        if not current_player == player_id:
            return state_utility

        return state_utility



    def update_policy(self):
        ''' Update policy based on the current regrets
        '''
        for obs in self.regrets:
            self.policy[obs] = self.regret_matching(obs)

    def regret_matching(self, obs):
        ''' Apply regret matching

        Args:
            obs (string): The state_str
        '''
        regret = self.regrets[obs]
        positive_regret_sum = sum([r for r in regret if r > 0])

        action_probs = np.zeros(self.env.num_actions)
        if positive_regret_sum > 0:
            for action in range(self.env.num_actions):
                action_probs[action] = max(0.0, regret[action] / positive_regret_sum)
        else:
            for action in range(self.env.num_actions):
                action_probs[action] = 1.0 / self.env.num_actions
        return action_probs

    def action_probs(self, obs, legal_actions, policy):
        ''' Obtain the action probabilities of the current state

        Args:
            obs (str): state_str
            legal_actions (list): List of leagel actions
            player_id (int): The current player
            policy (dict): The used policy

        Returns:
            (tuple) that contains:
                action_probs(numpy.array): The action probabilities
                legal_actions (list): Indices of legal actions
        '''
        if obs not in policy.keys():
            action_probs = np.array([1.0/self.env.num_actions for _ in range(self.env.num_actions)])
            self.policy[obs] = action_probs
        else:
            action_probs = policy[obs]
        action_probs = remove_illegal(action_probs, legal_actions)
        return action_probs

    def eval_step(self, state):
        ''' Given a state, predict action based on average policy

        Args:
            state (numpy.array): State representation

        Returns:
            action (int): Predicted action
            info (dict): A dictionary containing information
        '''
        probs = self.action_probs(state['obs'].tostring(), list(state['legal_actions'].keys()), self.average_policy)
        action = np.random.choice(len(probs), p=probs)

        info = {}
        info['probs'] = {state['raw_legal_actions'][i]: float(probs[list(state['legal_actions'].keys())[i]]) for i in range(len(state['legal_actions']))}

        return action, info

    def get_state(self, player_id):
        ''' Get state_str of the player

        Args:
            player_id (int): The player id

        Returns:
            (tuple) that contains:
                state (str): The state str
                legal_actions (list): Indices of legal actions
        '''
        state = self.env.get_state(player_id)
        state = self.reduce_state(state)
        return state['obs'].tostring(), list(state['legal_actions'].keys())

    def reduce_state(self, state):
        '''Put the state returned by env in one of a few predefined groups

        Args:
            state (dict): The state returned by env.get_state(player_id)
        
        Returns:
            reduced_state (dict): A dictionary containing fewer, lower-dim observations
        '''
        # maybe learn state groupings
        # this may not be an efficient way to encode state-- doing it the way rlcard did
        raw_state = state
        raw_obs = state['raw_obs']
        equity = calculate_equity(raw_obs['hand'], raw_obs['public_cards'])
        new_obs = { # don't need to make it into a dict before putting it in an array but wanted to be clear
            'pot': raw_obs['pot'],
            'my_chips': raw_obs['my_chips'],
            'equity': equity - (equity % 5),
            'stage': raw_obs['stage'],
        }    
        list_obs = np.array(list(new_obs.values()))
        raw_state['obs'] = list_obs
        return raw_state

    def save(self):
        ''' Save model
        '''
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        policy_file = open(os.path.join(self.model_path, 'policy.pkl'),'wb')
        pickle.dump(self.policy, policy_file)
        policy_file.close()

        average_policy_file = open(os.path.join(self.model_path, 'average_policy.pkl'),'wb')
        pickle.dump(self.average_policy, average_policy_file)
        average_policy_file.close()

        regrets_file = open(os.path.join(self.model_path, 'regrets.pkl'),'wb')
        pickle.dump(self.regrets, regrets_file)
        regrets_file.close()

        iteration_file = open(os.path.join(self.model_path, 'iteration.pkl'),'wb')
        pickle.dump(self.iteration, iteration_file)
        iteration_file.close()

    def load(self):
        ''' Load model
        '''
        if not os.path.exists(self.model_path):
            return

        policy_file = open(os.path.join(self.model_path, 'policy.pkl'),'rb')
        self.policy = pickle.load(policy_file)
        policy_file.close()

        average_policy_file = open(os.path.join(self.model_path, 'average_policy.pkl'),'rb')
        self.average_policy = pickle.load(average_policy_file)
        average_policy_file.close()

        regrets_file = open(os.path.join(self.model_path, 'regrets.pkl'),'rb')
        self.regrets = pickle.load(regrets_file)
        regrets_file.close()

        iteration_file = open(os.path.join(self.model_path, 'iteration.pkl'),'rb')
        self.iteration = pickle.load(iteration_file)
        iteration_file.close()
