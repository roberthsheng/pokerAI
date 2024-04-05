from pettingzoo.classic import texas_holdem_no_limit_v6
import time
from human_agent import HumanAgent
import numpy as np
import collections
from datetime import datetime
import os
import pickle
from equity_calculator import calculate_equity
from rlcard.utils.utils import *
from timing_utils import time_function

class CFRAgent():
    ''' Implement CFR (chance sampling) algorithm
    '''

    def __init__(self, env, model_path='./cfr_model'):
        ''' Initilize Agent

        Args:
            env (Env): Env class
        '''
        self.use_raw = False # this agent uses the raw rlcard env. This means raw_env.env (if using pettingzoo)
        # or rlcard.make("no-limit-holdem", config), need "allow_step_back": True 
        self.env = env
        # self.env.step = time_function(self.env.step)
        # self.env.step_back = time_function(self.env.step_back)
        datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.model_path = f"cfr_models/{model_path}_{datetime_str}"
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
            state_for_act = self.reduce_state(state_for_act)
            action, info = self.eval_step(state_for_act)
    
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

            # Keep traversing the child state
            self.env.step(action)
            utility = self.traverse_lazy(player_id) # this just computes utility if we follow policy exactly from here on out 
            self.env.step_back()

            state_utility += action_prob * utility
            action_utilities[action] = utility

        if not current_player == player_id:
            return state_utility

        # If it is current player, we record the policy and compute regret
        player_state_utility = state_utility[current_player]
        if obs not in self.regrets:
            self.regrets[obs] = np.zeros(self.env.num_actions)
        if obs not in self.average_policy:
            self.average_policy[obs] = np.zeros(self.env.num_actions)
        for action in legal_actions:
            action_prob = action_probs[action]
            regret = (action_utilities[action][current_player]
                    - player_state_utility)
            self.regrets[obs][action] += regret
            self.average_policy[obs][action] += self.iteration * action_prob
            
#        print(obs)
#        print(self.regrets[obs])
#        print(self.average_policy[obs])
#        print("\n\n\n\n\n")
#
    def traverse_lazy(self, player_id):
        ''' Traverse the game tree, get the outcome from following policy exactly at every step 

        Args:
            player_id: The player to get the payoff for 

        Returns:
            state_utilities (nparray): zeros for all other players, payoffs for player_id 
        '''
        if self.env.is_over():
            # print(f"got this payoff {self.env.get_payoffs()}")
            return self.env.get_payoffs()
        num_steps = 0
        while True:
            if self.env.is_over():
                utility = self.env.get_payoffs()
                break
            
            current_player = self.env.get_player_id()
            state_for_act = self.env.get_state(player_id)
            state_for_act = self.reduce_state(state_for_act)
            action, info = self.eval_step(state_for_act)
            num_steps += 1
            self.env.step(action)

        for i in range(num_steps):
            self.env.step_back()

        return utility


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
        probs = self.action_probs(state['obs'], list(state['legal_actions'].keys()), self.average_policy)
        # print(probs)
        action = np.random.choice(len(probs), p=probs)

        info = {}
        info['probs'] = {state['raw_legal_actions'][i]: float(probs[list(state['legal_actions'].keys())[i]]) for i in range(len(state['legal_actions']))}

        return action, info

    def step(self, observation):# obs unneeded but to match, gets state from env
        player_id = self.env.get_player_id()
        state_for_act = self.env.get_state(player_id)
        state_for_act = self.reduce_state(state_for_act)
        action, info = self.eval_step(state_for_act)
        
        state_str, _ = self.get_state(player_id)
        if state_str in self.regrets:
            print('encountered')
            print(self.regrets[state_str])
        else:
            print('havent encountered')
        print(f"cfr_agent chose: {action}")
        return action
 
        
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
        return state['obs'], list(state['legal_actions'].keys())

    def reduce_state(self, state):
        '''Put the state returned by env in one of a few predefined groups

        Args:
            state (dict): The state returned by env.get_state(player_id)
        
        Returns:
            reduced_state (dict): An identical dictionary containing fewer, lower-dim observations under 'obs'
        '''
        # maybe learn state groupings
        # this may not be an efficient way to encode state-- doing it the way rlcard did
        raw_state = state
        raw_obs = state['raw_obs']
        equity = calculate_equity(raw_obs['hand'], raw_obs['public_cards'])
        new_obs = { 
            'pot': raw_obs['pot'] - raw_obs['pot'] % 10,
            'my_chips': raw_obs['my_chips'] - raw_obs['my_chips'] % 10,
            'equity': round(equity * 20)/20,
            'stage': raw_obs['stage'],
        }    

        tuple_obs = tuple(new_obs.values())
        raw_state['obs'] = tuple_obs
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

    def load(self, optional_path=None):
        ''' Load model
        '''
        if optional_path:
            self.model_path = optional_path

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


# if running main, play a game with human vs equity agent
def main():
    # Initialize the game environment
    env = texas_holdem_no_limit_v6.env(render_mode="human", num_players=2)

    while True:
        env.reset()
        
        # Initialize agents
        cfr = CFRAgent(env.unwrapped.env) 
        cfr.load('./cfr_models/cfr_model_20240404_171341')
        print(len(cfr.regrets))
        agents = [HumanAgent(env.action_space(env.agents[0]).n), CFRAgent(env.unwrapped.env)]
        agent_dict = dict(zip(env.agents, agents))
    
        # Main game loop
        for agent in env.agent_iter():
            observation, reward, done, truncation, info = env.last()
            if done:
                env.step(None)
            else:
                # Fetch the corresponding agent (human or AI)
                current_agent = agent_dict[agent]
                
                # Perform an action
                action = current_agent.step(observation)
                env.step(action)
    
            if done or truncation: 
                print("Game Over")
                break
    
if __name__ == "__main__":
    main()

