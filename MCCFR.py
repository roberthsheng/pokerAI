import numpy as np
import collections
from datetime import datetime
import os
import pickle
from equity_calculator import calculate_equity
from rlcard.utils.utils import *
from pettingzoo.classic import texas_holdem_no_limit_v6
from human_agent import HumanAgent

class MCCFRAgent:
    def __init__(self, env, model_path='./mccfr_model', num_iterations=1000, num_traversals=10, epsilon=0.6):
        self.use_raw = False
        self.env = env
        datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.model_path = f"mccfr_models/{model_path}_{datetime_str}"
        self.policy = collections.defaultdict(list)
        self.average_policy = collections.defaultdict(np.array)
        self.regrets = collections.defaultdict(np.array)
        self.num_iterations = num_iterations
        self.num_traversals = num_traversals
        self.epsilon = epsilon

    def train(self):
        for i in range(self.num_iterations):
            for j in range(self.num_traversals):
                self.env.reset()
                probs = np.ones(self.env.num_players)
                self.traverse_tree(probs)

            self.update_policy()

    def traverse_tree(self, probs):
        if self.env.is_over():
            return self.env.get_payoffs()

        current_player = self.env.get_player_id()
        obs, legal_actions = self.get_state(current_player)

        if np.random.rand() < self.epsilon:
            action = np.random.choice(legal_actions)
        else:
            action_probs = self.action_probs(obs, legal_actions, self.policy)
            legal_action_indices = [i for i, a in enumerate(list(self.env.state_dict['legal_actions'].keys())) if a in legal_actions]
            action = np.random.choice(legal_action_indices, p=action_probs[legal_action_indices])
            action = list(self.env.state_dict['legal_actions'].keys())[action]

        self.env.step(action)
        utility = self.traverse_tree(probs)
        self.env.step_back()

        if obs not in self.regrets:
            self.regrets[obs] = np.zeros(self.env.num_actions)
        if obs not in self.average_policy:
            self.average_policy[obs] = np.zeros(self.env.num_actions)

        state_utility = utility[current_player]
        for a in legal_actions:
            if a == action:
                regret = utility[current_player] - state_utility
            else:
                regret = 0

            action_index = list(self.env.state_dict['legal_actions'].keys()).index(a)
            self.regrets[obs][action_index] += regret
            self.average_policy[obs][action_index] += probs[current_player] * (a == action)

        return utility

    def update_policy(self):
        for obs in self.regrets:
            self.policy[obs] = self.regret_matching(obs)

    def regret_matching(self, obs):
        legal_actions = list(self.env.state_dict['legal_actions'].keys())
        regret = self.regrets[obs]
        positive_regret = np.maximum(regret, 0)
        positive_regret_sum = np.sum(positive_regret)

        action_probs = np.zeros(self.env.num_actions)
        if positive_regret_sum > 0:
            action_probs[legal_actions] = positive_regret[legal_actions] / positive_regret_sum
        else:
            action_probs[legal_actions] = 1.0 / len(legal_actions)
        return action_probs

    def action_probs(self, obs, legal_actions, policy):
        if obs not in policy.keys():
            action_probs = np.array([1.0/self.env.num_actions for _ in range(self.env.num_actions)])
            self.policy[obs] = action_probs
        else:
            action_probs = policy[obs]
        action_probs = remove_illegal(action_probs, legal_actions)
        return action_probs

    def eval_step(self, state):
        probs = self.action_probs(state['obs'], list(state['legal_actions'].keys()), self.average_policy)
        action = np.random.choice(len(probs), p=probs)

        info = {}
        info['probs'] = {state['raw_legal_actions'][i]: float(probs[list(state['legal_actions'].keys())[i]]) for i in range(len(state['legal_actions']))}

        return action, info

    def step(self, observation):
        player_id = self.env.get_player_id()
        state_for_act = self.env.get_state(player_id)
        state_for_act = self.reduce_state(state_for_act)
        action, info = self.eval_step(state_for_act)
        return action

    def get_state(self, player_id):
        state = self.env.get_state(player_id)
        state = self.reduce_state(state)
        return state['obs'], list(state['legal_actions'].keys())

    def reduce_state(self, state):
        raw_state = state
        raw_obs = state['raw_obs']
        equity = calculate_equity(raw_obs['hand'], raw_obs['public_cards'])
        new_obs = {
            'pot': raw_obs['pot'] - raw_obs['pot'] % 5,
            'my_chips': raw_obs['my_chips'] - raw_obs['my_chips'] % 5,
            'equity': round(equity * 20)/20,
            'stage': raw_obs['stage'].value,
        }

        tuple_obs = tuple(new_obs.values())
        raw_state['obs'] = tuple_obs
        return raw_state

    def save(self):
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

    def load(self, optional_path=None):
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

def main():
    env = texas_holdem_no_limit_v6.env(render_mode="human", num_players=2)

    while True:
        env.reset()
        
        mccfr = MCCFRAgent(env.unwrapped.env)
        mccfr.train()

        agents = [HumanAgent(env.action_space(env.agents[0]).n), mccfr]
        agent_dict = dict(zip(env.agents, agents))
    
        for agent in env.agent_iter():
            observation, reward, done, truncation, info = env.last()
            if done:
                env.step(None)
            else:
                current_agent = agent_dict[agent]
                action = current_agent.step(observation)
                env.step(action)
    
            if done or truncation: 
                print("Game Over")
                break

if __name__ == "__main__":
    main()