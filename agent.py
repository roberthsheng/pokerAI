import numpy as np
from pettingzoo.classic import texas_holdem_v4
import copy
import logging

logging.basicConfig(level=logging.INFO)
logging.info('This is a debug message')


class CounterfactualRegretAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.regret_sum = np.zeros(num_actions)
        self.strategy_sum = np.zeros(num_actions)
        self.strategy = np.ones(num_actions) / num_actions

    def update_regret(self, action, regret):
        self.regret_sum[action] += regret[action]

    def update_strategy(self):
        self.strategy_sum += self.strategy
        self.strategy = self.get_strategy()

    def get_strategy(self):
        regrets = np.maximum(self.regret_sum, 0)
        normalizing_sum = np.sum(regrets)
        if normalizing_sum > 0:
            return regrets / normalizing_sum
        else:
            return np.ones(self.num_actions) / self.num_actions

    def get_average_strategy(self):
        normalizing_sum = np.sum(self.strategy_sum)
        if normalizing_sum > 0:
            return self.strategy_sum / normalizing_sum
        else:
            return np.ones(self.num_actions) / self.num_actions

    def choose_action(self, mask):
        masked_strategy = self.strategy * mask
        normalizing_sum = np.sum(masked_strategy)
        if normalizing_sum > 0:
            masked_strategy /= normalizing_sum
        else:
            # If the mask is all zeros, choose a random valid action
            valid_actions = np.where(mask == 1)[0]
            if len(valid_actions) > 0:
                return np.random.choice(valid_actions)
            else:
                return 0  # If no valid actions, return a default action (e.g., 0)
        return np.random.choice(self.num_actions, p=masked_strategy)

def train_agent(num_iterations):
    env = texas_holdem_v4.env()
    agents = {agent: CounterfactualRegretAgent(env.action_space(agent).n) for agent in env.possible_agents}

    for _ in range(num_iterations):
        env.reset()
        terminal = False
        while not terminal:
            current_player = env.agent_selection
            observation, reward, termination, truncation, info = env.last()
            mask = observation["action_mask"]

            if termination or truncation:
                terminal = True
                action = None
            else:
                action = agents[current_player].choose_action(mask)
                env.step(action)

            if not terminal:
                next_player = env.agent_selection
                if next_player != current_player:
                    # Save the current observation and action mask
                    current_observation = observation
                    current_mask = mask
                    
                    next_observation, _, _, _, _ = env.last()
                    next_mask = next_observation["action_mask"]
                    counterfactual_values = np.zeros(env.action_space(current_player).n)

                    for a in range(env.action_space(current_player).n):
                        if current_mask[a]:
                            # Create a new environment instance with the same configuration
                            counterfactual_env = texas_holdem_v4.env()
                            counterfactual_env.reset()                            
                            # Step the counterfactual environment with the current action
                            counterfactual_env.step(a)
                            _, next_reward, next_termination, next_truncation, _ = counterfactual_env.last()
                            counterfactual_values[a] = next_reward

                            if not (next_termination or next_truncation):
                                next_action = agents[next_player].choose_action(next_mask)
                                counterfactual_env.step(next_action)

                    regret = counterfactual_values - counterfactual_values[action]
                    agents[current_player].update_regret(action, regret)  # Pass the regret array
                    agents[current_player].update_strategy()

    return agents


## logging observation space to be human-readable
obs_dict = {0:'A', 1:'2', 2:'3', 3:'4', 4:'5', 6:'7', 7:'8', 8:'9', 9:'10', 10:'J', 11:'Q', 12:'K'}
def translate_obs(obs_space):
    log = ''
    obs = np.where(obs_space == 1)[0]
    for o in obs:
        if o < 52:
            if o <= 12:
                log += 'Spades'
            elif o <= 25:
                log += 'Hearts'
            elif o <= 38:
                log += 'Diamonds'
            elif o <= 51:
                log += 'Clubs'
            log += ' ' + obs_dict[o % 13] + '\n'
    logging.info(log)

num_iterations = 1000
trained_agents = train_agent(num_iterations)

env = texas_holdem_v4.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    logging.info('NEW STATE')
    observation, reward, termination, truncation, info = env.last()
    

    obs_space = observation['observation']
    translate_obs(obs_space)

    act_space = observation['action_mask']

    if termination or truncation:
        action = None
    else:
        mask = observation["action_mask"]
        action = trained_agents[agent].choose_action(mask)
    env.step(action)

env.close()