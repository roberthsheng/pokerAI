import numpy as np
import copy
import logging
from pettingzoo.classic import texas_holdem_v4
from nn import NeuralNetworkAgent  # Assuming this is your custom module

# Set up logging
logging.basicConfig(level=logging.INFO)

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
    cfr_agents = {agent: CounterfactualRegretAgent(env.action_space(agent).n) for agent in env.possible_agents}
    nn_agents = {agent: NeuralNetworkAgent(env.action_space(agent).n) for agent in env.possible_agents}

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
                action_cfr = cfr_agents[current_player].choose_action(mask)
                action_nn = nn_agents[current_player].choose_action(observation, mask)
                env.step(action_cfr)

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
                                next_action_cfr = cfr_agents[next_player].choose_action(next_mask)
                                next_action_nn = nn_agents[next_player].choose_action(next_observation, next_mask)
                                counterfactual_env.step(next_action_cfr)

                    target = np.max(counterfactual_values)
                    nn_agents[current_player].train(current_observation, action_nn, target)


    return cfr_agents, nn_agents

def evaluate_agent(cfr_agents, nn_agents, num_games=10):
    total_rewards_cfr = {agent: 0 for agent in cfr_agents.keys()}  # Initialize total rewards for CFR agents
    total_rewards_nn = {agent: 0 for agent in nn_agents.keys()}  # Initialize total rewards for NN agents

    for _ in range(num_games):
        env = texas_holdem_v4.env(render_mode=None)  # Turn off rendering for faster evaluation
        env.reset(seed=np.random.randint(10000))

        while True:
            agent = env.agent_selection
            observation, reward, termination, truncation, info = env.last()

            total_rewards_cfr[agent] += reward
            total_rewards_nn[agent] += reward

            if termination or truncation:
                action = None
            else:
                mask = observation["action_mask"]
                action = cfr_agents[agent].choose_action(mask)  # use the CFR agent's action for evaluation
            env.step(action)

            if all(env.terminations.values()):
                break  # exit loop if all are done

        env.close()

    # avg rewards
    average_rewards_cfr = {agent: total / num_games for agent, total in total_rewards_cfr.items()}
    average_rewards_nn = {agent: total / num_games for agent, total in total_rewards_nn.items()}

    return average_rewards_cfr, average_rewards_nn

num_iterations = 100
cfr_agents, nn_agents = train_agent(num_iterations)

# eval
num_games = 50
average_rewards_cfr, average_rewards_nn = evaluate_agent(cfr_agents, nn_agents, num_games)

print(f"Average rewards for CFR agents over {num_games} games:", average_rewards_cfr)
print(f"Average rewards for NN agents over {num_games} games:", average_rewards_nn)