import collections
from equity_calculator import calculate_equity
import numpy as np
from pettingzoo.classic import texas_holdem_no_limit_v6
from human_agent import HumanAgent
import os
import pickle

class QLearningAgent():
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.2, model_path='./qlearning_model'):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = collections.defaultdict(lambda: np.zeros(env.num_actions))
        self.model_path = model_path

    def choose_action(self, reduced_state):
        legal_actions = reduced_state['legal_actions']
        # convert observation array to tuple for use as Q-table key
        state_tuple = tuple(reduced_state['observation'].flatten())
        if np.random.rand() < self.epsilon:
            # Choosing randomly
            return np.random.choice(legal_actions)
        else:
            # exploit
            q_values = {action: self.q_table[state_tuple][action] for action in legal_actions if self.q_table[state_tuple][action] != 0}
            if not q_values:
                return np.random.choice(legal_actions)
            return max(q_values, key=q_values.get)
        
    def learn(self, state, action, reward, next_state, done):
        state_tuple = tuple(state.flatten())
        next_state_tuple = tuple(next_state.flatten()) if next_state is not None else None

        predict = self.q_table[state_tuple][action]
        target = reward if done or next_state is None else reward + self.gamma * np.max(self.q_table[next_state_tuple])
        self.q_table[state_tuple][action] += self.alpha * (target - predict)


    def step(self, observation):
        reduced_state = self.reduce_state(observation)
        action = self.choose_action(reduced_state)
        return action


    def reduce_state(self, observation):
        reduced_state = {
            'observation': observation['observation'],
            'legal_actions': np.where(observation['action_mask'] == 1)[0]
        }
        return reduced_state
    
    def save(self):
        ''' Save the Q-table and parameters of the model '''
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        with open(os.path.join(self.model_path, 'q_table.pkl'), 'wb') as f:
            pickle.dump(dict(self.q_table), f)

        params = {
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon
        }
        with open(os.path.join(self.model_path, 'params.pkl'), 'wb') as f:
            pickle.dump(params, f)

    def load(self):
        ''' Load the Q-table and parameters of the model '''
        q_table_path = os.path.join(self.model_path, 'q_table.pkl')
        if os.path.exists(q_table_path):
            with open(q_table_path, 'rb') as f:
                self.q_table = collections.defaultdict(lambda: np.zeros(self.env.num_actions), pickle.load(f))

        params_path = os.path.join(self.model_path, 'params.pkl')
        if os.path.exists(params_path):
            with open(params_path, 'rb') as f:
                params = pickle.load(f)
                self.alpha = params.get('alpha', self.alpha)
                self.gamma = params.get('gamma', self.gamma)
                self.epsilon = params.get('epsilon', self.epsilon)



def main():
    env = texas_holdem_no_limit_v6.env(render_mode="human", num_players=2)

    q_agent = QLearningAgent(env.unwrapped.env)

    while True:
        env.reset()
        human_agent = HumanAgent(env.action_space(env.agents[0]).n)
        agents = [human_agent, q_agent]
        agent_dict = dict(zip(env.agents, agents))

        prev_state = None
        prev_action = None

        for agent in env.agent_iter():
            observation, reward, done, truncation, info = env.last()
            print(observation)

            if done:
                env.step(None)
                # learn from prev action
                if prev_state is not None and current_agent == q_agent:
                    q_agent.learn(prev_state['obs'], prev_action, reward, None, done)
                prev_state = None
            else:
                current_agent = agent_dict[agent]

                if current_agent == q_agent:
                    current_state = q_agent.reduce_state(observation)
                    action = current_agent.step(observation)

                    if prev_state is not None:
                        q_agent.learn(prev_state['obs'], prev_action, reward, current_state['obs'], done)

                    prev_state = current_state
                    prev_action = action
                else:
                    action = current_agent.step(observation)

                env.step(action)

            if done or truncation:
                print("Game Over")
                break


if __name__ == "__main__":
    main()