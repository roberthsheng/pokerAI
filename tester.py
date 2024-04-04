import numpy as np
import matplotlib.pyplot as plt
import copy
import logging
from pettingzoo.classic import texas_holdem_v4

from agent import CounterfactualRegretAgent

class Tester:
    def __init__(self, cfr = CounterfactualRegretAgent(),  agent2 = CounterfactualRegretAgent, name2='cfr') -> None:
        self.agents = [cfr, agent2]
        self.names = ['cfr', name2]
    
    def change_agent(self, agent2, name2):
        self.agents[1] = agent2
        self.names[1] = name2

    def plot_stack(self, num_games, stacks):
        for stack in stacks:
            cfr_stack = stack['cfr']
            games = range(cfr_stack)
            plt.plot(games, cfr_stack)

            plt.title('CFR stack over', num_games, 'games vs.', str(self.names[1]))
        return plt

    
    def play(self, num_games, init_stack):
        stacks = []
        for _ in range(num_games):
            env = texas_holdem_v4.env(render_mode=None)  # Turn off rendering for faster evaluation
            env.reset(seed=np.random.randint(10000))

            stack = {'cfr': [init_stack], self.agent_name: [init_stack]}

            while True:
                agent = env.agent_selection
                observation, reward, termination, truncation, info = env.last()

                if termination or truncation:
                    stack[self.agents[0]].append(stack[self.agents[0]][-1] + reward)
                    stack[self.names[0]].append(stack[self.names[1]][-1] + reward)
                    action = None
                else:
                    mask = observation["action_mask"]
                    action = self.agents[agent].choose_action(mask)  # use the CFR agent's action for evaluation
                env.step(action)

                if all(env.terminations.values()):
                    break  # exit loop if all are done
            env.close()

            stacks.append[stack]
        return stacks


## tester
cfr1, cfr2 = CounterfactualRegretAgent(), CounterfactualRegretAgent()
t = Tester(cfr1, cfr2)
stacks = t.play(1000, 100)
t.plot_stack(stacks)
