import argparse
from typing import List
from equity_agent import EquityAgent
from human_agent import HumanAgent
from cfr_nolimit_agent import CFRAgent
from pettingzoo.classic import texas_holdem_no_limit_v6
import matplotlib.pyplot as plt
import numpy as np
from random_agent import RandomAgent

from value_agent import ValueAgent
from shove_agent import ShoveAgent
from lc_agent import LooseCannonAgent

def plot_total_payoffs(agent1_payoffs, agent2_payoffs):
    """
    This function takes the payoffs of two agents over time and plots the cumulative total payoffs over time.

    Parameters:
    - agent1_payoffs: List of payoffs for Agent 1
    - agent2_payoffs: List of payoffs for Agent 2
    """
    if len(agent1_payoffs) != len(agent2_payoffs):
        raise ValueError("The lists of payoffs must be of the same length.")
        return

      # Calculate running total payoffs
    
    # Generate time points
    time_points = list(range(1, len(agent1_payoffs) + 1))
    
    cumulative_total_payoffs = np.cumsum(agent1_payoffs)
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, agent1_payoffs, label='Agent 1 Payoffs', marker='o')
    plt.plot(time_points, agent2_payoffs, label='Agent 2 Payoffs', marker='x')
    plt.plot(time_points, cumulative_total_payoffs, label='Cumulative Total Payoffs Player 1', linestyle='--', marker='s')
    
    plt.title('Stage-wise and Cumulative Total Payoffs Over Time')
    plt.xlabel('Time')
    plt.ylabel('Payoffs')
    plt.legend()
    plt.grid(True)
    plt.show()

def play_poker_games(agent_names: List[str], number_of_games: int, render: int):
    """
    Function to simulate poker games between agents.
    
    :param agent_names: A list of agent names as strings.
    :param number_of_games: The number of games to be played.
    """
    # Implement the poker game logic here, using agent_names and other parameters.
    # This function should return a dictionary or similar structure with the evaluation metrics for each agent.

    if not render:
        env = texas_holdem_no_limit_v6.env(num_players=2)
    else: 
        env = texas_holdem_no_limit_v6.env(render_mode="human", num_players=2)

    env.reset()

    if len(agent_names) != 2:
        raise ValueError("Invalid argument: there should only be two agents")

    agents = [get_agent(agent_name, env, i) for i, agent_name in enumerate(agent_names)]
    agent_dict = dict(zip(env.agents, agents))
    payoffs_1  = [] # payoffs for the first agent
    payoffs_2 = [] # payoffs for the second agent
    won = []
        
    for i in range(number_of_games):
        env.reset()

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
                payoffs_1.append(env.unwrapped.env.get_payoffs()[0])
                payoffs_2.append(env.unwrapped.env.get_payoffs()[1])
                if payoffs_1[-1] < 0:
                    won.append(1)
                else:
                    won.append(0)
                print(f"Game over: agent 0 won {payoffs_1[-1]}")
                break
    
    return payoffs_1, payoffs_2, won
         

def get_agent(agent_name, env, player_id):
    """
    Function to get an initialized agent given a name.

    :param agent_name: A string detailing the agent to get. 
    :param env: The pettingzoo environment (wrapped, not raw)`.
    
    returns an agent that implements a step function which takes an observation, and returns an action that we can use to step the environment.
    """

    if agent_name == "cfr":
        agent = CFRAgent(env.unwrapped.env) 
        agent.load('./cfr_models/cfr_model_20240405_214610')
        print(len(agent.regrets))
        

    elif agent_name == "human": 
        agent = HumanAgent(env.action_space(env.agents[player_id]).n)

    elif agent_name == "equity":
        agent = EquityAgent(env)

    elif agent_name == "random":
        agent = RandomAgent(env, player_id)
    
    elif agent_name == "value":
        agent = ValueAgent(env, player_id)
    
    elif agent_name == "shove":
        agent = ShoveAgent(env)

    else:
        raise ValueError(f"Invalid argument: {agent_name} not a valid agent name")

    # TODO: implement a way to get your agents. Make sure they have an step function implemented that takes an observation and returns an action.  
    return agent
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play poker games between different agents and evaluate their performance.")
    parser.add_argument("agent_names", nargs='+', help="Names of the agents participating in the games.")
    parser.add_argument("--number_of_games", type=int, default=100, help="Number of games to be played.")
    parser.add_argument("--render", type=int, default=True, help="whether or not to render the games being played.")

    args = parser.parse_args()

    payoffs1, payoffs2, won = play_poker_games(
        agent_names=args.agent_names,
        number_of_games=args.number_of_games,
        render= args.render
    )

    # Print or process the results of the games
    plot_total_payoffs(payoffs1, payoffs2)
