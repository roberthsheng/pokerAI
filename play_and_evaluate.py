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
from equity_calculator import calculate_winpercent
from qlearning_agent import QLearningAgent
import os

def plot_evaluation_results(agent1_payoffs, agent2_payoffs, oracle_win_percentages, agent1_name, agent2_name, pot_sizes):
    """
    This function plots the cumulative total payoffs over time and the oracle win percentage of agent 1.

    Parameters:
    - agent1_payoffs: List of payoffs for Agent 1
    - agent2_payoffs: List of payoffs for Agent 2
    - oracle_win_percentages: List of oracle win percentages for Agent 1
    - agent1_name: Name of Agent 1
    - agent2_name: Name of Agent 2
    """
    if len(agent1_payoffs) != len(agent2_payoffs):
        raise ValueError("The lists of payoffs must be of the same length.")
        return

    # Calculate cumulative payoffs
    cumulative_payoffs_1 = np.cumsum(agent1_payoffs)
    cumulative_payoffs_2 = np.cumsum(agent2_payoffs)

    # Time points for x-axis
    time_points = np.arange(1, len(agent1_payoffs) + 1)

    plt.figure(figsize=(14, 10))

    # Plot for cumulative payoffs
    plt.subplot(2, 2, 1)
    plt.plot(time_points, cumulative_payoffs_1, label=f"Cumulative {agent1_name}", marker='o')
    plt.plot(time_points, cumulative_payoffs_2, label=f"Cumulative {agent2_name}", marker='x')
    plt.title('Cumulative Payoffs Over Time')
    plt.xlabel('Game Number')
    plt.ylabel('Cumulative Payoff')
    plt.legend()
    plt.grid(True)

    # Plot for oracle win percentage
    plt.subplot(2, 2, 2)
    plt.plot(time_points, oracle_win_percentages, label="Oracle Win Percentage", color='green', marker='s')
    plt.title('Oracle Win Percentage of Player 1 Over Time')
    plt.xlabel('Game Number')
    plt.ylabel('Win Percentage')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    agent1_avg_pot_size = [sum(pot_sizes[:i+1]) / (i+1) if payoff > 0 else 0 for i, payoff in enumerate(agent1_payoffs)]
    agent2_avg_pot_size = [sum(pot_sizes[:i+1]) / (i+1) if payoff > 0 else 0 for i, payoff in enumerate(agent2_payoffs)]
    plt.plot(time_points, agent1_avg_pot_size, label=f"{agent1_name} Average Pot Size Won", marker='o')
    plt.plot(time_points, agent2_avg_pot_size, label=f"{agent2_name} Average Pot Size Won", marker='x')
    plt.title('Average Pot Size Won Over Time')
    plt.xlabel('Game Number')
    plt.ylabel('Average Pot Size')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    # Save the figure to the 'figs' directory
    if not os.path.exists('figs'):
        os.makedirs('figs')
    # plt.savefig(f'figs/{agent1_name}_vs_{agent2_name}.png')
    plt.show()

def get_oracle_win(env):
    """
    Returns the win percentage of player 1 in the environment at the ending state

    :param env: An env which should be in the done state
    """
    p1_state = env.unwrapped.env.get_state(0)
    p2_state = env.unwrapped.env.get_state(1)
    p1_hand = p1_state['raw_obs']['hand']
    p2_hand = p2_state['raw_obs']['hand']
    board = p1_state['raw_obs']['public_cards']
    return calculate_winpercent(p1_hand, p2_hand, board)

def play_poker_games(agent_names: List[str], number_of_games: int, render: int, oracle: int):
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
    pot_sizes = []
    won = []
    oracle_wp = [] 
    
    for i in range(number_of_games):
        env.reset()

        for agent in env.agent_iter():
            observation, reward, done, truncation, info = env.last()

            if done:
                if oracle:
                    win_percent = get_oracle_win(env)
                    oracle_wp.append(win_percent)

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
                pot_sizes.append(abs(payoffs_1[-1]) + abs(payoffs_2[-1]))
                if payoffs_1[-1] < 0:
                    won.append(1)
                else:
                    won.append(0)
                print(f"Game over: agent 0 won {payoffs_1[-1]}")
                break
    
    return payoffs_1, payoffs_2, won, oracle_wp, pot_sizes
         

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

    elif agent_name == "qlearning":
        agent = QLearningAgent(env = env.unwrapped.env, model_path='./qlearning_models') # need to first run ```python test_rlcard_qlearning_agent.py````
    else:
        raise ValueError(f"Invalid argument: {agent_name} not a valid agent name")

    # TODO: implement a way to get your agents. Make sure they have an step function implemented that takes an observation and returns an action.  
    return agent
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play poker games between different agents and evaluate their performance.")
    parser.add_argument("agent_names", nargs='+', help="Names of the agents participating in the games.")
    parser.add_argument("--number_of_games", type=int, default=100, help="Number of games to be played.")
    parser.add_argument("--render", type=int, default=True, help="whether or not to render the games being played.")
    parser.add_argument("--oracle", type=int, default=True, help="whether or not to track oracle outcomes.")


    args = parser.parse_args()

    payoffs1, payoffs2, won, oracle_results, pot_sizes = play_poker_games(
        agent_names=args.agent_names,
        number_of_games=args.number_of_games,
        render=args.render,
        oracle=args.oracle
    )

    # Print or process the results of the games
    plot_evaluation_results(
        agent1_payoffs=payoffs1,
        agent2_payoffs=payoffs2,
        oracle_win_percentages=oracle_results,
        agent1_name=args.agent_names[0],
        agent2_name=args.agent_names[1],
        pot_sizes=pot_sizes
    )

