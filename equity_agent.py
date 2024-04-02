from pettingzoo.classic import texas_holdem_no_limit_v6
from human_agent import HumanAgent
from equity_calculator import calculate_equity

class EquityAgent(object):
    def __init__(self, env):
        self.env = env
        
    def get_state(self, observation):
        # need to get hand/community card data which seems inaccessible via vanilla observations
        unwrapped_env = self.env.unwrapped
        raw_data = unwrapped_env.env.game.get_state(unwrapped_env.env.get_player_id())
        hole_cards = raw_data['hand'] 
        community_cards = raw_data['public_cards'] 
        pot = raw_data['pot']
        current_bet = max(raw_data['all_chips'])
        amount_to_play = current_bet - raw_data['my_chips']
        state = {
            'raw_obs': observation,
            'hole_cards': hole_cards,
            'community_cards': community_cards, 
            'pot': pot,
            'amount_to_play': amount_to_play
        }
        return state 

    def step(self, observation):
        state = self.get_state(observation)
        print(state)
        if state["amount_to_play"] == 0: # check if there is no bet
            return 1
        
        hand_equity = calculate_equity(state['hole_cards'], state['community_cards'])
        pot_odds = state['amount_to_play']/(state['pot'] + state['amount_to_play']) 
        
        print(hand_equity)
        print(pot_odds)
        if pot_odds <= hand_equity: # call if equity > pot odds (vs all hands), this is +EV
            return 1 # meaning call

        else:
            return 0 # fold

# if running main, play a game with human vs equity agent
def main():
    # Initialize the game environment
    env = texas_holdem_no_limit_v6.env(render_mode="human", num_players=2)

    env.reset()
    
    # Initialize agents
    
    agents = [HumanAgent(env.action_space(env.agents[0]).n), EquityAgent(env)]
    agent_dict = dict(zip(env.agents, agents))

    agent_test = EquityAgent(env)
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
