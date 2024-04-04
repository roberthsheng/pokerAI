from rlcard_cfr_agent import CFRAgent
from pettingzoo.classic import texas_holdem_no_limit_v6
import rlcard
from cfr_nolimit_agent import CFRAgent


def main():
    config = {
        "allow_step_back": True,
        "seed": None,
        "game_num_players": 2, 
    }
    
    env = rlcard.make("no-limit-holdem", config)


    # Initialize the game environment
    # env = texas_holdem_no_limit_v6.raw_env(render_mode="human", num_players=2, allow_step_back=True)
    cfr = CFRAgent(env)

    cfr.train()
    print(cfr.regrets)

main()

