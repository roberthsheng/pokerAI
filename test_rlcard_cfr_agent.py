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
    for i in range(100000):
        cfr.train()
        if i % 50 == 0:
            print(i)
    print(len(cfr.regrets))
    print(f"regrets {cfr.regrets}\n\n")
    print(f"policy {cfr.policy}\n\n")
 #   print(f"avg {cfr.average_policy}\n\n")


#
#        print("\n\n\n\n\n")
#
        
    cfr.save()
 

main()

