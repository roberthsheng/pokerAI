from labml import experiment, monit, tracker
from typing import Callable, Dict, List, NewType, cast
import numpy as np
from pettingzoo.classic import texas_holdem_no_limit_v6

NUM_PLAYERS = 4    

Player = NewType("Player", int)
Action = NewType("Action", int)

class History:
    
    def is_terminal(self) -> bool:
        """Outputs whether the history is a finished game state."""
        raise NotImplementedError
    
    def terminal_utility(self, i: Player) -> float:
        """Ouputs utility of specified player for a terminal history."""
        raise NotImplementedError
    
    def player(self) -> Player:
        """Outputs the player to act in the history."""
        raise NotImplementedError
    
    def is_chance(self) -> bool:
        """Is the next game step a chance node?"""
        raise NotImplementedError
    
    def sample_chance(self) -> Action:
        """Samples a chance event."""
        raise NotImplementedError
    
    def __add__(self, action: Action):
        """Adds an action to the history."""
        raise NotImplementedError
    
    def info_set(self) -> str:  
        """Get the information set (strategy) for current player."""
        raise NotImplementedError
    
    def new_info_set(self) -> 'InfoSet':
        """Create a new information set for the current player."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """String representation of the history."""
        raise NotImplementedError

class InfoSet:
    key: str
    strategy: Dict[Action, float]
    regret: Dict[Action, float]
    cumulative_strategy: Dict[Action, float]

    def __init__ (self, key: str):
        self.key = key
        self.regret = {a: 0 for a in self.actions()}
        self.cumulative_strategy = {a: 0 for a in self.actions()}
        self.calculate_strategy()

    def actions(self) -> List[Action]:
        """Outputs the list of possible actions."""
        raise NotImplementedError
    
    @staticmethod
    def from_dict(data: Dict[str, any]) -> 'InfoSet':
        """Loads information set from a preexisting dictionary."""
        raise NotImplementedError
    
    def to_dict(self) -> Dict[str, any]:
        """Converts the information set to a dictionary."""
        return {
            "key": self.key,
            "regret": self.regret,
            "average_strategy": self.cumulative_strategy,
        }
    
    def load_dict(self, data: Dict[str, any]):
        """Loads information from a dictionary."""
        self.regret = data["regret"]
        self.cumulative_strategy = data["average_strategy"]
        self.calculate_strategy()

    def calculate_strategy(self):
        """Calculate CFR strategy."""
        regret = {a: max(r, 0) for a, r in self.regret.items()}
        regret_sum = sum(regret.values())
        if regret_sum > 0:
            self.strategy = {a: r / regret_sum for a, r in regret.items()} # normalize values
        else:
            action_count = len(list(a for a in self.regret))
            self.strategy = {a: 1 / action_count for a, _ in regret.items()} # uniform
    
    def get_average_strategy(self):
        """Outputs the average strategy."""
        cumulative_strat = {a: self.cumulative_strategy.get(a, 0.) for a in self.actions()}
        strat_sum = sum(cumulative_strat.values())
        if strat_sum > 0:
            return {a: s / strat_sum for a, s in cumulative_strat.items()} # normalize values
        else:
            action_count = len(list(a for a in cumulative_strat.items()))
            return {a: 1 / action_count for a, _ in cumulative_strat.items()}
        
    def __repr__(self) -> str:
        raise NotImplementedError
    
class CFR:
    info_sets: Dict[str, InfoSet]
    def __init__(self, *, create_new_history: Callable[[], History], epochs: int, n_players: int = 2):
        self.n_players = n_players
        self.epochs = epochs
        self.create_new_history = create_new_history
        self.info_sets = {}
        self.tracker = InfoSetTracker()

    def _get_info_set(self, h: History):
        """Get info set of current player for a given history."""
        info_set_key = h.info_set()
        if info_set_key not in self.info_sets:
            self.info_sets[info_set_key] = h.new_info_set()
        return self.info_sets[info_set_key]
    
    def traverse_tree(self, h: History, i: Player, pi_i: float, pi_neg_i: float) -> float:
        """Traverse the game tree.
        pi_i: probability of history for player i under given strategy
        pi_neg_i: probability of history for player(s) NOT i under given strategy
        """
        if h.is_terminal():
            return h.terminal_utility(i)
        elif h.is_chance():
            a = h.sample_chance()
            return self.traverse_tree(h + a, i, pi_i, pi_neg_i)
        
        info_set = self._get_info_set(h)
        actions = info_set.actions()
        v = 0
        va = {}
        for a in info_set.actions:
            if i == h.player():
                va[a] = self.traverse_tree(h + a, i, pi_i * info_set.strategy[a], pi_neg_i)
            else:
                va[a] = self.traverse_tree(h + a, i, pi_i, pi_neg_i * info_set.strategy[a])
            
            v += info_set.strategy[a] * va[a]

        if h.player() == i:
            # Update cumulative strategy and regret totals if current player is i
            for a in info_set.actions():
                info_set.cumulative_strategy[a] += pi_i * info_set.strategy[a]
            for a in info_set.actions():
                info_set.regret[a] += pi_neg_i * (va[a] - v)
            info_set.calculate_strategy()

        return v
    
    def iterate(self):
        for t in monit.iterate('Training', self.epochs):
            for i in range(self.n_players):
                self.traverse_tree(self.create_new_history(), cast(Player, i), 1, 1)

            tracker.add_global_step()
            self.tracker(self.info_sets)
            tracker.save()

            if (t + 1) % 1000 == 0: # save every 1000 iterations
                experiment.save_checkpoint()
                
class InfoSetTracker:
    def __init__(self):
        tracker.set_histogram(f'strategy.*')
        tracker.set_histogram(f'average_strategy.*')
        tracker.set_histogram(f'regret.*')
    def __call__(self, info_sets: Dict[str, InfoSet]):
        for info_set in info_sets.values():
            avg_strat = info_set.get_average_strategy()
            for a in info_set.actions():
                tracker.add({
                    f'strategy.{info_set.key}.{a}': info_set.strategy[a],
                    f'average_strategy.{info_set.key}.{a}': avg_strat[a],
                    f'regret.{info_set.key}.{a}': info_set.regret[a],
                })

# def main():
#     env = texas_holdem_no_limit_v6.env(render_mode="human", num_players=NUM_PLAYERS)
#     env.reset(seed=42)
    
#     for agent in env.agent_iter():
#         observation, reward, termination, truncation, info = env.last()
#         if termination or truncation:
#             action = None

#         mask = observation["action_mask"]
#         action = env.action_space(agent).sample(mask)

#         env.step(action)
#     env.close()

# if __name__ == "__main__":
#     main()