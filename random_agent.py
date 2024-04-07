class RandomAgent(object):
    def __init__(self, env, agent):
        self.env = env
        name = "player_" + str(agent)
        self.agent = name 
    def step(self, observation):
        mask = observation["action_mask"]
        return self.env.action_space(self.agent).sample(mask)  




