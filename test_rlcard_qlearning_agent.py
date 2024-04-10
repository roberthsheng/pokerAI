from pettingzoo.classic import texas_holdem_no_limit_v6
import numpy as np
from qlearning_agent import QLearningAgent

def train_q_learning_agent(episodes=1000000):
    # Initialize the game environment
    env = texas_holdem_no_limit_v6.env(render_mode="rgb-array", num_players=2)

    # Initialize Q-learning agent
    q_agent = QLearningAgent(env.unwrapped.env)
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay_steps = 900000

    def get_epsilon(episode):
        return max(epsilon_end, epsilon_start - (episode / epsilon_decay_steps) * (epsilon_start - epsilon_end))

    for episode in range(episodes):
        epsilon = get_epsilon(episode)
        q_agent.epsilon = epsilon  # Update the agent's exploration rate

        env.reset()
        total_reward = 0

        for agent in env.agent_iter():
            observation, reward, done, truncation, info = env.last()

            if done:
                env.step(None)  # Indicate to PettingZoo that the agent is done for this episode
                continue

            # Assume q_agent is associated with the first agent in env.agents list
            if agent == env.agents[0]:
                current_state = q_agent.reduce_state(observation)
                action = q_agent.step(observation)
                env.step(action)

                # Observe the next state and reward from the environment
                next_observation, reward, done, truncation, info = env.last()
                next_state = q_agent.reduce_state(next_observation) if not done else None

                # Let Q-learning agent learn from the transition
                q_agent.learn(current_state['observation'], action, reward, next_state['observation'] if next_state else None, done)

                total_reward += reward
            else:
                # If not the Q-learning agent's turn, take random action or predefined policy action
                action = np.random.choice(np.where(observation['action_mask'] == 1)[0])
                env.step(action)

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

    q_agent.save()

train_q_learning_agent()
