import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

class NeuralNetworkAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(72,)))  # Updated to match the actual observation shape
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.num_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.01))
        return model

    def train(self, state, action, target):
        state_vec = self.observation_to_vector(state)
        state_vec = np.reshape(state_vec, [1, 72])  # Updated to match the actual input shape
        target_f = self.model.predict(state_vec)
        target_f[0][action] = target
        self.model.fit(state_vec, target_f, epochs=1, verbose=0)

    def choose_action(self, state, mask):
        state_vec = self.observation_to_vector(state)
        state_vec = np.reshape(state_vec, [1, 72])  # Updated to match the actual input shape
        masked_q_values = self.model.predict(state_vec)[0] * mask
        normalizing_sum = np.sum(masked_q_values)
        if normalizing_sum > 0:
            masked_q_values /= normalizing_sum
        else:
            valid_actions = np.where(mask == 1)[0]
            if len(valid_actions) > 0:
                return np.random.choice(valid_actions)
            else:
                return 0
        return np.argmax(masked_q_values)

    def observation_to_vector(self, observation):
        return observation['observation']
