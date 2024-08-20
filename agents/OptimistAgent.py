import numpy as np
from agents.BaseAgent import BaseAgent

class OptimisticAgent(BaseAgent):
    def __init__(self, num_of_actions, epsilon=0.1, initial_value=5, alpha=0.1):
        self.Q = [initial_value] * num_of_actions
        self.N = [0] * num_of_actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.num_of_actions = num_of_actions

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_of_actions)
        else:
            return np.argmax(self.Q)

    def learn(self, action, reward):
        self.Q[action] += self.alpha * (reward - self.Q[action])