import numpy as np
from agents.BaseAgent import BaseAgent

class GradientBanditAgent(BaseAgent):
    def __init__(self, num_of_actions, alpha, baseline=True):
        self.num_of_actions = num_of_actions
        self.alpha = alpha
        self.baseline = baseline
        self.H = np.zeros(num_of_actions)  # H(a)
        self.pi_dist = np.ones(num_of_actions) / num_of_actions  # pi(a)
        self.avg_reward = 0.0
        self.time_step = 0

    def get_action(self) -> int:
        self.pi_dist = np.exp(self.H) / np.sum(np.exp(self.H)) # Dist softmax
        return np.random.choice(self.num_of_actions, p=self.pi_dist)

    def learn(self, action, reward) -> None:
        self.time_step += 1
        if self.baseline:
            self.avg_reward += (reward - self.avg_reward) / self.time_step

        baseline = self.avg_reward if self.baseline else 0
        # Actualizar valores simultaneamente
        one_hot = np.zeros(self.num_of_actions)
        one_hot[action] = 1
        self.H += self.alpha * (reward - baseline) * (one_hot - self.pi_dist)
