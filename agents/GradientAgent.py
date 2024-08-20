import numpy as np
from agents.BaseAgent import BaseAgent

class GradientBanditAgent(BaseAgent):
    def __init__(self, n_actions, alpha, baseline=True):
        self.n_actions = n_actions
        self.alpha = alpha
        self.baseline = baseline
        self.preferences = np.zeros(n_actions)  # H(a)
        self.action_probabilities = np.ones(n_actions) / n_actions  # π(a)
        self.average_reward = 0.0  # R̄
        self.time_step = 0

    def get_action(self) -> int:
        """Selecciona una acción basada en la distribución softmax."""
        self.action_probabilities = np.exp(self.preferences) / np.sum(np.exp(self.preferences))
        return np.random.choice(self.n_actions, p=self.action_probabilities)

    def learn(self, action, reward) -> None:
        """Actualiza las preferencias basadas en la acción tomada y la recompensa recibida."""
        self.time_step += 1
        if self.baseline:
            self.average_reward += (reward - self.average_reward) / self.time_step

        baseline = self.average_reward if self.baseline else 0
        one_hot = np.zeros(self.n_actions)
        one_hot[action] = 1
        self.preferences += self.alpha * (reward - baseline) * (one_hot - self.action_probabilities)
