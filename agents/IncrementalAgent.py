import random
import numpy as np
from agents.BaseAgent import BaseAgent

class IncrementalAgent(BaseAgent):
    def __init__(self, num_of_actions: int, epsilon: float = 0.1):
        self.num_of_actions = num_of_actions
        self.epsilon = epsilon
        self.Q = np.zeros(num_of_actions)  # Estimaciones de valor Q(a)
        self.N_counts = np.zeros(num_of_actions)  # Contador de selecciones de cada acción

    def get_action(self) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.num_of_actions)  # Selección aleatoria (exploración)
        else:
            # Selección de la mejor acción con desempate aleatorio
            max_value = np.max(self.Q)
            candidates = np.where(self.Q == max_value)[0]
            return np.random.choice(candidates)

    def learn(self, action, reward) -> None:
        self.N_counts[action] += 1
        # Actualización incremental de Q(a)
        self.Q[action] += (1 / self.N_counts[action]) * (reward - self.Q[action])