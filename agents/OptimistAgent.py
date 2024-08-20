import random
import numpy as np
from agents.BaseAgent import BaseAgent

class OptimisticAgent(BaseAgent):
    def __init__(self, num_of_actions, epsilon=0.1, initial_value=5, alpha=0.1):
        self.num_of_actions = num_of_actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.Q = [initial_value] * num_of_actions
        self.N = np.zeros(num_of_actions)

    def get_action(self):
        if random.random() < self.epsilon:
            return random.randrange(self.num_of_actions)  # Selección aleatoria (exploración)
        else:
            # Selección de la mejor acción con desempate aleatorio (breaking tie randomly, como aparece en la secció del libro)
            mejor_valor = np.max(self.Q)
            posibles_candidatos = np.where(self.Q == mejor_valor)[0]
            return np.random.choice(posibles_candidatos)

    def learn(self, action, reward):
        # Actualización incremental de Q(a) según la ecuación (2.5)
        self.Q[action] += self.alpha * (reward - self.Q[action])