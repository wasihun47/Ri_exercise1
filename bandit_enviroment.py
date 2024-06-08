import numpy as np
class Environment:
    def __init__(self, k, reward_distributions):
        self.k = k  # Number of arms
        self.reward_distributions = reward_distributions  # Reward distributions for each arm

    def get_reward(self, action):
        # Return a reward from the distribution corresponding to the selected action
        return np.random.normal(self.reward_distributions[action])

    def optimal_strategy(self):
        # Return the arm with the highest expected reward
        return np.argmax(self.reward_distributions)