import numpy as np
class rewards_historys:
    def __init__(self, k, epsilon=0.1):
        self.k = k  # Number of arms
        self.action_values = np.zeros(k)  # Estimated values
        self.arm_counts = np.zeros(k)  # Number of times each arm was pulled
        self.epsilon = epsilon  # Probability of exploration

    def select_action(self):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)  # Explore: select a random action
        else:
            return np.argmax(self.action_values)  # Exploit: select the best-known action

    def update_values(self, action, reward):
        # Update the estimated values based on the reward received
        self.arm_counts[action] += 1
        alpha = 1 / self.arm_counts[action]
        self.action_values[action] += alpha * (reward - self.action_values[action])

    def run(self, environment, iterations=1000):
        # Run the bandit algorithm for a specified number of iterations
        rewards_history = []

        for _ in range(iterations):
            action = self.select_action()
            reward = environment.get_reward(action)
            self.update_values(action, reward)
            rewards_history.append(reward)

        return rewards_history
