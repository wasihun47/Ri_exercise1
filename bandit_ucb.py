import numpy as np
class UCB_Bandit:
    def __init__(self, k):
        self.k = k  # Number of arms
        self.action_values = np.zeros(k)  # Estimated values
        self.arm_counts = np.zeros(k)  # Number of times each arm was pulled

    def select_action(self, t):
        # UCB action selection
        ucb_values = np.zeros(self.k)
        total_counts = np.sum(self.arm_counts)
        for arm in range(self.k):
            if self.arm_counts[arm] == 0:
                ucb_values[arm] = float('inf')  # Select each arm at least once
            else:
                bonus = np.sqrt((2 * np.log(total_counts)) / self.arm_counts[arm])
                ucb_values[arm] = self.action_values[arm] + bonus
        return np.argmax(ucb_values)

    def update_values(self, action, reward):
        # Update the estimated values based on the reward received
        self.arm_counts[action] += 1
        alpha = 1 / self.arm_counts[action]
        self.action_values[action] += alpha * (reward - self.action_values[action])

    def run(self, environment, iterations=1000):
        # Run the bandit algorithm for a specified number of iterations
        rewards_history = []

        for t in range(1, iterations + 1):
            action = self.select_action(t)
            reward = environment.get_reward(action)
            self.update_values(action, reward)
            rewards_history.append(reward)

        return rewards_history
