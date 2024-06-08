import numpy as np
class value_iterationsa:
     def value_iteration(self, iterations=1000):
        # Perform value iteration for a specified number of iterations
        value_estimates = np.zeros(self.k)
        rewards_history = []

        for _ in range(iterations):
            action = np.argmax(value_estimates)  # Choose the best action based on current estimates
            reward = self.get_reward(action)  # Simulate pulling the arm
            rewards_history.append(reward)
            # Update the value estimate for the chosen action
            self.arm_counts[action] += 1
            alpha = 1 / self.arm_counts[action]
            value_estimates[action] += alpha * (reward - value_estimates[action])

        return value_estimates, rewards_history






