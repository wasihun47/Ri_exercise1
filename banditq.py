import numpy as np
class qlearnBandit:
    def __init__(self, k, alpha=0.5, epsilon=0.1):
        self.k = k  # Number of arms
        self.q_values = np.zeros(k)  # Q values
        self.arm_counts = np.zeros(k)  # Number of times each arm was pulled
        self.alpha = alpha  # Learning rate
        self.epsilon = epsilon  # Exploration rate

    def select_action(self):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.k)  # Explore: select a random action
        else:
            return np.argmax(self.q_values)  # Exploit: select the action with max Q value

    def update_q_values(self, action, reward):
        # Update the Q value for the selected action based on the received reward
        self.arm_counts[action] += 1
        self.q_values[action] += self.alpha * (reward - self.q_values[action])

    def q_learning(self, environment, iterations=1000):
        # Perform Q-Learning for a specified number of iterations
        rewards_history = []

        for _ in range(iterations):
            action = self.select_action()  # Choose an action
            reward = environment.get_reward(action)  # Simulate pulling the arm
            rewards_history.append(reward)
            # Update the Q value for the chosen action
            self.update_q_values(action, reward)

        return self.q_values, rewards_history

