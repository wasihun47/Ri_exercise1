import numpy as np
class policy_iteration:
    def __init__(self, k,):
        self.k = k  # Number of arms
       # self.reward_distributions = reward_distributions  # Reward distributions for each arm
        self.action_values = np.zeros(k)  # Estimated values
        self.arm_counts = np.zeros(k)  # Number of times each arm was pulled
        self.policy = np.ones(k) / k  # Initialize a uniform policy
    
   # def get_reward(self, action):
        # Return a reward from the distribution corresponding to the selected action
    #    return np.random.normal(self.reward_distributions[action])

    
    #def optimal_strategy(self):
        # Return the arm with the highest expected reward
     #   return np.argmax(self.reward_distributions)
    def select_action(self):
        # Select an action according to the policy
        return np.random.choice(self.k, p=self.policy)

    def update_values(self, action, reward):
        # Update the estimated values based on the reward received
        self.arm_counts[action] += 1
        q_n = self.action_values[action]
        q_n_plus_1 = q_n + (1.0 / self.arm_counts[action]) * (reward - q_n)
        self.action_values[action] = q_n_plus_1

    def update_policy(self):
        # Update the policy based on the current action values
        best_action = np.argmax(self.action_values)
        for action in range(self.k):
            if action == best_action:
                self.policy[action] = 1 - (self.k - 1) * 0.01
            else:
                self.policy[action] = 0.01

    def policy_iteration(self, environment, iterations=1000):
        # Perform policy iteration for a specified number of iterations
        rewards_history = []

        for _ in range(iterations):
            action = self.select_action()  # Choose an action based on the policy
            reward = environment.get_reward(action)  # Simulate pulling the arm
            rewards_history.append(reward)
            # Update the value estimate for the chosen action and the policy
            self.update_values(action, reward)
            self.update_policy()

        return self.policy, rewards_history
    


