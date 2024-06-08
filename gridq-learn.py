import numpy as np
import matplotlib.pyplot as plt

class QLearning:
    def __init__(self, env, learning_rate=0.5, discount_factor=0.95, exploration_rate=0.3, iterations=10000):
        self.env = env
        self.lr = learning_rate
        self.df = discount_factor
        self.er = exploration_rate
        self.iterations = iterations
        self.q_table = np.zeros((len(env.states), len(env.actions)))

    def learn(self):
        for i in range(self.iterations):
            state = self.env.reset()
            while not state == self.env.goal:
                action = self.get_action(state)
                reward, next_state = self.env.step(action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state

    def get_action(self, state):
        if np.random.uniform(0, 1) < self.er:
            return np.random.choice(self.env.actions)
        else:
            return np.argmax(self.q_table[self.env.states.index(state)])

    def update_q_table(self, state, action, reward, next_state):
        old_value = self.q_table[self.env.states.index(state), self.env.actions.index(action)]
        next_max = np.max(self.q_table[self.env.states.index(next_state)])
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.df * next_max)
        self.q_table[self.env.states.index(state), self.env.actions.index(action)] = new_value

# Initialize GridWorld environment
env = GridWorld(5, 5, (0, 0), (4, 4), [(1, 2), (2, 2), (3, 2)])

# Initialize QLearning agent
q_agent = QLearning(env)

# Perform Q-learning
q_agent.learn()

# Print the learned Q-table
print(q_agent.q_table)
