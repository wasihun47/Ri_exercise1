class GridWorld:
    def __init__(self, n, m, start, goal, obstacles):
        self.n = n
        self.m = m
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.states = [(i, j) for i in range(n) for j in range(m) if (i, j) not in obstacles]
        self.actions = ['up', 'down', 'left', 'right']
        #self.actions = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
        self.rewards = {s: -1 for s in self.states}
        self.rewards[goal] = 10
   
    def reset(self):
        self.state = self.start
        return self.state
       
    def step(self, action):
        next_state = self.transition(self.start, action)
        reward = self.reward(next_state)
        self.start = next_state
        return reward, next_state

    def transitionss(self, state, action):
        if action == 'up':
            next_state = (state[0] - 1, state[1])
        elif action == 'down':
            next_state = (state[0] + 1, state[1])
        elif action == 'left':
            next_state = (state[0], state[1] - 1)
        else:  # action == 'right'
            next_state = (state[0], state[1] + 1)

        if next_state in self.states:
            return next_state
        else:
            return state

    def rewardss(self, state):
        return self.rewards[state]

    
    def transition(self, s, a):
        if a == 'up' and (s[0] - 1, s[1]) not in self.obstacles and s[0] > 0:
            return (s[0] - 1, s[1])
        elif a == 'down' and (s[0] + 1, s[1]) not in self.obstacles and s[0] < self.n - 1:
            return (s[0] + 1, s[1])
        elif a == 'left' and (s[0], s[1] - 1) not in self.obstacles and s[1] > 0:
            return (s[0], s[1] - 1)
        elif a == 'right' and (s[0], s[1] + 1) not in self.obstacles and s[1] < self.m - 1:
            return (s[0], s[1] + 1)
        else:
            return s

    def reward(self, s):
        if s == self.goal:
            return 10
        elif s in self.obstacles:
            return -10
        else:
            return -1


        
 




