import numpy as np
import random
from gameModule import SnakeGame

class SnakeQAgent():
    def __init__(self):
        # define initial parameters
        self.discount_rate = 0.95
        self.learning_rate = 0.01
        self.eps = 1.0
        self.eps_discount = 0.9992
        self.min_eps = 0.001
        self.num_episodes = 10000
        self.table = np.zeros((2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4))
        self.env = SnakeGame()
        self.score = []
        self.survived = []
        
    # epsilon-greedy action choice
    def get_action(self, state):
        # select random action (exploration)
        if random.random() < self.eps:
            return random.choice([0, 1, 2, 3])
        
        # select best action (exploitation)
        return np.argmax(self.table[state])
    
    def train(self):
        for i in range(1, self.num_episodes + 1):
            self.env  = SnakeGame()
            steps_without_food = 0
            length = len(self.env.snake)
                
            current_state = self.env.get_q_state()
            self.eps = max(self.eps * self.eps_discount, self.min_eps)
            done = False
            while not done:
                # choose action and take it
                action = self.get_action(current_state)
                new_state, reward, done = self.env.step(action)
                
                # Bellman Equation Update
                self.table[current_state][action] = (1 - self.learning_rate)\
                    * self.table[current_state][action] + self.learning_rate\
                    * (reward + self.discount_rate * max(self.table[new_state])) 
                current_state = new_state
                
                steps_without_food += 1
                if length != len(self.env.snake):
                    length = len(self.env.snake)
                    steps_without_food = 0
                if steps_without_food == 1000:
                    # break out of loops
                    break
            
            # keep track of important metrics
            self.score.append(len(self.env.snake) - 1)
            self.survived.append(self.env.survived)

    def eat(self):
        """
        This function is useless here, it is a placeholder for a function needed in the other
        algorithm.
        """
        pass