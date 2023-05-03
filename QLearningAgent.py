import numpy as np
import random
from gameModule import SnakeGame

class SnakeQAgent():
    def __init__(self, env, args="None"):
        # define initial parameters
        self.discount_rate = 0.95
        self.learning_rate = 0.01
        self.eps = 1.0
        self.eps_discount = 0.9992
        self.min_eps = 0.001
        self.num_episodes = 10000
        self.score = []
        self.survived = []
        self.table = self.create_table(args)

    def create_table(self ,args):
        if args == "None":
            return np.zeros((2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4))
        else:
            file = "q_tables/" + str(args)
            loaded_Q_table = np.load(file)
            return loaded_Q_table
        
    # epsilon-greedy action choice
    def get_action(self, state):
        # select random action (exploration)
        if random.random() < self.eps:
            return random.choice([0, 1, 2, 3])
        
        # select best action (exploitation)
        return np.argmax(self.table[state])
    
    def step(action, env): #TODO
        reward = 0
        new_pos = (env.snake[0][0] + action[0], env.snake[0][1] + action[1])
        if not new_pos.is_alive(): 
            reward = -10
        if new_pos
        return new_state, reward, done

    def train(self):
        for i in range(1, self.num_episodes + 1):
            env  = SnakeGame()
            env.start_run()
            while env.is_alive():
                #TODO check
                env.next_tick()
            steps_without_food = 0
            length = len(env.snake)
                
            current_state = self.env.get_q_state()
            self.eps = max(self.eps * self.eps_discount, self.min_eps)
            done = False
            while not done:
                # choose action and take it
                action = self.get_action(current_state)
                new_state, reward, done = self.step(self.adapt_action(action), env)
                
                # Bellman Equation Update
                self.table[current_state][action] = (1 - self.learning_rate)\
                    * self.table[current_state][action] + self.learning_rate\
                    * (reward + self.discount_rate * max(self.table[new_state])) 
                current_state = new_state
                
                steps_without_food += 1
                if length != len(env.snake):
                    length = len(env.snake)
                    steps_without_food = 0
                if steps_without_food == 1000:
                    # break out of loops
                    break
            
            # keep tracpoetry run python train.py k of important metrics
            self.score.append(len(env.snake) - 1)
            self.survived.append(env.survived)
        np.save('q_tables/q_table.npy', self.table)


    def eat(self):
        """
        This function is useless here, it is a placeholder for a function needed in the other
        algorithm.
        """
        pass

    def adapt_action(self, action="None"):
        if action == "None":
            action = random.choice([(0, -1), (0, 1), (-1, 0), (1, 0)])
        else:
            action = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        return action
    
