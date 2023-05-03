import os
import numpy as np
import random
from gameModule import SnakeGame

<<<<<<< Updated upstream
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
        
=======

def adapt_action(action="None"):
    if action == "None":
        action = random.choice([(0, -1), (0, 1), (-1, 0), (1, 0)])
    else:
        action = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
    return action

class SnakeQAgent:
    def __init__(self, args="None"):
        self.num_episodes = 1_000_000_000
        self.table = self.create_table(args)

    def create_table(self, args):
        if args == "None":
            return np.zeros((2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4))
        else:
            file = "q_tables/" + str(args)
            loaded_q_table = np.load(file)
            return loaded_q_table

>>>>>>> Stashed changes
    # epsilon-greedy action choice
    def get_action(self, state):
        return np.argmax(self.table[state])
<<<<<<< Updated upstream
    
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
=======

    def train(self):
        highest_score = 0

        for i in range(1, self.num_episodes + 1):
            curr_game = TrainingSnakeGame()
            curr_game.start_run()

            while curr_game.is_alive():
                curr_game.next_tick(self)

            if curr_game.score > highest_score:
                highest_score = curr_game.score

            print(f"Episode {i} finished. Highest_Score: {highest_score}")

        save_q_table(self.table)
>>>>>>> Stashed changes

    def eat(self):
        """
        This function is useless here, it is a placeholder for a function needed in the other
        algorithm.
        """
<<<<<<< Updated upstream
        pass
=======
        pass


def save_q_table(table):
    q_tables_dir = "q_tables/"
    if not os.path.exists(q_tables_dir):
        os.makedirs(q_tables_dir)
    np.save(q_tables_dir + str(random.randint(0, 1000000)), table)


class TrainingSnakeGame(SnakeGame):
    def __init__(self):
        super(TrainingSnakeGame, self).__init__()
        self.discount_rate = 0.95
        self.learning_rate = 0.01
        self.eps = 1.0
        self.eps_discount = 0.9992
        self.min_eps = 0.001

    def get_action(self, state, agent):
        if random.random() < self.eps:
            return random.choice([0, 1, 2, 3])
        else:
            return agent.get_action(state)

    def next_tick(self, agent):
        self.eps = max(self.eps * self.eps_discount, self.min_eps)

        current_state = self.get_q_state()
        action = self.get_action(current_state, agent)

        self.set_next_move(adapt_action(action))
        self.move_snake()

        new_state = self.get_q_state()
        reward = 1 if self.foodEaten else -10 if not self.is_alive() else None

        if reward is not None:
            self.bellman(agent.table, current_state, action, new_state, reward)

    def bellman(self, table, current_state, action, new_state, reward):
        table[current_state][action] = (1 - self.learning_rate) \
                                       * table[current_state][action] + self.learning_rate \
                                       * (reward + self.discount_rate * max(table[new_state]))
>>>>>>> Stashed changes
