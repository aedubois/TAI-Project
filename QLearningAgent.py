import os
import numpy as np
import random
from gameModule import SnakeGame, RIGHT, LEFT, UP, DOWN

MOVES = [LEFT, RIGHT, UP, DOWN]

class SnakeQAgent:
    def __init__(self, q_table_file_name="None", game=None):
        self.num_episodes = 1_000_000
        self.table = self.get_q_table(q_table_file_name)
        self.current_game = game

    def get_q_table(self, file_name):
        if file_name == "None":
            return np.zeros((2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4))
        else:
            return np.load("q_tables/" + str(file_name))

    def save_q_table(self, file_name):
        q_tables_dir = "q_tables/"
        if not os.path.exists(q_tables_dir):
            os.makedirs(q_tables_dir)
        np.save(q_tables_dir + file_name, self.table)

    def choose_next_move(self, state):
        q_state = self.current_game.get_q_state()
        return MOVES[np.argmax(self.table[q_state])]

    def train(self):
        highest_score = 0

        for i in range(1, self.num_episodes + 1):
            self.current_game = TrainingSnakeGame()
            self.current_game.start_run()

            while self.current_game.is_alive():
                self.current_game.next_tick(self)

            if self.current_game.score > highest_score:
                highest_score = self.current_game.score

            print(f"Episode {i} finished. Highest_Score: {highest_score}. Current_Score: {self.current_game.score}")

        self.save_q_table(highest_score)

    def eat(self):
        """
        This function is useless here, it is a placeholder for a function needed in the other
        algorithm.
        """
        pass

    def reset_state(self):
        """
        This function is useless here, it is a placeholder for a function needed in the other
        algorithm.
        """
        pass


class TrainingSnakeGame(SnakeGame):
    def __init__(self):
        super(TrainingSnakeGame, self).__init__()
        self.discount_rate = 0.95
        self.learning_rate = 0.01
        self.eps = 1.0
        self.eps_discount = 0.95
        self.min_eps = 0.001

    def next_tick(self, agent):
        self.eps = max(self.eps * self.eps_discount, self.min_eps)

        current_state = self.get_q_state()
        next_move = self.get_next_move(current_state, agent)

        self.set_next_move(next_move)
        self.move_snake()

        new_state = self.get_q_state()
        reward = 1 if self.foodEaten else -10 if not self.alive else -1

        if reward is not None:
            self.bellman(agent.table, current_state, MOVES.index(next_move), new_state, reward)

    # epsilon-greedy action choice
    def get_next_move(self, state, agent):
        if random.random() < self.eps:
            return MOVES[random.choice([0, 1, 2, 3])]
        else:
            return agent.choose_next_move(state)

    def bellman(self, table, current_state, action, new_state, reward):
        table[current_state][action] = (1 - self.learning_rate) \
                                       * table[current_state][action] + self.learning_rate \
                                       * (reward + self.discount_rate * max(table[new_state]))