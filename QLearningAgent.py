import os
import numpy as np
import random
from gameModule import SnakeGame, RIGHT, LEFT, UP, DOWN


class SnakeQAgent:
    def __init__(self, q_table_file_name="None"):
        self.num_episodes = 1_000_000_000
        self.table = self.get_q_table(q_table_file_name)

    def get_q_table(self, file_name):
        if file_name == "None":
            return np.zeros((2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4))
        else:
            return np.load("q_tables/" + str(file_name))

    def save_q_table(self):
        q_tables_dir = "q_tables/"
        if not os.path.exists(q_tables_dir):
            os.makedirs(q_tables_dir)
        np.save(q_tables_dir + str(random.randint(0, 1000000)), self.table)

    def choose_next_action(self, state):
        return np.argmax(self.table[state])

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

        self.save_q_table()

    def eat(self):
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
        self.eps_discount = 0.9992
        self.min_eps = 0.001

    def next_tick(self, agent):
        self.eps = max(self.eps * self.eps_discount, self.min_eps)

        current_state = self.get_q_state()
        action = self.get_action(current_state, agent)

        self.set_next_move(self.adapt_action(action))
        self.move_snake()

        new_state = self.get_q_state()
        reward = 1 if self.foodEaten else -10 if not self.alive else None

        if reward is not None:
            self.bellman(agent.table, current_state, action, new_state, reward)

    def get_q_state(self):
        """Build state for Q-Learner agent"""
        head_r, head_c = self.snake[0]
        direction = self.get_direction()
        food_r, food_c = self.food

        state = [
            int(direction == "left"), int(direction == "right"), int(direction == "up"), int(direction == "down"),
            int(food_r < head_r), int(food_r > head_r), int(food_c < head_c), int(food_c > head_c),
            self.is_unsafe(head_r + 1, head_c), self.is_unsafe(head_r - 1, head_c),
            self.is_unsafe(head_r, head_c + 1), self.is_unsafe(head_r, head_c - 1)]

        return tuple(state)

    def get_direction(self):
        if len(self.snake) == 1:
            return "right"

        head_r, head_c = self.snake[0]
        neck_r, neck_c = self.snake[1]

        if head_r > neck_r:
            return "right"
        if head_r < neck_r:
            return "left"
        if head_c > neck_c:
            return "down"
        else:
            return "up"

    def is_unsafe(self, r, c):
        """
        Check if the next move is unsafe
        """
        if self.is_collision((r, c)):
            return 1
        else:
            return 0

    # epsilon-greedy action choice
    def get_action(self, state, agent):
        if random.random() < self.eps:
            return random.choice([0, 1, 2, 3])
        else:
            return agent.choose_next_action(state)

    def adapt_action(self, action="None"):
        if action == "None":
            return random.choice([LEFT, RIGHT, UP, DOWN])
        else:
            return [LEFT, RIGHT, UP, DOWN][action]

    def bellman(self, table, current_state, action, new_state, reward):
        table[current_state][action] = (1 - self.learning_rate) \
                                       * table[current_state][action] + self.learning_rate \
                                       * (reward + self.discount_rate * max(table[new_state]))
