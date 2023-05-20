import os
import random
import numpy as np
from gameModule import SnakeGame, RIGHT, LEFT, UP, DOWN, is_collision
from helpers import plot

MOVES = [LEFT, RIGHT, UP, DOWN]


class SnakeQAgent:
    def __init__(self, q_table_file_name="None"):
        self.q_table = get_q_table(q_table_file_name)

    def choose_next_move(self, state):
        q_state = get_q_state(state)
        return MOVES[np.argmax(self.q_table[q_state])]

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
        self.eps_discount = 0.992
        self.min_eps = 0.001
        self.num_episodes = 10_000
        self.agent = SnakeQAgent()

    def train(self):
        total_score = 0
        plot_scores = []
        plot_mean_scores = []

        for episode_idx in range(self.num_episodes):
            self.start_run()
            self.train_one_game()

            print(f"Episode {episode_idx + 1} finished. Highest_Score: {self.best_score}. Current_Score: {self.score}",
                  "current espilon: ", self.eps)

            plot_scores.append(self.score)
            total_score += self.score
            mean_score = total_score / self.num_episodes + 1
            plot_mean_scores.append(mean_score)

        plot(plot_scores, plot_mean_scores, self.best_score, get_figures_dir())

        save_q_table(self.agent.q_table, str(self.best_score))

    def train_one_game(self):
        self.eps = max(self.eps * self.eps_discount, self.min_eps)
        while self.is_alive():
            self.next_tick()

    def next_tick(self):
        current_state = self.get_state()
        current_ql_state = get_q_state(current_state)
        next_move = get_next_move(current_state, self.eps, self.agent)

        self.set_next_move(next_move)
        self.move_snake()

        new_state = self.get_state()
        new_ql_state = get_q_state(new_state)
        reward = self.get_reward()

        bellman(self.agent.q_table, current_ql_state, MOVES.index(next_move), new_ql_state, reward, self.learning_rate,
                self.discount_rate)

    def get_reward(self):
        return 1 if self.food_eaten else -10 if not self.is_alive() else -0.1


def get_q_tables_dir():
    return os.path.join(os.path.dirname(__file__), "q_learning_q_tables/")


def get_figures_dir():
    return os.path.join(os.path.dirname(__file__), "q_learning_figures/")


def get_q_table(file_name):
    if file_name == "None":
        return np.zeros((2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4))
    else:
        return np.load(get_q_tables_dir() + file_name)


def save_q_table(q_table, file_name):
    if not os.path.exists(get_q_tables_dir()):
        os.makedirs(get_q_tables_dir())
    np.save(get_q_tables_dir() + str(file_name), q_table)


def bellman(table, current_state, action, new_state, reward, learning_rate, discount_rate):
    table[current_state][action] = (1 - learning_rate) \
                                   * table[current_state][action] + learning_rate \
                                   * (reward + discount_rate * max(table[new_state]))


def get_next_move(state, eps, agent):
    """epsilon-greedy action choice"""
    if random.random() < eps:
        return MOVES[random.choice([0, 1, 2, 3])]
    else:
        return agent.choose_next_move(state)


def get_q_state(state):
    grid, score, alive, snake, food, direction, rows, columns = state

    head_r, head_c = snake[0]
    food_r, food_c = food

    state = [
        int(direction == "left"),
        int(direction == "right"),
        int(direction == "up"),
        int(direction == "down"),
        int(food_r < head_r),
        int(food_r > head_r),
        int(food_c < head_c),
        int(food_c > head_c),
        int(is_collision((head_r + 1, head_c), rows, columns, grid)),
        int(is_collision((head_r - 1, head_c), rows, columns, grid)),
        int(is_collision((head_r, head_c + 1), rows, columns, grid)),
        int(is_collision((head_r, head_c - 1), rows, columns, grid))]

    return tuple(state)
