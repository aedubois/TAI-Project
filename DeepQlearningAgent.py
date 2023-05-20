import os
import random
from collections import deque

import torch

from model import LinearQNet, QTrainer
from gameModule import SnakeGame, is_collision, RIGHT, DOWN, LEFT, UP
from helpers import plot


def get_models_dir():
    return os.path.join(os.path.dirname(__file__), "deep_q_learning_models/")


def get_figures_dir():
    return os.path.join(os.path.dirname(__file__), "deep_q_learning_figures/")


def load_model(model_file_name="None"):
    model = LinearQNet(11, 256, 3)
    if model_file_name != "None":
        model.load_state_dict(torch.load(get_models_dir() + model_file_name))
    return model


class DeepQLearningAgent:
    def __init__(self, model_file_name="None"):
        self.model = load_model(model_file_name)

    def choose_next_move(self, state):
        direction = state[5]

        predicted_move = predict_next_move(get_deep_q_state(state), self.model)
        return adapt_move(predicted_move, direction)

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
        self.max_memory = 100_000
        self.batch_size = 1000

        self.learning_rate = 0.001
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.num_episodes = 1_000

        self.memory = deque(maxlen=self.max_memory)

        self.model = load_model()
        self.trainer = QTrainer(self.model, lr=self.learning_rate, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def choose_next_move(self, deep_q_state):
        self.epsilon = 80 - self.n_games
        should_explore = random.randint(0, 200) < self.epsilon

        if should_explore:
            return get_random_move()
        else:
            return predict_next_move(deep_q_state, self.model)

    def train(self):
        total_score = 0
        plot_scores = []
        plot_mean_scores = []

        while self.n_games < self.num_episodes:
            self.n_games += 1
            self.train_one_game()

            if self.score > 50:
                self.model.save(get_models_dir(), str(self.score) + ".pth")

            plot_scores.append(self.score)
            total_score += self.score
            mean_score = total_score / self.n_games
            plot_mean_scores.append(mean_score)

            print('Game', self.n_games, 'Score', self.score, 'Record:', self.best_score)

        plot(plot_scores, plot_mean_scores, self.best_score, get_figures_dir())

    def train_one_game(self):
        self.start_run()
        while self.is_alive():
            self.next_tick()
        self.train_long_memory()

    def next_tick(self):
        deep_q_state_old = get_deep_q_state(self.get_state())
        final_move = self.choose_next_move(deep_q_state_old)
        adapted_move = adapt_move(final_move, self.get_direction())

        self.set_next_move(adapted_move)
        self.move_snake()

        deep_q_state_new = get_deep_q_state(self.get_state())
        reward = self.get_reward()

        done = not self.is_alive()
        self.train_short_memory(deep_q_state_old, final_move, reward, deep_q_state_new, done)
        self.remember(deep_q_state_old, final_move, reward, deep_q_state_new, done)

    def get_reward(self):
        return 1 if self.food_eaten else -10 if not self.is_alive() else 0


def get_deep_q_state(state):
    grid, score, alive, snake, food, direction, rows, columns = state

    head = snake[0]

    point_l = (head[0] - 1, head[1])
    point_r = (head[0] + 1, head[1])
    point_u = (head[0], head[1] - 1)
    point_d = (head[0], head[1] + 1)

    dir_l = direction == "up"
    dir_r = direction == "down"
    dir_u = direction == "left"
    dir_d = direction == "right"

    state = [
        # Danger straight
        int((dir_r and is_collision(point_r, rows, columns, grid)) or
            (dir_l and is_collision(point_l, rows, columns, grid)) or
            (dir_u and is_collision(point_u, rows, columns, grid)) or
            (dir_d and is_collision(point_d, rows, columns, grid))),

        # Danger right
        int((dir_u and is_collision(point_r, rows, columns, grid)) or
            (dir_d and is_collision(point_l, rows, columns, grid)) or
            (dir_l and is_collision(point_u, rows, columns, grid)) or
            (dir_r and is_collision(point_d, rows, columns, grid))),

        # Danger left
        int((dir_d and is_collision(point_r, rows, columns, grid)) or
            (dir_u and is_collision(point_l, rows, columns, grid)) or
            (dir_r and is_collision(point_u, rows, columns, grid)) or
            (dir_l and is_collision(point_d, rows, columns, grid))),

        # Move direction
        int(dir_l),
        int(dir_r),
        int(dir_u),
        int(dir_d),

        # Food location
        int(food[0] < head[0]),  # food left
        int(food[0] > head[0]),  # food right
        int(food[1] < head[1]),  # food up
        int(food[1] > head[1])  # food down
    ]

    return tuple(state)


def adapt_move(next_move, direction):
    moves = ["right", "down", "left", "up"]
    start_point = moves.index(direction)

    if next_move[0] == 1:
        return adapt_dir(direction)
    if next_move[1] == 1:
        new_point = (start_point + 1) % 4
        return adapt_dir(moves[new_point])
    if next_move[2] == 1:
        new_point = (start_point - 1) % 4
        return adapt_dir(moves[new_point])


def adapt_dir(direction):
    moves_map = {"right": RIGHT, "down": DOWN, "left": LEFT, "up": UP}
    return moves_map[direction]


def get_random_move():
    final_move = [0, 0, 0]
    move = random.randint(0, 2)
    final_move[move] = 1
    return final_move


def predict_next_move(deep_q_state, model):
    final_move = [0, 0, 0]
    state = torch.tensor(deep_q_state, dtype=torch.float)
    prediction = model(state)
    move = torch.argmax(prediction).item()
    final_move[move] = 1
    return final_move
