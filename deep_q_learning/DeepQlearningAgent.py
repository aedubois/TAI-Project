import os
import random
from collections import deque

import torch

from deep_q_learning.model import LinearQNet, QTrainer
from gameModule import SnakeGame, is_collision, RIGHT, DOWN, LEFT, UP
from helpers import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


def get_models_dir():
    return os.path.join(os.path.dirname(__file__), "models/")


def get_figures_dir():
    return os.path.join(os.path.dirname(__file__), "figures/")


def load_model(model_file_name="None"):
    model = LinearQNet(11, 256, 3)
    if model_file_name != "None":
        model.load_state_dict(torch.load(get_models_dir() + model_file_name))
    return model


class DeepQLearningAgent:
    def __init__(self, model_file_name="None"):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = load_model(model_file_name)

        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_next_move(self, deep_q_state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(deep_q_state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def choose_next_move(self, state):
        deep_q_state = get_deep_q_state(state)
        direction = state[5]

        final_move = [0, 0, 0]
        state0 = torch.tensor(deep_q_state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move[move] = 1
        final_move = adapt_move(final_move, direction)
        return final_move

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


NUM_TRAINING_EPISODES = 1_000


def train():
    game = SnakeGame()
    agent = DeepQLearningAgent("None")

    game.start_run()
    total_score = 0
    plot_scores = []
    plot_mean_scores = []
    while True:
        deep_q_state_old = get_deep_q_state(game.get_state())
        final_move = agent.train_next_move(deep_q_state_old)

        adapted_move = adapt_move(final_move, game.get_direction())
        game.set_next_move(adapted_move)

        game.move_snake()
        deep_q_state_new = get_deep_q_state(game.get_state())
        reward = get_reward(game.is_alive(), game.food_eaten)

        done = not game.is_alive()

        agent.train_short_memory(deep_q_state_old, final_move, reward, deep_q_state_new, done)
        agent.remember(deep_q_state_old, final_move, reward, deep_q_state_new, done)

        score = game.score
        best_score = game.best_score

        if done:
            game.start_run()

            agent.n_games += 1
            agent.train_long_memory()

            if score > 50:
                agent.model.save(get_models_dir(), "state_dict" + str(score) + ".pth")

            print('Game', agent.n_games, 'Score', score, 'Record:', best_score)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

        if agent.n_games > NUM_TRAINING_EPISODES:
            plot(plot_scores, plot_mean_scores, best_score, get_figures_dir())
            break


class TrainingSnakeGame(SnakeGame):
    def __init__(self):
        super(TrainingSnakeGame, self).__init__()

    def set_next_move(self, move):
        adapted_move = adapt_move(move)
        super().set_next_move(adapted_move)


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


def get_reward(is_alive, food_eaten):
    if not is_alive:
        return -10
    elif food_eaten:
        return 1
    else:
        return 0


if __name__ == '__main__':
    train()
