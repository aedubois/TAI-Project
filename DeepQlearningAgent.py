import os
import numpy as np
import random
import torch
from Model import Qnet, QTrainer
from gameModule import SnakeGame, RIGHT, LEFT, UP, DOWN
from collections import deque
from game import SnakeGameAI
from enum import Enum
from collections import namedtuple

MOVES = [LEFT, RIGHT, UP, DOWN]
DMOVES = [[1,0,0],[0,1,0],[0,0,1]]

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
class DeepQlearningAgent:
    def __init__(self, deep_q_model_file_name="None", game=None):

        self.current_game = game
        self.discount_rate = 0.90
        self.eps = 1
        self.eps_discount = 0.992
        self.min_eps = 0.001
        self.num_episodes = 10_000
        self.moves = [LEFT, RIGHT, UP, DOWN]
        self.model = Qnet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.discount_rate)

        self.memory = deque(maxlen=MAX_MEMORY)

        if deep_q_model_file_name != "None":
            self.model = self.model.load_model(deep_q_model_file_name, 11, 256, 3)


    def choose_next_move(self, state):
        final_move = [0, 0, 0]
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move[move] = 1
        return final_move
    def convert_move(self,next_move):
        moves = ["right", "down", "left", "up"]
        direct = self.current_game.get_direction()
        start_point = moves.index(direct)
        if next_move[0] == 1:
            return self.convert_dir(direct)
        if next_move[1] == 1:
            new_point = (start_point+1)%4
            return self.convert_dir(moves[new_point])
        if next_move[2] == 1:
            new_point = (start_point-1)%4
            return self.convert_dir(moves[new_point])

    def convert_dir(self, direction):
        if direction == "left":
            return LEFT
        elif direction == "right":
            return RIGHT
        elif direction == "up":
            return UP
        elif direction == "down":
            return DOWN
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

    def train(self):
        highest_score = 0

        self.current_game = TrainingSnakeGame()
        self.current_game.train_one_game(self)

        print(f"Episode finished. Highest_Score: {highest_score}. Current_Score: {self.current_game.score}", "current espilon: ", self.eps)



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

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')
class TrainingSnakeGame(SnakeGameAI):
    def __init__(self):
        super(TrainingSnakeGame, self).__init__()



    def train_one_game(self, agent):
        agent.eps = max(agent.eps * agent.eps_discount, agent.min_eps)
        record = 0
        n_games = 0
        while True:
            n_games, record = self.next_tick(agent, record, n_games)

    def next_tick(self, agent, record, n_games):
        current_state = agent.get_state(self)

        next_move = self.get_action(current_state,n_games, agent)

        reward, done, score = self.play_step(next_move)

        new_state = agent.get_state(self)

        # train short memory
        agent.train_short_memory(current_state, next_move, reward, new_state, done)

        agent.remember(current_state, next_move, reward, new_state, done)

        if done:
            n_games += 1
            self.reset()
            agent.train_long_memory()
            if score > record:
                record = score
                agent.model.save()
            print('Game', n_games, 'Score', score, 'Record:', record)
        return n_games,record


    def convert_move(self,next_move):
        moves = ["right", "down", "left", "up"]
        direct = self.get_direction()
        start_point = moves.index(direct)
        if next_move[0] == 1:
            return self.convert_dir(direct)
        if next_move[1] == 1:
            new_point = (start_point+1)%4
            return self.convert_dir(moves[new_point])
        if next_move[2] == 1:
            new_point = (start_point-1)%4
            return self.convert_dir(moves[new_point])

    def convert_dir(self, direction):
        if direction == "left":
            return LEFT
        elif direction == "right":
            return RIGHT
        elif direction == "up":
            return UP
        elif direction == "down":
            return DOWN

    # epsilon-greedy action choice
    def get_next_move(self, state, agent):
        if random.random() < agent.eps:

            return DMOVES[random.choice([0, 1, 2])]
        else:
            return agent.choose_next_move(state)

    # TODO: consider more cases like the possibility of a near body collision,
    #  the age of the agent, and such
    def get_reward(self):
        if self.foodEaten:
            return 10
        if not self.is_alive():
            return -10
        return -0.1

    def get_action(self, state, n_games, agent):
        # random moves: tradeoff exploration / exploitation
        epsilon = 160 - n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = agent.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

