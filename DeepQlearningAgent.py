import torch
import random
from collections import deque
from gameModule import SnakeGame, LEFT, RIGHT, UP, DOWN
from model import LinearQNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class DeepQLearningAgent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = LinearQNet(11, 256, 3)
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

    def choose_next_move(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

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


def train():
    agent = DeepQLearningAgent()

    game = TrainingSnakeGame()
    game.start_run()

    while True:
        state_old = game.get_deep_q_state()
        final_move = agent.choose_next_move(state_old)
        game.set_next_move(final_move)
        game.move_snake()
        state_new = game.get_deep_q_state()
        reward = game.get_reward()
        done = not game.is_alive()
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        score = game.score
        best_score = game.best_score

        if done:
            game.start_run()

            agent.n_games += 1
            agent.train_long_memory()

            # TODO: only save if the score is a new record
            agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', best_score)


class TrainingSnakeGame(SnakeGame):
    def __init__(self):
        super(TrainingSnakeGame, self).__init__()

    def set_next_move(self, move):
        adapted_move = self.adapt_move(move)
        super().set_next_move(adapted_move)

    def adapt_move(self, next_move):
        moves = ["right", "down", "left", "up"]
        direction = self.get_direction()

        start_point = moves.index(direction)

        if next_move[0] == 1:
            return self.adapt_dir(direction)
        if next_move[1] == 1:
            new_point = (start_point + 1) % 4
            return self.adapt_dir(moves[new_point])
        if next_move[2] == 1:
            new_point = (start_point - 1) % 4
            return self.adapt_dir(moves[new_point])

    def adapt_dir(self, direction):
        moves_map = {"right": RIGHT, "down": DOWN, "left": LEFT, "up": UP}
        return moves_map[direction]

    def get_reward(self):
        if not self.is_alive():
            return -10
        elif self.foodEaten:
            return 1
        else:
            return 0


if __name__ == '__main__':
    train()
