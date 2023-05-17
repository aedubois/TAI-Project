import torch
import random
from collections import deque
from gameModule import SnakeGame
from model import LinearQNet, QTrainer
from DeepQplot import deep_plot


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class DeepQLearningAgent:
    def __init__(self, model_file_name="None", game=None):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = self.load_model(model_file_name)
        self.game = game

        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def load_model(self, model_file_name="None"):
        model = LinearQNet(11, 256, 3)
        if model_file_name != "None":
            model.load_state_dict(torch.load("model/"+model_file_name))
        return model
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

    def train_next_move(self, state):
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

    def choose_next_move(self,state):
        state = self.game.get_deep_q_state()
        final_move = [0, 0, 0]
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move[move] = 1
        final_move = self.game.adapt_move(final_move)
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
    game = SnakeGame()
    agent = DeepQLearningAgent("None", game)

    game.start_run()
    total_score = 0
    plot_scores = []
    plot_mean_scores = []
    while True:
        state_old = game.get_deep_q_state()
        final_move = agent.train_next_move(state_old)

        adapted_move = game.adapt_move(final_move)
        game.set_next_move(adapted_move)

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
            if score > 30:

                agent.model.save("stateDict" + str(score) + ".pth")

            print('Game', agent.n_games, 'Score', score, 'Record:', best_score)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

        if agent.n_games > 700:
            deep_plot(plot_scores, plot_mean_scores, best_score)
            break


class TrainingSnakeGame(SnakeGame):
    def __init__(self):
        super(TrainingSnakeGame, self).__init__()

    def set_next_move(self, move):
        adapted_move = self.adapt_move(move)
        super().set_next_move(adapted_move)


if __name__ == '__main__':
    train()
