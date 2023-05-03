from gameModule import GUISnakeGame
from QLearningAgent import SnakeQAgent
from QLearningAgent import SnakeQAgent

def main():
    game = GUISnakeGame()
    game.init_pygame()
    agent = SnakeQAgent(game)
    agent.train()

if __name__ == "__main__":
    main()
