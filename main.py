import argparse
import pickle
from pathlib import Path

from a_star.Astar_snake import IA_Astar
from deep_q_learning.DeepQlearningAgent import DeepQLearningAgent
from gameModule import GUISnakeGame
from genetic.dna import Dna
from genetic.genetic_snake import Snake
from q_learning.QLearningAgent import SnakeQAgent


def main():
    parser = argparse.ArgumentParser(description="Snake game.")
    group_play = parser.add_mutually_exclusive_group(required=False)
    group_play.add_argument(
        "-p",
        "--player",
        action="store_true",
        help="Player mode: the player controls the game",
    )
    group_play.add_argument(
        "-x",
        "--ai",
        action="store_true",
        help="AI mode: the AI controls the game (requires an 'algorithm' argument)",
    )
    group_algorithm = parser.add_mutually_exclusive_group(required=False)
    group_algorithm.add_argument(
        "-g",
        "--genetic",
        help="Genetic algorithm: plays a move based of trained neural network, please select weight file",
    )
    group_algorithm.add_argument(
        "-s",
        "--sshaped",
        action="store_true",
        help="S-Shaped algorithm: browses the whole "
             "grid each time in an 'S' shape. Only "
             "works if height of grid is even.",
    )
    group_algorithm.add_argument(
        "-a",
        "--astar",
        action="store_true",
        help="A* algorithm: classical A* algorithm, with "
             "Manhattan distance as heuristic",
    )
    args = parser.parse_args()

    game = GUISnakeGame()
    game.init_pygame()

    q_learning_player = False
    deep_q_learning_player = False

    if args.player:
        agent = None
    elif args.ai:
        if args.astar or args.sshaped:
            agent = IA_Astar(args, game)
        elif args.genetic:
            with open(Path(args.genetic), "rb") as f:
                weights, bias = pickle.load(f)
            agent = Snake(Dna(weights, bias))
        elif q_learning_player:
            agent = SnakeQAgent("73.npy")
        elif deep_q_learning_player:
            agent = DeepQLearningAgent("stateDict55.pth")

    while game.is_running():
        game.next_tick(agent)

    game.cleanup_pygame()


if __name__ == "__main__":
    main()
