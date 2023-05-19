import os

import matplotlib.pyplot as plt
from IPython import display

plt.ion()


def plot(scores, mean_scores, highest_score, dir_name):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    save_figure(highest_score, dir_name)


def save_figure(highest_score, dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    plt.savefig(dir_name + str(highest_score))
