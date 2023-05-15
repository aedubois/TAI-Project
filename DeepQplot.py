import os
import matplotlib
import matplotlib.pyplot as plt

from IPython import display

plt.ion()

def deep_plot(scores, mean_scores, highest_score):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    save_figure(highest_score)

def save_figure(highest_score):
    figures_dir = "deep_q_figures/"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    plt.savefig(figures_dir + str(highest_score))