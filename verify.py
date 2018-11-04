#!/bin/python

import argparse
from csv import reader
import itertools

import matplotlib
matplotlib.use("Agg")
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize'] = [8.0, 6.0]
matplotlib.rcParams['figure.dpi'] = 80
matplotlib.rcParams['savefig.dpi'] = 100

matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['legend.fontsize'] = 'large'
matplotlib.rcParams['figure.titlesize'] = 'medium'

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def capitalize(snake_str):
    parts = snake_str.split('_')
    return ' '.join([*map(str.title, parts)])

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Matriz de Confusão',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, rotation_mode='anchor', horizontalalignment="right")
    plt.yticks(tick_marks, classes)

    fmt = '{0:.2f}' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j], 2),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Verdadeira categoria')
    plt.xlabel('Categoria prevista')
    plt.tight_layout()

parser = argparse.ArgumentParser(description='Classificador de questões relacionadas com cinema.')
parser.add_argument('exp_file', metavar='EXPECTED')
parser.add_argument('obt_file', metavar='OBTAINED')

args = parser.parse_args()

exp_file = open(args.exp_file, 'r')
obt_file = open(args.obt_file, 'r')

exp_vec = [line.strip() for line in exp_file]
obt_vec = [line.strip() for line in obt_file]

labels = sorted(list(set(exp_vec + obt_vec)))

confusion_matrix = sklearn.metrics.confusion_matrix(exp_vec, obt_vec, labels=labels)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(confusion_matrix, classes=list(map(capitalize, labels)), normalize=False)

plt.savefig('confusion_matrix.pdf', bbox_inches='tight', pad_inches=0)

print("Accuracy:", sklearn.metrics.accuracy_score(exp_vec, obt_vec, normalize=True))






