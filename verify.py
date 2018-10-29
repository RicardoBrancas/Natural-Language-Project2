#!/bin/python

import sklearn.metrics
from csv import reader
import argparse

parser = argparse.ArgumentParser(description='Classificador de questões relacionadas com cinema.')
parser.add_argument('obt_file', metavar='OBTAINED')
parser.add_argument('exp_file', metavar='EXPECTED')

args = parser.parse_args()

obt_file = open(args.obt_file, 'r')
exp_file = open(args.exp_file, 'r')


obt_vec = [line[0].strip() for line in reader(obt_file)]
exp_vec = [line[0].strip() for line in reader(exp_file)]

confusion_matrix = sklearn.metrics.confusion_matrix(obt_vec, exp_vec)

print("Confusion matrix:\n " + str(confusion_matrix))





