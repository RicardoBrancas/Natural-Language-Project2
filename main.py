#!/bin/python3

import numpy as np
import gensim
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import csv
from sklearn.preprocessing import normalize
import argparse
import sklearn.metrics.pairwise

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

parser = argparse.ArgumentParser(description='Classificador de questões relacionadas com cinema.')
parser.add_argument('train_file', metavar='TRAINING')
parser.add_argument('test_file', metavar='TESTING')

args = parser.parse_args()

classes = {}
obs = {}

with open(args.train_file, 'r') as file:
    reader = csv.reader(file, delimiter='\t')

    for row in reader:
        category = row[0]
        line = row[1]
        tokenizer = RegexpTokenizer(r'[A-Za-z]+')
        tokens = tokenizer.tokenize(line)

        sb = np.zeros((300))
        filtered_words = filter(lambda token: token not in stopwords.words('english'), tokens)
        for token in filtered_words:
            sb += model[token]

        sb = sb / np.linalg.norm(sb)
       
        if category in classes:
            classes[category] += sb
            obs[category] += 1
        else:
            classes[category] = sb
            obs[category] = 1

    for key, value in classes.items():
        classes[key] = value / obs[key]

with open(args.test_file, 'r') as file:
    reader = csv.reader(file, delimiter='\t')

    for row in reader:
        line = row[0]
        tokenizer = RegexpTokenizer(r'[A-Za-z]+')
        tokens = tokenizer.tokenize(line)

        sb = np.zeros((300))

        filtered_words = filter(lambda token: token not in stopwords.words('english'), tokens)
        for token in filtered_words:
            sb += model[token]

        sb = sb / np.linalg.norm(sb)

        max_key = None
        max_similarity = 0
        for key, value in classes.items():
            similarity = sklearn.metrics.pairwise.cosine_similarity(value.reshape(1, -1), sb.reshape(1, -1))

            if similarity > max_similarity:
                max_similarity = similarity
                max_key = key

        print(max_key)


