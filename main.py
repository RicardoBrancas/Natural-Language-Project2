#!/bin/python3

import numpy as np
import gensim
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import csv
from sklearn.preprocessing import normalize
import argparse
import sklearn.metrics.pairwise
import re

def replaceNames(sentence, oldNames, newName):
    for name in oldNames:
        if name and len(name[0]) > 0:
            if(name[0] in sentence):
                sentence = re.sub(r'\b' + name[0] + r'\b', " " + newName + " ", sentence)
    return sentence
    

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

parser = argparse.ArgumentParser(description='Classificador de questões relacionadas com cinema.')
parser.add_argument('train_file', metavar='TRAINING')
parser.add_argument('test_file', metavar='TESTING')

args = parser.parse_args()

classes = {}
obs = {}

movieNames = list(csv.reader(open('recursos/list_movies.txt'), delimiter='\t'))
characterNames = list(csv.reader(open('recursos/list_characters.txt'), delimiter='\t'))
companyNames = list(csv.reader(open('recursos/list_companies.txt'), delimiter='\t'))
genreNames = list(csv.reader(open('recursos/list_genres.txt'), delimiter='\t'))
jobNames = list(csv.reader(open('recursos/list_jobs.txt'), delimiter='\t'))
peopleNames = list(csv.reader(open('recursos/list_people.txt'), delimiter='\t'))

with open(args.train_file, 'r') as file:
    reader = csv.reader(file, delimiter='\t')

    for row in reader:
        category = row[0]
        line = row[1]

        #Replace key words
        line = replaceNames(line, movieNames, 'movie')
        line = replaceNames(line, characterNames, 'character')
        line = replaceNames(line, genreNames, 'genre')
        line = replaceNames(line, jobNames, 'job')
        line = replaceNames(line, peopleNames, 'person')

        tokenizer = RegexpTokenizer(r'[A-Za-z]+')
        tokens = tokenizer.tokenize(line)

        sb = np.zeros((300))
        filtered_words = filter(lambda token: token not in stopwords.words('english'), tokens)
        for token in filtered_words:
            if token in model:
                sb += model[token]
        norm = np.linalg.norm(sb)
        if norm != 0:
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

        #Replace key words
        line = replaceNames(line, movieNames, 'movie')
        line = replaceNames(line, characterNames, 'character')
        line = replaceNames(line, genreNames, 'genre')
        line = replaceNames(line, jobNames, 'job')
        line = replaceNames(line, peopleNames, 'person')

        tokenizer = RegexpTokenizer(r'[A-Za-z]+')
        tokens = tokenizer.tokenize(line)

        sb = np.zeros((300))

        filtered_words = filter(lambda token: token not in stopwords.words('english'), tokens)
        for token in filtered_words:
            if token in model:
                sb += model[token]

        norm = np.linalg.norm(sb)
        if norm != 0:
            sb = sb / np.linalg.norm(sb)

        max_key = None
        max_similarity = 0
        for key, value in classes.items():
            similarity = sklearn.metrics.pairwise.cosine_similarity(value.reshape(1, -1), sb.reshape(1, -1))

            if similarity > max_similarity:
                max_similarity = similarity
                max_key = key

        print(max_key)


