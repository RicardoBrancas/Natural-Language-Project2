#!/bin/python3

import numpy as np
import gensim
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import csv
from sklearn.preprocessing import normalize


model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

with open('corpora/QuestoesConhecidas.txt', 'r') as file:
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
       

        
