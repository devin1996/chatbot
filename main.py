import nltk
from nltk.stem.lancaster import LancasterStemmer
from numpy.polynomial.tests.test_classes import classes

stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json

with open("intents.json") as file:
    data = json.load(file)

# print(data)
# print(data["intents"])
words = []
labels = []
docs_x = []
#what intents it a part of
docs_y = []


for intent in data["intents"]:
    for pattern in intent["pattern"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(pattern)
        docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(classes))]

for x, doc in enumerate(docs_x)
    bag = []

    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            # if mensioned words are available it will become 1
            bag.append(1)
        else:
            bag.append(0)
output_row = list(out_empty)