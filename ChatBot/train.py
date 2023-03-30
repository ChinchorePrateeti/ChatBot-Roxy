
from nltk.stem import WordNetLemmatizer
import random  # to create random response from the list
import json
import pickle  # for serialization
import numpy as np
import tensorflow

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential

import nltk
# nltk.download()


# this import reduces the word to its root form


# lemmatizing individual words
lemmatizer = WordNetLemmatizer

# reading contents of json file as text, then passing that text to loads function, we get json object which is a dictionary.
intents = json.loads(open('conversation.json').read())

words = []
classes = []
documents = []
ignore_symbols = [",", "?", ".", "!"]

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordlist = nltk.word_tokenize(pattern)
        words.append(wordlist)
        documents.append((wordlist, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
print(documents)


words = [lemmatizer.lemmatize(word)
         for word in words if word not in ignore_symbols]
words = sorted(set(words))
print(words)
