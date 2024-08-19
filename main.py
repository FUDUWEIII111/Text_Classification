from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
import nltk


def lowercase(text):

    return text.lower()


def punctuation_removal(text):
    translator = str.maketrans('', '', string.punctuation)

    return text.translate(translator)


def tokenize(text):

    return nltk.word_tokenize(text)


def remove_stopword(tokens):
    stop_words = nltk.corpus.stopwords.words('english')

    return [token for token in tokens if token not in stop_words]


def stemming(tokens):
    stemmer = nltk.PorterStemmer()

    return [stemmer.stem(token) for token in tokens]


def process_text(text):
    text = lowercase(text)
    text = punctuation_removal(text)
    tokens = tokenize(text)
    tokens = remove_stopword(tokens)
    tokens = stemming(tokens)

    return tokens


def create_dictionary(messages):
    dictionary = []
    for tokens in messages:
        for token in tokens:
            if token not in dictionary:
                dictionary.append(token)

    return dictionary


def create_features(tokens, dictionary):
    features = np.zeros(len(dictionary))

    for token in tokens:
        if token in dictionary:
            features[dictionary.index(token)] += 1

    return features


data_path = '2cls_spam_text_cls.csv'
df = pd.read_csv(data_path)
messages = df['Message'].values.tolist()
labels = df['Category'].values.tolist()
messages = [process_text(message) for message in messages]
dictionary = create_dictionary(messages)
X = np.array([create_features(tokens, dictionary) for tokens in messages])
le = LabelEncoder()
y = le.fit_transform(labels)
print(f'Classes: {le. classes_}')
