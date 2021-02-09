import pytest

import os
import csv

from nlpwiz.classification import naive_bayes

@pytest.fixture()
def imdb_sent_data():
    data_path = os.path.join("tests", "test_data", "classification", "imdb_labelled.txt")
    data_lines = [line.strip() for line in open(data_path).readlines()]
    data = [(line[:-1].strip(), int(line[-1])) for line in data_lines]
    return data

def test_naive_bayes(imdb_sent_data, train_test_split=[0.8, 0.2]):
    l = len(imdb_sent_data)
    train_data = imdb_sent_data[int(l*train_test_split[0]):]
    test_data = imdb_sent_data[:int(l*train_test_split[1])]

    train_texts = [d[0] for d in train_data]
    train_labels = [d[1] for d in train_data]
    test_texts = [d[0] for d in test_data]
    test_labels = [d[1] for d in test_data]

    classifier,corpus = naive_bayes.train(train_texts, train_labels)
    accuracy = naive_bayes.test(classifier, corpus, test_texts, test_labels)
    assert accuracy >=0.5
