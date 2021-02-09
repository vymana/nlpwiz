import logging

import numpy as np
from scipy.spatial.distance import cosine
from scipy import spatial
import gensim.downloader as api

logger = logging.getLogger(__name__)


def tokenize(text):
    return text.lower().split()


class Word2Vec:
    """
    Wrapper over gensim: https://radimrehurek.com/gensim/models/word2vec.html
    """
    def __init__(self, model_name="glove-wiki-gigaword-100"):
        self.model = api.load(model_name)

    def word_vector(self, word):
        if word in self.model:
            return self.model[word]
        return self.model["unk"]

    def sentence_vector(self, sentence):
        tokens = tokenize(sentence)
        if len(tokens) == 0:
            tokens = ["unk"]
        vectors = [self.word_vector(token) for token in tokens]
        return np.average(np.array(vectors), axis=0).tolist()

    def sentence_similarity(self, sentence1, sentence2):
        s1_afv = self.sentence_vector(sentence1)
        s2_afv = self.sentence_vector(sentence2)
        sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
        return sim
