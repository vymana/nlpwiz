import pytest

from nlpwiz.lexical import wordnet

def test_annotation():
    word = "happy"
    syns = wordnet.synonyms([word])
    assert word in syns and "glad" in syns[word]


def test_lemmatization():
    word = "happy"
    ants = wordnet.antonyms([word])
    assert word in ants and "unhappy" in ants[word]
