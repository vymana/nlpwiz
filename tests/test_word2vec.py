import pytest

from nlpwiz.embedding import word2vec


@pytest.fixture()
def word2vec_model():
    model = word2vec.Word2Vec()
    return model


def test_sentence_similarity(word2vec_model):
    txt = "This is a test"
    sim = word2vec_model.sentence_similarity(txt, txt)
    assert sim == 1.0
