"""
requires the server to be running
"""

import pytest

from nlpwiz.server import client


def test_test():
    res = client.model_test("test")
    assert res == "test method"


def test_word2vec():
    txt = "Somewhere over the rainbow"
    sim = client.run("sentence_similarity", kwargs={"text1": txt, "text2": txt})
    assert sim == 1.0


