import pytest

from nlpwiz.summarization import sumy_summarizers

TEST_TEXT = "Jack went to New York by the morning flight. Fligt got delayed. He was late but he had nothing else to do. He waited on the airport. He still had nothing to do. He was bored and he slept. He missed his flight. He came home as he had nothing to do."

def test_sumy_summarization():
    summary = sumy_summarizers.summarize(TEST_TEXT, 20)
