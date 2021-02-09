import pytest
import os

from nlpwiz.ir import tfidf

"""
def test_tfidf():
    data_path = os.path.join("tests", "test_data", "wikiqa_test_questions.txt")
    model_file =os.path.join("tests", "test_data", "tfidf_test.model")

    model = tfidf.TFIDF(data_path=data_path)
    model.train(model_path=model_file)

    model = tfidf.TFIDF(data_path=data_path, model_path=model_file)
    txt = "how a water pump works"
    sim_doc_ids = [sim["id"] for sim in model.similar_docs(doc=txt)]
    assert 2 == sim_doc_ids[0] #the same query should be the first result
"""