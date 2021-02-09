import pytest
import os

from nlpwiz.embedding import doc2vec

"""
def test_doc2vec():
    data_path = os.path.join("tests", "test_data", "wikiqa_test_questions.txt")
    model_file =os.path.join("tests", "test_data", "doc2vec_test.model")

    model = doc2vec.Doc2Vec(data_path=data_path)
    model.train(model_path=model_file)

    model = doc2vec.Doc2Vec(data_path=data_path, model_path=model_file)
    txt = "how a water pump works"
    sim_doc_ids = [sim["id"] for sim in model.similar_docs(doc=txt)]
    assert 2 in sim_doc_ids
"""