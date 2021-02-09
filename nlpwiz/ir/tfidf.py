import os
import logging
import json

from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from nlpwiz.utils import text_utils
from nlpwiz.base import ModelBase

logger = logging.getLogger(__name__)

class TFIDF(ModelBase):
    def __init__(self, model_path=None, data_path=None):
        """
        :param model_path:
        :param data_path: contains sentences to train the model on or on which the model was trained
        """
        self.knn, self.vectorizer, self.transformer = None, None, None
        self.data_path = data_path
        self.docs = []
        self.num_nn = 10
        #self.pretrained_emb = None
        if model_path:
            self.load(model_path)
        if data_path is not None:
            self.docs = [text_utils.process_text(t) for t in open(data_path, 'r').readlines()]


    def load(self, model_path=None):
        model_path = model_path or self.model_path
        super(TFIDF, self).load(model_path)
        self.knn, self.vectorizer, self.transformer = self.model


    def save(self, model_path=None):
        self.model = [self.knn, self.vectorizer, self.transformer]
        super(TFIDF, self).save(model_path)


    def train(self, model_path):
        num_nn = self.num_nn if len(self.docs) >= self.num_nn else len(self.docs)
        logger.info("training tfidf on {} docs".format(len(self.docs)))

        self.vectorizer = CountVectorizer()
        vectorizer_fit = self.vectorizer.fit_transform(self.docs)

        self.transformer = TfidfTransformer()
        X = self.transformer.fit_transform(vectorizer_fit)

        self.knn = NearestNeighbors(n_neighbors=num_nn, n_jobs=5)
        self.knn.fit(X)

        # save model
        model_path = model_path or self.model_path
        logger.info("saving model {}".format(model_path))
        self.model = [self.knn, self.vectorizer, self.transformer]
        self.save(model_path)

    def similar_docs(self, doc=None, docs=[], count=10):
        """
        return documents most similar to input documents set
        """
        #import ipdb; ipdb.set_trace()
        if doc is not None:
            docs = [doc]
        docs = [text_utils.lemmatize_text(doc) for doc in docs]
        vec = self.vectorizer.transform(docs)
        tvec = self.transformer.transform(vec)
        sims, docids = self.knn.kneighbors(tvec, return_distance=True)
        #return [self.docs[docid] for docid in docids[0][:count]], [1-sim for sim in sims[0][:count]]
        results = []
        for idx in range(len(docids[0])):
            docid = docids[0][idx]
            results.append({
                "id": docid,
                "text": self.docs[docid],
                "score": 1-sims[0][idx], #distance to similarity
            })
        results = sorted(results, key=lambda x: -x["score"])
        return results[:count]
