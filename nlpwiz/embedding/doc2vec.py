import os
import logging

import gensim.models as g

from nlpwiz.utils import text_utils

logger = logging.getLogger(__name__)


class Doc2Vec:
    """
    TODO: Use pretrained word2vec embeddings - https://github.com/RaRe-Technologies/gensim/issues/1338
    """
    def __init__(self, model_path=None, data_path=None, binary=False, mode="eval"):
        """
        :param model_path:
        :param data_path: contains sentences to train the model on or on which the model was trained
        :param binary:
        """
        self.model = None
        self.model_name = "doc2vec"
        self.data_path = data_path
        self.docs = []
        self.preprocessed_path = None
        self.pretrained_emb = None
        self.model_path = None
        if model_path and os.path.exists(model_path):
            self.model = self.load(model_path, binary=binary)
            self.model_path = model_path
        if data_path is not None:
            #self.dataset = dataset.DataSet()
            #self.dataset.load(data_path)
            self.docs = [text_utils.process_text(t) for t in open(data_path, 'r').readlines()]
            if mode == "train":
                #preprocess text only for training
                logger.info("loading data {}".format(data_path))
                self.preprocessed_docs = self.docs
                self.preprocessed_path = os.path.join(os.path.dirname(data_path), "preprocessed")
                with open(self.preprocessed_path, 'w') as fp:
                    fp.write("\n".join(self.preprocessed_docs))

    def load(self, model_path, binary=False):
        logger.info("loading {}".format(model_path))
        self.model = g.doc2vec.Doc2Vec.load(model_path)
        return self.model

    def train(self, model_path=None, vector_size=300, window_size=15, min_count=1, sampling_threshold=1e-5, negative_size=5, worker_count=10, num_epochs=1000):
        dm = 0  # 0 = dbow; 1 = dmpv
        hs = 1 #hierarchical softmax
        sampling_threshold = 0
        logger.info("training doc2vec on {}".format(self.preprocessed_path))
        docs = g.doc2vec.TaggedLineDocument(self.preprocessed_path)
        #model = g.Doc2Vec(docs, vector_size=vector_size, window=window_size, min_count=min_count, sample=sampling_threshold, workers=worker_count, hs=0, dm=dm, negative=negative_size, dbow_words=1, dm_concat=1, pretrained_emb=self.pretrained_emb, iter=train_epoch)
        model = g.Doc2Vec(docs, vector_size=vector_size, window=window_size, min_count=min_count,
                        workers=worker_count, hs=hs, dm=dm, dbow_words=1, dm_concat=1, pretrained_emb=self.pretrained_emb, iter=num_epochs)

        #model.build_vocab(docs)
        #model.train(docs, total_examples=len(docs), epochs=model.iter)

        model_path = model_path or self.model_path
        if model_path:
            # save model
            logger.info("saving model {}".format(model_path))
            model.save(model_path)

        self.model = model
        return model

    def infer(self, infile=None, inlines=None, outfile=None, start_alpha=0.01, infer_epoch=1000):
        if infile is not None:
            inlines = open(infile, 'r').readlines()

        inlines_words = [text_utils.lemmatize_text(doc).split() for doc in inlines]
        doc_vectors = []
        for inline_words in inlines_words:
            #self.model.random.seed(1)
            doc_vector = self.model.infer_vector(inline_words, alpha=start_alpha, steps=infer_epoch)
            doc_vectors.append(doc_vector)

        if outfile is not None:
            with open(outfile, 'w') as fout:
                fout.write(" ".join([str(x) for x in doc_vectors]) + "\n")
        return doc_vectors

    def similar_docs(self, doc, count=10):
        doc = text_utils.lemmatize_text(doc)
        words = doc.split()
        #sims = self.model.wv.most_similar(positive=words)  # gives you top 10 document tags and their cosine similarity
        #self.model.random.seed(1)
        self.model.docvecs.init_sims(replace=True)
        new_vector = self.model.infer_vector(words)
        sims = self.model.docvecs.most_similar([new_vector], topn=count)  # gives you top 10 document tags and their cosine similarity
        results = []
        for sim in sims:
            results.append({
                "id": sim[0],
                "score": sim[1],
                "text": self.docs[sim[0]],
            })
        return results
