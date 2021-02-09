# https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html
# https://radimrehurek.com/gensim/tutorial.html

from nltk.corpus import stopwords
import gensim

def gensim_lda(documents, num_topics=10, num_passes=20, language="english"):
    #https://radimrehurek.com/gensim/tut1.html
    stoplist = stopwords.words(language)
    texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
    dictionary = gensim.corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=num_passes)
    ldamodel.print_topics(20)

def topic_modeling():
    #https://de.dariah.eu/tatom/topic_model_python.html
    pass