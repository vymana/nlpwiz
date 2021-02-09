"""
Text similarity methods
"""

from sklearn.feature_extraction.text import TfidfVectorizer

from nlpwiz.tagging import spacy


def cosine_similarity(texts):
    """
    cosine similarity of texts
    :param docs: list of texts
    :return: NxN similarity matrix
    """
    tokenized_texts = [" ".join(spacy.tokenize_and_stem(text)) for text in texts]

    vect = TfidfVectorizer(min_df=1)
    tfidf = vect.fit_transform(tokenized_texts)

    #get cosine similarity
    cos_similarity = (tfidf * tfidf.T).A
    return cos_similarity

