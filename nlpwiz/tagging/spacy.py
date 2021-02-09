import re
import multiprocessing

import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

PUNCTS = ['-', '$', '&', '.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{','}']
STOP_WORDS = set(stopwords.words('english'))
SEPERATORS = [" ", "\t", "\n"]
STOP_WORDS.update(PUNCTS)

spacy_model = spacy.load('en')
stemmer=SnowballStemmer("english")

def parse(text):
    """
    spacy annotations: https://stackoverflow.com/questions/40288323/what-do-spacys-part-of-speech-and-dependency-tags-mean
    """
    text = re.sub('[ ]+', ' ', text).strip()  # Convert multiple whitespaces into one

    doc = spacy_model(text)
    token_tags = []
    for tok in doc:
        tags = {"text": tok.text, "lemma": tok.lemma_, "pos": tok.pos_,
                "tag": tok.tag_, "dep": tok.dep_, "shape": tok.shape_,
                     "is_alpha": tok.is_alpha, "is_stop": tok.is_stop}
        token_tags.append(tags)

    return token_tags


def lemmatize(text):
    doc = spacy_model(text)
    return [tok.lemma_ for tok in doc]


def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)

    return [stemmer.stem(t) for t in filtered_tokens]
