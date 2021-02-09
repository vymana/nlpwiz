
import nltk
from nltk.corpus import wordnet


def synonyms(words):
    """
    get synonyms for the input words 
    """
    synonyms = {}
    for word in words:
        syns = []
        synsets= wordnet.synsets(word)
        for synset in synsets:
            lemmas = synset.lemmas()
            if len(lemmas) > 1:
                syns += [l.name() for l in lemmas if l.name() != word]
        synonyms[word] = syns
    return synonyms


def antonyms(words):
    """
    get antonyms using wordnet synsets
    """
    antonyms = {}
    for word in words:
        ants = []
        synsets = wordnet.synsets(word)
        for synset in synsets:
            syn_ants = []
            for lemma in synset.lemmas():
                for lant in lemma.antonyms():
                    syn_ants.append(lant.name())
            ants += syn_ants
        antonyms[word] = ants
    return antonyms

