from nlpwiz.tagging import spacy

def process_text(txt):
    txt = txt.strip(" \n").lower()
    return txt

def lemmatize_text(t):
    t = t.strip(" \n").lower()
    return " ".join(spacy.lemmatize(t))
