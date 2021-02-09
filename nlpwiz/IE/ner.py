
from nlpwiz.tagging import spacy



def ner(text):
    """
    Using spacy for ner
    """
    doc = spacy.spacy_model(text)
    named_entities = []
    for tok in doc.ents:
        named_entities.append({"text": tok.text, "start": tok.start_char, "end": tok.end_char, "label": tok.label_})
    return named_entities

