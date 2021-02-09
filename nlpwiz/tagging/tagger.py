
import re
import multiprocessing

from nltk.corpus import stopwords

from nlpwiz.tagging import spacy
from nlpwiz.ie import ner

PUNCTS = ['-', '$', '&', '.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{','}']
STOP_WORDS = set(stopwords.words('english'))
SEPERATORS = [" ", "\t", "\n"]
STOP_WORDS.update(PUNCTS)


def annotate_record(record, features=["ner"], text_field="text"):
    text = record[text_field]
    record[text_field+".tags"] = spacy.parse(text)
    if "ner" in features:
        record[text_field+".ner"] = ner.ner(text)
    if "words" in features:
        splits = re.split(r'\W+', text)
        record[text_field+".words"] = [w for w in splits if not (w in STOP_WORDS)]
    return record



def annotate_records(records, features=["ner"], text_field="text", parallel=True, cpu_count=8):
    if parallel:
        pool = multiprocessing.Pool(processes=cpu_count)
        results = pool.starmap(annotate_record,
                           [(record, features, text_field) for record in records])
        pool.close()
    else:
        results = []
        for record in records:
            result = annotate_record(record, features, text_field)
            results.append(result)
    return results
