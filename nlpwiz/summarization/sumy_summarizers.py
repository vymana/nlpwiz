
import spacy

from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

nlp = spacy.load('en')

def _len_summ(text, perc):
    doc1 = nlp(text)
    n_s=0
    for sentence in doc1.sents:
        n_s+=1
        n = n_s*perc//100
    return (n)


def _parse_text(inp_text):
    parser = PlaintextParser.from_string(inp_text,Tokenizer("english"))
    return (parser)


def summarize(text, perc, method="textrank"):
    if method == "textrank":
        summarizer = TextRankSummarizer()
    elif method == "lsa":
        summarizer = LsaSummarizer()
    else:
        raise Exception(f"invalid summarization method - {method}")

    summary_len = _len_summ(text, perc)
    parsed = _parse_text(text)
    summary = summarizer(parsed.document, summary_len)
    return summary
