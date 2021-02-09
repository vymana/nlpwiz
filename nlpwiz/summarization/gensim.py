from gensim.summarization import summarizer

def summarize(text, ratio=0.1):
    return summarizer.summarize(text, ratio=ratio)
