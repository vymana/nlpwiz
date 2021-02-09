
import os
import sys
import copy
from multiprocessing import Process, Lock


class TestModel:
    def test(self):
        return "TEST MODEL SUCCESS"

def test_model():
    ## Model interfaces
    return TestModel()


# Model Cache

_MODELS_CACHE = {}
_LOADER_LOCKS = {}


##Word2Vec

def word2vec():
    model_key = "word2vec"
    if model_key not in _MODELS_CACHE:
        print("loading word2vec")
        from nlpwiz.embedding import word2vec
        model = word2vec.Word2Vec()
        _MODELS_CACHE[model_key] = model
    return _MODELS_CACHE[model_key]




### utility functions
ulock = Lock()

def get(model_name, *args, **kwargs):
    """
    factory method to load a model in process-safe manner
    """
    ulock.acquire()

    if model_name in _LOADER_LOCKS:
        lock = _LOADER_LOCKS[model_name]
    else:
        lock = Lock()
        _LOADER_LOCKS[model_name] = lock
    lock.acquire()
    try:
        method = eval(model_name)
        model = method(*args, **kwargs)
    finally:
        lock.release()
        ulock.release()
    return model


## Define model functions exposed by the service

def invoke(model_name, method_name, params, model_params={}):
    model=get(model_name, **model_params)
    method = getattr(model, method_name)
    results = method(**params)
    return results


def word_vector(word, model_name="word2vec"):
    model = get(model_name)
    return model.word_vector(word)


def sentence_vector(text, model_name="word2vec"):
    model = get(model_name)
    return model.sentence_vector(text)


def sentence_similarity(text1, text2, model_name="word2vec"):
    model = get(model_name)
    return model.sentence_similarity(text1, text2)


if __name__ == "__main__":
    pass
