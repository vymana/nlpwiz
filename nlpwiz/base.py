
import pickle

class ModelBase():
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None

    def load(self, model_path=None):
        model_path = model_path or self.model_path
        self.model = pickle.load(open(model_path, "rb"))

    def save(self, model_path=None):
        model_path = model_path or self.model_path
        pickle.dump(self.model, open(model_path, "wb"))
