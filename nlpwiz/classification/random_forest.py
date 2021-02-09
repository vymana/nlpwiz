import os
import pickle
import logging

from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

class RandomForest:
    def __init__(self, model_path=None, n_estimators=500, criterion='entropy', max_depth=10, min_samples_leaf=1,
                 max_features=0.4, n_jobs=4):
        self.model_path = model_path
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs

        if model_path is None or not os.path.exists(model_path):
            self.model = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                       max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                       max_features=self.max_features, n_jobs=self.n_jobs)

        else:
            logger.info("loading existing model {}".format(self.model_path))
            self.model = pickle.load(open(model_path, 'rb'))

    def train(self, X, y):
        self.model.fit(X, y)
        logger.info("writing trained model to {}".format(self.model_path))
        pickle.dump(self.model, open(self.model_path, 'wb'))

    def predict(self, X_test):
        preds = self.model.predict_proba(X_test)[:, 1]
        return preds
