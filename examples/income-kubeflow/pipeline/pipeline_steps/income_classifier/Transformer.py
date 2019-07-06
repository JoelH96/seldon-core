
import dill
import logging

class Transformer(object):
    def __init__(self):

        with open('/mnt/income_class.model', 'rb') as model_file:
            self.clf_model = dill.load(model_file)

    def predict(self, X, feature_names):
        logging.warning(X)
        prediction = self.clf_model.predict_proba(X)
        logging.warning(prediction)
        return prediction


