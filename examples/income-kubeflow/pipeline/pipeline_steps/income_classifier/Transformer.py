
import dill
import logging

class Transformer(object):
    def __init__(self):

        with open('/mnt/income_class.model', 'rb') as model_file:
            self.clf_model = dill.load(model_file)

        with open('/mnt/preprocessor.model', 'rb') as prepross_file:
            self.preprocessor = dill.load(prepross_file)

        self.class_names = ['<=$50K', '>$50K']
        self.feature_names = ['Age', 'Workclass', 'Education', 'Marital Status', 'Occupation', 'Relationship',
                              'Race', 'Sex', 'Capital Gain', 'Capital Loss', 'Hours per week', 'Country']

    def predict(self, X, feature_names):
        logging.warning(X)
        prediction = self.clf_model.predict_proba(self.preprocessor.transform(X))
        logging.warning(prediction)
        return prediction


