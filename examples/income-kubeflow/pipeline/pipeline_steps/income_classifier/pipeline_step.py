import click
import numpy as np
import dill
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from joblib import dump
from alibi.datasets import adult


@click.command()
@click.option('--tabular-data', default="/mnt/tabular_data.data")
@click.option('--preprocessor-path', default="/mnt/preprocessor.model")
@click.option('--model-path', default="/mnt/income_class.model")
@click.option('--out-path', default="/mnt/clf_prediction.data")
@click.option('--action', default="predict", 
        type=click.Choice(['predict', 'train']))

def run_pipeline(
        tabular_data,
        preprocessor_path,
        model_path,
        out_path, 
        action):

    if action == "train":

        # load data
        data, labels, feature_names, category_map = adult()

        # define train and test set
        np.random.seed(0)
        data_perm = np.random.permutation(np.c_[data, labels])
        data = data_perm[:, :-1]
        labels = data_perm[:, -1]

        idx = 30000
        X_train, Y_train = data[:idx, :], labels[:idx]
        X_test, Y_test = data[idx + 1:, :], labels[idx + 1:]

        # feature transformation pipeline
        ordinal_features = [x for x in range(len(feature_names)) if x not in list(category_map.keys())]
        ordinal_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                            ('scaler', StandardScaler())])

        categorical_features = list(category_map.keys())
        categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                                ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(transformers=[('num', ordinal_transformer, ordinal_features),
                                                    ('cat', categorical_transformer, categorical_features)])
        preprocessor.fit(data)

        # train an RF model
        np.random.seed(0)
        clf = RandomForestClassifier(n_estimators=50)
        clf.fit(preprocessor.transform(X_train), Y_train)

        # Store the preprocessor and model
        with open(preprocessor_path, "wb") as pre_process:
            dill.dump(preprocessor, pre_process)

        with open(model_path, "wb") as model_f:
            dill.dump(clf, model_f)

    elif action == "predict":
        with open(model_path, "rb") as model_f:
            clf = dill.load(model_f)

        with open(preprocessor_path, "rb") as pre_process:
            preprocessor = dill.load(pre_process)

        with open(tabular_data, 'rb') as tab_data:
            X_test = dill.load(tab_data)

    # Make a prediction
    y = clf.predict_proba(preprocessor.transform(X_test))

    with open(out_path, "wb") as out_f:
        dill.dump(y, out_f)

if __name__ == "__main__":
    run_pipeline()