from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import numpy as np

def split_data(data_tbl, features, test_size=0.2, seed=None):
    """Return a tuple of length four with train data, test data, train feature, test feature."""
    return train_test_split(data_tbl, features, test_size=test_size, random_state=seed)


def train_model(data_tbl, features, microbes, method='random_forest', n_estimators=20, n_neighbours=21):
    """Return a trained model to predict features from data."""
    if (method == "random_forest"):
        classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
    elif (method == "gaussian"):
        kernel_gpc = 1.0 * RBF(1.0)
        classifier = GaussianProcessClassifier(kernel=kernel_gpc, random_state=0)
    elif (method == "knn"):
        classifier = KNeighborsClassifier(n_neighbors=n_neighbours)
    elif (method == "linear_svc"):
        classifier = svm.SVC(kernel='linear', probability=True)
    elif (method == "svm"):
        classifier = svm.SVC(
            gamma='scale', decision_function_shape='ovo', kernel="rbf", probability=True
        )
    elif (method == "neural_network"):
        classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2))
    classifier.fit(data_tbl, features)
    if(method=="random_forest"):
        importances = classifier.feature_importances_
        indices = np.argsort(importances)
        for importance, microbe_names in zip(importances, microbes):
            if (importance>0):
                print (microbe_names, "=", importance)		
    return classifier


def predict_with_model(model, data_tbl):
    """Return a dictionary with evaluation data for the model on the data."""
    return model.predict(data_tbl)


def multi_predict_with_model(model, data_tbl):
    """Return a dictionary with evaluation data for all the classes of the model on the data."""
    return model.predict_proba(data_tbl)


def predict_top_classes(model, data_tbl, features, top_hits=[1, 2, 3, 5, 10]):
    prediction = multi_predict_with_model(model, data_tbl)
    hit_values = []
    for i in top_hits:
        top_n_hits = np.argsort(-prediction, axis=1)[:, :i]
        hits = 0
        for i, val in enumerate(features):
            hits += 1 if val in top_n_hits[i] else 0
        hit_values.append(hits / len(features))
    return hit_values
