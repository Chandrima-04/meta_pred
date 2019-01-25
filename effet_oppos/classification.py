from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

def split_data(data_tbl, features, test_size=0.2):
    """Return a tuple of length four with train data, test data, train feature, test feature."""
    return train_test_split(data_tbl, features, test_size=test_size)


def train_model(data_tbl, features, method='random_forest', n_estimators=20, n_neighbours=21):
    """Return a trained model to predict features from data."""
    if (method == "random_forest"):
        classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
    elif (method == "gaussian"):
        kernel_gpc = 1.0 * RBF(1.0)
        classifier = GaussianProcessClassifier(kernel=kernel_gpc, random_state=0)
    elif (method == "knn"):
        classifier = KNeighborsClassifier(n_neighbours= n_neighbours)
    elif (method == "svm"):
        classifier = svm.SVC(kernel='linear')
    classifier.fit(data_tbl, features)
    return classifier

def predict_with_model(model, data_tbl):
    """Return a dictionary with evaluation data for the model on the data."""
    return model.predict(data_tbl)

def multi_predict_with_model(model, data_tbl):
    """Return a dictionary with evaluation data for all the classes of the model on the data."""
    return model.predict_log_proba(data_tbl)
