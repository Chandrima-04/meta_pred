import numpy as np
from sklearn.model_selection import (
    train_test_split, 
    KFold
)
from sklearn.ensemble import (
    RandomForestClassifier, 
    ExtraTreesClassifier, 
    AdaBoostClassifier
)
from sklearn.tree import DecisionTreeClassifier 
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def split_data(data_tbl, features, test_size=0.2, seed=None):
    """Return a tuple of length four with train data, test data, train feature, test feature."""
    return train_test_split(data_tbl, features, test_size=test_size, random_state=seed)

def get_classifier(data_tbl, features, method='random_forest', n_estimators=20, n_neighbours=21, n_components=10, seed=None):
    """Fits the model with chosen classifier along with given parameters."""
    if method == "random_forest":
        classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    elif method == "decision_tree":
        classifier = DecisionTreeClassifier(random_state=seed)
    elif method == "extra_tree":
        classifier = ExtraTreesClassifier(n_estimators=n_estimators, criterion='entropy', random_state=seed)
    elif method == "adaboost":
        classifier = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=1)
    elif method == "gaussian":
        kernel_gpc = 1.0 * RBF(1.0)
        classifier = GaussianProcessClassifier(kernel=kernel_gpc, random_state=seed)
    elif method == "gaussianNB":
        classifier = GaussianNB()
    elif method == "knn":
        classifier = KNeighborsClassifier(n_neighbors=n_neighbours)
    elif method == "linear_svc":
        classifier = svm.SVC(kernel='linear', probability=True)
    elif method == "svm":
        classifier = svm.SVC(
            gamma='scale', decision_function_shape='ovo', kernel="rbf", probability=True
        )
    elif method == "neural_network":
        classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2))
    elif method == "LDA":
        classifier = LinearDiscriminantAnalysis(n_components=n_components)
    return classifier
    

def k_fold_crossvalid(data_tbl, features, method='random_forest', n_estimators=20, n_neighbours=21, n_components=10, k_fold=5, seed=None):
    """Return a model for saving after training using k-fold cross-validation."""
    scores = []
    classifier_score = 0
    classifier = get_classifier(data_tbl, features, method=method, n_estimators=n_estimators, n_neighbours=n_neighbours, n_components=n_components, seed=None)
    cv = KFold(n_splits=k_fold, random_state=seed, shuffle=False)
    X = np.array(data_tbl)
    y = np.array(features)
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.fit(X_train, y_train)
        scores.append(classifier.score(X_test, y_test))
        if classifier_score < classifier.score(X_test, y_test):
             X_train_best, X_test_best, y_train_best, y_test_best = X_train, X_test, y_train, y_test
    return (classifier.fit(X_train_best, y_train_best), np.mean(scores), np.std(scores))


def train_model(data_tbl, features, method='random_forest', n_estimators=20, n_neighbours=21, n_components=10, seed=None):
    """Return a trained model to predict features from data."""
    classifier = get_classifier(data_tbl, features, method=method, n_estimators=n_estimators, n_neighbours=n_neighbours, n_components=n_components, seed=None)
    classifier.fit(data_tbl, features)
    return classifier


def predict_with_model(model, data_tbl):
    """Return a dictionary with evaluation data for the model on the data."""
    return model.predict(data_tbl)


def multi_predict_with_model(model, data_tbl):
    """Return a dictionary with evaluation data for all the classes of the model on the data."""
    return model.predict_proba(data_tbl)

def predict_top_classes(model, data_tbl, features):
    """Return the accuracy of the top-most important class."""
    prediction = multi_predict_with_model(model, data_tbl)
    hit_values = []
    for j in (1,2,3,5,10):
        top_n_hits = np.argsort(-prediction, axis=1)[:, :j]
        hits = 0
        for i, val in enumerate(features):
            top_hits = top_n_hits[i]
            if any( top_hits == -1 ): 
                hits += 1 if (val + 1) in top_hits else 0
            else:
                hits += 1 if val in top_hits else 0
        hit_values.append(hits / len(features))
    return hit_values
	
def feature_importance(microbes, model):
    """Return the top features of importance as selected by Random Forest Classifier."""
    importances = model.feature_importances_
    feature_val = sorted(zip(microbes, importances), key=lambda x: x[1], reverse=True)
    return feature_val
