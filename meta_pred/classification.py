import numpy as np
import pandas as pd
from time import time
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    LeaveOneGroupOut,
    cross_val_score,
)
from sklearn.metrics import accuracy_score
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    VotingClassifier
)
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def split_data(data_tbl, features, test_size=0.2, seed=None):
    """Return a tuple of length four with train data, test data, train feature, test feature."""
    return train_test_split(data_tbl, features, test_size=test_size, random_state=seed)

def get_classifier(data_tbl, features, method='random_forest', n_estimators=1000, n_neighbours=21, seed=None):
    """Fits the model with chosen classifier along with given parameters."""
    if method == "random_forest":
        classifier = RandomForestClassifier(n_estimators=n_estimators, criterion="entropy", bootstrap=True, random_state=seed)
    elif method == "decision_tree":
        classifier = DecisionTreeClassifier(random_state=seed)
    elif method == "extra_tree":
        classifier = ExtraTreesClassifier(n_estimators=n_estimators, criterion='entropy', random_state=seed)
    elif method == "adaboost":
        classifier = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=4)
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
    elif method == "logistic_regression":
        classifier = LogisticRegression(solver='lbfgs', C=1e5, max_iter= 1000000)
    elif method == "LDA":
        classifier = LinearDiscriminantAnalysis(solver='svd')
    elif method == "mixed":
    	clf1 = LogisticRegression(solver='lbfgs', C=1e5, max_iter= 1000000)
    	clf2 = RandomForestClassifier(n_estimators=n_estimators, criterion="entropy", bootstrap=True, warm_start=True, random_state=seed)
    	clf3 = svm.SVC(kernel='linear', probability=True)
    	classifier = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('lsvc', clf3)], voting='soft')
    return classifier


def k_fold_crossvalid(data_tbl, features, method='random_forest', n_estimators=1000, n_neighbours=21, k_fold=10, seed=None):
    """Return a model for saving after training using k-fold cross-validation."""
    scores = []
    classifier_score = 0
    classifier = get_classifier(data_tbl, features, method=method, n_estimators=n_estimators, n_neighbours=n_neighbours, seed=None)
    cv = StratifiedKFold(n_splits=k_fold, shuffle=False)
    for train_index, test_index in cv.split(data_tbl, features):
        X_train, X_test = data_tbl[train_index], data_tbl[test_index]
        y_train, y_test = features[train_index], features[test_index]
        classifier.fit(X_train, y_train)
        scores.append(classifier.score(X_test, y_test))
        if classifier_score < classifier.score(X_test, y_test):
            classifier_score = classifier.score(X_test, y_test)
    return np.mean(scores), np.std(scores), classifier_score

def leave_one_group_out(data_tbl, features, group_name, method='random_forest', n_estimators=1000, n_neighbours=21, seed=None):
    scores = []
    classifier_score = 0
    classifier = get_classifier(data_tbl, features, method=method, n_estimators=n_estimators, n_neighbours=n_neighbours, seed=None)
    leave_one_group = LeaveOneGroupOut()
    for train_index, test_index in leave_one_group.split(data_tbl, y=features, groups=group_name):
        X_train, X_test = data_tbl[train_index], data_tbl[test_index]
        y_train, y_test = features[train_index], features[test_index]
        print (X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        classifier.fit(X_train, y_train)
        scores.append(classifier.score(X_test, y_test))
        if classifier_score < classifier.score(X_test, y_test):
            classifier_score = classifier.score(X_test, y_test)
    return np.mean(scores), np.std(scores), classifier_score

def train_model(data_tbl, features, method='random_forest', n_estimators=1000, n_neighbours=21, seed=None):
    """Return a trained model to predict features from data."""
    t0 = time()
    print (type(data_tbl), type(features), data_tbl.shape, features.shape)
    classifier = get_classifier(data_tbl, features, method=method, n_estimators=n_estimators, n_neighbours=n_neighbours, seed=None)
    classifier.fit(data_tbl, features)
    return classifier, (time() - t0)


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
