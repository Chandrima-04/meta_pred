import pandas as pd
import numpy as np
from random import randint
import os.path
import click
from itertools import product
from sklearn.metrics import (
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    accuracy_score,
)
from .preprocessing import (
    feature_extraction,
    group_feature_extraction,
    normalize_data,
)
from .classification import (
    split_data,
    get_classifier,
	k_fold_crossvalid,
    leave_one_group_out,
    train_model,
    predict_with_model,
    multi_predict_with_model,
    predict_top_classes,
    feature_importance,
)

MODEL_NAMES = [
    'LDA',
    'random_forest',
    'decision_tree',
    'extra_tree',
    'adaboost',
    'knn',
    'gaussianNB',
    'linear_svc',
    'svm',
    'logistic_regression',
    'neural_network',
]

NORMALIZER_NAMES = [
    'raw',
    'standard_scalar',
    'total_sum',
    'binary'
]

NOISE_VALUES = [
    0,
    0.0000000001,
    0.000000001,
    0.00000001,
    0.0000001,
    0.000001,
    0.00001,
    0.0001,
    0.001,
    0.01,
    0.1,
    1,
    10,
    100,
    1000
]

@click.group()
def main():
    pass

test_size = click.option('--test-size', default=0.2, help='The relative size of the test data')
num_estimators = click.option('--num-estimators', default=100, help='Number of trees in our Ensemble Methods')
num_neighbours = click.option('--num-neighbours', default=21, help='Number of clusters in our knn/MLknn')
n_components = click.option('--n-components', default=100,
              help='Number of components for dimensionality reduction in Linear Discriminant Analysis')
model_name = click.option('--model-name', default='random_forest', help='The model type to train')
normalize_method = click.option('--normalize-method', default='standard_scalar', help='Normalization method')
feature_name = click.option('--feature-name', default='city', help='The feature to predict')
normalize_threshold = click.option('--normalize-threshold', default='0.0001',
                      help='Normalization threshold for binary normalization.')



@main.command('kfold')
@click.option('--k-fold', default=10, help='The value of k for cross-validation')
@test_size
@num_estimators
@num_neighbours
@n_components
@model_name
@normalize_method
@feature_name
@normalize_threshold
@click.option('--test-filename', default="test_sample.csv", help='Filename to save test dataset')
@click.option('--model-filename', default="model_k.pkl", help='Filename to save Model')
@click.argument('metadata_file', type=click.File('r'))
@click.argument('data_file', type=click.File('r'))
@click.argument('out_dir')
def kfold_cv(k_fold, test_size, num_estimators, num_neighbours,  n_components, model_name, normalize_method,
             feature_name, normalize_threshold, test_filename, model_filename, metadata_file, data_file, out_dir):
    """Train and evaluate a model with k-fold cross-validation. echo the model results to stderr."""
    raw_data, microbes, feature, name_map = feature_extraction(data_file, metadata_file, feature_name=feature_name)
    click.echo(f'Training {model_name} using {normalize_method} to predict {feature_name}',err=True)
    tbl, seed = {}, randint(0, 1000)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        os.mkdir(str(out_dir + '/' + 'confusion_matrix'))
    else:
        os.mkdir(str(out_dir + '/' + 'confusion_matrix'))
    normalized = normalize_data(raw_data, method=normalize_method, threshold=normalize_threshold)
    split_train_data, split_test_data, split_train_feature, split_test_feature = split_data(
        normalized, feature, test_size=test_size, seed=seed
    )
    model, mean_score, std_score = k_fold_crossvalid(
        split_train_data, split_train_feature, method=model_name,
        n_estimators=num_estimators, n_neighbours=num_neighbours, n_components=n_components, k_fold=k_fold, seed=seed
    )

    click.echo(f'Average cross-validation score {mean_score} and standard deviation {std_score}',err=True)
    predictions = predict_with_model(model, split_test_data).round()
    file_name = str(model_name + '_' + normalize_method)
    model_results = []
    model_results.append(accuracy_score(split_test_feature, predictions.round()))
    model_results.append(precision_score(split_test_feature, predictions, average="micro"))
    model_results.append(recall_score(split_test_feature, predictions, average="micro"))
    tbl[file_name] = model_results
    conf_matrix = pd.DataFrame(confusion_matrix(split_test_feature, predictions.round()))
    conf_matrix.to_csv(os.path.join(str(out_dir + '/' + 'confusion_matrix' + '/'), file_name  + "." + 'csv'))
    col_names = [
        'Accuracy',
        'Precision',
        'Recall',
    ]

    out_metrics = pd.DataFrame.from_dict(tbl, columns=col_names, orient='index')
    out_metrics.to_csv(os.path.join(out_dir, str(model_name + '_' + normalize_method) + "." + 'csv'))

@main.command('one')
@test_size
@num_estimators
@num_neighbours
@n_components
@model_name
@normalize_method
@feature_name
@normalize_threshold
@click.option('--model-filename', default=None, help='Filename of previously saved model')
@click.argument('metadata_file', type=click.File('r'))
@click.argument('data_file', type=click.File('r'))
@click.argument('out_dir')
def eval_one(test_size, num_estimators, num_neighbours, n_components, model_name, normalize_method,
             feature_name, normalize_threshold, model_filename, metadata_file, data_file, out_dir):
    """Train and evaluate a model. Print the model results to stderr."""
    raw_data, microbes, feature, name_map = feature_extraction(data_file, metadata_file, feature_name=feature_name)
    click.echo(f'Training {model_name} using {normalize_method} to predict {feature_name}',err=True)
    tbl, seed = {}, randint(0, 1000)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        os.mkdir(str(out_dir + '/' + 'confusion_matrix'))
        os.mkdir(str(out_dir + '/' + 'classification_report'))
    else:
        os.mkdir(str(out_dir + '/' + 'confusion_matrix'))
        os.mkdir(str(out_dir + '/' + 'classification_report'))
    normalized = normalize_data(raw_data, method=normalize_method, threshold=normalize_threshold)
    train_data, test_data, train_feature, test_feature = split_data(
            normalized, feature, test_size=test_size, seed=seed
    )

    model = train_model(
            train_data, train_feature, method=model_name,
            n_estimators=num_estimators, n_neighbours=num_neighbours, n_components=n_components, seed=seed
    )
    predictions = predict_with_model(model, test_data).round()
    conf_matrix = pd.DataFrame(confusion_matrix(test_feature, predictions.round()))
    conf_matrix.to_csv(os.path.join(str(out_dir + '/' + 'confusion_matrix' + '/'), str(model_name + '_' + normalize_method) + "." + 'csv'))

    model_results = []
    model_results.append(accuracy_score(test_feature, predictions.round()))
    model_results.append(precision_score(test_feature, predictions, average="micro"))
    model_results.append(recall_score(test_feature, predictions, average="micro"))
    col_names = [
        'Accuracy',
        'Precision',
        'Recall',
    ]
    tbl[str(model_name + ' ' + normalize_method)] = model_results
    out_metrics = pd.DataFrame.from_dict(tbl, columns=col_names, orient='index')
    out_metrics.to_csv(os.path.join(out_dir, str(model_name + '_' + normalize_method) + "." + 'csv'))

@main.command('all')
@test_size
@num_estimators
@num_neighbours
@n_components
@feature_name
@normalize_threshold
@click.option('--noisy', default=True, help='Add noise to data')
@click.argument('metadata_file', type=click.File('r'))
@click.argument('data_file', type=click.File('r'))
@click.argument('out_dir')
def eval_all(test_size, num_estimators, num_neighbours, n_components, feature_name, normalize_threshold, noisy,
             metadata_file, data_file, out_dir):
    """Evaluate all models and all normalizers."""
    raw_data, microbes, feature, name_map = feature_extraction(data_file, metadata_file, feature_name=feature_name)
    click.echo(f'Training all models using multiple normalization to predict {feature_name}',err=True)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        os.mkdir(str(out_dir + '/' + 'confusion_matrix'))
        os.mkdir(str(out_dir + '/' + 'pd_confusion_matrix'))
    else:
        os.mkdir(str(out_dir + '/' + 'confusion_matrix'))
        os.mkdir(str(out_dir + '/' + 'pd_confusion_matrix'))
    model_results = []
    noise_data = [0]
    if noisy==True:
        noise_data = NOISE_VALUES

    tbl, seed = {}, randint(0, 1000)
    for model_name, norm_name in product(MODEL_NAMES, NORMALIZER_NAMES):
        click.echo(
            f'Training {model_name} using {norm_name} to predict {feature_name}',
            err=True
        )
        normalized = normalize_data(raw_data, method=norm_name, threshold=normalize_threshold)
        train_data, test_data, train_feature, test_feature = split_data(
            normalized, feature, test_size=test_size, seed=seed
        )

        for i in noise_data:

            click.echo(f'Gaussian noise {i} has been added',err=True)

            # Adding noise to train data to check for over-fitting
            train_noise = np.random.normal(0, i,(train_data.shape[0], train_data.shape[1]))
            train_data = train_data+ train_noise

            model = train_model(
                    train_data, train_feature, method=model_name,
                    n_estimators=num_estimators, n_neighbours=num_neighbours, n_components=n_components, seed=seed
            )

            predictions = predict_with_model(model, test_data).round()
            model_results = predict_top_classes(model, test_data, test_feature)
            model_results.append(precision_score(test_feature, predictions, average="micro"))
            model_results.append(recall_score(test_feature, predictions, average="micro"))
            model_results.insert(0,i);
            model_results.insert(0,norm_name);
            model_results.insert(0,model_name);
            tbl[str(model_name + '_' + norm_name + '_' + str(i))] = model_results
            conf_matrix = pd.DataFrame(confusion_matrix(test_feature, predictions.round()))
            conf_matrix.to_csv(os.path.join(str(out_dir + '/' + 'confusion_matrix' + '/'), str(model_name + '_' + norm_name + '_' + str(i)) + "." + 'csv'))
            CV_table = pd.crosstab(name_map[test_feature], name_map[predictions], rownames=['Actual ' + feature_name], colnames=['Predicted ' + feature_name])
            CV_table.to_csv(os.path.join(str(out_dir + '/' + 'pd_confusion_matrix' + '/'), str(model_name + '_' + norm_name + '_' + str(i)) + "." + 'csv'))


    col_names = [
        'Classifier',
        'Preprocessing',
        'Noise',
        'Accuracy',
        'Top_2_accuracy',
        'Top_3_accuracy',
        'Top_5_accuracy',
        'Top_10_accuracy',
        'Precision',
        'Recall',
    ]

    out_metrics = pd.DataFrame.from_dict(tbl, columns=col_names, orient='index')
    out_metrics.to_csv(os.path.join(out_dir, 'output_metrics' + "." + 'csv'))

@main.command('leave-one')
@num_estimators
@num_neighbours
@n_components
@model_name
@normalize_method
@feature_name
@click.option('--group-name', default='city', help='The group to be considered')
@normalize_threshold
@click.option('--test-filename', default="test_sample.csv", help='Filename to save test dataset')
@click.argument('metadata_file', type=click.File('r'))
@click.argument('data_file', type=click.File('r'))
@click.argument('out_dir')
def leave_one(num_estimators, num_neighbours,  n_components, model_name, normalize_method,
             feature_name, group_name, normalize_threshold, test_filename, metadata_file, data_file, out_dir):
    """Train and evaluate a model and validate using a third-party group. echo the model results to stderr."""
    raw_data, microbes, feature, name_map, group_feature, group_map = group_feature_extraction(data_file,
                                metadata_file, feature_name=feature_name, group_name=group_name)
    click.echo(f'Training {model_name} using {normalize_method} to predict {feature_name}',err=True)
    tbl, seed = {}, randint(0, 1000)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        os.mkdir(str(out_dir + '/' + 'confusion_matrix'))
    else:
        os.mkdir(str(out_dir + '/' + 'confusion_matrix'))
    normalized = normalize_data(raw_data, method=normalize_method, threshold=normalize_threshold)

    model, mean_score, std_score, test_data, test_feature = leave_one_group_out(
        normalized, feature, group_feature, method=model_name,
        n_estimators=num_estimators, n_neighbours=num_neighbours, n_components=n_components, seed=seed
    )

    predictions = predict_with_model(model, test_data).round()

    conf_matrix = pd.DataFrame(confusion_matrix(test_feature, predictions.round()))
    conf_matrix.to_csv(os.path.join(str(out_dir + '/' + 'confusion_matrix' + '/'), str(model_name + '_' + normalize_method) + "." + 'csv'))

    model_results = []
    model_results.append(accuracy_score(test_feature, predictions.round()))
    model_results.append(precision_score(test_feature, predictions, average="micro"))
    model_results.append(recall_score(test_feature, predictions, average="micro"))
    col_names = [
        'Accuracy',
        'Precision',
        'Recall',
    ]
    tbl[str(model_name + ' ' + normalize_method)] = model_results
    out_metrics = pd.DataFrame.from_dict(tbl, columns=col_names, orient='index')
    out_metrics.to_csv(os.path.join(out_dir, str(model_name + '_' + normalize_method) + "." + 'csv'))




if __name__ == '__main__':
    main()
