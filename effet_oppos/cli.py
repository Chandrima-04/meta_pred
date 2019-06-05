import pandas as pd
import numpy as np 
import os.path
import click
from random import randint
from sklearn.externals import joblib
from itertools import product
from sklearn.metrics import (
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    accuracy_score,
)
from .preprocessing import (
    parse_raw_data,
    parse_feature,
    normalize_data,
)
from .classification import (
    split_data,
    get_classifier,
	k_fold_crossvalid,
    train_model,
    predict_with_model,
    multi_predict_with_model,
    predict_top_classes,
    feature_importance,
)

MODEL_NAMES = [
    'random_forest', 
    'decision_tree', 
    'extra_tree', 
    'adaboost', 
    'knn', 
    'LDA', 
    'neural_network', 
    'gaussianNB',
    'linear_svc',
    'svm'
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

@main.command('kfold')
@click.option('--k-fold', default=5, help='The value of k for cross-validation')
@click.option('--test-size', default=0.2, help='The relative size of the test data')
@click.option('--num-estimators', default=20, help='Number of trees in our random forest and/or adaboast')
@click.option('--num-neighbours', default=21, help='Number of clusters in our knn')
@click.option('--n-components', default=10, 
              help='Number of components for dimensionality reduction in Linear Discriminant Analysis')
@click.option('--model-name', default='random_forest', help='The model type to train')
@click.option('--normalize-method', default='standard_scalar', help='Normalization method')
@click.option('--feature-name', default='city', help='The feature to predict')
@click.option('--normalize-threshold', default='0.0001',
              help='Normalization threshold for binary normalization.')
@click.option('--test-filename', default="test_sample.csv", help='Filename to save test dataset')
@click.option('--model-filename', default="model_k.pkl", help='Filename to save Model')
@click.argument('metadata_file', type=click.File('r'))
@click.argument('data_file', type=click.File('r'))
@click.argument('out_dir')
def kfold_cv(k_fold, test_size, num_estimators, num_neighbours,  n_components, model_name, normalize_method, 
             feature_name, normalize_threshold, test_filename, model_filename, metadata_file, data_file, out_dir):
    """Train and evaluate a model with k-fold cross-validation. echo the model results to stderr."""    
    raw_data, microbes = parse_raw_data(data_file)
    tbl, seed = {}, randint(0, 1000)
    feature, name_map = parse_feature(metadata_file, raw_data.index, feature_name=feature_name)    
    click.echo(f'Training {model_name} using {normalize_method} to predict {feature_name}',err=True)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        os.mkdir(str(out_dir + '/' + 'confusion_matrix'))
    else:
        os.mkdir(str(out_dir + '/' + 'confusion_matrix'))
    new_index = feature!= -1
    raw_data = raw_data[new_index == True]
    feature = feature[new_index == True]
    normalized = normalize_data(raw_data, method=normalize_method, threshold=normalize_threshold)
    split_train_data, split_test_data, split_train_feature, split_test_feature = split_data(
        normalized, feature, test_size=test_size, seed=seed
    )
    model, mean_score, std_score = k_fold_crossvalid(
        split_train_data, split_train_feature, method=model_name, 
        n_estimators=num_estimators, n_neighbours=num_neighbours, n_components=n_components, k_fold=k_fold, seed=seed
    )
	
    click.echo(f'Average cross-validation score {mean_score} and standard deviation {std_score}',err=True)
    joblib.dump(model, model_filename)

 
    for i in range(0,2):
        if i == 0:
            predictions = predict_with_model(model, split_test_data).round()
            file_name = str(model_name + '_' + normalize_method)
        else:
            predictions = predict_with_model(model, split_test_data).round()
            file_name = str(model_name + '_' + normalize_method + '_noisy')
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
@click.option('--test-size', default=0.2, help='The relative size of the test data')
@click.option('--num-estimators', default=20, help='Number of trees in our random forest and/or adaboast')
@click.option('--num-neighbours', default=21, help='Number of clusters in our knn')
@click.option('--n-components', default=10, 
              help='Number of components for dimensionality reduction in Linear Discriminant Analysis')
@click.option('--model-name', default='random_forest', help='The model type to train')
@click.option('--normalize-method', default='standard_scalar', help='Normalization method.')
@click.option('--feature-name', default='city', help='The feature to predict')
@click.option('--normalize-threshold', default='0.0001',
              help='Normalization threshold for binary normalization.')
@click.option('--model-filename', default=None, help='Filename of previously saved model')
@click.argument('metadata_file', type=click.File('r'))
@click.argument('data_file', type=click.File('r'))
@click.argument('out_dir')
def eval_one(test_size, num_estimators, num_neighbours, n_components, model_name, normalize_method, 
             feature_name, normalize_threshold, model_filename, metadata_file, data_file, out_dir):
    """Train and evaluate a model. Print the model results to stderr."""
    
    raw_data, microbes = parse_raw_data(data_file)
    tbl, seed = {}, randint(0, 1000)
    feature, name_map = parse_feature(metadata_file, raw_data.index, feature_name=feature_name)
	
    click.echo(f'Training {model_name} using {normalize_method} to predict {feature_name}',err=True)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        os.mkdir(str(out_dir + '/' + 'confusion_matrix'))
        os.mkdir(str(out_dir + '/' + 'classification_report'))
    else:
        os.mkdir(str(out_dir + '/' + 'confusion_matrix')) 
        os.mkdir(str(out_dir + '/' + 'classification_report'))		
    
    if model_filename==None:
        new_index = feature!= -1
        raw_data = raw_data[new_index == True]
        feature = feature[new_index == True]
        normalized = normalize_data(raw_data, method=normalize_method, threshold=normalize_threshold) 
        train_data, test_data, train_feature, test_feature = split_data(
            normalized, feature, test_size=test_size, seed=seed
	    )
		
        model = train_model(
            train_data, train_feature, method=model_name,
            n_estimators=num_estimators, n_neighbours=num_neighbours, n_components=n_components, seed=seed
	    )    

    else:
        normalized = normalize_data(raw_data, method=normalize_method, threshold=normalize_threshold) 
        test_data = normalized
        model = joblib.load(model_filename)
		
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
	
	
    #if model_name == 'random_forest':
    #   val = 0
    #    feature_list = feature_importance(microbes, model)
    #    for microbes_name, values in feature_list:
            #if val < 100:
    #        click.echo("{}={}".format(microbes_name, values))
            #val +=1

@main.command('all')
@click.option('--test-size', default=0.2, help='The relative size of the test data')
@click.option('--num-estimators', default=20, help='Number of trees in our random forest and/or adaboast')
@click.option('--num-neighbours', default=21, help='Number of clusters in our knn') 
@click.option('--n-components', default=10, 
               help='Number of components for dimensionality reduction in Linear Discriminant Analysis')
@click.option('--feature-name', default='city', help='The feature to predict')
@click.option('--normalize-threshold', default='0.0001',
              help='Normalization threshold for binary normalization.')
@click.option('--noisy', default=True, help='Add noise to data')
@click.argument('metadata_file', type=click.File('r'))
@click.argument('data_file', type=click.File('r'))
@click.argument('out_dir')
def eval_all(test_size, num_estimators, num_neighbours, n_components, feature_name, normalize_threshold, noisy, 
             metadata_file, data_file, out_dir):                
    """Evaluate all models and all normalizers."""
    raw_data, microbes = parse_raw_data(data_file)
    feature, name_map = parse_feature(metadata_file, raw_data.index, feature_name=feature_name)
    new_index = feature!= -1
    raw_data = raw_data[new_index == True]
    feature = feature[new_index == True]
	
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        os.mkdir(str(out_dir + '/' + 'confusion_matrix'))
        os.mkdir(str(out_dir + '/' + 'pd_confusion_matrix'))
    else:
        os.mkdir(str(out_dir + '/' + 'confusion_matrix'))
        os.mkdir(str(out_dir + '/' + 'pd_confusion_matrix'))
	
    click.echo(noisy)
    noise_data = [0]	
    if noisy==True:
        noise_data = NOISE_VALUES
    click.echo(noise_data)

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
            click.echo(classification_report(test_feature, predictions))
		
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
	

if __name__ == '__main__':
    main()
