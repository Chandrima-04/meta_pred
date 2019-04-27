from sklearn.externals import joblib
import pandas as pd
import click
from itertools import product
from random import randint
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
    k_fold_crossvalid,
    split_data,
    train_model,
    predict_with_model,
    multi_predict_with_model,
    predict_top_classes,
    feature_importance,
)

MODEL_NAMES = ['random_forest', 'gaussian', 'knn', 'svm', 'linear_svc', 'neural_network']
NORMALIZER_NAMES = ['raw', 'standard_scalar', 'total_sum', 'binary']


@click.group()
def main():
    pass

@main.command('kfold')
@click.option('--k-fold', default=5, help='The value of k for cross-validation')
@click.option('--test-size', default=0.2, help='The relative size of the test data')
@click.option('--num-estimators', default=20, help='Number of trees in our random forest')
@click.option('--num-neighbours', default=21, help='Number of clusters in our knn')
@click.option('--model-name', default='random_forest', help='The model type to train')
@click.option('--normalize-method', default='standard_scalar', help='Normalization method.')
@click.option('--feature-name', default='city', help='The feature to predict')
@click.option('--normalize-threshold', default='0.0001',
              help='Normalization threshold for binary normalization.')
@click.option('--test-filename', default="test_sample.csv", help='Filename to save test dataset')
@click.option('--model-filename', default="model_k.pkl", help='Filename to save Model')
@click.argument('metadata_file', type=click.File('r'))
@click.argument('data_file', type=click.File('r'))
def kfold_cv(k_fold, test_size, num_estimators, num_neighbours, model_name, normalize_method, 
             feature_name, normalize_threshold, test_filename, model_filename, metadata_file, data_file):
    """Train and evaluate a model with k-fold cross-validation. Print the model results to stderr."""    
    raw_data, microbes = parse_raw_data(data_file)
    seed = randint(0, 1000)
    feature, name_map = parse_feature(metadata_file, raw_data.index, feature_name=feature_name)    
    click.echo(f'Training {model_name} using {normalize_method} to predict {feature_name}',err=True)

    normalized = normalize_data(raw_data, method=normalize_method, threshold=normalize_threshold)
    split_train_data, split_test_data, split_train_feature, split_test_feature = split_data(
        normalized, feature, test_size=test_size, seed=seed
    ) 
	
    #new_csv = pd.merge(left=split_test_data, right=split_test_feature, how='outer')
    #new_csv.to_csv(test_filename, index=False)
	
    model = k_fold_crossvalid(
        split_train_data, split_train_feature, method=model_name, 
        n_estimators=num_estimators, n_neighbours=num_neighbours, k_fold=k_fold
    )
	
    click.echo("\n In here")
    joblib.dump(model, model_filename)

    predictions = predict_with_model(model, split_test_data).round()
    click.echo(confusion_matrix(split_test_feature, predictions.round()))
    click.echo(classification_report(split_test_feature, predictions.round()))
    click.echo(accuracy_score(split_test_feature, predictions.round()))

    multiclass_prediction = multi_predict_with_model(model,split_test_data)
    click.echo(multiclass_prediction)

@main.command('one')
@click.option('--test-size', default=0.2, help='The relative size of the test data')
@click.option('--num-estimators', default=20, help='Number of trees in our random forest')
@click.option('--num-neighbours', default=21, help='Number of clusters in our knn')
@click.option('--model-name', default='random_forest', help='The model type to train')
@click.option('--normalize-method', default='standard_scalar', help='Normalization method.')
@click.option('--feature-name', default='city', help='The feature to predict')
@click.option('--normalize-threshold', default='0.0001',
              help='Normalization threshold for binary normalization.')
@click.option('--model-filename', default=None, help='Filename of previously saved model')
@click.argument('metadata_file', type=click.File('r'))
@click.argument('data_file', type=click.File('r'))
def eval_one(test_size, num_estimators, num_neighbours, model_name, normalize_method, 
             feature_name, normalize_threshold, model_filename, metadata_file, data_file):
    """Train and evaluate a model. Print the model results to stderr."""
    
    raw_data, microbes = parse_raw_data(data_file)
    seed = randint(0, 1000)
    feature, name_map = parse_feature(metadata_file, raw_data.index, feature_name=feature_name)
    click.echo(feature)
    click.echo(name_map)
    
    click.echo(f'Training {model_name} using {normalize_method} to predict {feature_name}',err=True)
    
    normalized = normalize_data(raw_data, method=normalize_method, threshold=normalize_threshold)    
    
    if(model_filename==None):
	
        train_data, test_data, train_feature, test_feature = split_data(
            normalized, feature, test_size=test_size, seed=seed
	    )

        model = train_model(
            train_data, train_feature, method=model_name,
            n_estimators=num_estimators, n_neighbours=num_neighbours
	    )

    else:
        test_data = normalized
        model = joblib.load(model_filename)
    #if (model_name == 'random_forest'):
            #feature_list = feature_importance(microbes, model)
            #for microbes_name, values in feature_list:
                #if values > 0:
                    #click.echo("{}={}".format(microbes_name, values))
    predictions = predict_with_model(model, test_data).round()
    click.echo(predictions)
    if(model_filename==None):
        click.echo(confusion_matrix(test_feature, predictions.round()))
        click.echo(classification_report(test_feature, predictions.round()))
        click.echo(accuracy_score(test_feature, predictions.round()))

    multiclass_prediction = multi_predict_with_model(model,test_data)
    click.echo(multiclass_prediction)

@main.command('all')
@click.option('--test-size', default=0.2, help='The relative size of the test data')
@click.option('--num-estimators', default=20, help='Number of trees in our random forest')
@click.option('--num-neighbours', default=21, help='Number of clusters in our knn') 
@click.option('--feature-name', default='city', help='The feature to predict')
@click.option('--normalize-threshold', default='0.0001',
              help='Normalization threshold for binary normalization.')
@click.argument('metadata_file', type=click.File('r'))
@click.argument('data_file', type=click.File('r'))
@click.argument('out_file', type=click.File('w'))
def eval_all(test_size, num_estimators, num_neighbours, feature_name, normalize_threshold,
             metadata_file, data_file, out_file):                
    """Evaluate all models and all normalizers."""
    raw_data, microbes = parse_raw_data(data_file)
    feature, name_map = parse_feature(metadata_file, raw_data.index, feature_name=feature_name)

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
        model = train_model(
            train_data, train_feature, method=model_name,
            n_estimators=num_estimators, n_neighbours=num_neighbours
        )
        if (model_name == 'random_forest'):
            feature_list = feature_importance(microbes, model)
            for microbes_name, values in feature_list:
                if values > 0:
                    click.echo("{}={}".format(microbes_name, values))
		
        predictions = predict_with_model(model, test_data).round()
		
        model_results = predict_top_classes(model, test_data, test_feature)
        model_results.append(precision_score(test_feature, predictions, average="macro"))
        model_results.append(recall_score(test_feature, predictions, average="macro"))
        print(model_results)
        model_results.insert(0,norm_name);
        model_results.insert(0,model_name);
        tbl[str(model_name + ' ' + norm_name)] = model_results
    print(tbl)
    col_names = [
        'Classifier',
        'Preprocessing',
        'Accuracy',
        'Top_2_accuracy',
        'Top_3_accuracy',
        'Top_5_accuracy',
        'Top_10_accuracy',
        'Precision',
        'Recall',
    ]
    df = pd.DataFrame.from_dict(tbl, columns=col_names, orient='index')
    df.to_csv(out_file)


if __name__ == '__main__':
    main()
