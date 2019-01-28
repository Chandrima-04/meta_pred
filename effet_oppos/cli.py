import pandas as pd
import click
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report, accuracy_score

from .preprocessing import (
    parse_raw_data,
    parse_feature,
    normalize_data,
)
from .classification import (
    split_data,
    train_model,
    predict_with_model,
    multi_predict_with_model,
    predict_top_classes,
)


@click.command()
@click.option('--test-size', default=0.2, help='The relative size of the test data')
@click.option('--num-estimators', default=20, help='Number of trees in our random forest')
@click.option('--num-neighbours', default=21, help='Number of clusters in our knn')
@click.option('--model-type', default='random_forest', help='The model type to train')
@click.option('--feature-name', default='city', help='The feature to predict')
@click.option('--normalize-method', default='standard_scalar', help='Normalization method.')
@click.option('--normalize-threshold', default='0.0001', help='Normalization threshold for binary normalization.')
@click.argument('metadata_filename')
@click.argument('data_filename')
def main(test_size, num_estimators, num_neighbours, model_type, feature_name, normalize_method, normalize_threshold,
         metadata_filename, data_filename):
    """Train and evaluate a model. Print the model results to stderr."""
    model_name = ['random_forest', 'gaussian', 'knn', 'svm', 'linear_svc', 'neural_network']
    normalize_model = ['raw', 'standard_scalar', 'total_sum', 'binary']
    i = 0
    top1, top3, top5, top10, model_classifier, model_normalization, precision, recall = [],[], [], [], [], [], [], []
    for i in range(len(model_name)):
        j = 0
        for j in range(len(normalize_model)):
            model_type = model_name[i]
            normalize_method = normalize_model[j]
            click.echo(f'Training {model_type} using {normalize_method} to predict {feature_name}', err=True)
            raw_data = parse_raw_data(data_filename)
            normalized = normalize_data(raw_data, method=normalize_method, threshold=normalize_threshold)
            feature, name_map = parse_feature(metadata_filename, raw_data.index, feature_name=feature_name)
            train_data, test_data, train_feature, test_feature = split_data(
                normalized, feature, test_size=test_size
            )
            model = train_model(
                train_data, train_feature, method=model_type, n_estimators=num_estimators, n_neighbours=num_neighbours
            )   
            predictions = predict_with_model(model, test_data)
            click.echo(confusion_matrix(test_feature, predictions.round()))
            click.echo(classification_report(test_feature, predictions.round()))
            click.echo(accuracy_score(test_feature, predictions.round()))

            multiclass_prediction = multi_predict_with_model(model,test_data)
            click.echo(multiclass_prediction)

            predict_top_class = predict_top_classes(model, test_data, test_feature)
            click.echo(predict_top_class)
            top1.append(predict_top_class[0])
            top3.append(predict_top_class[1])
            top5.append(predict_top_class[2])
            top10.append(predict_top_class[3])
            model_classifier.append(model_type)
            model_normalization.append(normalize_method)
            precision_value = precision_score(test_feature, predictions.round(), average="macro") 
            recall_value  = recall_score(test_feature, predictions.round(), average="macro") 
            precision.append((precision_value))
            recall.append((recall_value))

    raw_data = {'Classifier': model_classifier,
                'Preprocessing': model_normalization,
                'Accuracy': top1,
                'Top_3_accuracy': top3,
                'Top_5_accuracy': top5,
                'Top_10_accuracy': top10,
                'Precision' : precision,
                'Recall': recall

                }
    df = pd.DataFrame(raw_data, columns = ['Classifier', 'Preprocessing', 'Accuracy', 'Top_3_accuracy', 'Top_5_accuracy', 'Top_10_accuracy', 'Precision', 'Recall'])
    df.to_csv("example_classifier.csv")


if __name__ == '__main__':
    main()
