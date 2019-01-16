
import click
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

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
)


@click.command()
@click.option('--test-size', default=0.2, help='The relative size of the test data')
@click.option('--num-estimators', default=20, help='Number of trees in our random forest')
@click.option('--model-type', default='random_forest', help='The model type to train')
@click.option('--feature-name', default='city', help='The feature to predict')
@click.option('--normalize-method', default='total_sum', help='Normalization method.')
@click.argument('metadata_filename')
@click.argument('data_filename')
def main(test_size, num_estimators, model_type, feature_name, normalize_method,
         metadata_filename, data_filename):
    """Train and evaluate a model. Print the model results to stderr."""
    click.echo(f'Training {model_type} to predict {feature_name}', err=True)
    raw_data = parse_raw_data(data_filename)
    normalized = normalize_data(raw_data, method=normalize_method)
    feature, name_map = parse_feature(metadata_filename, raw_data.index, feature_name=feature_name)
    train_data, test_data, train_feature, test_feature = split_data(
        normalized, feature, test_size=test_size
    )
    model = train_model(
        train_data, train_feature, method=model_type, n_estimators=num_estimators
    )
    predictions = predict_with_model(model, test_data)
    click.echo(confusion_matrix(test_feature, predictions.round()))
    click.echo(classification_report(test_feature, predictions.round()))
    click.echo(accuracy_score(test_feature, predictions.round()))

    multiclass_prediction = multi_predict_with_model(model,test_data)
    click.echo(multiclass_prediction)


if __name__ == '__main__':
    main()
