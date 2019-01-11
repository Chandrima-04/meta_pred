
import click

from .preprocessing import (
    parse_raw_data,
    parse_feature,
    normalize_data,
)
from .classification import (
    train_model,
    test_model,
)


@click.command()
@click.option('--model-type', default='random_forest', help='The model type to train')
@click.option('--feature', default='city', help='The feature to predict')
@click.option('--normalize-method', default='total_sum', help='Normalization method.')
@click.argument('metadata_filename')
@click.argument('data_filename')
def main(model_type, feature, normalize_method, metadata_filename, data_filename):
    """Train and evaluate a model. Print the model results to stderr."""
    click.echo(f'Training {model_type} to predict {feature}', err=True)


if __name__ == '__main__':
    main()
