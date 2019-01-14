
import pandas as pd
from sklearn.preprocessing import StandardScaler


def parse_raw_data(filename):
    """Return a pandas dataframe containing the raw data to classify.

    Data points should be in rows, features in columns. This function
    should do 'unopinionated' data cleaning: filling in NAs, normalizing
    names, etc.
    """
    tbl = pd.read_csv(filename, index_col=0)
    tbl = tbl.fillna(0)
    return tbl


def parse_feature(metadata_filename, sample_names, feature_name='city'):
    """Return a tuple of factorized features for our samples and a map
    from factor values to feature names.
    """
    metadata = pd.read_csv(metadata_filename, index_col=0)
    metadata = metadata.loc[sample_names]
    feature = metadata[feature_name]
    factorized, name_map = pd.factorize(feature)
    return factorized, name_map


def normalize_data(data_tbl, method='raw'):
    """Return a pandas dataframe ready for classification.

    Normalize/preprocess the dataset according to the
    specified method.
    """
    processor = {
        'raw': normalize_data_raw,
        'total_sum': normalize_data_total_sum,
    }[method.lower()]
    return processor(data_tbl)


def normalize_data_raw(data_tbl):
    """Return the data table without normalization."""
    return data_tbl


def normalize_data_total_sum(data_tbl):
    """Return a data table with each value divided by its row sum."""
    sc = StandardScaler()
    return sc.fit_transform(data_tbl)
