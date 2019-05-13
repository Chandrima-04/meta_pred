
import pandas as pd
from sklearn.preprocessing import StandardScaler


def parse_raw_data(filename):
    """Return a pandas dataframe containing the raw data to classify.

    Data points should be in rows, features in columns. This function
    should do 'unopinionated' data cleaning: filling in NAs, normalizing
    names, etc.
    """
    tbl = pd.read_csv(filename, index_col=0)
    header = list(tbl.columns.values)
    tbl = tbl.fillna(0)
    new_tbl = tbl[tbl.sum(axis=1) != 0]
    return new_tbl, header


def parse_feature(metadata_filename, sample_names, feature_name='city'):
    """Return a tuple of factorized features for our samples and a map
    from factor values to feature names.
    """
    metadata = pd.read_csv(metadata_filename, index_col=0)
    metadata = metadata.loc[sample_names]
    feature = metadata[feature_name]
    factorized, name_map = pd.factorize(feature)
    return factorized, name_map


def normalize_data(data_tbl, method='standard_scalar',threshold=0.0001):
    """Return a pandas dataframe ready for classification.

    Normalize/preprocess the dataset according to the
    specified method.
    """
    processor = {
        'raw': normalize_data_raw,
        'standard_scalar': normalize_data_standard_scalar,
        'total_sum' : normalize_data_total_sum,
        'binary' : normalize_data_binary,
    }[method.lower()]
    return processor(data_tbl)


def normalize_data_raw(data_tbl):
    """Return the data table without normalization."""
    return data_tbl


def normalize_data_standard_scalar(data_tbl):
    """Return a data table with each column standardized to have a mean of 0."""
    sc = StandardScaler()
    return sc.fit_transform(data_tbl)

def normalize_data_total_sum(data_tbl):
    """Return a data table with each value divided by its row sum."""
    return ((data_tbl.T / data_tbl.T.sum()).T)

def normalize_data_binary(data_tbl, threshold=0.0001):
    """Return a data table with 1 and 0 based on a threshold"""
    data_tbl[data_tbl < threshold] = 0
    data_tbl[data_tbl >= threshold] = 1
    return data_tbl
