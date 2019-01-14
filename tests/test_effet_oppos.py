"""Test suite for effet oppos."""

from unittest import TestCase
from os.path import join, dirname
from math import isclose

from effet_oppos.preprocessing import (
    parse_raw_data,
    parse_feature,
    normalize_data,
)
from effet_oppos.classification import (
    split_data,
    train_model,
    predict_with_model,
)


SAMPLE_DATA = join(dirname(__file__), 'taxa.csv')
SAMPLE_METADATA = join(dirname(__file__), 'complete_metadata.csv')


class TestEffetOppos(TestCase):
    """Test suite for packet building."""

    def test_parse_raw(self):
        """Test that we can build a taxonomy table."""
        data_tbl = parse_raw_data(SAMPLE_DATA)
        self.assertTrue(data_tbl.shape[0])
        self.assertTrue(data_tbl.shape[1])

    def test_parse_feature(self):
        """Test we can make AMR table."""
        data_tbl = parse_raw_data(SAMPLE_DATA)
        metadata_tbl = parse_feature(SAMPLE_METADATA, data_tbl.index)[0]
        self.assertTrue(metadata_tbl.shape[0])

    def test_normalize_data(self):
        """Test that normalization works."""
        data_tbl = parse_raw_data(SAMPLE_DATA)
        data_tbl = normalize_data(data_tbl, method='total_sum')
        self.assertTrue(data_tbl.shape[0])
        self.assertTrue(data_tbl.shape[1])

    def test_split_data(self):
        """Test that splitting data works."""
        data_tbl = parse_raw_data(SAMPLE_DATA)
        metadata_tbl = parse_feature(SAMPLE_METADATA, data_tbl.index)[0]
        training, testing, _, _ = split_data(data_tbl, metadata_tbl)
        self.assertTrue(
            isclose(training.shape[0] * 0.25, testing.shape[0], abs_tol=5)
        )

    def test_train_model(self):
        """Test that training model works."""
        data_tbl = parse_raw_data(SAMPLE_DATA)
        metadata_tbl = parse_feature(SAMPLE_METADATA, data_tbl.index)[0]
        training, _, feature, _ = split_data(data_tbl, metadata_tbl)
        model = train_model(training, feature)
