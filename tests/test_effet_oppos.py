"""Test suite for effet oppos."""

from unittest import TestCase
from os.path import join, dirname

from effet_oppos.preprocessing import (
    parse_raw_data,
    parse_feature,
    normalize_data,
)
from effet_oppos.classification import (
    train_model,
    test_model,
)


SAMPLE_DATA = join(dirname(__file__), '')
SAMPLE_METADATA = join(dirname(__file__), '')


class TestEffetOppos(TestCase):
    """Test suite for packet building."""

    def test_parse_raw(self):
        """Test that we can build a taxonomy table."""
        data_tbl = parse_raw_data(SAMPLE_DATA)
        self.assertTrue(data_tbl.shape[0])
        self.assertTrue(data_tbl.shape[1])

    def test_parse_feature(self):
        """Test we can make AMR table."""
        data_tbl = parse_feature(SAMPLE_METADATA)
        self.assertTrue(data_tbl.shape[0])
        self.assertTrue(data_tbl.shape[1])
