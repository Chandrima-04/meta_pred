import numpy as np
import pandas as pd

from meta_pred.preprocessing import (
	parse_raw_data,
	)
	
data = {"line1": [24, np.nan, 4, 21, 19], "line2": [np.nan, 30, 13, 9, np.nan]}
df = pd.DataFrame(data)

class TestPreprocessing():
	"""Test Preprocessing"""
	
	def raw_data_nan_conversion(self):
		assert parse_raw_data(df) == {[24, 0, 4, 21, 19], [0, 30, 13, 9, 0]}, ["line1", "line2"]
