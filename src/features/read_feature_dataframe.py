"""
Created on Tue Jan 28 2020

@author Name Redacted Surname Redacted
"""

import pandas as pd
import os
from . import data
import numpy as np


def read_feature_dataframe(name):
    df = pd.read_csv("{}.csv".format(os.path.join(data.feature_path, name)))
    df = df.replace({np.nan: None})
    return df


if __name__ == "__main__":
    read_feature_dataframe("artist_birth_date")
