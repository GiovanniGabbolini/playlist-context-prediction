"""
Created on Tue Jan 28 2020

@author Name Redacted Surname Redacted
"""

import numpy as np
import os
from . import data


def read_feature_dictionary(name):
    return np.load(
        "{}.npy".format(os.path.join(data.feature_path, name)), allow_pickle="TRUE"
    ).item()
