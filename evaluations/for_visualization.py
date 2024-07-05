import pandas as pd
import numpy as np


def calculate_polarization(medians):
    # print(medians, np.mean([abs(50 - x) for x in medians]))
    return np.mean([abs(50 - x) for x in medians])
