import numpy as np
import pandas as pd


def rmse(y_pr: pd.core.series.Series, y_valid: pd.core.series.Series) -> float:
    n = len(y_valid)
    return np.sqrt(np.sum((y_pr - y_valid) ** 2) / n)


def mae(y_pr: pd.core.series.Series, y_valid: pd.core.series.Series) -> float:
    n = len(y_valid)
    return np.sum(np.abs(y_pr - y_valid)) / n


def rmspe(y_pr: pd.core.series.Series, y_valid: pd.core.series.Series) -> float:
    n = len(y_valid)
    return np.sum(((y_pr - y_valid) / y_valid) ** 2) / n
