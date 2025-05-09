import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate(actual, pred):
    return {
        'MAE': mean_absolute_error(actual, pred),
        'RMSE': np.sqrt(mean_squared_error(actual, pred))
    }