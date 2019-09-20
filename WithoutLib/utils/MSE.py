import numpy as np

def mean_square_error(test, predict):
    """
    evaluate the mean squared error between true data and predicted data
    """
    return np.mean((test - predict) ** 2)