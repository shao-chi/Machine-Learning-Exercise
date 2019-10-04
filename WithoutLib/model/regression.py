# https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/supervised_learning/regression.py
from __future__ import print_function, division
import numpy as np
import math

def LinearRegression(X, Y, n_iter=100, learn_rate=0.001):
    """
    train a linear regression using gradient descent
    Return: training_error(array), weight(array)
    """
    # insert constant ones for bias weights
    X = np.insert(X, 0, 1, axis=1)

    # initialize weights randomly [-1/N, 1/N]
    # N: square root of the number of features
    N = math.sqrt(X.shape[1])
    limit  = 1/N
    weight = np.random.uniform(-limit, limit, (X.shape[1], ))

    training_error = list()

    # do gradient descent n_iter times
    for _ in range(n_iter):
        # X * weight
        predict = X.dot(weight)

        # calculate loss
        error = Y - predict
        mse = np.mean((error) ** 2) # objective function
        training_error.append(mse)

        # gradient
        # derivative of objective function to weight
        grad_w = -(error.dot(X)) # * 2
        # grad_bias = error

        # update weight
        weight -= grad_w * learn_rate

    return training_error, weight

def LassoRegression(X, Y, degree, reg_factor, n_iter=100, learn_rate=0.001):
    """
    degree: the degree of polynomial
    reg_factor: the amount of regularization and feature shrinkage
    """
    