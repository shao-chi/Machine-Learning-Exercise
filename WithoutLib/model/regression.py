# https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/supervised_learning/regression.py
from __future__ import print_function, division
import numpy as np
import math
from ..utils.normalize import normalization

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

def Regularization_L1(alpha, w, grad=False):
    """
    L1 Penalty: λ∑|βj|
    λ: tuninig parameter (λ → ∞ 時，懲罰效果最大，迫使所有係數都趨近於0)
    """
    if grad == False:
        return alpha * np.linalg.norm(w) # 平方總和開根號 -> n維歐幾里德空間的直覺長度
    else:
        return alpha * np.sign(w)

def Regularization_L2(alpha, w, grad=False):
    """
    L2 Penalty: λ∑βj^2
    """
    if grad == False:
        return alpha * 0.5 * w.T.dot(w) # W^T * W
    else:
        return alpha * w

def LassoRegression(X, Y, degree, reg_alpha, n_iter=3000, learn_rate=0.001):
    """
    Least absolute shrinkage and selection operator (Lasso)
    Regression with L1 Penalty

    degree: the degree of polynomial
    reg_alpha: the amount og regularization (feature shrinkage)
    """
    
def RidgeRegression():
    """
    Regression with L2 Penalty
    """
