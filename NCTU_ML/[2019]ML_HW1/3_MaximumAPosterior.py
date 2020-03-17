import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
from math import sqrt

# read dataset
X_df = pd.read_csv('Dataset/dataset_X.csv')
T_df = pd.read_csv('Dataset/dataset_T.csv')

X_array = X_df.values
T_array = T_df.values
X_array = np.array([np.array([x[2], x[8], x[9]]) for x in X_array[:, 1:]]).astype(np.float64)
T_array = T_array[:, 1].astype(np.float64).reshape(len(X_array), 1)

def polynomial_basis(X, M):
    """
    generate polynomial basis funciton on X
    M: order of polynomial funciton
    ex. M = 2, [a, b] -> [1, a, b, a^2, ab, b^2]
    """
    n_samples, n_features = X.shape
    w_coef = 1
    for m in range(M + 1):
        w_coef += n_features ** m
    poly_x = np.empty((n_samples, w_coef), dtype=np.float64)

    element_index = [combinations_with_replacement(range(n_features), i) for i in range(M + 1)]
    element_index = [i for sub in element_index for i in sub]

    for i, index in enumerate(element_index):
        poly_x[:, i] = np.prod(X[:, index], axis=1)

    return poly_x

def gaussian_basis(X):
    sigma_inv = np.linalg.pinv(np.cov(X.T))
    # mean = X.mean(0)
    gauss_x = np.array([(-np.dot((x - x.mean(0)).reshape((len(x),1)).T, sigma_inv) * (x - x.mean(0)) / 2).reshape((len(x))) for x in X])

    return gauss_x

def sigmoid_basis(X):
    return 1 / (1 + np.exp(-X))

def ML_fit(train_basis_x, train_T):
    tmp = np.matmul(train_basis_x.T, train_basis_x)
    tmp = np.linalg.pinv(tmp)
    tmp = np.matmul(tmp, train_basis_x.T)
    W = np.matmul(tmp, train_T)

    return W

def MAP_fit(train_basis_x, train_T, a, b):
    a_I = a * np.identity(train_basis_x.shape[1])
    tmp = b * np.matmul(train_basis_x.T, train_basis_x)
    tmp = a_I + tmp
    tmp = np.linalg.pinv(tmp)
    tmp = np.matmul(tmp, train_basis_x.T)
    W = np.matmul(tmp, train_T)

    return W

def ML_RMS(predict, Y):
    error = np.sum((predict - Y) ** 2) / 2
    rms = sqrt(2 * error / len(predict))

    return rms

def MAP_RMS(predict, Y, W, a, b):
    error = (b * np.sum((predict - Y) ** 2) / 2) + (a * np.matmul(W.T, W))
    rms = sqrt(2 * error / len(predict))

    return rms
    
a = 0.5
b = 1
k_fold = len(X_array) // 4
X_array = np.array(np.split(X_array, 4, axis=0))
T_array = np.array(np.split(T_array, 4, axis=0))
# print(X_array.shape)
for m in range(1, 7):
    print('\nPolynomial (M = {}) + Sigmoid'.format(m))
    for f in range(4):
        print('Fold {}: {} ~ {}'.format(f+1, k_fold*f, k_fold*(f+1)))
        valid_x = X_array[f]
        valid_t = T_array[f]
        k = [0, 1, 2, 3]
        k.remove(f)
        train_x = np.concatenate((X_array[k[0]], X_array[k[1]], X_array[k[2]]), axis=0)
        train_t = np.concatenate((T_array[k[0]], T_array[k[1]], T_array[k[2]]), axis=0)
        poly_x_t = polynomial_basis(train_x, m)
        # sig_x_t = sigmoid_basis(train_x)
        poly_x_v = polynomial_basis(valid_x, m)
        # sig_x_v = sigmoid_basis(valid_x)
        # train_x_basis = np.array([np.concatenate((poly_x_t[i], sig_x_t[i])) for i in range(len(train_x))]).astype(np.float64)
        # valid_x_basis = np.array([np.concatenate((poly_x_v[i], sig_x_v[i])) for i in range(len(valid_x))]).astype(np.float64)
        train_x_basis = sigmoid_basis(poly_x_t)
        valid_x_basis = sigmoid_basis(poly_x_v)
        w = MAP_fit(train_x_basis, train_t, a, b)

        train_predict = np.matmul(train_x_basis, w)
        valid_predict = np.matmul(valid_x_basis, w)
        print('RMS of train = ', MAP_RMS(train_predict, train_t, w, a, b))
        print('RMS of valid = ', MAP_RMS(valid_predict, valid_t, w, a, b))


