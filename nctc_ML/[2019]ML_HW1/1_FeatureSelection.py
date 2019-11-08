import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
from math import sqrt

# read dataset
X_df = pd.read_csv('Dataset/dataset_X.csv')
T_df = pd.read_csv('Dataset/dataset_T.csv')

X_array = X_df.values
T_array = T_df.values

# training set size = 0.8 * dataset size = 0.8 * 1096 = 876.8
# validation set size = 1096 - 877
n = int(len(X_array) * 0.75)
train_x = X_array[:n, 1:].astype(np.float64)
train_t = T_array[:n, 1].astype(np.float64).reshape(len(train_x), 1)
valid_x = X_array[n:, 1:].astype(np.float64)
valid_t = T_array[n:, 1].astype(np.float64).reshape(len(valid_x), 1)

def fit_poly(M, train_X, train_T, valid_X, valid_T):
    """
    fit the data by applying a polynomial function.
    minimizie the error function.
    input:
        M: order of polynomial function
        X: observed data
        T: targets
    outputs:
        RMS_train: root-mean-square error of training set
        RMS_valid: root-mean-square error of validation set
    """
    train_samples, n_features = train_X.shape
    valid_samples = valid_X.shape[0]

    # Φ: basis function (Y = WΦ(X))
    w_coef = 1
    for m in range(M + 1):
        w_coef +=  n_features ** m
    train_basis_x = np.empty((train_samples, w_coef), dtype=np.float64)
    valid_basis_x = np.empty((valid_samples, w_coef), dtype=np.float64)
    
    element_index = [combinations_with_replacement(range(n_features), i) for i in range(M + 1)]
    element_index = [i for sub in element_index for i in sub]
    
    for i, index in enumerate(element_index):
        train_basis_x[:, i] = np.prod(train_X[:, index], axis=1)
        valid_basis_x[:, i] = np.prod(valid_X[:, index], axis=1)

    # train_basis_x_tran = train_basis_x.transpose()
    # polynomial coefficients
    tmp = np.matmul(train_basis_x.T, train_basis_x)
    tmp = np.linalg.pinv(tmp)
    tmp = np.matmul(tmp, train_basis_x.T)
    train_T = train_T.reshape(len(train_basis_x), 1)
    W = np.matmul(tmp, train_T)
    # W = np.linalg.inv(train_basis_x_tran * train_basis_x) * train_basis_x_tran * train_T

    train_predict = np.matmul(train_basis_x, W)
    valid_predict = np.matmul(valid_basis_x, W)
    train_error = np.sum((train_predict - train_T) ** 2) / 2 / len(train_predict)
    valid_error = np.sum((valid_predict - valid_T) ** 2) / 2 / len(valid_predict)
    RMS_train = sqrt(2 * train_error)
    RMS_valid = sqrt(2 * valid_error)

    return RMS_train, RMS_valid

# 1. (a)
M_1_TrainRMS, M_1_ValidRMS = fit_poly(1, train_x, train_t, valid_x, valid_t)
M_2_TrainRMS, M_2_ValidRMS = fit_poly(2, train_x, train_t, valid_x, valid_t)

print('M = 1')
print('RMS of train = ', M_1_TrainRMS)
print('RMS of valid = ', M_1_ValidRMS)
print('M = 2')
print('RMS of train = ', M_2_TrainRMS)
print('RMS of valid = ', M_2_ValidRMS)

# 1. (b)
feature_name = X_df.columns[1:]
for f in range(train_x.shape[1]):
    X_df_drop = X_df.drop(columns=[feature_name[f]])
    X_array = X_df_drop.values
    train_x = X_array[:n, 1:]
    valid_x = X_array[n:, 1:]

    TrainRMS, ValidRMS = fit_poly(1, train_x, train_t, valid_x, valid_t)

    print(f+1, feature_name[f])
    print('M = 1, RMS of train = ', TrainRMS)
    print('M = 1, RMS of valid = ', ValidRMS)
