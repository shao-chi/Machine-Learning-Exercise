from scipy.io import loadmat # just for loading .mat file
import numpy as np
from math import exp, sqrt
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mv_normal
from numpy.random import multivariate_normal

data = loadmat('./dataset/1_data.mat')
"""
>>> data.keys()
dict_keys(['__header__', '__version__', '__globals__', 'x', 't'])
"""
x = np.array(list(data.values())[3])
t = np.array(list(data.values())[4])

def sigmoid_basis(X, M, s):
    sigmoid_x = list()
    for i in range(len(X)):
        sig_x =  list()
        for j in range(M):
            mu_j = 2 * j / M
            a = (X[i] - mu_j) / s
            sig_x.append(1 / (1 + exp(-a)))
        sigmoid_x.append(np.array(sig_x))
    return np.array(sigmoid_x)


def posterior(basis_x, t, beta, S0, M0):
    t = t.reshape(len(basis_x), 1)
    SN = np.linalg.inv(np.linalg.inv(S0) + beta * basis_x.T.dot(basis_x))
    MN = SN.dot(beta * basis_x.T.dot(t))
    post = multivariate_normal(MN.flatten(), SN)
    return post, SN, MN

def contour(MN, SN, n):
    x, y = np.mgrid[-3:6:.01, -3:6:.01]
    pos = np.empty(x.shape + (2, ))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    plt.suptitle("Prior distributions of weights (N = {})".format(n))
    plt.contourf(x, y, mv_normal(MN[:2], SN[:2, :2]).pdf(pos), 20)
    plt.xlabel('W0')
    plt.ylabel('W1')
    plt.show()

N = [5, 10, 30, 80, 100]
M = 3
s = 0.1

alpha = 10 ** (-6)
beta = 1
S0 = np.linalg.inv(alpha*np.identity(M))
M0 = np.zeros(M)

# cmap = plt.get_cmap('viridis')
# plt.scatter(x, t, color=cmap(0.9), s=10)
# plt.show()

step = 5
sort_x = sorted(x)
basis_x = sigmoid_basis(sort_x, M, s)
# generate five curve samples from the parameter posterior distribution
for n in N:
    train_x = x[100-n:]
    train_t = t[100-n:]

    # Color map
    cmap = plt.get_cmap('viridis')
    plt.suptitle("Samples from the posterior distributions (N = {})".format(n))
    plt.xlabel('X')
    plt.ylabel('T')

    prior = multivariate_normal(M0, S0)
    rms = 0
    for _ in range(step):
        train_basis_x = sigmoid_basis(train_x, M, s)
        post, SN, MN = posterior(train_basis_x, train_t, beta, S0, M0)

        predict = post.dot(basis_x.T)
        train_error = np.sum((predict - t) ** 2) / 2 / len(predict)
        rms += sqrt(2 * train_error)

        plt.scatter(train_x, train_t, color=cmap(0.9), s=10)
        plt.plot(sort_x, predict, color='orange', linewidth=1, label="Prediction")

    print("N = {}".format(n))
    print("RMS: %.2f" % (rms/step))
    plt.show()

    contour(MN.flatten(), SN, n)

    y_up = []
    y_down = []
    y_mean = []
    for t_x in sort_x:
        b_x = sigmoid_basis([t_x], M, s).flatten()
        mean = MN.T.dot(b_x)
        stddev = np.sqrt(1/beta + b_x.T.dot(SN).dot(b_x))
        y_up.append(mean+stddev)
        y_down.append(mean-stddev)
        y_mean.append(mean)

    plt.suptitle("Predictive distributions (N = {})".format(n))
    plt.scatter(train_x, train_t, color=cmap(0.9), s=10)
    plt.plot(sort_x, y_up, color=cmap(0.2), linewidth=0.5)
    plt.plot(sort_x, y_down, color=cmap(0.2), linewidth=0.5)
    plt.plot(sort_x, y_mean, c='red', linewidth=0.5)
    plt.show()

