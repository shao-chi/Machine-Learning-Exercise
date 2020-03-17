import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt

def exponential_quadratic(x, xx, theta):
    n = len(x)
    return (theta[0] * np.exp(-(theta[1]/2) * (np.subtract.outer(x, xx)**2))).reshape(n, n) + theta[2] + theta[3]*x.dot(xx.T)

def ARD_exponential_quadratic(x, xx, theta):
    n = len(x)
    eta = np.random.rand(n).reshape(n, 1)/5
    # eta = 0.05
    # return (theta[0] * np.exp(-(np.power(x-xx,2).reshape(n, 1) * eta / 2))) + theta[1] + theta[2]*x.dot(xx.T)
    return (theta[0] * np.exp(-(np.subtract.outer(x, xx) **2 ) * eta / 2)).reshape(n, n) + theta[1] + theta[2]*x.dot(xx.T)
    # return (theta[0] * np.exp(-((x[0] - xx[0])**2) * eta) / 2) + theta[1] + theta[2]* x[0] * xx[0]

def guassian_process(xNN, x, t, kernel, theta, c_n):
    k = np.array([kernel(x[i:i+1], xNN, theta) for i in range(len(x))]).reshape(len(x), 1)
    c_n = np.linalg.inv(c_n)
    # print(c_n)
    mean = k.T.dot(c_n).dot(t)
    sigma = kernel(xNN, xNN, theta) + 1 - k.T.dot(c_n).dot(k)
    # print(mean, sigma)
    return mean[0], sigma[0]

def rms(mean, t):
    return np.sqrt((np.sum(mean - t) ** 2) / len(t))

data = loadmat('gp.mat')
x = data['x'] # (100, 1)
t = data['t'] # (100, 1)

x_train = x[:60]
t_train = t[:60]
x_test = x[60:]
t_test = t[60:]

theta = np.array([[0, 0, 0, 1], [1, 4, 0, 0], [1, 4, 0, 5], [1, 32, 5, 5]])
# theta = np.array([[1, 4, 0, 0], [5, 4, 0, 0], [8, 4, 0, 0], [15, 4, 0, 0]])
# theta = np.array([[8, 4, 5, 0], [8, 4, 10, 0], [8, 4, 15, 0], [8, 4, 50, 0]])
theta = [[8, 4, i, 5] for i in range(0,101)]
rms_train = []
rms_test = []
for ta in theta:
    c_n = exponential_quadratic(x_train, x_train, ta) + np.identity(len(x_train))
    # print(c_n)
    x_axis = np.linspace(0, 2, 300)
    y_axis = np.array( \
        [guassian_process(np.array([xNN]), x_train, t_train, exponential_quadratic, ta, c_n) \
            for xNN in x_axis]) \
        .reshape(len(x_axis), 2)

    mean = y_axis[:, 0]
    sigma = np.sqrt(y_axis[:, 1])

    train = np.array( \
        [guassian_process(np.array([xNN]), x_train, t_train, exponential_quadratic, ta, c_n) \
            for xNN in x_train]) \
        .reshape(len(x_train), 2)
    mean_train = train[:, 0]

    test = np.array( \
        [guassian_process(np.array([xNN]), x_train, t_train, exponential_quadratic, ta, c_n) \
            for xNN in x_test]) \
        .reshape(len(x_test), 2)
    mean_test = test[:, 0]

    RMS_train = rms(mean_train, t_train)
    RMS_test = rms(mean_test, t_test)
    print('θ = {}'.format(ta))
    print('Training rms = {}'.format(RMS_train))
    print('Testing rms = {}'.format(RMS_test))

    rms_train.append(RMS_train)
    rms_test.append(RMS_test)

    # plt.figure()
    # plt.suptitle('θ = {}'.format(ta))
    # plt.title('Training RMS = {}, Testing RMS = {}'.format(RMS_train, RMS_test))
    # plt.plot(x_axis, mean, c='red')
    # plt.fill_between(x_axis, mean+sigma, mean-sigma, color='orange')
    # plt.scatter(x_train, t_train, marker='.', c='blue')
    # plt.scatter(x_test, t_test, marker='.', c='green')
    # plt.show()

rms_axis = np.linspace(0, 100, 101)
plt.figure()
plt.suptitle('θ = {}'.format(ta))
plt.title('Training RMS = {}, Testing RMS = {}'.format(RMS_train, RMS_test))
plt.plot(rms_axis, rms_train, c='blue')
plt.plot(rms_axis, rms_test, c='green')
plt.show()

# theta = np.array([[0, 0, 1], [1, 0, 0], [1, 0, 5], [2, 50, 20]])
# theta = [[0.3, 5000, 2000]]
# for ta in theta:
#     c_n = ARD_exponential_quadratic(x_train, x_train, ta) + np.identity(len(x_train))
#     # print(c_n)
#     x_axis = np.linspace(0, 2, 300)
#     y_axis = np.array( \
#         [guassian_process(np.array([xNN]), x_train, t_train, ARD_exponential_quadratic, ta, c_n) \
#             for xNN in x_axis]) \
#         .reshape(len(x_axis), 2)

#     mean = y_axis[:, 0]
#     sigma = np.sqrt(y_axis[:, 1])

#     train = np.array( \
#         [guassian_process(np.array([xNN]), x_train, t_train, ARD_exponential_quadratic, ta, c_n) \
#             for xNN in x_train]) \
#         .reshape(len(x_train), 2)
#     mean_train = train[:, 0]

#     test = np.array( \
#         [guassian_process(np.array([xNN]), x_train, t_train, ARD_exponential_quadratic, ta, c_n) \
#             for xNN in x_test]) \
#         .reshape(len(x_test), 2)
#     mean_test = test[:, 0]

#     RMS_train = rms(mean_train, t_train)
#     RMS_test = rms(mean_test, t_test)
#     print('θ = {}'.format(ta))
#     print('Training rms = {}'.format(RMS_train))
#     print('Testing rms = {}'.format(RMS_test))

#     plt.figure()
#     plt.suptitle('θ = {}'.format(ta))
#     plt.title('Training RMS = {}, Testing RMS = {}'.format(RMS_train, RMS_test))
#     plt.plot(x_axis, mean, c='red')
#     plt.fill_between(x_axis, mean+sigma, mean-sigma, color='orange')
#     plt.scatter(x_train, t_train, marker='.', c='blue')
#     plt.scatter(x_test, t_test, marker='.', c='green')
#     plt.show()