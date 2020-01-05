import pandas as pd
import numpy as np 
from sklearn import svm
from matplotlib import pyplot as plt

def linear_kernel(x):
    return x

def poly_kernel(x):
    return np.vstack((x[:, 0]**2, np.sqrt(2)*x[:, 0]*x[:, 1], x[:, 1]**2)).T 

def PCA(x, n_c):
    # x -= np.mean(x, axis = 0)
    cov = np.cov((x - np.mean(x, axis = 0)), rowvar = False)
    # print(cov)
    eigen_val, eigen_vec = np.linalg.eigh(cov)
    idx = np.argsort(eigen_val)[::-1][:n_c]
    eigen_vec = eigen_vec[:, idx]
    eigen_val = eigen_val[idx]
    pca_x = np.dot((x - np.mean(x, axis = 0)), eigen_vec)
    return pca_x
    # return idx

def predict(x, w, b, kernel):
    pred = []
    c = [(0, 1), (0, 2), (1, 2)]
    for i in range(len(x)):
        votes = np.array([0, 0, 0])
        for sv in range(len(c)):
            c1, c2 = c[sv]
            weight = w[sv]
            bias = b[sv]
            y = weight.dot(kernel(x[i]).T) + bias
            if y > 0:
                votes[c1] += 1
            else:
                votes[c2] += 1
        # print(votes)
        pred.append(votes.argmax())
    return np.array(pred)

x_train = pd.read_csv('x_train.csv', header=None)
t_train = pd.read_csv('t_train.csv', header=None)
x_train = x_train.values
t_train = t_train.values

pca_x = PCA(x_train, 2)
pca_x = (pca_x - pca_x.mean(axis=0)) / pca_x.std(axis=0)
x_c0 = pca_x[:100]
x_c1 = pca_x[100:200]
x_c2 = pca_x[200:]

### class 0, 1
x = pca_x[:200]
t = np.concatenate((np.ones((100)), np.zeros((100))))
c01_svm = svm.SVC(kernel='linear')
c01_svm.fit(x, t)
c01_s_v = c01_svm.support_vectors_
w01 = c01_svm.coef_
b01 = c01_svm.intercept_
c01_decision = w01.dot(x.T)+b01

### class 0, 2
x = np.concatenate((pca_x[:100], pca_x[200:]))
t = np.concatenate((np.ones((100)), np.zeros((100))))
c02_svm = svm.SVC(kernel='linear')
c02_svm.fit(x, t)
c02_s_v = c02_svm.support_vectors_
w02 = c02_svm.coef_
b02 = c02_svm.intercept_
c02_decision = w02.dot(x.T)+b02

### class 1, 2
x = pca_x[100:]
t = np.concatenate((np.ones((100)), np.zeros((100))))
c12_svm = svm.SVC(kernel='linear')
c12_svm.fit(x, t)
c12_s_v = c12_svm.support_vectors_
w12 = c12_svm.coef_
b12 = c12_svm.intercept_
c12_decision = w12.dot(x.T)+b12

weight = np.array([w01[0], w02[0], w12[0]])
bias = np.array([b01[0], b02[0], b12[0]])
s_v = np.concatenate((c01_s_v, c02_s_v, c12_s_v))
# decision = np.array([c01_decision[0], c02_decision[0], c12_decision[0]])
min_x = np.min(pca_x[:, 0]) - 1
max_x = np.max(pca_x[:, 0]) + 1
min_y = np.min(pca_x[:, 1]) - 1
max_y = np.max(pca_x[:, 1]) + 1
xx, yy = np.meshgrid(np.arange(min_x, max_x, 0.01), np.arange(min_y, max_y, 0.01))
xy = np.hstack((xx.flatten().reshape((len(xx.flatten()), 1)), yy.flatten().reshape((len(yy.flatten()), 1))))
pred = predict(xy, weight, bias, linear_kernel).reshape(xx.shape)

plt.scatter(x_c0[:, 1], x_c0[:, 0], marker='o', c='green', label='class 1')
plt.scatter(x_c1[:, 1], x_c1[:, 0], marker='X', c='orange', label='class 2')
plt.scatter(x_c2[:, 1], x_c2[:, 0], marker='^', c='blue', label='class 3')
plt.scatter(s_v[:, 1], s_v[:, 0], marker='2', c='black',  label='support vecter')
# plt.plot(xx, yy)
plt.contour(xx, yy, pred.reshape(xx.shape), colors='k', levels=[-1, 0, 1], alpha=0.2, linestyles=['--', '-', '--'])
plt.legend()
plt.show()


### class 0, 1
x = pca_x[:200]
t = np.concatenate((np.ones((100)), np.zeros((100))))
c01_svm = svm.SVC(kernel='poly', degree=2)
c01_svm.fit(x, t)
c01_s_v = c01_svm.support_vectors_

### class 0, 2
x = np.concatenate((pca_x[:100], pca_x[200:]))
t = np.concatenate((np.ones((100)), np.zeros((100))))
c02_svm = svm.SVC(kernel='poly', degree=2)
c02_svm.fit(x, t)
c02_s_v = c02_svm.support_vectors_

### class 1, 2
x = pca_x[100:]
t = np.concatenate((np.ones((100)), np.zeros((100))))
c12_svm = svm.SVC(kernel='poly', degree=2)
c12_svm.fit(x, t)
c12_s_v = c12_svm.support_vectors_

s_v = np.concatenate((c01_s_v, c02_s_v, c12_s_v))
# decision = np.array([c01_decision[0], c02_decision[0], c12_decision[0]])
min_x = np.min(pca_x[:, 0]) - 1
max_x = np.max(pca_x[:, 0]) + 1
min_y = np.min(pca_x[:, 1]) - 1
max_y = np.max(pca_x[:, 1]) + 1
xx, yy = np.meshgrid(np.arange(min_x, max_x, 0.01), np.arange(min_y, max_y, 0.01))
xy = np.hstack((xx.flatten().reshape((len(xx.flatten()), 1)), yy.flatten().reshape((len(yy.flatten()), 1))))

c01_decision = c01_svm.decision_function(xy).reshape((len(xy), 1))
c02_decision = c02_svm.decision_function(xy).reshape((len(xy), 1))
c12_decision = c12_svm.decision_function(xy).reshape((len(xy), 1))
decision = np.concatenate((c01_decision, c02_decision, c12_decision), axis=1)

pred = []
c = [(0, 1), (0, 2), (1, 2)]
for i in range(len(xy)):
    votes = np.array([0, 0, 0])
    for sv in range(len(c)):
        c1, c2 = c[sv]
        d = decision[i][sv]
        if d > 0:
            votes[c1] += 1
        else:
            votes[c2] += 1
    pred.append(votes.argmax())

pred = np.array(pred)
plt.scatter(x_c0[:, 1], x_c0[:, 0], marker='o', c='green', label='class 1')
plt.scatter(x_c1[:, 1], x_c1[:, 0], marker='X', c='orange', label='class 2')
plt.scatter(x_c2[:, 1], x_c2[:, 0], marker='^', c='blue', label='class 3')
plt.scatter(s_v[:, 1], s_v[:, 0], marker='2', c='black',  label='support vecter')
# plt.plot(xx, yy)
plt.contourf(xx, yy, pred.reshape(xx.shape), alpha=0.3, cmap=plt.cm.coolwarm)
plt.legend()
plt.show()