import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

def KNN(train_x, train_t, valid_x, valid_t, K):

    def distance(a, b):
        return np.sqrt(np.sum(np.power(a - b, 2)))

    Class = np.zeros((len(valid_x), 3)) # 3 classes
    for i in range(len(valid_x)):

        distances = [distance(train_x[j], valid_x[i]) for j in range(len(train_x))]
        neighbors = np.argsort(distances)[:K]

        for j in range(len(neighbors)):
            Class[i][int(train_t[neighbors[j]])] += 1

    predict = np.argmax(Class, axis=1)
    print('KNN (K = {})'.format(K))
    print(valid_t)
    print(predict)

    accuracy = np.sum(predict == valid_t) / len(valid_t)
    print(accuracy)

    return accuracy

data = pd.read_csv('./dataset/Pokemon.csv')
data = data.values

class_name = list(set(data[:, 2]))
legendary_name = list(set(data[:, 11]))

for i in range(len(data)):
    data[i][2] = class_name.index(data[i][2])
    data[i][11] = legendary_name.index(data[i][11])

target = data[:, 2].astype(float)
feature = data[:, 3:].astype(float)

norm_feature = (feature - feature.mean(axis=0)) / feature.std(axis=0)

train_x = norm_feature[:120,:]
train_t = target[:120]
valid_x = norm_feature[120:, :]
valid_t = target[120:]

acc = []
x_axis = np.linspace(1, 10, 10)
for k in range(1, 11):
    acc.append(KNN(train_x, train_t, valid_x, valid_t, k))

plt.suptitle("K-nearest neighbors Accuracy")
plt.plot(x_axis, acc, linewidth=1)
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.show()

def PCA(x, n_c):
    x -= np.mean(x, axis = 0)
    cov = np.cov(x, rowvar = False)
    eigen_val , eigen_vec = np.linalg.eigh(cov)
    idx = np.argsort(eigen_val)[::-1][:n_c]
    eigen_vec = eigen_vec[:, idx]
    eigen_val = eigen_val[idx]
    pca_x = np.dot(x, eigen_vec)

    return pca_x

dim = [7, 6, 5]
for d in dim:
    pca_x = PCA(norm_feature, d)
    train_x = pca_x[:120,:]
    valid_x = pca_x[120:, :]

    acc = []
    for k in range(1, 11):
        print('PCA {}'.format(d))
        acc.append(KNN(train_x, train_t, valid_x, valid_t, k))

    plt.suptitle("K-nearest neighbors Accuracy (PCA {})".format(d))
    plt.plot(x_axis, acc, linewidth=1)
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.show()