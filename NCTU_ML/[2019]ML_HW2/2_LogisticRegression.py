from PIL import Image
import numpy as np
from math import log
import matplotlib.pyplot as plt

train_x = list()
valid_x = list()
train_t = list()
valid_t = list()
for c in range(0, 5):
    for i in range(1, 11):
        img = Image.open('./dataset/Faces/s{}/{}.pgm'.format(c+1, i)) 
        data = np.array(img).flatten()
        one_hot = np.zeros(5)
        one_hot[c] = 1
        if i <= 5: 
            train_x.append(data)
            train_t.append(one_hot)
        else: 
            valid_x.append(data)
            valid_t.append(one_hot)

train_x = np.array(train_x).astype(float)
train_t = np.array(train_t).astype(float)
valid_x = np.array(valid_x).astype(float)
valid_t = np.array(valid_t).astype(float)

# normalization
train_x = (train_x - train_x.mean(axis=0)) / train_x.std(axis=0)
valid_x = (valid_x - valid_x.mean(axis=0)) / valid_x.std(axis=0)

def error(y, t):
    err = 0
    for i in range(len(y)):
        for j in range(t.shape[1]):
            err -= t[i][j] * log(y[i][j])

    return err

def gradient_descent(train_x, train_t, valid_x, valid_t, epochs, learn_rate):
    w = np.zeros((train_x.shape[1], train_t.shape[1]))
    train_error = []
    valid_error = []
    accuracy_train = []
    accuracy_valid = []
    for e in range(epochs):
        train_y = train_x.dot(w)
        valid_y = valid_x.dot(w)
        exp_train_y = np.exp(train_y)
        exp_valid_y = np.exp(valid_y)
        softmax_train_y = []
        softmax_valid_y = []
        train_true = 0
        valid_true = 0 
        for j in range(len(train_y)):
            t_soft = np.array(exp_train_y[j]/exp_train_y[j].sum())
            v_soft = np.array(exp_valid_y[j]/exp_valid_y[j].sum())
            softmax_train_y.append(t_soft)
            softmax_valid_y.append(v_soft)
            if np.argmax(t_soft) == np.argmax(train_t[j]):
                train_true += 1
            if np.argmax(v_soft) == np.argmax(valid_t[j]):
                valid_true += 1
        
        w = w + learn_rate * train_x.T.dot(train_t)

        train_error.append(error(softmax_train_y, train_t))
        valid_error.append(error(softmax_valid_y, valid_t))

        accuracy_train.append(train_true / len(train_x))
        accuracy_valid.append(valid_true / len(valid_x))

    predict_train = np.argmax(softmax_train_y, axis=1) + 1
    predict_valid = np.argmax(softmax_valid_y, axis=1) + 1

    plt.suptitle("Gradient Descent Error (learning rate = {})".format(learn_rate))
    x_axis = np.linspace(1, epochs, epochs)
    plt.plot(x_axis, train_error, color='blue', linewidth=1, label="Training Error")
    plt.plot(x_axis, valid_error, color='orange', linewidth=1, label="Validation Error")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    
    plt.suptitle("Gradient Descent Accuracy (learning rate = {})".format(learn_rate))
    plt.plot(x_axis, accuracy_train, color='blue', linewidth=1, label="Training Accuracy")
    plt.plot(x_axis, accuracy_valid, color='orange', linewidth=1, label="Validation Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

    print("Gradient Descent (learning rate = {})".format(learn_rate))
    print(train_error, accuracy_train)
    print(valid_error, accuracy_valid)

    return predict_train, predict_valid

learning_rate = [0.0001, 0.00001]
for l_r in learning_rate:
    predict_train, predict_valid = gradient_descent(train_x, train_t, valid_x, valid_t, 30, l_r)
    print(predict_train)
    print(predict_valid)

def PCA(x, n_c, state):
    x -= np.mean(x, axis = 0)
    cov = np.cov(x, rowvar = False)
    eigen_val , eigen_vec = np.linalg.eigh(cov)
    idx = np.argsort(eigen_val)[::-1][:n_c]
    eigen_vec = eigen_vec[:, idx]
    eigen_val = eigen_val[idx]
    pca_x = np.dot(x, eigen_vec) 
    print(pca_x.shape)
    print(eigen_vec.shape)
    print(eigen_val.shape)
    # np.save('./{}_pca_x_{}.npy'.format(state, n_c), pca_x)
    # np.save('./{}_eigen_vec_{}.npy'.format(state, n_c), eigen_vec)
    # np.save('./{}_eigen_val_{}.npy'.format(state, n_c), eigen_val)

dim = [2, 5, 10]
# for d in dim:
#     PCA(train_x, d, 'train')
#     PCA(valid_x, d, 'valid')

# PCA(train_x, 10304, 'all')

pca_x = np.load('./dataset/Faces/all_pca_x_10304.npy')
eigen_vec = np.load('./dataset/Faces/all_eigen_vec_10304.npy')
mean = train_x.mean(axis=0)
std = eigen_vec.std(axis=0).mean()
for v in range(0, 10, 2):
    fig, ax = plt.subplots()
    ax.scatter(pca_x[0:10, v], pca_x[0:10, v+1])
    ax.scatter(pca_x[11:20, v], pca_x[11:20, v+1])
    ax.scatter(pca_x[21:30, v], pca_x[21:30, v+1])
    ax.scatter(pca_x[31:40, v], pca_x[31:40, v+1])
    ax.scatter(pca_x[41:50, v], pca_x[41:50, v+1])
    for axis, color in zip(eigen_vec[v:v+2], ['red', 'green']):
        start = mean[v:v+2]
        end = (mean + std * axis)[v:v+2]
        print(start, std, axis[v:v+2])
        plt.annotate('', xy=end, xytext=start, arrowprops=dict(facecolor=color, width=0.1))
    ax.set_aspect('equal')
    plt.show()

def newton_raphson(train_x, train_t, valid_x, valid_t, epochs, x_dim):
    w = np.zeros((train_x.shape[1], train_t.shape[1]))
    train_error = []
    valid_error = []
    accuracy_train = []
    accuracy_valid = []
    for e in range(epochs):
        train_y = train_x.dot(w)
        valid_y = valid_x.dot(w)
        exp_train_y = np.exp(train_y)
        exp_valid_y = np.exp(valid_y)
        softmax_train_y = []
        softmax_valid_y = []
        train_true = 0
        valid_true = 0 
        R = np.identity(train_x.shape[0])
        for j in range(len(train_y)):
            t_soft = np.array(exp_train_y[j]/exp_train_y[j].sum())
            v_soft = np.array(exp_valid_y[j]/exp_valid_y[j].sum())
            softmax_train_y.append(t_soft)
            softmax_valid_y.append(v_soft)
            if np.argmax(t_soft) == np.argmax(train_t[j]):
                train_true += 1
            if np.argmax(v_soft) == np.argmax(valid_t[j]):
                valid_true += 1

            y_n = t_soft[np.argmax(train_t[j])]
            R[j][j] = y_n * (1 - y_n)

        softmax_train_y = np.array(softmax_train_y)
        R = np.array(R)
        z = train_x.dot(w) - np.linalg.inv(R).dot(softmax_train_y - train_t)
        w = np.linalg.inv(train_x.T.dot(R).dot(train_x)).dot(train_x.T).dot(R).dot(z)

        
        train_error.append(error(softmax_train_y, train_t))
        valid_error.append(error(softmax_valid_y, valid_t))

        accuracy_train.append(train_true / len(train_x))
        accuracy_valid.append(valid_true / len(valid_x))

    predict_train = np.argmax(softmax_train_y, axis=1) + 1
    predict_valid = np.argmax(softmax_valid_y, axis=1) + 1

    plt.suptitle("Newton Raphson Error (PCA {})".format(x_dim))
    x_axis = np.linspace(1, epochs, epochs)
    plt.plot(x_axis, train_error, color='blue', linewidth=1, label="Training Error")
    plt.plot(x_axis, valid_error, color='orange', linewidth=1, label="Validation Error")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    
    plt.suptitle("Newton Raphson Accuracy (PCA {})".format(x_dim))
    plt.plot(x_axis, accuracy_train, color='blue', linewidth=1, label="Training Accuracy")
    plt.plot(x_axis, accuracy_valid, color='orange', linewidth=1, label="Validation Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

    print("Newton Raphson (PCA {})".format(x_dim))
    print(train_error, accuracy_train)
    print(valid_error, accuracy_valid)

    return predict_train, predict_valid

for d in dim:
    pca_train_x = np.load('./dataset/Faces/train_pca_x_{}.npy'.format(d))
    pca_valid_x = np.load('./dataset/Faces/valid_pca_x_{}.npy'.format(d))
    predict_train, predict_valid = newton_raphson(pca_train_x, train_t, pca_valid_x, valid_t, 10, d)
    print(predict_train)
    print(predict_valid)
