import numpy as np 
from matplotlib import pyplot as plt

# read the data
xcor = np.load('denoise.npy')
# n = np.random.rand(1000)/-100+np.random.rand(1000)/-10+np.random.rand(1000)/10
x_axis = np.linspace(0, len(xcor)-1, len(xcor))
plt.plot(x_axis, xcor)
plt.show()

# set matrix D
D = np.zeros((999, 1000))
for i in range(D.shape[0]):
    D[i][i] = -1
    D[i][i+1] = 1

# calculate X
I = np.identity((1000))
lamda = [0.001, 0.1, 1, 10, 100, 500, 1000, 10000]
for l in lamda:
    X = np.linalg.inv(I + l * D.T.dot(D)).dot(xcor)
    noise = xcor - X

    plt.plot(x_axis, noise, c='orange')
    plt.plot(x_axis, X, c='blue')
    plt.title("λ = {}".format(l))
    plt.xlabel('i')
    plt.ylabel('x')
    plt.show()

# sequential learning
X = xcor
for step in range(0,8):
    XX = np.linalg.inv(I + 10 * D.T.dot(D)).dot(X)
    noise = X - XX

    plt.plot(x_axis, noise, c='orange')
    plt.plot(x_axis, XX, c='blue')
    plt.title("λ = 10 (step = {})".format(step + 1))
    plt.xlabel('i')
    plt.ylabel('x')
    plt.show()

    X = XX

    # n = np.linalg.inv(I + l * D.T.dot(D)).dot(noise)
    # plt.plot(x_axis, noise-n, c='orange')
    # plt.plot(x_axis, n, c='blue')
    # plt.title("λ = {}".format(l))
    # plt.xlabel('i')
    # plt.ylabel('x')
    # plt.show()
