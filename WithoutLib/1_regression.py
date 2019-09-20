from model.regression import LinearRegression
from utils.MSE import mean_square_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

def plot_loss(loss):
    """
    Training error plot
    """
    n = len(loss)
    training, = plt.plot(range(n), loss, label="Training Loss")
    plt.legend(handles=[training])
    plt.title("Error Plot")
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Iterations')
    plt.show()

def plot_result(x_train, y_train, x_test, y_test, X, Y_predict, mse):
    # Color map
    cmap = plt.get_cmap('viridis')

    # Plot the results
    m1 = plt.scatter(x_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(x_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X, Y_predict, color='black', linewidth=1, label="Prediction")
    plt.suptitle("Linear Regression")
    plt.title("MSE of testing: %.2f" % mse, fontsize=10)
    plt.xlabel('Day')
    plt.ylabel('Temperature in Celcius')
    plt.legend((m1, m2), ("Training data", "Test data"), loc='lower right')
    plt.show()

X, Y = make_regression(n_samples=300, n_features=1, noise=20)

split = 240
x_train = X[:split]
y_train = Y[:split]
x_test = X[split:]
y_test = Y[split:]

# train linear regression model
loss, weight = LinearRegression(x_train, y_train, n_iter=150, learn_rate=0.0001)

plot_loss(loss)

x_test_p = np.insert(x_test, 0, 1, axis=1)
predict = x_test_p.dot(weight)
mse = mean_square_error(y_test, predict)
print ("LinearRegression - Mean squared error of Testing: %s" % (mse))

X_p = np.insert(X, 0, 1, axis=1)
Y_predict = X_p.dot(weight)
mse_all = mean_square_error(Y, Y_predict)
print ("LinearRegression - Mean squared error of all data: %s" % (mse_all))

plot_result(x_train, y_train, x_test, y_test, X, Y_predict, mse)