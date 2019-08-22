import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import datasets, linear_model
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def mse(test, predict):
    result = 0
    for t, p in zip(test, predict):
        result += (p - t) ** 2
    result /= len(test)
    return result

# create data set
x, y = make_regression(n_samples=500, n_features=1, noise=25, random_state=25)

# split data set into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

# create a linear regression object
linear_regress = linear_model.LinearRegression()

# train the model by training set
linear_regress.fit(x_train, y_train)

# makke predictions using testing set
y_predict = linear_regress.predict(x_test)

# evaluate mean square error
mse_sk = mean_squared_error(y_test, y_predict)
mse_no_modules = mse(y_test, y_predict)

print("MSE by sklearn: ", mse_sk)
print("MSE by myself: ", mse_no_modules)

# plot the test data
sns.set_style("darkgrid")
sns.regplot(x_test, y_test, fit_reg=False)

# Remove ticks from the plot
# plt.xticks([])
# plt.yticks([])

# plt.tight_layout()
plt.show()

# plot the predicted data
sns.set_style("darkgrid")
sns.regplot(x_test, y_test, fit_reg=False)
plt.plot(x_test, y_predict, color='black')

plt.show()

# nonlinear data set
y_nonlinear = y ** 2

x_train, x_test, y_train, y_test = train_test_split(x, y_nonlinear, \
                                                        random_state=0)

# create regression object based on gradient descent
sgd_regress = linear_model.SGDRegressor(max_iter=10000, tol=0.001)

# train the model
sgd_regress.fit(x_train, y_train)

# predict
y_predict = sgd_regress.predict(x_test)

mse_sk = mean_squared_error(y_test, y_predict)
mse_no_modules = mse(y_test, y_predict)

print("MSE by sklearn: ", mse_sk)
print("MSE by myself: ", mse_no_modules)

# plot the nonlinear data
sns.set_style("darkgrid")
sns.regplot(x, y_nonlinear, fit_reg=False)
plt.plot(x_test, y_predict, color='black')

plt.show()

# exponential regression
y_exp = np.exp((y + abs(y.min())) / 75)

# plot the exponential data
sns.set_style("darkgrid")
sns.regplot(x, y_exp, fit_reg=False)

plt.show()