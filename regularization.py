import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import numpy as np
from MSE import mse

DEGREE = 12
ALPHA = 5.0

# create a data set for analysis
x, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=0)
y = y ** 2

# pipeline lets us set the steps for our modeling
# quadratic (or higher degree) model
model = Pipeline([('poly', PolynomialFeatures(degree=DEGREE)), \
                    ('linear', LinearRegression(fit_intercept=False))])

# Ridge regression
ridgeModel = Pipeline([('poly', PolynomialFeatures(degree=DEGREE)), \
                    ('ridge', Ridge(alpha=ALPHA))])

# Lasso regression
lassoModel = Pipeline([('poly', PolynomialFeatures(degree=DEGREE)), \
                    ('lasso', Lasso(alpha=ALPHA))])

model = model.fit(x, y)
ridgeModel = ridgeModel.fit(x, y)
lassoModel= lassoModel.fit(x, y)

# predict
x_plot = np.linspace(min(x)[0], max(x)[0], 100)
x_plot = x_plot[:, np.newaxis]
y_predict = model.predict(x_plot)
y_ridgePredict = ridgeModel.predict(x_plot)
y_lassoPredict = lassoModel.predict(x_plot)

# evaluate mean square error
mse_sk = mean_squared_error(y, y_predict)
mse_i = mse(y, y_predict)
mse_sk_ridge = mean_squared_error(y, y_ridgePredict)
mse_i_ridge = mse(y, y_ridgePredict)
mse_sk_lasso = mean_squared_error(y, y_lassoPredict)
mse_i_lasso = mse(y, y_lassoPredict)

print("degree: ", DEGREE)
print("MSE by sklearn: ", mse_sk)
print("MSE by myself: ", mse_i)
print("MSE (Ridge) by sklearn: ", mse_sk_ridge)
print("MSE (Ridge) by myself: ", mse_i_ridge)
print("MSE (Lasso) by sklearn: ", mse_sk_lasso)
print("MSE (Lasso) by myself: ", mse_i_lasso)

# plot
sns.set_style("darkgrid")
plt.plot(x_plot, y_predict, color='black')
plt.plot(x_plot, y_ridgePredict, color='red')
plt.plot(x_plot, y_lassoPredict, color='green')
plt.scatter(x, y, marker='o')
plt.xticks(())
plt.yticks(())
plt.tight_layout()
plt.show()