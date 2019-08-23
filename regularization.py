import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np

# create a data set for analysis
x, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=0)
y = y ** 2

# pipeline lets us set the steps for our modeling
# quadratic model
model = Pipeline([('poly', PolynomialFeatures(degree=2)), \
                    ('linear', LinearRegression(fit_intercept=False))])

model = model.fit(x, y)

# predict
x_plot = np.linspace(min(x)[0], max(x)[0], 100)
x_plot = x_plot[:, np.newaxis]
y_predict = model.predict(x_plot)

# plot
sns.set_style("darkgrid")
plt.plot(x_plot, y_predict, color='black')
plt.scatter(x, y, marker='o')
plt.xticks(())
plt.yticks(())
plt.tight_layout()
plt.show()