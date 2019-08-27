import pandas as pd 
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold

# percentage of training data
TRAIN_SPLIT = 0.7
# K of K fold
NUM_SPLITS = 5

# dataset columns
columns = ['age', 'sex', 'bmi', 'map', 'tc', 'ldl', 'hdl', 'tch', 'ltg', 'glu']

# load the diabetes dataset
dataset = datasets.load_diabetes()

# create a pandas dataframe
df = pd.DataFrame(dataset.data, columns=columns)

# split dataset via the holdout method
x_train, x_test, y_train, y_test = train_test_split(
    df, dataset.target, train_size=TRAIN_SPLIT, test_size=1-TRAIN_SPLIT
)

total = len(df.index)
print("Total data points: {}\n".format(total))

print("Holdout method: split out training and testing dataset")
print("# training data points: {} ({}%)" \
        .format(len(x_train), 100*len(x_train)/total))
print("# testing data points: {} ({}%)" \
        .format(len(x_test), 100*len(x_test)/total))

# perform k fold
dataset.target = [[dataset.target[i]] for i in range(len(dataset.target))]
data_target = np.concatenate((dataset.data, dataset.target), axis=1)

kfold = KFold(n_splits=NUM_SPLITS)
split_data = kfold.split(data_target)

print("\nK-Folds method: split out k folds of test data, \
    work until all points has been used for testing")
print("K-Fold split (with n_splits = {}):".format(NUM_SPLITS))
for train, test in split_data:
    print("# train: ", len(train), " ({})".format(100*len(train)/total), \
            ", test: ", len(test), " ({})".format(100*len(test)/total))
