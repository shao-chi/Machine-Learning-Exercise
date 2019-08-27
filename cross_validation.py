import pandas as pd 
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, LeavePOut

# percentage of training data
TRAIN_SPLIT = 0.7
# K of K fold
NUM_SPLITS = 5
# p of leave-p-out
P_VAL = 3

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
"""
1: [ T T - - - - ]
2: [ - - T T - - ]
3: [ - - - - T T ]
"""
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

# Leave-P-Out Cross Validation (LPOCV)
"""
1: [ T T - - ]
2: [ T - T - ]
3: [ T - - T ]
4: [ - T T - ]
5: [ - T - T ]
6: [ - - T T ]
"""
def print_result(split_data):
    """
    Prints the result of either a LPOCV or LOOCV operation
    Args:
        split_data: The resulting (train, test) split data
    """
    for train, test in split_data:
        bar = ["-"] * ((len(train) + len(test)))
        for i in test:
            bar[i] = "T"
            
        print("# [ {} ]".format(" ".join(bar)))

data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])

# 2 method of leave out
loocv = LeaveOneOut()
lpocv = LeavePOut(p=P_VAL)

print("\nLeave P / ONE out")

split_data = loocv.split(data)
print("Leave ONE out cross validation")
print_result(split_data)

split_data = lpocv.split(data)
print("Leave P out cross validaiton")
print_result(split_data)