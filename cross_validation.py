import pandas as pd 
from sklearn import datasets
from sklearn.model_selection import train_test_split

# percentage of training data
TRAIN_SPLIT = 0.7

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
print("Holdout method: split out training and testing dataset")
print("Total data points: {}".format(total))
print("# training data points: {} ({}%)" \
        .format(len(x_train), 100*len(x_train)/total))
print("# testing data points: {} ({}%)" \
        .format(len(x_test), 100*len(x_test)/total))