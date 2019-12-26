import pandas as pd
import numpy as np 

x_train = pd.read_csv('x_train.csv', header=None)
t_train = pd.read_csv('t_train.csv', header=None)
x_train = x_train.values
t_train = t_train.values