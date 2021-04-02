import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../datafile/MNIST/mnist_train.csv')
print(df)
#2-D visualization using PCA

d = df.drop('label',axis=1)
print(d.head())


