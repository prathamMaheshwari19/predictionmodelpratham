import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

df = pd.read_csv('/content/sample_data/mnist_test.csv')
print(df.head(),"\n")
df.head()
df.shape
df.describe()