# 01.00 setup module

# data analysis modules
import pandas as pd
import numpy as np

# plot module and config settings
import matplotlib.pyplot as plt
%matplotlib inline

# plot module and config settings
import seaborn as sns
sns.set_theme()
sns.set(color_codes=True)
sns.set_palette(sns.color_palette("muted"))
sns.set(rc={'figure.figsize':(11.7,8.27)})

# ml module
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

# read data and view contents
# https://geohub.lacity.org/datasets/clean-streets-index-2016-q1
df = pd.read_csv('https://opendata.arcgis.com/datasets/661fe82f121a4eb795a1d3884a06f1da_0.csv')
print('Output first 5 rows:')
print(df.head(), '\n')
print('Output dataset info:')
print(df.info(), '\n')
print('Output unique values for segment name:')
print(df['FullName'].unique(), '\n')

# filter data for model train/predict
# https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html
df_subset = df[[
    'OBJECTID',
    'SegmentID',
    'Seg_ID_Sco',
    'Seg_Bk_Sco',
    'Seg_LL_Sco',
    'Seg_Wd_Sco',
    'SegScore'
]]

# split train and test data
# https://stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas
print('Output df shape:')
print(df.shape, '\n')
train, test = train_test_split(df_subset, test_size=0.2)
print('Output train df shape:')
print(train.shape, '\n')
print('Output test df shape:')
print(test.shape, '\n')

# train rf classifier
# https://stackabuse.com/classification-in-python-with-scikit-learn-and-pandas/
y_train = train.iloc[:,6]
X_train = train.iloc[:,:6]
y_test = test.iloc[:,6]
X_test = test.iloc[:,:6]

# encode labels as continuous
# https://stackoverflow.com/questions/41925157/logisticregression-unknown-label-type-continuous-using-sklearn-in-python
# lab_enc = preprocessing.LabelEncoder()
# y_train = lab_enc.fit_transform(y_train)
# y_test = lab_enc.fit_transform(y_test)
y_train = y_train.astype('int64')
y_test = y_test.astype('int64')
print('Output X_train df info:')
print(X_train.info(), '\n')
print('Output X_test df info:')
print(X_test.info(), '\n')
print('Output y_train df info:')
print(y_train.dtype, '\n')
print('Output y_test df info:')
print(y_test.dtype, '\n')
