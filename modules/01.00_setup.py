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
