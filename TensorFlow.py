import tensorflow as tf
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v2.feature_column as fc
from IPython.display import clear_output
from six.moves import urllib


# Linear Regresion
# Importing train and test data from csv file
dftrain=pd.read_csv('eval.csv')
dftest=pd.read_csv('train.csv')

train_output=dftrain.pop('survived')
test_output=dftest.pop('survived')

# Basic information about dataset
plt.style.use('bmh')
fig,ax=plt.subplots(figsize=(10,7))
plt.hist(dftrain['age'],bins=20)
plt.title('Histogram of passanger age',fontsize=20)
plt.xlabel('Age')
plt.ylabel('Percent')
fig,ax=plt.subplots(figsize=(10,7))
dftrain.sex.value_counts().plot(kind='bar')
plt.title('Number of woman and man',fontsize=20)
plt.xlabel('Woman/Man')
plt.xticks(rotation=0)
plt.ylabel('Number')
plt.show()

Categorical_columns = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
Numeric_columns = ['age', 'fare']

feature_columns = []
for feature_name in Categorical_columns:
  vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in Numeric_columns:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)