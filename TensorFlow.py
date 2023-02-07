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
plt.ylabel('Number')
fig,ax=plt.subplots(figsize=(10,7))
dftrain.sex.value_counts().plot(kind='bar')
plt.title('Number of woman and man',fontsize=20)
plt.xlabel('Woman/Man')
plt.xticks(rotation=0)
plt.ylabel('Number')
plt.show()
# Converting categorical data into numerical
Categorical_columns = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
Numeric_columns = ['age', 'fare']
feature_columns = []
for feature_name in Categorical_columns:
  vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
for feature_name in Numeric_columns:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

# Creating input function
def make_input_fn(data_df,label_df,num_epochs=100,shuffle=True,batch_size=32):
    def input_function():
        ds=tf.data.Dataset.from_tensor_slices((dict(data_df),label_df))
        if shuffle:
            ds=ds.shuffle(1000)
        ds=ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

train_input_fn=make_input_fn(dftrain,train_output)
eval_input_fn=make_input_fn(dftest,test_output,num_epochs=1,shuffle=False)

# Creating the model
linear_est=tf.estimator.LinearClassifier(feature_columns=feature_columns)

# Training the Model
linear_est.train(train_input_fn)
result=linear_est.evaluate(eval_input_fn)
clear_output()
print('Accuracy is: ',result['accuracy'])
result=list(linear_est.predict(eval_input_fn))
print('Probability of survive is equal =  ',result[10]['probabilities'][1])
print('Did the passenger survive? ( 0 - No, 1 - Yes) : ',
       test_output[10])
print('Probability of survive is equal =  ',result[1]['probabilities'][1])
print('Did the passenger survive? ( 0 - No, 1 - Yes) : ',
       test_output[1])

