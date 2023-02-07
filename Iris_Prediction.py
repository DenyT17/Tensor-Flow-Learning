from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v2.feature_column as fc
from IPython.display import clear_output
from six.moves import urllib
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
# Importing data from csv files
training_data=pd.read_csv('iris_training.csv')
test_data=pd.read_csv('iris_test.csv')
# Renaming column name
training_data=training_data.rename(
    columns={'120':'SepalLength',
             '4':'SepalWidth',
             'setosa':'PetalLength',
             'versicolor':'PetalWidth',
             'virginica':'Species'})
test_data=test_data.rename(
    columns={'30':'SepalLength',
             '4':'SepalWidth',
             'setosa':'PetalLength',
             'versicolor':'PetalWidth',
             'virginica':'Species'})

# Creating output data
test_output=test_data.pop('Species')
training_output=training_data.pop('Species')
# Creating input function
def input_function(input_data,output_data,training=True,batch_size=250):
    dataset=tf.data.Dataset.from_tensor_slices((dict(input_data),output_data))
    if training:
        dataset=dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)

# Creating feature columns
feature_columns=[]
for key in training_data.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key))

# Building model
classifier=tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[30,10],
    n_classes=3)

# Training model
classifier.train(
    input_fn=lambda:input_function(
    training_data,training_output,training=True),
                 steps=5000)
# Predict Species, for a flower with user-specified dimensions
def input_fn(features,batch_size=256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features= ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict={}

print('Please, give me the following dimensions od the flower: ')
for feature in features:
    valid = True
    while valid:
        val=input(feature + ": ")
        if not val.isdigit():valid=False
    predict[feature]=[float(val)]
predictions=classifier.predict(input_fn=lambda:input_fn(predict))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

print('According of the prediction, your flower is "{}",'
      ' and probability of it is ({:.1f}%)'.format(SPECIES[class_id], 100 * probability))

