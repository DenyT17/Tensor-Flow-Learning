## Tensor Flow Learing  🧮

## Technologies 💡
![Tensor Flow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white)
![PyCharm](https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=black&labelColor=green)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
## Description 
This project will be my notebook where I will put my progress in learning Tensor Flow. 
I will create programs based on the tutorial that can be found [TensorFlow 2.0 Complete Course - Python Neural Networks for Beginners Tutorial.](https://www.youtube.com/watch?v=tPYj3fFJGjk&t=7743s)

As progress is made, the repository will be updated.

## Dataset📁
Dataset used in this project you can find below:

[Training Data](https://storage.googleapis.com/tf-datasets/titanic/train.csv)

[Test Data](https://storage.googleapis.com/tf-datasets/titanic/eval.csv)

## Linear Regression 

#### First, I create basic graphs, thanks to which I can gain some information about dataset.
![Figure_1](https://user-images.githubusercontent.com/122997699/217101786-d173b928-ab50-4772-9297-7b3ca6ddefc9.png)
![Figure_2](https://user-images.githubusercontent.com/122997699/217101883-77f74f5e-8f84-40e3-a652-9c677df81a42.png)

#### Next I must converting categorical data into numerical. I do this thanks to this for loop:
 ```python
for feature_name in Categorical_columns:
  vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
for feature_name in Numeric_columns:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
 ```
 Categorical_columns and Numeric_columns are tables with name of each column of dataset. 
 #### Then I create a function that will turn the input into an object data.Dataset required by TensorFlow model.
 ##### - Batch size is parameter about the amount of data in one "packet"
 ##### - Thanks to Shuffle parameter I can shuffle data
 ##### - Epochs number specifies how many times the model will retrieve the same data.
  ```python
 def make_input_fn(data_df,label_df,num_epochs=10,shuffle=True,batch_size=32):
    def input_function():
        ds=tf.data.Dataset.from_tensor_slices((dict(data_df),label_df))
        if shuffle:
            ds=ds.shuffle(1000)
        ds=ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function
  ```
  ##### Use of make_input_fn function:
```python
train_input_fn=make_input_fn(dftrain,train_output)
eval_input_fn=make_input_fn(dftest,test_output,num_epochs=1,shuffle=False)
```
#### Now, I must create model. When choosing the right model, I have to provide pre-created and saved in feature_columns, information about the data that will be provided, 
```python
linear_est=tf.estimator.LinearClassifier(feature_columns=feature_columns)
```
#### Finally, I can start training the model.
```python
linear_est.train(train_input_fn)
result=linear_est.evaluate(eval_input_fn)
```
