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
In this moment repository include examples : 

-Linear Regression

-DNNClassifier

-Neural Networks

-Deep Computer Vision

-Data Augmentation

-Pretrained Model

As progress is made, the repository will be updated.

## Dataset📁
Datasets used in this project you can find below:

Titanic dataset:

[Training Data](https://storage.googleapis.com/tf-datasets/titanic/train.csv)

[Test Data](https://storage.googleapis.com/tf-datasets/titanic/eval.csv)


Iris dataset:

[Training Data](https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv)

[Test Data](https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv)
## Linear Regression 
The entire script of the program is placed in the file TensorFlow.py

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
#### At this moment accuracy is approx 0.8 Fot this reason sometimes prediction doesn't corect. I put below examples of correct and incorrect prediction. 
![image](https://user-images.githubusercontent.com/122997699/217205100-fdab11ae-4eac-49fd-afc9-71693197057a.png)

## Classification
The entire script of the program is placed in the file Iris_Prediction.py

In this case i use Iris dataset and DNNClassifier (Deep Neural Network) model. The task of this part of the project is to predict the Iris flower species with the dimensions provided by the user.
#### First I rename column od training and test dataset, and creating output data. 
```python
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
```
#### In next step i define input_function. Unlike to Linear Regression i create only one function, wchich returnc resulting components with additional outer dimension, which is batch_size. In this case, when i will training model i will must use lambda fnction. 
```python
def input_function(input_data,output_data,training=True,batch_size=250):
    dataset=tf.data.Dataset.from_tensor_slices((dict(input_data),output_data))
    if training:
        dataset=dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)
```
#### Because all column of input data has numeric value, I create feature columns thanks to keys() method.
for key in training_data.keys():
```python
    feature_columns.append(tf.feature_column.numeric_column(key=key))
```
#### In next step i create DNNClassifier model. In addition to the feature_columns, i must declare:

- hidden_units - number hidden units per layer
- n_classes - number of label classes
```python
classifier=tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[30,10],
    n_classes=3)
```
#### Now i can training model. As I wrote before i must use lambda function. 
```python
classifier.train(
    input_fn=lambda:input_function(
    training_data,training_output,training=True),
                 steps=5000)
```
#### To be able to make a prediction of species with the dimensions provided by the user, I must create another input function. 
```python
features= ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
def input_fn(features,batch_size=256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)
```
#### In next step i creat dictionary with dimension provided by the usser. 
```python
for feature in features:
    valid = True
    while valid:
        val=input(feature + ": ")
        if not val.isdigit():valid=False
    predict[feature]=[float(val)]
```
#### At the end I can make a prediction of species.
```python
predictions=classifier.predict(input_fn=lambda:input_fn(predict))
```
#### Prediction example:
![image](https://user-images.githubusercontent.com/122997699/217300798-8885570d-8b85-4197-84a7-f59e19d7033c.png)

## Neural Networks 
In this part i use Neural Network to predict type of clothes on images.

The entire script of the program is placed in the file Iris_Prediction.py
#### First I upload MNIST Fashion Dataset from keras.
```python
fashion_mnist=keras.datasets.fashion_mnist 
(tr_in_images,tr_out_images),(ts_in_images,ts_out_images)=fashion_mnist.load_data()
```
Training input data include 60000 pictures with dimension 28x28 px. Every pixel is represented  number between 0 to 255, when 0 is black and 255 is white.

Output data include number betwen 0 to 9. Every number is represented one type of clothes,  from the table: 
```python
class_names=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```
#### To facilitate the model learning process,i use  data processing. I change range of every pixel from 0-255 to 0-1. 
```python
tr_in_images=tr_in_images/255.0
ts_in_images=ts_in_images/255.0
```
#### Now I can build model. I will use Sequential model from Keras. I must declare number of neurons in input, hidden and output layers. Additional i must choose activation functions.
```python
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), 
    keras.layers.Dense(128,activation='relu'), 
    keras.layers.Dense(10,activation='softmax') 
])
```
In input layer will be 784 neurons, becouse every images have 784 pixels. 
Number of neurons in hidden layer is 128, and activation function is Rectified Linear Unit
In output layer will be 10 neurons, because output data has 10 different type of clothes. Aditional i chose Relu softmax activation function.

#### Next step is compile the model. I must choose loss function, optimizer and metrics.
```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

#### Now I can training model.
```python
model.fit(tr_in_images,tr_out_images,epochs=5)
```
#### At the end I testing my model. Additional i display test accuracy and calue of loss function. 
```python
test_loss,test_acc=model.evaluate(ts_in_images,ts_out_images,verbose=1)
print('Test accuracy: ',test_acc)
print('Loss : ',test_loss)
```
![image](https://user-images.githubusercontent.com/122997699/217659852-6a78fdae-b78d-4af3-af53-5432def5f8a9.png)

#### Now i choose picture number 5 and 10 from test data, and make prediction. 
```python
# Prediction
predictions=model.predict(ts_in_images)
print(class_names[np.argmax(predictions[5])])
print(class_names[ts_out_images[5]])
plt.figure()
plt.imshow(ts_in_images[5])
plt.colorbar()
print(class_names[np.argmax(predictions[10])])
print(class_names[ts_out_images[10]])
plt.figure()
plt.imshow(ts_in_images[10])
plt.colorbar()
plt.show()
```
##### Images number 5:
![Figure_1_NN](https://user-images.githubusercontent.com/122997699/217661822-36fb5735-4a42-4c8c-9f24-ebed692f78d4.png)
##### Images number 10:
![Figure_2_NN](https://user-images.githubusercontent.com/122997699/217661744-79aadc04-0160-4909-ba93-ae52fb67275b.png)
##### Confirmation of correct prediction.
![image](https://user-images.githubusercontent.com/122997699/217661719-d5159e7d-a222-4e2a-a076-5ce470da9edb.png)

## Deep Computer Vision
The entire script of the program is placed in the file Deep Computer Vision.py
Deep Computer Vision work thanks to convolutional neural network. The purpose of convolutional layers is to find patterns from images that can be used to classify an image or part of an image.

##### First I load data and normalize pixel values to be between 0 and 1.
```python
(tr_in_images,tr_out_images),(ts_in_images,ts_out_images)=datasets.cifar10.load_data()
tr_in_images,ts_in_images=tr_in_images/255.0,ts_in_images/255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
```
##### In next step I nuild model. 
```python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
```
I choose Sequential model.
In first layer I declarate shape of input data. Additional I put number and size of filter. I also apply the activation function relu to the output of each convolution operation. 
In the next line, I do a max pooling operation using 2x2 samples and step 2.
In second and third layers I put number and size od fileter and choos the same activation function. After each of layer I do pooling operation.

##### Now similary as in Neural Network I add Dense layer.
```python
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
```

##### In this moment I can train my model. I choose loss function, optimizer and metrics. 
```python
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(tr_in_images, tr_out_images, epochs=4,
                    validation_data=(ts_in_images, ts_out_images))
```

##### At the end i can evaluate model and make prediction. 
```python
test_loss,test_acc=model.evaluate(ts_in_images,
                                  ts_out_images,
                                  verbose=2)
predictions=model.predict(ts_in_images)
```
##### Examples of prediction: 
###### Prediction images:
![Figure_1](https://user-images.githubusercontent.com/122997699/217955106-f6cc92e8-c023-4cd7-93ad-c9b59ab074f1.png)
![Figure_2](https://user-images.githubusercontent.com/122997699/217955111-7eacfb67-3c73-4e21-9f49-0acdcf749656.png)
![Figure_3](https://user-images.githubusercontent.com/122997699/217955109-0710b59b-29b6-4c9d-b68f-39d340f9c3b1.png)

###### And prediction : 
![image](https://user-images.githubusercontent.com/122997699/217955551-40651965-c295-4ae3-89fb-35f03c12a827.png)

Accuracy is 0.6725 so prediction sometimes does not correct. 

## Data Augmentation
The entire script of the program is placed in the file Deep Data Augmentation.py
Thanks to Data Augmentation I can multiple number of image in my dataset.ImageDataGenerator from keras create new images based on my images. New images are created through edit old images based on operation like: horizontal flip, height and width shift, zoom, rotation.

##### First I must create data generator object. In this step i declare parameters, which will be respected along generation new images. 
```python
datagen=ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest')
```
##### The ImageDataGenerator class has three methods:
- flow()
- flow_from_directory()
- flow_from_dataframe()
In this example I use first method. For this reason I must prepare the data accordingly.
```python
test_img=tr_in_images[15]
img=imag.img_to_array(test_img)
img=img.reshape((1,)+img.shape)
```
##### Now I create for loop thanks to witch I generate new image and display they. 
```python
i=0
for batch in datagen.flow(img,save_prefix='test',save_format='jpeg'):
    plt.figure(i)
    plot=plt.imshow(imag.img_to_array(batch[0]))
    i+=1
    if i > 5:
        break
plt.show()
```
##### Examples of new images:
![Figure_01](https://user-images.githubusercontent.com/122997699/218143830-93f1469c-a304-4b01-833e-8754b74b0808.png)
![Figure_11](https://user-images.githubusercontent.com/122997699/218143850-e33bb762-e303-461b-a0d9-fd8d3b749f8e.png)
![Figure_5](https://user-images.githubusercontent.com/122997699/218143866-3bf26b28-a514-4a6a-b9db-c171d7964558.png)

## Pretrained Model 
The entire script of the program is placed in the file Deep Pretrained Model.py
