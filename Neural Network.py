import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Loading MNIST Fashion Dataset from keras

fashion_mnist=keras.datasets.fashion_mnist
(tr_in_images,tr_out_images),(ts_in_images,ts_out_images)=fashion_mnist.load_data()
# print(tr_in_images.shape) (60000, 28, 28)
# print(tr_in_images[0,24,12]) 244 - Pixel value 0 being black 255 being white
# print(tr_out_images.shape) (60000,)
# print(tr_out_images[:20]) # First 20 training labels
class_names=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Data Preprocessing
tr_in_images=tr_in_images/255.0
ts_in_images=ts_in_images/255.0


# Buliding model

model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])

# Compile the model

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Training model

model.fit(tr_in_images,tr_out_images,epochs=5)

# Testing model
test_loss,test_acc=model.evaluate(ts_in_images,ts_out_images,verbose=1)

print('Test accuracy: ',test_acc)
print('Loss : ',test_loss)

# Prediction
predictions=model.predict(ts_in_images)
print('Prediction :',class_names[np.argmax(predictions[5])])
print('Real type of cloths: ',class_names[ts_out_images[5]])
plt.figure()
plt.imshow(ts_in_images[5])
plt.title('Image number 5')
plt.colorbar()
print('Prediction :',class_names[np.argmax(predictions[10])])
print('Real type of cloths: ',class_names[ts_out_images[10]])
plt.figure()
plt.imshow(ts_in_images[10])
plt.title('Image number 10')
plt.colorbar()
plt.show()