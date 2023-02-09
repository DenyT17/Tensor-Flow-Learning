import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
# Load data
(tr_in_images,tr_out_images),(ts_in_images,ts_out_images)=datasets.cifar10.load_data()
# Data Preprocessing
tr_in_images,ts_in_images=tr_in_images/255.0,ts_in_images/255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Building Convolutions Base
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Adding Dense Layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# Training
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(tr_in_images, tr_out_images, epochs=4,
                    validation_data=(ts_in_images, ts_out_images))

# Evaluating the Model
test_loss,test_acc=model.evaluate(ts_in_images,
                                  ts_out_images,
                                  verbose=2)
predictions=model.predict(ts_in_images)
plt.figure()
print('First Prediction :',class_names[np.argmax(predictions[10])])
print('Real type of first clothes: ',class_names[ts_out_images[10][0]])
pred_img = 10  # change this to look at other images
plt.imshow(ts_in_images[pred_img] ,cmap=plt.cm.binary)
plt.title('First images')
plt.show()
plt.figure()
print('Second Prediction :',class_names[np.argmax(predictions[13])])
print('Real type of second clothes: ',class_names[ts_out_images[13][0]])
pred_img = 13 # change this to look at other images
plt.imshow(ts_in_images[pred_img] ,cmap=plt.cm.binary)
plt.title('Second images')
plt.show()
plt.figure()
print('Third Prediction :',class_names[np.argmax(predictions[113])])
print('Real type of third clothes: ',class_names[ts_out_images[113][0]])
pred_img = 113 # change this to look at other images
plt.imshow(ts_in_images[pred_img] ,cmap=plt.cm.binary)
plt.title('Third images')
plt.show()

